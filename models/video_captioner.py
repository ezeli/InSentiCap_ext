# coding:utf8
import math
import torch
from torch import nn
from collections import defaultdict
from copy import deepcopy

from .captioner import BeamCandidate, MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding, Encoder, \
    LabelSmoothingCriterion


class DecoderLayer(nn.Module):
    def __init__(self, settings):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(settings)

        self.mh_atts = nn.ModuleDict()
        for feat_type in ['two_d', 'three_d', 'audio', 'sem']:
            self.mh_atts[feat_type] = nn.ModuleDict({
                'con': MultiHeadAttention(settings),
                'sen': MultiHeadAttention(settings),
            })

        self.feed_forward = PositionwiseFeedForward(settings)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(settings['d_model']) for _ in range(10)])
        self.drop = nn.Dropout(settings['dropout_p'])

        self.fuse_scores = {}

    def _sublayer(self, x, sublayer, n):
        return self.drop(sublayer(self.layer_norms[n](x)))  # x + self.drop(sublayer(self.layer_norms[n](x)))

    def _add_res_connection(self, x, sublayer, n):
        return x + self._sublayer(x, sublayer, n)  # x + self.drop(sublayer(self.layer_norms[n](x)))

    def _fuse_gate(self, x, att_feats):
        scores = att_feats.matmul(x.unsqueeze(-1)) / math.sqrt(att_feats.shape[-1])  # [bs, seq_len, 2 or 4, 1]
        scores = scores.transpose(2, 3).softmax(-1)  # [bs, seq_len, 1, 2 or 4]
        att_feats = scores.matmul(att_feats).squeeze(2)  # [bs, seq_len, d_model]
        # att_feats = att_feats.mean(2)
        return att_feats, scores.squeeze(2)

    def forward(self, captions, seq_masks, sem_feats, vis_feats):
        captions = self._add_res_connection(captions, lambda x: self.self_att(x, x, x, seq_masks), 0)

        cpt_words, senti_words = sem_feats
        cpt_words = self._sublayer(captions, lambda x: self.mh_atts['sem']['con'](x, cpt_words, cpt_words), 1)
        senti_words = self._sublayer(captions, lambda x: self.mh_atts['sem']['sen'](x, senti_words, senti_words), 2)
        sem_feats = torch.stack([cpt_words, senti_words], dim=2)  # [bs, seq_len, 2, d_model]
        sem_feats, sem_scores = self._fuse_gate(captions, sem_feats)
        self.fuse_scores['sem_scores'] = sem_scores
        if vis_feats:
            vis_fuse_feats = []
            ln_idx = 3
            for feat_type, (con_feats, sen_feats, feat_masks) in vis_feats.items():
                con_feats = self._sublayer(
                    captions, lambda x: self.mh_atts[feat_type]['con'](x, con_feats, con_feats, feat_masks), ln_idx)
                sen_feats = self._sublayer(
                    captions, lambda x: self.mh_atts[feat_type]['sen'](x, sen_feats, sen_feats, feat_masks), ln_idx+1)
                ln_idx += 2
                f_feats = torch.stack([con_feats, sen_feats], dim=2)  # [bs, seq_len, 2, d_model]
                f_feats, f_scores = self._fuse_gate(captions, f_feats)
                self.fuse_scores[f'{feat_type}_scores'] = sem_scores
                vis_fuse_feats.append(f_feats)

            fuse_feats = captions + (sem_feats + sum(vis_fuse_feats)) / (1 + len(vis_fuse_feats))
        else:
            fuse_feats = captions + sem_feats

        return self._add_res_connection(fuse_feats, self.feed_forward, -1)


class Decoder(nn.Module):
    def __init__(self, settings):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(settings) for _ in range(settings['N_dec'])])
        self.layer_norm = nn.LayerNorm(settings['d_model'])

    def forward(self, captions, seq_masks, sem_feats, vis_feats):
        for layer in self.layers:
            captions = layer(captions, seq_masks, sem_feats, vis_feats)
        return self.layer_norm(captions)


class Captioner(nn.Module):
    def __init__(self, idx2word, sentiment_categories, settings):
        super(Captioner, self).__init__()
        self.idx2word = idx2word
        self.pad_id = idx2word.index('<PAD>')
        self.unk_id = idx2word.index('<UNK>')
        self.sos_id = idx2word.index('<SOS>') if '<SOS>' in idx2word else self.pad_id
        self.eos_id = idx2word.index('<EOS>') if '<EOS>' in idx2word else self.pad_id
        self.neu_idx = sentiment_categories.index('neutral')
        self.max_seq_len = settings['max_seq_len']

        self.d_model = settings['d_model']
        self.vocab_size = len(idx2word)
        drop = nn.Dropout(settings['dropout_p'])
        self.word_embed = nn.Sequential(nn.Embedding(self.vocab_size, settings['d_model'],
                                                     padding_idx=self.pad_id),
                                        nn.ReLU(),
                                        drop)
        self.pe = PositionalEncoding(settings)
        self.senti_label_embed = nn.Sequential(nn.Embedding(len(sentiment_categories), settings['d_model']),
                                               nn.ReLU(),
                                               drop)

        num_sentis = len(sentiment_categories)
        self.vis_encoders = nn.ModuleDict()
        for feat_type in ['two_d', 'three_d', 'audio']:
            self.vis_encoders[feat_type] = nn.ModuleDict({
                'emb': nn.Sequential(nn.Linear(settings[f'{feat_type}_feat_dim'], settings['d_model']),
                                     nn.ReLU(),
                                     nn.Dropout(settings['dropout_p'])),
                'con': Encoder(settings),
                'sen': nn.ModuleList([Encoder(settings) for _ in range(num_sentis)])
            })

        self.sem_encoder = nn.ModuleDict({
            'con': Encoder(settings),
            'sen': Encoder(settings),
        })

        self.decoder = Decoder(settings)
        self.classifier = nn.Linear(settings['d_model'], self.vocab_size)

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'xe')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, 'forward_' + mode)(*args, **kwargs)

    def get_mask_by_lengths(self, lengths):
        if lengths is None:
            return None
        mask = next(self.parameters()).new_zeros(len(lengths), max(lengths), dtype=torch.bool)  # bs*max_len
        for i, l in enumerate(lengths):
            mask[i, :l] = True
        mask = mask.unsqueeze(-2)
        return mask

    def _sequence_encode(self, captions, lengths=None):
        seq_len = captions.size(-1)
        captions = self.word_embed(captions)  # bs*seq_len*d_model
        captions = self.pe(captions)

        if lengths is None:
            seq_masks = captions.new_ones(1, seq_len, seq_len, dtype=torch.bool).tril(diagonal=0)  # 1*seq_len*seq_len
        else:
            assert seq_len == max(lengths)
            seq_masks = self.get_mask_by_lengths(lengths)  # bs*1*seq_len
            seq_masks = seq_masks & seq_masks.new_ones(1, seq_len, seq_len).tril(diagonal=0)  # bs*seq_len*seq_len
        return captions, seq_masks

    def _vis_encode(self, two_d_feats_tensor, two_d_feats_lengths,
                    three_d_feats_tensor, three_d_feats_lengths,
                    audio_feats_tensor, audio_feats_lengths,
                    senti_labels):
        res = {}
        for feat_type, feat_tensor, feat_len in [('two_d', two_d_feats_tensor, two_d_feats_lengths),
                                                 ('three_d', three_d_feats_tensor, three_d_feats_lengths),
                                                 ('audio', audio_feats_tensor, audio_feats_lengths)]:
            feat_tensor = feat_tensor.reshape(feat_tensor.size(0), -1, feat_tensor.size(-1))
            feat_masks = self.get_mask_by_lengths(feat_len)
            emb_feats = self.vis_encoders[feat_type]['emb'](feat_tensor)
            con_feats = self.vis_encoders[feat_type]['con'](emb_feats, feat_masks)
            sen_feats = []
            for i, senti_label in enumerate(senti_labels):
                senti_label = int(senti_label)
                sen_feat = self.vis_encoders[feat_type]['sen'][senti_label](emb_feats[i:i + 1],
                                                                            feat_masks[i:i + 1] if feat_masks is not None else None)
                sen_feats.append(sen_feat)
            sen_feats = torch.cat(sen_feats, 0)
            res[feat_type] = [con_feats, sen_feats, feat_masks]

        return res

    def _sem_encode(self, cpt_words, senti_words):
        cpt_words = self.word_embed(cpt_words)
        cpt_words = self.sem_encoder['con'](cpt_words, None)

        senti_words = self.word_embed(senti_words)
        senti_words = self.sem_encoder['sen'](senti_words, None)
        return cpt_words, senti_words

    def _decode(self, captions, lengths, senti_labels, sem_feats, vis_feats=None):
        captions, seq_masks = self._sequence_encode(captions, lengths)  # bs*seq_len*d_model, bs*seq_len*seq_len
        captions = captions + senti_labels.unsqueeze(1)

        dec_out = self.decoder(captions, seq_masks, sem_feats, vis_feats)  # bs*seq_len*d_model
        dec_out = self.classifier(dec_out).log_softmax(dim=-1)  # bs*seq_len*vocab
        return dec_out

    def forward_xe(self, two_d_feats_tensor, two_d_feats_lengths,
                   three_d_feats_tensor, three_d_feats_lengths,
                   audio_feats_tensor, audio_feats_lengths,
                   senti_labels, cpt_words, senti_words, captions, lengths):
        vis_feats = self._vis_encode(two_d_feats_tensor, two_d_feats_lengths,
                                     three_d_feats_tensor, three_d_feats_lengths,
                                     audio_feats_tensor, audio_feats_lengths,
                                     senti_labels)
        senti_labels = self.senti_label_embed(senti_labels)  # [bs, d_model]
        sem_feats = self._sem_encode(cpt_words, senti_words)

        dec_out = self._decode(captions[:, :-1], lengths, senti_labels,
                               sem_feats, vis_feats)
        return dec_out

    def forward_seq2seq(self, senti_labels, cpt_words, senti_words, senti_captions, lengths):
        senti_labels = self.senti_label_embed(senti_labels)  # [bs, d_model]
        sem_feats = self._sem_encode(cpt_words, senti_words)

        dec_out = self._decode(senti_captions[:, :-1], lengths, senti_labels,
                               sem_feats)
        return dec_out

    def forward_rl(self, two_d_feats_tensor, two_d_feats_lengths,
                   three_d_feats_tensor, three_d_feats_lengths,
                   audio_feats_tensor, audio_feats_lengths,
                   senti_labels, cpt_words, senti_words, sample_max):
        batch_size = two_d_feats_tensor.size(0)

        vis_feats = self._vis_encode(two_d_feats_tensor, two_d_feats_lengths,
                                     three_d_feats_tensor, three_d_feats_lengths,
                                     audio_feats_tensor, audio_feats_lengths,
                                     senti_labels)
        senti_labels = self.senti_label_embed(senti_labels)  # [bs, d_model]
        sem_feats = self._sem_encode(cpt_words, senti_words)

        seq = two_d_feats_tensor.new_zeros((batch_size, self.max_seq_len), dtype=torch.long)
        seq_logprobs = two_d_feats_tensor.new_zeros((batch_size, self.max_seq_len))
        seq_masks = two_d_feats_tensor.new_zeros((batch_size, self.max_seq_len))
        it = two_d_feats_tensor.new_zeros(batch_size, dtype=torch.long).fill_(self.sos_id)  # first input <SOS>
        unfinished = it == self.sos_id
        pre_words = it.unsqueeze(1)  # bs*1
        for t in range(self.max_seq_len):
            logprobs = self._decode(pre_words, None, senti_labels,
                                    sem_feats, vis_feats)  # bs*seq_len*vocab
            logprobs = logprobs[:, -1]  # bs*vocab

            if sample_max:
                sample_logprobs, it = torch.max(logprobs, 1)
            else:
                prob_prev = torch.exp(logprobs)
                it = torch.multinomial(prob_prev, 1)
                sample_logprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
            it = it.reshape(-1).long()
            sample_logprobs = sample_logprobs.reshape(-1)

            seq_masks[:, t] = unfinished
            it = it * unfinished.type_as(it)  # bs
            seq[:, t] = it
            seq_logprobs[:, t] = sample_logprobs
            pre_words = torch.cat([pre_words, it.unsqueeze(1)], dim=1)  # bs*seq_len

            unfinished = unfinished * (it != self.eos_id)
            if unfinished.sum() == 0:
                break

        return seq, seq_logprobs, seq_masks

    def sample(self, two_d_feats_tensor, three_d_feats_tensor, audio_feats_tensor,
               senti_label, cpt_words, senti_words,
               beam_size=3, decoding_constraint=1):
        self.eval()
        two_d_feats_tensor = two_d_feats_tensor.unsqueeze(0)  # [1, num, att_feat]
        three_d_feats_tensor = three_d_feats_tensor.unsqueeze(0)  # [1, num, att_feat]
        audio_feats_tensor = audio_feats_tensor.unsqueeze(0)  # [1, num, att_feat]
        cpt_words = cpt_words.unsqueeze(0)  # [1, num]
        senti_words = senti_words.unsqueeze(0)  # [1, num]

        vis_feats = self._vis_encode(two_d_feats_tensor, None,
                                     three_d_feats_tensor, None,
                                     audio_feats_tensor, None,
                                     senti_label)
        senti_label = self.senti_label_embed(senti_label)  # [bs, d_model]
        sem_feats = self._sem_encode(cpt_words, senti_words)

        # log_prob_sum, log_prob_seq, word_id_seq, scores_seq
        candidates = [BeamCandidate(0., [], [self.sos_id], defaultdict(list))]
        for t in range(self.max_seq_len):
            tmp_candidates = []
            end_flag = True
            for candidate in candidates:
                log_prob_sum, log_prob_seq, word_id_seq, scores_seq = candidate
                if t > 0 and word_id_seq[-1] == self.eos_id:
                    tmp_candidates.append(candidate)
                else:
                    end_flag = False
                    pre_words = two_d_feats_tensor.new_tensor(word_id_seq, dtype=torch.long).unsqueeze(0)  # 1*seq_len
                    logprobs = self._decode(pre_words, None, senti_label,
                                            sem_feats, vis_feats)  # 1*seq_len*vocab
                    logprobs = logprobs[:, -1]  # 1*vocab
                    logprobs = logprobs.squeeze(0)  # vocab_size
                    if self.pad_id != self.eos_id:
                        logprobs[self.pad_id] += float('-inf')  # do not generate <PAD>, <SOS> and <UNK>
                        logprobs[self.sos_id] += float('-inf')
                        logprobs[self.unk_id] += float('-inf')
                    if decoding_constraint:  # do not generate last step word
                        logprobs[word_id_seq[-1]] += float('-inf')

                    fuse_score = defaultdict(float)
                    for layer in self.decoder.layers:
                        fuse_scores = layer.fuse_scores
                        for s_name in fuse_scores:
                            fuse_score[f'{s_name}_con'] += float(fuse_scores[s_name][0, -1, 0])
                            fuse_score[f'{s_name}_sen'] += float(fuse_scores[s_name][0, -1, 1])
                    for s_name, s_val in fuse_score.items():
                        scores_seq[s_name].append(s_val)

                    output_sorted, index_sorted = torch.sort(logprobs, descending=True)
                    for k in range(beam_size):
                        log_prob, word_id = output_sorted[k], index_sorted[k]  # tensor, tensor
                        log_prob = float(log_prob)
                        word_id = int(word_id)
                        tmp_candidates.append(BeamCandidate(log_prob_sum + log_prob,
                                                            log_prob_seq + [log_prob],
                                                            word_id_seq + [word_id],
                                                            deepcopy(scores_seq)))
            candidates = sorted(tmp_candidates, key=lambda x: x.log_prob_sum, reverse=True)[:beam_size]
            if end_flag:
                break

        # captions, scores
        caption = ' '.join([self.idx2word[idx] for idx in candidates[0].word_id_seq[1:-1]])
        fuse_scores = {}
        for s_name, s_vals in candidates[0].scores_seq.items():
            fuse_scores[s_name] = s_vals[:-1]
        score = candidates[0].log_prob_sum
        return caption, (fuse_scores, score)

    def get_optim_criterion(self, lr, weight_decay=0, smoothing=0.1):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay), \
               LabelSmoothingCriterion(smoothing), nn.MSELoss()  # xe, domain align
