# coding:utf8
import math
import torch
from torch import nn
from collections import namedtuple


BeamCandidate = namedtuple('BeamCandidate',
                           ['log_prob_sum', 'log_prob_seq', 'word_id_seq'])


class MultiHeadAttention(nn.Module):
    def __init__(self, settings):
        super(MultiHeadAttention, self).__init__()
        assert settings['d_model'] % settings['h'] == 0
        self.h = settings['h']
        self.d_k = settings['d_model'] // settings['h']
        self.linears = nn.ModuleList([nn.Linear(settings['d_model'], settings['d_model']) for _ in range(4)])
        self.drop = nn.Dropout(settings['dropout_p'])

    def _attention(self, query, key, value, mask=None):
        # Scaled Dot Product Attention
        scores = query.matmul(key.transpose(-2, -1)) \
                 / math.sqrt(self.d_k)  # bs*h*n1*n2
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = scores.softmax(-1)
        return scores.matmul(value)  # bs*h*n1*d_k

    def forward(self, query, key, value, mask=None):
        """
            query: bs*n1*d_model
            key: bs*n2*d_model
            value: bs*n2*d_model
            mask: bs*(n2 or 1)*n2
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # bs*1*(n2 or 1)*n2
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [self.drop(l(x)).reshape(batch_size, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears[:3], (query, key, value))]  # bs*h*n*d_k

        # 2) Apply attention on all the projected vectors in batch.
        x = self._attention(query, key, value, mask)  # bs*h*n1*d_k

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).reshape(batch_size, -1, self.h * self.d_k)  # bs*n1*d_model
        return self.drop(self.linears[-1](x))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, settings):
        super(PositionwiseFeedForward, self).__init__()
        self.pff = nn.Sequential(
            nn.Linear(settings['d_model'], settings['d_ff']),
            nn.ReLU(),
            # nn.Dropout(settings['dropout_p']),
            nn.Linear(settings['d_ff'], settings['d_model'])
        )

    def forward(self, x):
        return self.pff(x)


class PositionalEncoding(nn.Module):
    def __init__(self, settings):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(settings['max_seq_len'], settings['d_model'])
        position = torch.arange(0, settings['max_seq_len']).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, settings['d_model'], 2).float() *
                             -(math.log(10000.0) / settings['d_model']))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.drop = nn.Dropout(settings['dropout_p'])

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.drop(x)


class EncoderLayer(nn.Module):
    def __init__(self, settings):
        super(EncoderLayer, self).__init__()
        self.multi_head_att = MultiHeadAttention(settings)
        self.feed_forward = PositionwiseFeedForward(settings)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(settings['d_model']) for _ in range(2)])
        self.drop = nn.Dropout(settings['dropout_p'])

    def _add_res_connection(self, x, sublayer, n):
        return x + self.drop(sublayer(self.layer_norms[n](x)))  # x + self.drop(sublayer(self.layer_norms[n](x)))

    def forward(self, x, mask):
        x = self._add_res_connection(x, lambda x: self.multi_head_att(x, x, x, mask), 0)
        return self._add_res_connection(x, self.feed_forward, 1)


class Encoder(nn.Module):
    def __init__(self, settings):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(settings) for _ in range(settings['N_enc'])])
        self.layer_norm = nn.LayerNorm(settings['d_model'])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, settings):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(settings)

        self.vis_con_att = MultiHeadAttention(settings)
        self.vis_sen_att = MultiHeadAttention(settings)
        self.sem_con_att = MultiHeadAttention(settings)
        self.sem_sen_att = MultiHeadAttention(settings)

        self.feed_forward = PositionwiseFeedForward(settings)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(settings['d_model']) for _ in range(6)])
        self.drop = nn.Dropout(settings['dropout_p'])

    def _sublayer(self, x, sublayer, n):
        return self.drop(sublayer(self.layer_norms[n](x)))  # x + self.drop(sublayer(self.layer_norms[n](x)))

    def _add_res_connection(self, x, sublayer, n):
        return x + self._sublayer(x, sublayer, n)  # x + self.drop(sublayer(self.layer_norms[n](x)))

    def _fuse_gate(self, x, att_feats):
        scores = att_feats.matmul(x.unsqueeze(-1))  # [bs, seq_len, 2 or 4, 1]
        scores = scores.transpose(2, 3).softmax(-1)  # [bs, seq_len, 1, 2 or 4]
        att_feats = scores.matmul(att_feats).squeeze(2)  # [bs, seq_len, d_model]
        # att_feats = att_feats.mean(2)
        return att_feats

    def forward(self, captions, seq_masks, cpt_words, senti_words, region_feats, spatial_feats):
        captions = self._add_res_connection(captions, lambda x: self.self_att(x, x, x, seq_masks), 0)

        cpt_words = self._sublayer(captions, lambda x: self.sem_con_att(x, cpt_words, cpt_words), 1)
        senti_words = self._sublayer(captions, lambda x: self.sem_sen_att(x, senti_words, senti_words), 2)
        sem_feats = torch.stack([cpt_words, senti_words], dim=2)  # [bs, seq_len, 2, d_model]
        sem_feats = self._fuse_gate(captions, sem_feats)
        if region_feats is not None:
            region_feats = self._sublayer(captions, lambda x: self.vis_con_att(x, region_feats, region_feats), 3)
            spatial_feats = self._sublayer(captions, lambda x: self.vis_sen_att(x, spatial_feats, spatial_feats), 4)
            vis_feats = torch.stack([region_feats, spatial_feats], dim=2)  # [bs, seq_len, 4, d_model]
            vis_feats = self._fuse_gate(captions, vis_feats)
            fuse_feats = (sem_feats + vis_feats) / 2
        else:
            fuse_feats = sem_feats
        fuse_feats = captions + fuse_feats

        return self._add_res_connection(fuse_feats, self.feed_forward, -1)


class Decoder(nn.Module):
    def __init__(self, settings):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(settings) for _ in range(settings['N_dec'])])
        self.layer_norm = nn.LayerNorm(settings['d_model'])

    def forward(self, captions, seq_masks, cpt_words, senti_words, region_feats, spatial_feats):
        for layer in self.layers:
            captions = layer(captions, seq_masks, cpt_words, senti_words, region_feats, spatial_feats)
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
        # self.cpt2fc = nn.Sequential(nn.Linear(settings['d_model'], settings['d_model']),
        #                             nn.ReLU())
        self.vis_con_encoder = nn.ModuleDict({
            'emb': nn.Sequential(nn.Linear(settings['att_feat_dim'], settings['d_model']),
                                 nn.ReLU(),
                                 drop),
            'enc': Encoder(settings)
        })

        self.vis_sen_encoder = nn.ModuleList([
            self._get_vis_sen_head(settings) for _ in range(len(sentiment_categories))
        ])

        self.sem_con_encoder = Encoder(settings)
        self.sem_sen_encoder = Encoder(settings)

        self.decoder = Decoder(settings)
        self.classifier = nn.Linear(settings['d_model'], self.vocab_size)

    def _get_vis_sen_head(self, settings):
        drop = nn.Dropout(settings['dropout_p'])
        convs = nn.Sequential()
        in_channels = settings['att_feat_dim']
        for i in range(2):
            convs.add_module(
                'conv_%d' % i, nn.Conv2d(in_channels, in_channels // 2, 3))
            in_channels //= 2
        convs.add_module('relu1', nn.ReLU())
        convs.add_module('drop1', drop)
        for i in range(2, 4):
            convs.add_module(
                'conv_%d' % i, nn.Conv2d(in_channels, in_channels, 3))
        convs.add_module('relu2', nn.ReLU())
        convs.add_module('drop2', drop)
        vis_head = nn.ModuleDict({
            'conv': convs,
            'emb': nn.Sequential(nn.Linear(in_channels, settings['d_model']),
                                 nn.ReLU(),
                                 drop),
            'ln': nn.LayerNorm(settings['d_model'])
        })
        return vis_head

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'xe')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, 'forward_' + mode)(*args, **kwargs)

    def _sequence_encode(self, captions, lengths=None):
        seq_len = captions.size(-1)
        captions = self.word_embed(captions)  # bs*seq_len*d_model
        captions = self.pe(captions)

        if lengths is None:
            seq_masks = captions.new_ones(1, seq_len, seq_len, dtype=torch.bool).tril(diagonal=0)  # 1*seq_len*seq_len
        else:
            assert seq_len == max(lengths)
            seq_masks = captions.new_zeros(len(lengths), seq_len, dtype=torch.bool)  # bs*seq_len
            for i, l in enumerate(lengths):
                seq_masks[i, :l] = True
            seq_masks = seq_masks.unsqueeze(-2)  # bs*1*seq_len
            seq_masks = seq_masks & seq_masks.new_ones(1, seq_len, seq_len).tril(diagonal=0)  # bs*seq_len*seq_len
        return captions, seq_masks

    def _vis_encode(self, region_feats, spatial_feats, senti_labels):
        region_feats = region_feats.reshape(region_feats.size(0), -1, region_feats.size(-1))
        region_feats = self.vis_con_encoder['emb'](region_feats)
        region_feats = self.vis_con_encoder['enc'](region_feats, None)

        map_dim = int((spatial_feats.numel() // spatial_feats.size(0) // spatial_feats.size(-1)) ** 0.5)
        spatial_feats = spatial_feats.reshape(spatial_feats.size(0), map_dim, map_dim, spatial_feats.size(-1))
        spatial_feats = spatial_feats.permute(0, 3, 1, 2).contiguous()  # [bz, fc_feat_dim, 14, 14]
        s_feats = []
        for i, senti_label in enumerate(senti_labels):
            senti_label = int(senti_label)
            head = self.vis_sen_encoder[senti_label]
            s_feat = head['conv'](spatial_feats[i:i+1])  # [1, 512, 6, 6]
            s_feat = s_feat.squeeze(0).reshape(s_feat.size(1), -1).permute(1, 0).contiguous()  # [36, 512]
            s_feat = head['emb'](s_feat)  # [36, 512]
            s_feat = head['ln'](s_feat)  # [36, 512]
            s_feats.append(s_feat)
        spatial_feats = torch.stack(s_feats, dim=0)  # [bz, 36, 512]
        return region_feats, spatial_feats

    def _sem_encode(self, cpt_words, senti_words):
        cpt_words = self.word_embed(cpt_words)
        cpt_words = self.sem_con_encoder(cpt_words, None)

        senti_words = self.word_embed(senti_words)
        senti_words = self.sem_sen_encoder(senti_words, None)
        return cpt_words, senti_words

    def _decode(self, captions, lengths, senti_labels, cpt_words, senti_words, region_feats=None, spatial_feats=None):
        captions, seq_masks = self._sequence_encode(captions, lengths)  # bs*seq_len*d_model, bs*seq_len*seq_len
        captions = captions + senti_labels.unsqueeze(1)

        dec_out = self.decoder(captions, seq_masks, cpt_words, senti_words, region_feats, spatial_feats)  # bs*seq_len*d_model
        dec_out = self.classifier(dec_out).log_softmax(dim=-1)  # bs*seq_len*vocab
        return dec_out

    def forward_xe(self, region_feats, spatial_feats, senti_labels, cpt_words, senti_words, captions, lengths):
        region_feats, spatial_feats = self._vis_encode(region_feats, spatial_feats, senti_labels)
        senti_labels = self.senti_label_embed(senti_labels)  # [bs, d_model]
        cpt_words, senti_words = self._sem_encode(cpt_words, senti_words)

        dec_out = self._decode(captions[:, :-1], lengths, senti_labels,
                               cpt_words, senti_words, region_feats, spatial_feats)
        return dec_out

    def forward_seq2seq(self, senti_labels, cpt_words, senti_words, senti_captions, lengths):
        senti_labels = self.senti_label_embed(senti_labels)  # [bs, d_model]
        cpt_words, senti_words = self._sem_encode(cpt_words, senti_words)

        dec_out = self._decode(senti_captions[:, :-1], lengths, senti_labels,
                               cpt_words, senti_words)
        return dec_out

    def forward_rl(self, region_feats, spatial_feats, senti_labels, cpt_words, senti_words, sample_max):
        batch_size = region_feats.size(0)

        region_feats, spatial_feats = self._vis_encode(region_feats, spatial_feats, senti_labels)
        senti_labels = self.senti_label_embed(senti_labels)  # [bs, d_model]
        cpt_words, senti_words = self._sem_encode(cpt_words, senti_words)

        seq = region_feats.new_zeros((batch_size, self.max_seq_len), dtype=torch.long)
        seq_logprobs = region_feats.new_zeros((batch_size, self.max_seq_len))
        seq_masks = region_feats.new_zeros((batch_size, self.max_seq_len))
        it = region_feats.new_zeros(batch_size, dtype=torch.long).fill_(self.sos_id)  # first input <SOS>
        unfinished = it == self.sos_id
        pre_words = it.unsqueeze(1)  # bs*1
        for t in range(self.max_seq_len):
            logprobs = self._decode(pre_words, None, senti_labels,
                                    cpt_words, senti_words, region_feats, spatial_feats)  # bs*seq_len*vocab
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

    def sample(self, region_feats, spatial_feats, senti_labels, cpt_words, senti_words,
               beam_size=3, decoding_constraint=1):
        self.eval()
        region_feats = region_feats.unsqueeze(0)  # [1, 36, att_feat]
        spatial_feats = spatial_feats.unsqueeze(0)  # [1, 14, 14, att_feat]
        cpt_words = cpt_words.unsqueeze(0)  # [1, num]
        senti_words = senti_words.unsqueeze(0)  # [1, num]

        region_feats, spatial_feats = self._vis_encode(region_feats, spatial_feats, senti_labels)
        senti_labels = self.senti_label_embed(senti_labels)  # [1, d_model]
        cpt_words, senti_words = self._sem_encode(cpt_words, senti_words)

        # log_prob_sum, log_prob_seq, word_id_seq
        candidates = [BeamCandidate(0., [], [self.sos_id])]
        for t in range(self.max_seq_len):
            tmp_candidates = []
            end_flag = True
            for candidate in candidates:
                log_prob_sum, log_prob_seq, word_id_seq = candidate
                if t > 0 and word_id_seq[-1] == self.eos_id:
                    tmp_candidates.append(candidate)
                else:
                    end_flag = False
                    pre_words = region_feats.new_tensor(word_id_seq, dtype=torch.long).unsqueeze(0)  # 1*seq_len
                    logprobs = self._decode(pre_words, None, senti_labels,
                                            cpt_words, senti_words, region_feats, spatial_feats)  # 1*seq_len*vocab
                    logprobs = logprobs[:, -1]  # 1*vocab
                    logprobs = logprobs.squeeze(0)  # vocab_size
                    if self.pad_id != self.eos_id:
                        logprobs[self.pad_id] += float('-inf')  # do not generate <PAD>, <SOS> and <UNK>
                        logprobs[self.sos_id] += float('-inf')
                        logprobs[self.unk_id] += float('-inf')
                    if decoding_constraint:  # do not generate last step word
                        logprobs[word_id_seq[-1]] += float('-inf')

                    output_sorted, index_sorted = torch.sort(logprobs, descending=True)
                    for k in range(beam_size):
                        log_prob, word_id = output_sorted[k], index_sorted[k]  # tensor, tensor
                        log_prob = float(log_prob)
                        word_id = int(word_id)
                        tmp_candidates.append(BeamCandidate(log_prob_sum + log_prob,
                                                            log_prob_seq + [log_prob],
                                                            word_id_seq + [word_id]))
            candidates = sorted(tmp_candidates, key=lambda x: x.log_prob_sum, reverse=True)[:beam_size]
            if end_flag:
                break

        # captions, scores
        captions = [' '.join([self.idx2word[idx] for idx in candidate.word_id_seq[1:-1]])
                    for candidate in candidates]
        scores = [candidate.log_prob_sum for candidate in candidates]
        return captions, scores

    def get_optim_criterion(self, lr, weight_decay=0, smoothing=0.1):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay), \
               LabelSmoothingCriterion(smoothing), nn.MSELoss()  # xe, domain align


class LabelSmoothingCriterion(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCriterion, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.true_dist = None

    def forward(self, pred, target, lengths):
        max_len = max(lengths)
        pred = pred[:, :max_len]
        target = target[:, :max_len]
        mask = pred.new_zeros(len(lengths), max_len)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1

        vocab_size = pred.size(-1)
        pred = pred.reshape(-1, vocab_size)  # [bs*seq_len, vocab]
        target = target.reshape(-1)  # [bs*seq_len]
        mask = mask.reshape(-1)  # [bs*seq_len]

        true_dist = pred.clone()
        true_dist.fill_(self.smoothing / (vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        loss = (self.criterion(pred, true_dist.detach()).sum(1) * mask).sum() / mask.sum()
        return loss
