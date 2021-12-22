import torch
import torch.nn as nn

from .captioner import Encoder, PositionalEncoding


class SentenceSentimentClassifier(nn.Module):
    def __init__(self, idx2word, sentiment_categories, settings=None):
        super(SentenceSentimentClassifier, self).__init__()
        settings = dict()
        settings['d_model'] = 512  # model dim
        settings['d_ff'] = 2048  # feed forward dim
        settings['h'] = 8  # multi heads num
        settings['N_enc'] = 4  # encoder layers num
        settings['dropout_p'] = 0.1
        settings['max_seq_len'] = 50

        self.sentiment_categories = sentiment_categories
        self.neu_id = sentiment_categories.index('neutral')
        self.pad_id = idx2word.index('<PAD>')
        self.cls_id = idx2word.index('<SOS>')
        self.vocab_size = len(idx2word)
        self.word_embed = nn.Sequential(nn.Embedding(self.vocab_size, settings['d_model'],
                                                     padding_idx=self.pad_id),
                                        nn.ReLU(),
                                        nn.Dropout(settings['dropout_p']))
        self.pe = PositionalEncoding(settings)

        self.encoder = Encoder(settings)
        self.sent_senti_cls = nn.Sequential(
            nn.Linear(settings['d_model'], settings['d_model']),
            nn.ReLU(),
            nn.Dropout(settings['dropout_p']),
            nn.Linear(settings['d_model'], len(sentiment_categories)),
        )

    def forward(self, seqs, lengths):
        seqs = torch.cat([seqs.new_ones(seqs.size(0), 1).fill_(self.cls_id), seqs], dim=1)
        lengths = [l+1 for l in lengths]

        seqs = self.word_embed(seqs)  # [bs, seq_len, d_model]
        seqs = self.pe(seqs)

        seq_masks = seqs.new_zeros(len(lengths), max(lengths), dtype=torch.bool)  # bs*seq_len
        for i, l in enumerate(lengths):
            seq_masks[i, :l] = True
        seq_masks = seq_masks.unsqueeze(-2)  # bs*1*seq_len

        out = self.encoder(seqs, seq_masks)
        out = out[:, 0]  # [bs, d_model]
        pred = self.sent_senti_cls(out)  # [bs, 3]
        att_weights = self.encoder.layers[-1].multi_head_att.scores.sum(1)[:, 0, 1:].softmax(-1)  # bs*seq_len

        return pred, att_weights

    def sample(self, seqs, lengths):
        self.eval()
        pred, att_weights = self.forward(seqs, lengths)
        pred = pred.softmax(-1)
        result = []
        result_w = []
        scores = []
        for p in pred:
            res = int(p.argmax(-1))
            score = float(p[res])
            result.append(res)
            result_w.append(self.sentiment_categories[res])
            scores.append(score)

        return result, result_w, scores, att_weights

    def get_optim_and_crit(self, lr, weight_decay=0):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay), \
               nn.CrossEntropyLoss()
