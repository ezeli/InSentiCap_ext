import torch
import torch.nn as nn

from .captioner import Encoder

class VideoSentimentDetector(nn.Module):
    def __init__(self, sentiment_categories, settings):
        super(VideoSentimentDetector, self).__init__()
        self.sentiment_categories = sentiment_categories

        self.two_d_feats_emb = nn.Sequential(
            nn.Linear(settings['fc_feat_dim'], settings['d_model']),
            nn.ReLU(),
            nn.Dropout(settings['dropout_p']),
        )

        self.three_d_feats_emb = nn.Sequential(
            nn.Linear(settings['fc_feat_dim'], settings['d_model']),
            nn.ReLU(),
            nn.Dropout(settings['dropout_p']),
        )

        self.audio_feats_emb = nn.Sequential(
            nn.Linear(settings['audio_feat_dim'], settings['d_model']),
            nn.ReLU(),
            nn.Dropout(settings['dropout_p']),
        )

        N_enc = settings['N_enc']
        settings['N_enc'] = 4
        self.encoder = Encoder(settings)
        settings['N_enc'] = N_enc

        self.output = nn.Linear(settings['d_model'], len(sentiment_categories))

    def _get_masks(self, feats_lengths):
        if feats_lengths is None:
            att_masks = None
        else:
            att_masks = next(self.parameters()).new_zeros(len(feats_lengths), max(feats_lengths), dtype=torch.bool)  # bs*num_atts
            for i, l in enumerate(feats_lengths):
                att_masks[i, :l] = True
            att_masks = att_masks.unsqueeze(-2)  # bs*1*num_atts
        return att_masks

    def forward(self, two_d_feats_tensor, two_d_feats_lengths,
                three_d_feats_tensor, three_d_feats_lengths,
                audio_feats_tensor, audio_feats_lengths):
        two_d_feats_tensor = self.two_d_feats_emb(two_d_feats_tensor)
        three_d_feats_tensor = self.three_d_feats_emb(three_d_feats_tensor)
        audio_feats_tensor = self.audio_feats_emb(audio_feats_tensor)
        features = torch.cat([two_d_feats_tensor, three_d_feats_tensor, audio_feats_tensor], dim=1)  # bs*seq_len*512
        if two_d_feats_lengths is None:
            masks = None
        else:
            masks = torch.cat([self._get_masks(lens) for lens in [two_d_feats_lengths, three_d_feats_lengths, audio_feats_lengths]], dim=-1)  # bs*1*seq_len
        features = self.encoder(features, masks)[:, 0]  # bs*512
        return self.output(features)  # [bz, num]

    def sample(self, two_d_feats_tensor, three_d_feats_tensor, audio_feats_tensor):
        self.eval()
        two_d_feats_tensor = two_d_feats_tensor.unsqueeze(0)
        three_d_feats_tensor = three_d_feats_tensor.unsqueeze(0)
        audio_feats_tensor = audio_feats_tensor.unsqueeze(0)
        output = self.forward(two_d_feats_tensor, None,
                              three_d_feats_tensor, None,
                              audio_feats_tensor, None)[0]
        output = output.softmax(dim=-1)
        score, senti_label = output.max(dim=-1)  # bz

        return senti_label, self.sentiment_categories[senti_label], score

    def get_optim_criterion(self, lr, weight_decay=0):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay),\
               nn.CrossEntropyLoss()
