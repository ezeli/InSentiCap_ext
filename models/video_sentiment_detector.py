import torch
import torch.nn as nn


class VideoSentimentDetector(nn.Module):
    def __init__(self, sentiment_categories, settings):
        super(VideoSentimentDetector, self).__init__()
        self.sentiment_categories = sentiment_categories

        self.output = nn.Sequential(
            nn.Linear(settings['fc_feat_dim']*2 + settings['audio_feat_dim'], settings['fc_feat_dim']),
            nn.ReLU(),
            nn.Dropout(settings['dropout_p']),
            nn.Linear(settings['fc_feat_dim'], settings['fc_feat_dim'] // 2),
            nn.ReLU(),
            nn.Linear(settings['fc_feat_dim'] // 2, settings['fc_feat_dim'] // 4),
            nn.ReLU(),
            nn.Dropout(settings['dropout_p']),
            nn.Linear(settings['fc_feat_dim'] // 4, len(sentiment_categories)),
        )

    def forward(self, two_d_feats_tensor, three_d_feats_tensor, audio_feats_tensor):
        # [bz, fc_feat_dim],[bz, fc_feat_dim],[bz, audio_feat_dim]
        features = torch.cat([two_d_feats_tensor, three_d_feats_tensor, audio_feats_tensor], dim=-1)
        return self.output(features)  # [bz, num]

    def sample(self, two_d_feats_tensor, three_d_feats_tensor, audio_feats_tensor):
        # [bz, 14, 14, fc_feat_dim]
        self.eval()
        output = self.forward(two_d_feats_tensor, three_d_feats_tensor, audio_feats_tensor)
        output = output.softmax(dim=-1)
        scores, senti_labels = output.max(dim=-1)  # bz

        sentiments = []
        for i in senti_labels:
            sentiments.append(self.sentiment_categories[i])

        return senti_labels, sentiments, scores

    def get_optim_criterion(self, lr, weight_decay=0):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay),\
               nn.CrossEntropyLoss()
