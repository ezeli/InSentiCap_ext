import torch
import torch.nn as nn
import tqdm
from collections import defaultdict

from .captioner import Captioner
from .sentiment_detector import SentimentDetector
from .sent_senti_cls import SentenceSentimentClassifier

from self_critical.utils import get_ciderd_scorer, get_self_critical_reward, \
    get_lm_reward, RewardCriterion, get_cls_reward, get_senti_words_reward


def clip_gradient(optimizer, grad_clip=0.1):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class Detector(nn.Module):
    def __init__(self, idx2word, max_seq_len, sentiment_categories, lrs, settings):
        super(Detector, self).__init__()
        self.idx2word = idx2word
        self.pad_id = idx2word.index('<PAD>')
        self.max_seq_len = max_seq_len

        self.captioner = Captioner(idx2word, sentiment_categories, settings)
        self.senti_detector = SentimentDetector(sentiment_categories, settings)
        self.sent_senti_cls = SentenceSentimentClassifier(idx2word, sentiment_categories, settings)
        self.senti_detector.eval()
        self.sent_senti_cls.eval()

        self.cap_optim, self.cap_xe_crit, self.cap_da_crit = self.captioner.get_optim_criterion(lrs['cap_lr'])
        self.cap_rl_crit = RewardCriterion()

        self.cls_flag = 0.4
        self.seq_flag = 1.0
        self.senti_threshold = 0.7

    def set_ciderd_scorer(self, captions):
        self.ciderd_scorer = get_ciderd_scorer(captions, self.captioner.sos_id, self.captioner.eos_id)

    def set_lms(self, lms):
        self.lms = lms

    def forward(self, data, training):
        self.captioner.train(training)
        all_losses = defaultdict(float)
        device = next(self.parameters()).device

        if training:
            seq2seq_data = iter(data[1])
        caption_data = iter(data[0])
        for _ in tqdm.tqdm(range(min(500, len(data[0])))):
            fns, region_feats, spatial_feats, (caps_tensor, lengths), cpts_tensor, sentis_tensor, ground_truth = next(caption_data)
            region_feats = region_feats.to(device)
            spatial_feats = spatial_feats.to(device)
            caps_tensor = caps_tensor.to(device)
            cpts_tensor = cpts_tensor.to(device)
            sentis_tensor = sentis_tensor.to(device)

            senti_labels, _, _, _ = self.senti_detector.sample(spatial_feats, self.senti_threshold)
            senti_labels = senti_labels.detach()

            sample_captions, sample_logprobs, seq_masks = self.captioner(
                region_feats, spatial_feats, senti_labels, cpts_tensor, sentis_tensor,
                sample_max=0, mode='rl')
            self.captioner.eval()
            with torch.no_grad():
                greedy_captions, _, greedy_masks = self.captioner(
                    region_feats, spatial_feats, senti_labels, cpts_tensor, sentis_tensor,
                    sample_max=1, mode='rl')
            self.captioner.train(training)

            fact_reward = get_self_critical_reward(
                sample_captions, greedy_captions, fns, ground_truth,
                self.captioner.sos_id, self.captioner.eos_id, self.ciderd_scorer)
            fact_reward = torch.from_numpy(fact_reward).float().to(device)
            all_losses['fact_reward'] += float(fact_reward[:, 0].mean())

            cls_reward = get_cls_reward(
                sample_captions, seq_masks, greedy_captions, greedy_masks,
                senti_labels, self.sent_senti_cls)   # [bs, num_sentis]
            cls_reward = torch.from_numpy(cls_reward).float().to(device)
            all_losses['cls_reward'] += float(cls_reward.mean(-1).mean(-1))

            # lm_reward = get_lm_reward(
            #     sample_captions, greedy_captions, senti_labels,
            #     self.captioner.sos_id, self.captioner.eos_id, self.lms)
            # lm_reward = torch.from_numpy(lm_reward).float().to(device)
            # all_losses['lm_reward'] += float(lm_reward[:, 0].sum())

            rewards = fact_reward + self.cls_flag * cls_reward  # + 0.05 * senti_words_reward
            all_losses['all_rewards'] += float(rewards.mean(-1).mean(-1))
            cap_loss = self.cap_rl_crit(sample_logprobs, seq_masks, rewards)
            all_losses['cap_loss'] += float(cap_loss)

            with torch.no_grad():
                xe_senti_labels, _ = self.sent_senti_cls(caps_tensor[:, 1:], lengths)
                xe_senti_labels = xe_senti_labels.softmax(dim=-1)
                xe_senti_labels = xe_senti_labels.argmax(dim=-1).detach()

            pred = self.captioner(region_feats, spatial_feats, xe_senti_labels, cpts_tensor, sentis_tensor,
                                  caps_tensor, lengths, mode='xe')
            xe_loss = self.cap_xe_crit(pred, caps_tensor[:, 1:], lengths)
            all_losses['xe_loss'] += float(xe_loss)

            seq2seq_loss = 0.0
            if training:
                try:
                    (caps_tensor, lengths), cpts_tensor, sentis_tensor, senti_labels = next(seq2seq_data)
                except:
                    seq2seq_data = iter(data[1])
                    (caps_tensor, lengths), cpts_tensor, sentis_tensor, senti_labels = next(seq2seq_data)
                caps_tensor = caps_tensor.to(device)
                cpts_tensor = cpts_tensor.to(device)
                sentis_tensor = sentis_tensor.to(device)
                senti_labels = senti_labels.to(device)

                pred = self.captioner(senti_labels, cpts_tensor, sentis_tensor, caps_tensor, lengths,
                                      mode='seq2seq')
                seq2seq_loss = self.seq_flag * self.cap_xe_crit(pred, caps_tensor[:, 1:], lengths)
                all_losses['seq2seq_loss'] += float(seq2seq_loss)

            cap_loss = cap_loss + xe_loss + seq2seq_loss

            if training:
                self.cap_optim.zero_grad()
                cap_loss.backward()
                clip_gradient(self.cap_optim)
                self.cap_optim.step()

        # if training and data_type == 'fact':
        #     self.cls_flag = self.cls_flag * 2
        #     if self.cls_flag > 1.0:
        #         self.cls_flag = 1.0

        for k, v in all_losses.items():
            all_losses[k] = v / len(data)
        return all_losses

    def sample(self, region_feats, spatial_feats, cpt_words, senti_words,
               beam_size=3, decoding_constraint=1):
        self.eval()
        att_feats = spatial_feats.unsqueeze(0)
        senti_label, _, det_img_sentis, _ = self.senti_detector.sample(att_feats, self.senti_threshold)

        captions, _ = self.captioner.sample(
            region_feats, spatial_feats, senti_label, cpt_words, senti_words,
            beam_size, decoding_constraint)

        return captions, det_img_sentis
