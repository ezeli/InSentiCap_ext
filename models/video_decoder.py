import torch
import torch.nn as nn
import tqdm
import numpy as np
from collections import defaultdict

from self_critical.utils import get_ciderd_scorer, get_self_critical_reward, \
    get_lm_reward, RewardCriterion, get_cls_reward, get_senti_words_reward


def clip_gradient(optimizer, grad_clip=0.1):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class Detector():
    def __init__(self, captioner, cap_optim, sent_senti_cls):
        super(Detector, self).__init__()
        self.captioner = captioner
        self.cap_optim = cap_optim
        self.sent_senti_cls = sent_senti_cls

        _, self.cap_xe_crit, self.cap_da_crit = self.captioner.get_optim_criterion(0)
        self.cap_rl_crit = RewardCriterion()

        self.cls_flag = 0.5
        self.lm_flag = 0.1
        self.seq_flag = 0.1
        self.xe_flag = 0.1
        self.min_iter_cnt = 500

    def set_ciderd_scorer(self, captions):
        self.ciderd_scorer = get_ciderd_scorer(captions, self.captioner.sos_id, self.captioner.eos_id)

    def set_lms(self, lms):
        self.lms = lms

    def forward(self, data, training):
        self.captioner.train(training)
        all_losses = defaultdict(float)
        device = next(self.captioner.parameters()).device

        if training:
            seq2seq_data = iter(data[1])
        caption_data = iter(data[0])
        for _ in tqdm.tqdm(range(min(self.min_iter_cnt, len(data[0]))), ncols=100):
            fns, vis_sentis, (two_d_feats_tensor, two_d_feats_lengths), \
            (three_d_feats_tensor, three_d_feats_lengths), (audio_feats_tensor, audio_feats_lengths), \
            (caps_tensor, lengths), xe_senti_labels, cpts_tensor, sentis_tensor, ground_truth = next(caption_data)
            vis_sentis = vis_sentis.to(device)
            two_d_feats_tensor = two_d_feats_tensor.to(device)
            three_d_feats_tensor = three_d_feats_tensor.to(device)
            audio_feats_tensor = audio_feats_tensor.to(device)
            caps_tensor = caps_tensor.to(device)
            xe_senti_labels = xe_senti_labels.to(device)
            cpts_tensor = cpts_tensor.to(device)
            sentis_tensor = sentis_tensor.to(device)

            sample_captions, sample_logprobs, seq_masks = self.captioner(
                two_d_feats_tensor, two_d_feats_lengths, three_d_feats_tensor, three_d_feats_lengths,
                audio_feats_tensor, audio_feats_lengths, vis_sentis, cpts_tensor, sentis_tensor,
                sample_max=0, mode='rl')
            self.captioner.eval()
            with torch.no_grad():
                greedy_captions, _, greedy_masks = self.captioner(
                    two_d_feats_tensor, two_d_feats_lengths, three_d_feats_tensor, three_d_feats_lengths,
                    audio_feats_tensor, audio_feats_lengths, vis_sentis, cpts_tensor, sentis_tensor,
                    sample_max=1, mode='rl')
            self.captioner.train(training)

            fact_reward = get_self_critical_reward(
                sample_captions, greedy_captions, fns, ground_truth,
                self.captioner.sos_id, self.captioner.eos_id, self.ciderd_scorer)
            fact_reward = torch.from_numpy(fact_reward).float().to(device)
            all_losses['fact_reward'] += float(fact_reward[:, 0].mean())

            cls_reward = get_cls_reward(
                sample_captions, seq_masks, vis_sentis, self.sent_senti_cls, self.captioner.neu_idx)   # [bs, num_sentis]
            cls_reward = torch.from_numpy(cls_reward).float().to(device)
            all_losses['cls_reward'] += float(cls_reward.mean(-1).mean(-1))

            lm_reward = get_lm_reward(
                sample_captions, greedy_captions, vis_sentis,
                self.captioner.sos_id, self.captioner.eos_id, self.lms)
            lm_reward = torch.from_numpy(lm_reward).float().to(device)
            all_losses['lm_reward'] += float(lm_reward[:, 0].sum())

            rewards = fact_reward + self.cls_flag * cls_reward + self.lm_flag * lm_reward
            all_losses['all_rewards'] += float(rewards.mean(-1).mean(-1))
            cap_loss = self.cap_rl_crit(sample_logprobs, seq_masks, rewards)
            all_losses['cap_loss'] += float(cap_loss)

            # xe_loss = 0.0
            # xe_idx = (vis_sentis == xe_senti_labels).cpu().numpy()
            # if sum(xe_idx) > 0:
            pred = self.captioner(two_d_feats_tensor, two_d_feats_lengths, three_d_feats_tensor, three_d_feats_lengths,
                                  audio_feats_tensor, audio_feats_lengths, xe_senti_labels,
                                  cpts_tensor, sentis_tensor, caps_tensor, lengths, mode='xe')
            # xe_loss = self.xe_flag * self.cap_xe_crit(pred[xe_idx], caps_tensor[xe_idx][:, 1:], np.array(lengths)[xe_idx])
            xe_loss = self.xe_flag * self.cap_xe_crit(pred, caps_tensor[:, 1:], lengths)
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
            all_losses[k] = v / min(self.min_iter_cnt, len(data[0]))
        return all_losses
