import numpy as np
import torch
import torch.nn as nn
import tqdm
from collections import defaultdict

from .cider.pyciderevalcap.ciderD.ciderD import CiderD
from .bleu.bleu import Bleu


def _array_to_str(arr, sos_token, eos_token):
    arr = list(arr)
    if arr[0] == sos_token:
        arr = arr[1:]
    out = ''
    for i in range(len(arr)):
        if arr[i] == eos_token:
            break
        out += str(arr[i]) + ' '
    out += str(eos_token)
    return out.strip()


def _extract_feature(arr, sos_token, eos_token):
    arr = list(arr)
    if arr[0] == sos_token:
        arr = arr[1:]
    feature = {}
    for i in range(len(arr)):
        if arr[i] == eos_token:
            break
        feature[arr[i]] = True
    feature[eos_token] = True

    return feature


def get_ciderd_scorer(split_captions, sos_token, eos_token):
    print('====> get_ciderd_scorer begin')
    captions = {}
    for caps in split_captions.values():
        captions.update(caps)

    refs_idxs = []
    for caps in tqdm.tqdm(captions.values(), ncols=100):
        ref_idxs = []
        for cap in caps:
            ref_idxs.append(_array_to_str(cap[0], sos_token, eos_token))
        refs_idxs.append(ref_idxs)

    scorer = CiderD(refs=refs_idxs)
    print('====> get_ciderd_scorer end')
    return scorer


def get_self_critical_reward(sample_captions, greedy_captions, fns, ground_truth,
                             sos_token, eos_token, scorer):
    batch_size = len(fns)
    sample_captions = sample_captions.cpu().numpy()
    greedy_captions = greedy_captions.cpu().numpy()
    assert sample_captions.shape[0] == greedy_captions.shape[0] == batch_size
    sample_result = []
    greedy_result = []
    gts = {}
    for i, fn in enumerate(fns):
        sample_result.append({'image_id': fn, 'caption': [_array_to_str(sample_captions[i], sos_token, eos_token)]})
        greedy_result.append({'image_id': fn, 'caption': [_array_to_str(greedy_captions[i], sos_token, eos_token)]})
        caps = []
        for cap in ground_truth[fn]:
            caps.append(_array_to_str(cap, sos_token, eos_token))
        gts[fn] = caps
    all_result = sample_result + greedy_result
    if isinstance(scorer, CiderD):
        _, scores = scorer.compute_score(gts, all_result)
    elif isinstance(scorer, Bleu):
        _, scores = scorer.compute_score(gts, all_result)
        scores = np.array(scores[3])
    else:
        raise Exception('do not support this scorer: %s' % type(scorer))

    scores = scores[:batch_size] - scores[batch_size:]
    rewards = np.repeat(scores[:, np.newaxis], sample_captions.shape[1], 1)
    return rewards


def get_lm_reward(sample_captions, greedy_captions, senti_labels, sos_token, eos_token, lms):
    batch_size = sample_captions.size(0)
    sample_captions = sample_captions.cpu().numpy()
    greedy_captions = greedy_captions.cpu().numpy()
    senti_labels = senti_labels.cpu().numpy()
    scores = []
    for i in range(batch_size):
        sample_res = _array_to_str(sample_captions[i], sos_token, eos_token)
        greedy_res = _array_to_str(greedy_captions[i], sos_token, eos_token)

        senti_lm = lms[senti_labels[i]]
        scores.append(np.sign(senti_lm.perplexity(greedy_res) - senti_lm.perplexity(sample_res)))

        # senti_ppls = [[senti_id, senti_lm.perplexity(sample_res)] for senti_id, senti_lm in lms.items()]
        # senti_ppls.sort(key=lambda sp: sp[1])
        # scores.append(1 if senti_ppls[0][0] == senti_labels[i] else -1)

    scores = np.array(scores)
    rewards = np.repeat(scores[:, np.newaxis], sample_captions.shape[1], 1)
    return rewards


def get_cls_reward(sample_captions, sample_masks, senti_labels, sent_senti_cls, neu_idx):
    sample_lens = list(sample_masks.sum(dim=-1).type(torch.int).cpu().numpy())
    with torch.no_grad():
        sample_preds, sample_att_weights = sent_senti_cls(sample_captions, sample_lens)
        sample_preds = sample_preds.argmax(dim=-1)
        sample_preds = (sample_preds == senti_labels).type_as(sample_att_weights)
        sample_preds[sample_preds == 0] = -1
        # sample_preds = ((sample_preds == senti_labels) & (senti_labels != neu_idx)).type_as(sample_att_weights)
        '''sample_preds = sample_preds.unsqueeze(1)
        sample_scores = sample_preds * sample_att_weights
        sample_scores = sample_scores.detach().cpu().numpy()

    max_len = sample_captions.shape[1]
    rewards = np.pad(sample_scores, ((0, 0), (0, max_len-sample_scores.shape[1])))
    rewards = rewards - (rewards.sum(-1, keepdims=True) / sample_masks.cpu().numpy().sum(-1, keepdims=True))'''

    scores = sample_preds.detach().cpu().numpy()
    rewards = np.repeat(scores[:, np.newaxis], sample_captions.shape[1], 1)
    return rewards


def get_senti_words_reward(sample_captions, senti_labels, sentiment_words):
    batch_size = sample_captions.size(0)
    sample_captions = sample_captions.cpu().numpy()
    rewards = np.zeros(sample_captions.shape, dtype=float)
    accur_w = defaultdict(set)
    for i in range(batch_size):
        senti_id = int(senti_labels[i])
        for j, w in enumerate(sample_captions[i]):
            if w in sentiment_words[senti_id]:
                rewards[i, j] = sentiment_words[senti_id][w]
                accur_w[senti_id].add(w)

    return rewards, accur_w


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, seq_logprobs, seq_masks, reward):
        output = - seq_logprobs * seq_masks * reward
        output = torch.sum(output) / torch.sum(seq_masks)

        return output
