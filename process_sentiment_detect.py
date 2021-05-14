import torch
import h5py
import os
import numpy as np
import json
import tqdm

from models.sentiment_detector import SentimentDetector
from models.sent_senti_cls import SentenceSentimentClassifier


device = torch.device('cuda:1')


def process_img_senti():
    senti_detector_file = 'checkpoint/sentiment/model-10.pth'
    senti_threshold = 0.7
    batch_size = 50

    print("====> loading senti_detector_file '{}'".format(senti_detector_file))
    ch = torch.load(senti_detector_file, map_location=lambda s, l: s)
    sentiment_categories = ch['sentiment_categories']
    senti_detector = SentimentDetector(sentiment_categories, ch['settings'])
    senti_detector.load_state_dict(ch['model'])
    senti_detector.to(device)
    senti_detector.eval()

    dataset_name = 'coco'
    feats_file = os.path.join('./data/features', dataset_name, '%s_att.h5' % dataset_name)
    spatial_feats = h5py.File(feats_file, mode='r')
    vis_sentiments = {}
    fns = list(spatial_feats.keys())
    for i in tqdm.tqdm(range(0, len(fns), batch_size), ncols=100):
        spatial_feats.close()
        spatial_feats = h5py.File(feats_file, mode='r')
        tmp_fns = fns[i:i+batch_size]
        tmp_feats = []
        for fn in tmp_fns:
            tmp_feats.append(spatial_feats[fn][:])
        tmp_feats = torch.FloatTensor(np.array(tmp_feats)).to(device)
        _, _, senti_labels, _ = senti_detector.sample(tmp_feats, senti_threshold)
        for fn, senti in zip(tmp_fns, senti_labels):
            vis_sentiments[fn] = senti
    json.dump(vis_sentiments, open(os.path.join('./data/captions', dataset_name, 'vis_sentiments.json'), 'w'))


def process_caption_senti():
    max_seq_len = 16
    dataset_name = 'coco'
    corpus_type = 'part'
    idx2word = json.load(open(os.path.join('./data/captions', dataset_name, corpus_type, 'idx2word.json'), 'r'))
    img_captions = json.load(open(os.path.join('./data/captions', dataset_name, 'img_captions.json'), 'r'))

    ss_cls_file = os.path.join('./checkpoint', 'sent_senti_cls', dataset_name, corpus_type, 'model-best.pth')
    print("====> loading ss_cls_file '{}'".format(ss_cls_file))
    ch = torch.load(ss_cls_file, map_location=lambda s, l: s)
    sentiment_categories = ch['sentiment_categories']
    sent_senti_cls = SentenceSentimentClassifier(idx2word, sentiment_categories, ch['settings'])
    sent_senti_cls.load_state_dict(ch['model'])
    sent_senti_cls.to(device)
    sent_senti_cls.eval()

    word2idx = {}
    for i, w in enumerate(idx2word):
        word2idx[w] = i

    img_captions_senti = {}
    for split, captions in img_captions.items():
        img_captions_senti[split] = {}
        for fn, caps in tqdm.tqdm(captions.items(), ncols=100):
            img_captions_senti[split][fn] = []
            tmp_caps = []
            for cap in caps:
                tmp = [word2idx.get(w, None) or word2idx['<UNK>'] for w in cap] + [word2idx['<EOS>']]
                tmp_caps.append(tmp)
            lengths = [min(len(c), max_seq_len) for c in tmp_caps]
            caps_tensor = torch.LongTensor(len(tmp_caps), max(lengths)).fill_(sent_senti_cls.pad_id)
            for i, c in enumerate(tmp_caps):
                end = lengths[i]
                caps_tensor[i, :end] = torch.LongTensor(c[:end])
            caps_tensor = caps_tensor.to(device)
            _, rest_w, _, _ = sent_senti_cls.sample(caps_tensor, lengths)
            for cap, senti in zip(caps, rest_w):
                img_captions_senti[split][fn].append([cap, senti])
    json.dump(img_captions_senti, open(os.path.join('./data/captions', dataset_name, corpus_type, 'img_captions_senti.json'), 'w'))


if __name__ == '__main__':
    process_img_senti()
