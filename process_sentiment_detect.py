import torch
import h5py
import os
import numpy as np
import json
import tqdm
import pickle

from models.sentiment_detector import SentimentDetector
from models.video_sentiment_detector import VideoSentimentDetector
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


def process_vid_senti():
    senti_detector_file = 'checkpoint/sentiment/video/model-12_0.3088_0.6832_1012-2046.pth'
    max_feat_len = 10

    print("====> loading senti_detector_file '{}'".format(senti_detector_file))
    ch = torch.load(senti_detector_file, map_location=lambda s, l: s)
    sentiment_categories = ch['sentiment_categories']
    senti_detector = VideoSentimentDetector(sentiment_categories, ch['settings'])
    senti_detector.load_state_dict(ch['model'])
    senti_detector.to(device)
    senti_detector.eval()

    dataset_name = 'msrvtt'
    two_d_feature_file = os.path.join('./data/features', dataset_name, '%s_ResNet101.h5' % dataset_name)
    three_d_feature_file = os.path.join('./data/features', dataset_name, '%s_3DResNext101.h5' % dataset_name)
    audio_feature_file = os.path.join('./data/features', dataset_name, '%s_audio_VGGish.pickle' % dataset_name)
    audio_feats = pickle.load(open(audio_feature_file, 'rb'))

    vis_sentiments = {}
    fns = list(audio_feats.keys())
    for fn in tqdm.tqdm(fns, ncols=100):
        two_d_feats = h5py.File(two_d_feature_file, 'r')
        three_d_feats = h5py.File(three_d_feature_file, 'r')
        audio_feats = pickle.load(open(audio_feature_file, 'rb'))
        two_d_feats = two_d_feats[fn][:]
        three_d_feats = three_d_feats[fn][:]
        audio_feats = np.array(audio_feats[fn])

        end_idx = min(two_d_feats.shape[0], max_feat_len)
        feat_idxs = np.linspace(0, two_d_feats.shape[0], num=end_idx, endpoint=False, dtype=np.int)
        two_d_feats_tensor = torch.FloatTensor(two_d_feats[feat_idxs]).to(device)

        end_idx = min(three_d_feats.shape[0], max_feat_len)
        feat_idxs = np.linspace(0, three_d_feats.shape[0], num=end_idx, endpoint=False, dtype=np.int)
        three_d_feats_tensor = torch.FloatTensor(three_d_feats[feat_idxs]).to(device)

        end_idx = min(audio_feats.shape[0], max_feat_len)
        feat_idxs = np.linspace(0, audio_feats.shape[0], num=end_idx, endpoint=False, dtype=np.int)
        audio_feats_tensor = torch.FloatTensor(audio_feats[feat_idxs]).to(device)

        _, senti, _ = senti_detector.sample(two_d_feats_tensor, three_d_feats_tensor, audio_feats_tensor)
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
