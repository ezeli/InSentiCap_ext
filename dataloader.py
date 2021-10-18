# coding:utf8
import torch
from torch.utils import data
import numpy as np
import h5py
import random
import pickle


def create_collate_fn(name, pad_index=0, max_seq_len=17, num_concepts=5,
                      num_sentiments=10, mode='xe'):
    def caption_collate_fn(dataset):
        ground_truth = {}
        tmp = []
        for fn, vis_senti, region_feat, spatial_feat, caps_idx, cpts_idx, sentis_idx in dataset:
            ground_truth[fn] = [c[0][:max_seq_len] for c in caps_idx]
            if mode == 'rl':
                tmp_caps_idx = [c for c in caps_idx if c[1] == vis_senti]
                if tmp_caps_idx:
                    caps_idx = random.sample(tmp_caps_idx, 1)
                else:
                    caps_idx = random.sample(caps_idx, 1)
            for cap, senti in caps_idx:
                tmp.append([fn, vis_senti, region_feat, spatial_feat, cap, senti, cpts_idx, sentis_idx])
        dataset = tmp
        dataset.sort(key=lambda p: len(p[4]), reverse=True)
        fns, vis_sentis, region_feats, spatial_feats, caps, xe_senti_labels, cpts, sentis = zip(*dataset)
        region_feats = torch.FloatTensor(np.array(region_feats))
        spatial_feats = torch.FloatTensor(np.array(spatial_feats))
        vis_sentis = torch.LongTensor(np.array(vis_sentis))
        xe_senti_labels = torch.LongTensor(np.array(xe_senti_labels))
 
        lengths = [min(len(c), max_seq_len) for c in caps]
        caps_tensor = torch.LongTensor(len(caps), lengths[0]).fill_(pad_index)
        for i, c in enumerate(caps):
            end = lengths[i]
            caps_tensor[i, :end] = torch.LongTensor(c[:end])
        lengths = [l - 1 for l in lengths]

        cpts_tensor = torch.LongTensor(len(cpts), num_concepts).fill_(pad_index)
        for i, c in enumerate(cpts):
            end = min(len(c), num_concepts)
            cpts_tensor[i, :end] = torch.LongTensor(c[:end])

        sentis_tensor = torch.LongTensor(len(sentis), num_sentiments).fill_(pad_index)
        for i, s in enumerate(sentis):
            end = min(len(s), num_sentiments)
            sentis_tensor[i, :end] = torch.LongTensor(s[:end])

        return fns, vis_sentis, region_feats, spatial_feats, (
        caps_tensor, lengths), xe_senti_labels, cpts_tensor, sentis_tensor, ground_truth

    def scs_collate_fn(dataset):
        dataset.sort(key=lambda p: len(p[0]), reverse=True)
        caps, cpts, sentis, senti_ids = zip(*dataset)
        senti_ids = torch.LongTensor(np.array(senti_ids))

        lengths = [min(len(c), max_seq_len) for c in caps]
        caps_tensor = torch.LongTensor(len(caps), lengths[0]).fill_(pad_index)
        for i, c in enumerate(caps):
            end = lengths[i]
            caps_tensor[i, :end] = torch.LongTensor(c[:end])
        lengths = [l - 1 for l in lengths]

        cpts_tensor = torch.LongTensor(len(cpts), num_concepts).fill_(pad_index)
        for i, c in enumerate(cpts):
            end = min(len(c), num_concepts)
            cpts_tensor[i, :end] = torch.LongTensor(c[:end])

        sentis_tensor = torch.LongTensor(len(sentis), num_sentiments).fill_(pad_index)
        for i, c in enumerate(sentis):
            end = min(len(c), num_sentiments)
            sentis_tensor[i, :end] = torch.LongTensor(c[:end])

        return (caps_tensor, lengths), cpts_tensor, sentis_tensor, senti_ids

    def concept_collate_fn(dataset):
        fns, fc_feats, cpts = zip(*dataset)
        fc_feats = torch.FloatTensor(np.array(fc_feats))
        cpts_tensors = torch.LongTensor(np.array(cpts))
        return fns, fc_feats, cpts_tensors

    def senti_image_collate_fn(dataset):
        fns, att_feats, labels = zip(*dataset)
        att_feats = torch.FloatTensor(np.array(att_feats))
        labels = torch.LongTensor(np.array(labels))
        return fns, att_feats, labels

    def senti_sents_collate_fn(dataset):
        dataset.sort(key=lambda p: len(p[1]), reverse=True)
        sentis, caps = zip(*dataset)
        sentis = torch.LongTensor(np.array(sentis))

        lengths = [min(len(c), max_seq_len) for c in caps]
        caps_tensor = torch.LongTensor(len(caps), lengths[0]).fill_(pad_index)
        for i, c in enumerate(caps):
            end = lengths[i]
            caps_tensor[i, :end] = torch.LongTensor(c[:end])

        return sentis, (caps_tensor, lengths)

    if name == 'caption':
        return caption_collate_fn
    elif name == 'senti_sents':
        return senti_sents_collate_fn
    elif name == 'concept':
        return concept_collate_fn
    elif name == 'senti_image':
        return senti_image_collate_fn
    elif name == 'senti_corpus_with_sentis':
        return scs_collate_fn


class SCSDataset(data.Dataset):
    def __init__(self, senti_corpus_with_sentis):
        self.senti_corpus_with_sentis = senti_corpus_with_sentis

    def __getitem__(self, index):
        cap, cpts, sentis, senti_id = self.senti_corpus_with_sentis[index]
        return cap, cpts, sentis, senti_id

    def __len__(self):
        return len(self.senti_corpus_with_sentis)


class CaptionDataset(data.Dataset):
    def __init__(self, region_feats, spatial_feats, img_captions, vis_sentiments, img_det_concepts, img_det_sentiments):
        self.region_feats = region_feats
        self.spatial_feats = spatial_feats
        self.captions = list(img_captions.items())  # [(fn, [[1, 2],[3, 4],...]),...]
        self.vis_sentiments = vis_sentiments
        self.det_concepts = img_det_concepts  # {fn: [1,2,...])}
        self.det_sentiments = img_det_sentiments  # {fn: [1,2,...])}

    def __getitem__(self, index):
        fn, caps = self.captions[index]
        vis_senti = self.vis_sentiments.get(fn, -1)
        region_feats = h5py.File(self.region_feats, mode='r')
        spatial_feats = h5py.File(self.spatial_feats, mode='r')
        region_feats = region_feats[fn][:]
        spatial_feats = spatial_feats[fn][:]
        cpts = self.det_concepts[fn]
        sentis = self.det_sentiments[fn]
        return fn, vis_senti, np.array(region_feats), np.array(spatial_feats), caps, cpts, sentis

    def __len__(self):
        return len(self.captions)


class ConceptDataset(data.Dataset):
    def __init__(self, fc_feats, img_concepts, num_cpts):
        self.fc_feats = fc_feats
        self.concepts = list(img_concepts.items())
        self.num_cpts = num_cpts

    def __getitem__(self, index):
        fn, cpts_idx = self.concepts[index]
        f_fc = h5py.File(self.fc_feats, mode='r')
        fc_feat = f_fc[fn][:]
        cpts = np.zeros(self.num_cpts, dtype=np.int16)
        cpts[cpts_idx] = 1
        return fn, np.array(fc_feat), cpts

    def __len__(self):
        return len(self.concepts)


class SentiImageDataset(data.Dataset):
    def __init__(self, senti_att_feats, img_senti_labels):
        self.att_feats = senti_att_feats
        self.img_senti_labels = img_senti_labels  # [(fn, senti_label),...]

    def __getitem__(self, index):
        fn, senti_label = self.img_senti_labels[index]
        f_att = h5py.File(self.att_feats, mode='r')
        att_feat = f_att[fn][:]
        return fn, np.array(att_feat), senti_label

    def __len__(self):
        return len(self.img_senti_labels)


class SentiSentDataset(data.Dataset):
    def __init__(self, senti_sentences):
        self.senti_sentences = senti_sentences

    def __getitem__(self, index):
        senti, sent = self.senti_sentences[index]
        return senti, np.array(sent)

    def __len__(self):
        return len(self.senti_sentences)


def get_caption_dataloader(region_feats, spatial_feats, img_captions, vis_sentiments,
                           img_det_concepts, img_det_sentiments,
                           pad_index, max_seq_len, num_concepts, num_sentiments,
                           batch_size, num_workers=0, shuffle=True, mode='xe'):
    dataset = CaptionDataset(region_feats, spatial_feats, img_captions, vis_sentiments, img_det_concepts,
                             img_det_sentiments)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=create_collate_fn(
                                     'caption', pad_index, max_seq_len + 1,
                                     num_concepts, num_sentiments, mode))
    return dataloader


def get_senti_corpus_with_sentis_dataloader(senti_corpus_with_sentis,
                                            pad_index, max_seq_len, num_concepts, num_sentiments,
                                            batch_size, num_workers=0, shuffle=True):
    dataset = SCSDataset(senti_corpus_with_sentis)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=create_collate_fn(
                                     'senti_corpus_with_sentis', pad_index, max_seq_len + 1,
                                     num_concepts, num_sentiments))
    return dataloader


def get_concept_dataloader(fc_feats, img_concepts, num_cpts,
                           batch_size, num_workers=0, shuffle=True):
    dataset = ConceptDataset(fc_feats, img_concepts, num_cpts)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=create_collate_fn('concept'))
    return dataloader


def get_senti_image_dataloader(senti_att_feats, img_senti_labels,
                               batch_size, num_workers=0, shuffle=True):
    dataset = SentiImageDataset(senti_att_feats, img_senti_labels)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=create_collate_fn('senti_image'))
    return dataloader


def get_senti_sents_dataloader(senti_sentences, pad_index, max_seq_len,
                               batch_size=80, num_workers=2, shuffle=True):
    dataset = SentiSentDataset(senti_sentences)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=create_collate_fn(
                                     'senti_sents', pad_index=pad_index,
                                     max_seq_len=max_seq_len))
    return dataloader


def video_create_collate_fn(name, pad_index=0, max_seq_len=17, num_concepts=5,
                            num_sentiments=10, max_feat_len=15, mode='xe'):
    def caption_collate_fn(dataset):
        ground_truth = {}
        tmp = []
        for fn, vis_senti, two_d_feats, three_d_feats, audio_feats, caps_idx, cpts_idx, sentis_idx in dataset:
            ground_truth[fn] = [c[0][:max_seq_len] for c in caps_idx]
            if mode == 'rl':
                tmp_caps_idx = [c for c in caps_idx if c[1] == vis_senti]
                if tmp_caps_idx:
                    caps_idx = random.sample(tmp_caps_idx, 1)
                else:
                    caps_idx = random.sample(caps_idx, 1)
            for cap, senti in caps_idx:
                tmp.append([fn, vis_senti, two_d_feats, three_d_feats, audio_feats, cap, senti, cpts_idx, sentis_idx])
        dataset = tmp
        dataset.sort(key=lambda p: len(p[5]), reverse=True)
        fns, vis_sentis, two_d_feats, three_d_feats, audio_feats, caps, xe_senti_labels, cpts, sentis = zip(*dataset)
        vis_sentis = torch.LongTensor(np.array(vis_sentis))
        xe_senti_labels = torch.LongTensor(np.array(xe_senti_labels))

        two_d_feats_lengths = [min(c.shape[0], max_feat_len) for c in two_d_feats]
        two_d_feats_tensor = torch.FloatTensor(len(two_d_feats), max(two_d_feats_lengths),
                                               two_d_feats[0].shape[1]).fill_(0)
        for i, c in enumerate(two_d_feats):
            end_idx = two_d_feats_lengths[i]
            feat_idxs = np.linspace(0, c.shape[0], num=end_idx, endpoint=False, dtype=np.int)
            two_d_feats_tensor[i, :end_idx] = torch.FloatTensor(c[feat_idxs])

        three_d_feats_lengths = [min(c.shape[0], max_feat_len) for c in three_d_feats]
        three_d_feats_tensor = torch.FloatTensor(len(three_d_feats), max(three_d_feats_lengths),
                                                 three_d_feats[0].shape[1]).fill_(0)
        for i, c in enumerate(three_d_feats):
            end_idx = three_d_feats_lengths[i]
            feat_idxs = np.linspace(0, c.shape[0], num=end_idx, endpoint=False, dtype=np.int)
            three_d_feats_tensor[i, :end_idx] = torch.FloatTensor(c[feat_idxs])

        audio_feats_lengths = [min(c.shape[0], max_feat_len) for c in audio_feats]
        audio_feats_tensor = torch.FloatTensor(len(audio_feats), max(audio_feats_lengths),
                                               audio_feats[0].shape[1]).fill_(0)
        for i, c in enumerate(audio_feats):
            end_idx = audio_feats_lengths[i]
            feat_idxs = np.linspace(0, c.shape[0], num=end_idx, endpoint=False, dtype=np.int)
            audio_feats_tensor[i, :end_idx] = torch.FloatTensor(c[feat_idxs])

        lengths = [min(len(c), max_seq_len) for c in caps]
        caps_tensor = torch.LongTensor(len(caps), lengths[0]).fill_(pad_index)
        for i, c in enumerate(caps):
            end = lengths[i]
            caps_tensor[i, :end] = torch.LongTensor(c[:end])
        lengths = [l - 1 for l in lengths]

        cpts_tensor = torch.LongTensor(len(cpts), num_concepts).fill_(pad_index)
        for i, c in enumerate(cpts):
            end = min(len(c), num_concepts)
            cpts_tensor[i, :end] = torch.LongTensor(c[:end])

        sentis_tensor = torch.LongTensor(len(sentis), num_sentiments).fill_(pad_index)
        for i, s in enumerate(sentis):
            end = min(len(s), num_sentiments)
            sentis_tensor[i, :end] = torch.LongTensor(s[:end])

        return fns, vis_sentis, (two_d_feats_tensor, two_d_feats_lengths), (
        three_d_feats_tensor, three_d_feats_lengths), (audio_feats_tensor, audio_feats_lengths), (
               caps_tensor, lengths), xe_senti_labels, cpts_tensor, sentis_tensor, ground_truth

    def concept_collate_fn(dataset):
        fns, two_d_feats, three_d_feats, audio_feats, cpts = zip(*dataset)
        two_d_feats = torch.FloatTensor(np.array(two_d_feats))
        three_d_feats = torch.FloatTensor(np.array(three_d_feats))
        audio_feats = torch.FloatTensor(np.array(audio_feats))
        cpts_tensors = torch.LongTensor(np.array(cpts))
        return fns, two_d_feats, three_d_feats, audio_feats, cpts_tensors

    def senti_video_collate_fn(dataset):
        fns, two_d_feats, three_d_feats, audio_feats, senti_labels = zip(*dataset)
        senti_labels = torch.LongTensor(np.array(senti_labels))

        two_d_feats_lengths = [min(c.shape[0], max_feat_len) for c in two_d_feats]
        two_d_feats_tensor = torch.FloatTensor(len(two_d_feats), max(two_d_feats_lengths),
                                               two_d_feats[0].shape[1]).fill_(0)
        for i, c in enumerate(two_d_feats):
            end_idx = two_d_feats_lengths[i]
            feat_idxs = np.linspace(0, c.shape[0], num=end_idx, endpoint=False, dtype=np.int)
            two_d_feats_tensor[i, :end_idx] = torch.FloatTensor(c[feat_idxs])

        three_d_feats_lengths = [min(c.shape[0], max_feat_len) for c in three_d_feats]
        three_d_feats_tensor = torch.FloatTensor(len(three_d_feats), max(three_d_feats_lengths),
                                                 three_d_feats[0].shape[1]).fill_(0)
        for i, c in enumerate(three_d_feats):
            end_idx = three_d_feats_lengths[i]
            feat_idxs = np.linspace(0, c.shape[0], num=end_idx, endpoint=False, dtype=np.int)
            three_d_feats_tensor[i, :end_idx] = torch.FloatTensor(c[feat_idxs])

        audio_feats_lengths = [min(c.shape[0], max_feat_len) for c in audio_feats]
        audio_feats_tensor = torch.FloatTensor(len(audio_feats), max(audio_feats_lengths),
                                               audio_feats[0].shape[1]).fill_(0)
        for i, c in enumerate(audio_feats):
            end_idx = audio_feats_lengths[i]
            feat_idxs = np.linspace(0, c.shape[0], num=end_idx, endpoint=False, dtype=np.int)
            audio_feats_tensor[i, :end_idx] = torch.FloatTensor(c[feat_idxs])
        return fns, (two_d_feats_tensor, two_d_feats_lengths), (three_d_feats_tensor, three_d_feats_lengths), (
        audio_feats_tensor, audio_feats_lengths), senti_labels

    if name == 'concept':
        return concept_collate_fn
    elif name == 'senti_video':
        return senti_video_collate_fn
    elif name == 'caption':
        return caption_collate_fn


class VideoConceptDataset(data.Dataset):
    def __init__(self, two_d_feature_file, three_d_feature_file, audio_feature_file,
                 concepts, num_cpts):
        self.two_d_feature_file = two_d_feature_file
        self.three_d_feature_file = three_d_feature_file
        self.audio_feats = pickle.load(open(audio_feature_file, 'rb'))
        self.concepts = list(concepts.items())
        self.num_cpts = num_cpts

    def __getitem__(self, index):
        fn, cpts_idx = self.concepts[index]
        two_d_feats = h5py.File(self.two_d_feature_file, 'r')
        three_d_feats = h5py.File(self.three_d_feature_file, 'r')
        two_d_feats = two_d_feats[fn][:].mean(0)
        three_d_feats = three_d_feats[fn][:].mean(0)
        audio_feats = np.array(self.audio_feats[fn]).mean(0)
        cpts = np.zeros(self.num_cpts, dtype=np.int16)
        cpts[cpts_idx] = 1
        return fn, np.array(two_d_feats), np.array(three_d_feats), audio_feats, cpts

    def __len__(self):
        return len(self.concepts)


def get_video_concept_dataloader(two_d_feature_file, three_d_feature_file, audio_feature_file,
                                 concepts, num_cpts, batch_size, num_workers=0, shuffle=True):
    dataset = VideoConceptDataset(two_d_feature_file, three_d_feature_file, audio_feature_file,
                                  concepts, num_cpts)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=video_create_collate_fn('concept'))
    return dataloader


class SentiVideoDataset(data.Dataset):
    def __init__(self, two_d_feature_file, three_d_feature_file, audio_feature_file, vid_senti_labels):
        self.two_d_feature_file = two_d_feature_file
        self.three_d_feature_file = three_d_feature_file
        self.audio_feats = pickle.load(open(audio_feature_file, 'rb'))
        self.vid_senti_labels = vid_senti_labels  # [(fn, senti_label),...]

    def __getitem__(self, index):
        fn, senti_label = self.vid_senti_labels[index]
        two_d_feats = h5py.File(self.two_d_feature_file, 'r')
        three_d_feats = h5py.File(self.three_d_feature_file, 'r')
        two_d_feats = two_d_feats[fn][:]
        three_d_feats = three_d_feats[fn][:]
        audio_feats = np.array(self.audio_feats[fn])
        return fn, np.array(two_d_feats), np.array(three_d_feats), audio_feats, senti_label

    def __len__(self):
        return len(self.vid_senti_labels)


def get_senti_video_dataloader(two_d_feature_file, three_d_feature_file, audio_feature_file,
                               vid_senti_labels, batch_size, num_workers=0, shuffle=True):
    dataset = SentiVideoDataset(two_d_feature_file, three_d_feature_file, audio_feature_file,
                                vid_senti_labels)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=video_create_collate_fn('senti_video'))
    return dataloader


class VideoCaptionDataset(data.Dataset):
    def __init__(self, two_d_feature_file, three_d_feature_file, audio_feature_file,
                 img_captions, vis_sentiments, img_det_concepts, img_det_sentiments):
        self.two_d_feature_file = two_d_feature_file
        self.three_d_feature_file = three_d_feature_file
        self.audio_feats = pickle.load(open(audio_feature_file, 'rb'))

        self.captions = list(img_captions.items())  # [(fn, [[1, 2],[3, 4],...]),...]
        self.vis_sentiments = vis_sentiments
        self.det_concepts = img_det_concepts  # {fn: [1,2,...])}
        self.det_sentiments = img_det_sentiments  # {fn: [1,2,...])}

    def __getitem__(self, index):
        fn, caps = self.captions[index]
        vis_senti = self.vis_sentiments.get(fn, -1)

        two_d_feats = h5py.File(self.two_d_feature_file, 'r')
        three_d_feats = h5py.File(self.three_d_feature_file, 'r')
        two_d_feats = two_d_feats[fn][:]
        three_d_feats = three_d_feats[fn][:]
        audio_feats = np.array(self.audio_feats[fn])

        cpts = self.det_concepts[fn]
        sentis = self.det_sentiments[fn]
        return fn, vis_senti, np.array(two_d_feats), np.array(three_d_feats), audio_feats, caps, cpts, sentis

    def __len__(self):
        return len(self.captions)


def get_vid_caption_dataloader(two_d_feature_file, three_d_feature_file, audio_feature_file,
                               img_captions, vis_sentiments,
                               img_det_concepts, img_det_sentiments,
                               pad_index, max_seq_len, num_concepts, num_sentiments,
                               batch_size, num_workers=0, shuffle=True, mode='xe'):
    dataset = VideoCaptionDataset(two_d_feature_file, three_d_feature_file, audio_feature_file,
                                  img_captions, vis_sentiments, img_det_concepts, img_det_sentiments)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=video_create_collate_fn(
                                     'caption', pad_index, max_seq_len + 1,
                                     num_concepts, num_sentiments, mode=mode))
    return dataloader
