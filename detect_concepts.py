# coding:utf8
import torch
import json
import tqdm
import os
import h5py
import numpy as np
import pickle

from opts import parse_opt
from models.concept_detector import ConceptDetector

data_type = 'video'
if data_type == 'video':
    from models.video_concept_detector import VideoConceptDetector as ConceptDetector


opt = parse_opt()
print("====> loading checkpoint '{}'".format(opt.eval_model))
chkpoint = torch.load(opt.eval_model, map_location=lambda s, l: s)
idx2concept = chkpoint['idx2concept']
settings = chkpoint['settings']
dataset_name = chkpoint['dataset_name']
model = ConceptDetector(idx2concept, settings)
model.to(opt.device)
model.load_state_dict(chkpoint['model'])
model.eval()
_, criterion = model.get_optim_criterion(0)
print("====> loaded checkpoint '{}', epoch: {}, dataset_name: {}".
      format(opt.eval_model, chkpoint['epoch'], dataset_name))


if data_type == 'video':
    two_d_feats = h5py.File(os.path.join(opt.feats_dir, dataset_name, '%s_ResNet101.h5' % dataset_name), 'r')
    three_d_feats = h5py.File(os.path.join(opt.feats_dir, dataset_name, '%s_3DResNext101.h5' % dataset_name), 'r')
    audio_feats = pickle.load(open(os.path.join(opt.feats_dir, dataset_name, '%s_audio_VGGish.pickle' % dataset_name), 'rb'))

    predict_result = {}
    fns = list(two_d_feats.keys())
    for i in tqdm.tqdm(range(0, len(fns), 100), ncols=100):
        cur_fns = fns[i:i + 100]
        feats = [[], [], []]
        for fn in cur_fns:
            feats[0].append(two_d_feats[fn][:].mean(0))
            feats[1].append(three_d_feats[fn][:].mean(0))
            feats[2].append(np.array(audio_feats[fn]).mean(0))
        feats = [torch.FloatTensor(np.array(f)).to(opt.device) for f in feats]
        _, concepts, _ = model.sample(*feats, num=10)
        for j, fn in enumerate(cur_fns):
            predict_result[fn] = concepts[j]

    json.dump(predict_result, open(os.path.join(opt.captions_dir, dataset_name, 'vid_det_concepts.json'), 'w'))
else:
    fact_fc = h5py.File(os.path.join(opt.feats_dir, dataset_name, '%s_fc.h5' % dataset_name), 'r')
    senti_fc = h5py.File(os.path.join(opt.feats_dir, 'sentiment', 'feats_fc.h5'), 'r')

    predict_result = {}
    for fc in [fact_fc, senti_fc]:
        fns = list(fc.keys())
        for i in tqdm.tqdm(range(0, len(fns), 100), ncols=100):
            cur_fns = fns[i:i + 100]
            feats = []
            for fn in cur_fns:
                feats.append(fc[fn][:])
            feats = torch.FloatTensor(np.array(feats)).to(opt.device)
            _, concepts, _ = model.sample(feats, num=10)
            for j, fn in enumerate(cur_fns):
                predict_result[fn] = concepts[j]

    json.dump(predict_result, open(os.path.join(opt.captions_dir, dataset_name, 'img_det_concepts.json'), 'w'))
