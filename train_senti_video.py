# coding:utf8
import tqdm
import os
from collections import defaultdict
import time
import json
import sys
import pdb
import traceback
from bdb import BdbQuit
import torch

from opts import parse_opt
from models.video_sentiment_detector import VideoSentimentDetector
from dataloader import get_senti_video_dataloader


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train():
    senti_detector = VideoSentimentDetector(opt.sentiment_categories, opt.settings)
    senti_detector.to(opt.device)
    lr = opt.concept_lr
    optimizer, criterion = senti_detector.get_optim_criterion(lr)
    if opt.concept_resume:
        print("====> loading checkpoint '{}'".format(opt.concept_resume))
        chkpoint = torch.load(opt.concept_resume, map_location=lambda s, l: s)
        assert opt.settings == chkpoint['settings'], \
            'opt.settings and resume model settings are different'
        assert opt.sentiment_categories == chkpoint['sentiment_categories'], \
            'sentiment_categories and resume model sentiment_categories are different'
        senti_detector.load_state_dict(chkpoint['model'])
        optimizer.load_state_dict(chkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(opt.concept_resume, chkpoint['epoch']))

    vid_senti_labels = json.load(open(opt.vid_senti_labels, 'r'))

    senti_label2idx = {}
    for i, w in enumerate(opt.sentiment_categories):
        senti_label2idx[w] = i
    print('====> process video senti_labels begin')
    senti_labels_id = {}
    for split, senti_labels in vid_senti_labels.items():
        print('convert %s senti_labels to index' % split)
        senti_labels_id[split] = []
        for fn, senti_label in tqdm.tqdm(senti_labels, ncols=100):
            senti_labels_id[split].append([fn, senti_label2idx[senti_label]])
    vid_senti_labels = senti_labels_id
    print('====> process video senti_labels end')

    dataset_name = 'msrvtt'
    two_d_feature_file = os.path.join(opt.feats_dir, dataset_name, '%s_ResNet101.h5' % dataset_name)
    three_d_feature_file = os.path.join(opt.feats_dir, dataset_name, '%s_3DResNext101.h5' % dataset_name)
    audio_feature_file = os.path.join(opt.feats_dir, dataset_name, '%s_audio_VGGish.pickle' % dataset_name)
    train_data = get_senti_video_dataloader(
        two_d_feature_file, three_d_feature_file, audio_feature_file,
        vid_senti_labels['train'], opt.senti_bs, opt.senti_num_works)
    test_data = get_senti_video_dataloader(
        two_d_feature_file, three_d_feature_file, audio_feature_file,
        vid_senti_labels['test'], opt.senti_bs, opt.senti_num_works, shuffle=False)

    def forward(data, training=True):
        senti_detector.train(training)
        loss_val = 0.0
        for _, two_d_feats, three_d_feats, audio_feats, senti_labels in tqdm.tqdm(data, ncols=100):
            two_d_feats = two_d_feats.to(opt.device)
            three_d_feats = three_d_feats.to(opt.device)
            audio_feats = audio_feats.to(opt.device)
            senti_labels = senti_labels.to(opt.device)
            pred = senti_detector(two_d_feats, three_d_feats, audio_feats)
            loss = criterion(pred, senti_labels)
            loss_val += loss.item()
            if training:
                optimizer.zero_grad()
                loss.backward()
                clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()
        return loss_val / len(data)

    checkpoint = os.path.join(opt.checkpoint, 'sentiment', 'video')
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    previous_loss = None
    for epoch in range(opt.senti_epochs):
        print('--------------------epoch: %d' % epoch)
        train_loss = forward(train_data)
        with torch.no_grad():
            # test
            corr_num = defaultdict(int)
            all_num = defaultdict(int)
            for fns, two_d_feats, three_d_feats, audio_feats, senti_labels in tqdm.tqdm(test_data, ncols=100):
                two_d_feats = two_d_feats.to(opt.device)
                three_d_feats = three_d_feats.to(opt.device)
                audio_feats = audio_feats.to(opt.device)
                senti_labels = senti_labels.to(opt.device)
                pred, _, _ = senti_detector.sample(two_d_feats, three_d_feats, audio_feats)
                for gt_idx, pred_idx in zip(senti_labels, pred):
                    gt_idx = int(gt_idx)
                    pred_idx = int(pred_idx)
                    all_num[gt_idx] += 1
                    if gt_idx == pred_idx:
                        corr_num[gt_idx] += 1
            total_corr_rate = sum(corr_num.values()) / sum(all_num.values())
            corr_rate = {}
            for senti_idx in all_num:
                senti_name = senti_detector.sentiment_categories[senti_idx]
                corr_rate[senti_name] = corr_num[senti_idx] / all_num[senti_idx]

        if previous_loss is not None and total_corr_rate < previous_loss:
            lr = lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = total_corr_rate

        print('train_loss: %.4f, total_corr_rate: %.4f, corr_rate: %s' %
              (train_loss, total_corr_rate, corr_rate))
        if epoch > -1:
            chkpoint = {
                'epoch': epoch,
                'model': senti_detector.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': opt.settings,
                'sentiment_categories': opt.sentiment_categories,
            }
            checkpoint_path = os.path.join(checkpoint, 'model_%d_%.4f_%.4f_%s.pth' % (
                epoch, train_loss, total_corr_rate, time.strftime('%m%d-%H%M')))
            torch.save(chkpoint, checkpoint_path)


if __name__ == '__main__':
    try:
        opt = parse_opt()
        train()
    except BdbQuit:
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)
