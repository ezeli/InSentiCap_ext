# coding:utf8
import tqdm
import os
import time
from collections import defaultdict
import json
import sys
import pdb
import traceback
from bdb import BdbQuit
import torch
import random
import kenlm

from opts import parse_opt
from models.captioner import Captioner
from models.decoder import Detector
from models.sent_senti_cls import SentenceSentimentClassifier
from dataloader import get_caption_dataloader, get_senti_corpus_with_sentis_dataloader


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def train():
    dataset_name = opt.dataset_name
    corpus_type = opt.corpus_type

    idx2word = json.load(open(os.path.join(opt.captions_dir, dataset_name, corpus_type, 'idx2word.json'), 'r'))
    img_captions = json.load(
        open(os.path.join(opt.captions_dir, dataset_name, corpus_type, 'img_captions_senti.json'), 'r'))
    vis_sentiments = json.load(open(os.path.join(opt.captions_dir, dataset_name, 'vis_sentiments.json'), 'r'))
    img_det_concepts = json.load(open(os.path.join(opt.captions_dir, dataset_name, 'img_det_concepts.json'), 'r'))
    img_det_sentiments = json.load(
        open(os.path.join(opt.captions_dir, dataset_name, corpus_type, 'img_det_sentiments.json'), 'r'))
    senti_captions = json.load(
        open(os.path.join(opt.captions_dir, dataset_name, corpus_type, 'senti_captions.json'), 'r'))

    captioner = Captioner(idx2word, opt.sentiment_categories, opt.settings)
    captioner.to(opt.device)
    lr = opt.rl_lr
    optimizer, _, _ = captioner.get_optim_criterion(lr)
    if opt.rl_resume:
        print("====> loading checkpoint '{}'".format(opt.rl_resume))
        chkpoint = torch.load(opt.rl_resume, map_location=lambda s, l: s)
        assert idx2word == chkpoint['idx2word'], \
            'idx2word and resume model idx2word are different'
        assert opt.max_seq_len == chkpoint['max_seq_len'], \
            'opt.max_seq_len and resume model max_seq_len are different'
        assert opt.sentiment_categories == chkpoint['sentiment_categories'], \
            'opt.sentiment_categories and resume model sentiment_categories are different'
        assert dataset_name == chkpoint['dataset_name'], \
            'dataset_name and resume model dataset_name are different'
        assert corpus_type == chkpoint['corpus_type'], \
            'corpus_type and resume model corpus_type are different'
        captioner.load_state_dict(chkpoint['model'])
        optimizer.load_state_dict(chkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(opt.rl_resume, chkpoint['epoch']))
    else:
        rl_xe_resume = os.path.join(opt.checkpoint, 'xe', dataset_name, corpus_type, 'fuse_scores/model-best.pth')
        print("====> loading checkpoint '{}'".format(rl_xe_resume))
        chkpoint = torch.load(rl_xe_resume, map_location=lambda s, l: s)
        assert idx2word == chkpoint['idx2word'], \
            'idx2word and resume model idx2word are different'
        assert opt.sentiment_categories == chkpoint['sentiment_categories'], \
            'opt.sentiment_categories and resume model sentiment_categories are different'
        assert dataset_name == chkpoint['dataset_name'], \
            'dataset_name and resume model dataset_name are different'
        assert corpus_type == chkpoint['corpus_type'], \
            'corpus_type and resume model corpus_type are different'
        captioner.load_state_dict(chkpoint['model'])
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(rl_xe_resume, chkpoint['epoch']))

    ss_cls_file = os.path.join(opt.checkpoint, 'sent_senti_cls', dataset_name, corpus_type, 'model-best.pth')
    print("====> loading checkpoint '{}'".format(ss_cls_file))
    chkpoint = torch.load(ss_cls_file, map_location=lambda s, l: s)
    assert idx2word == chkpoint['idx2word'], \
        'idx2word and resume model idx2word are different'
    assert opt.sentiment_categories == chkpoint['sentiment_categories'], \
        'opt.sentiment_categories and resume model sentiment_categories are different'
    assert dataset_name == chkpoint['dataset_name'], \
        'dataset_name and resume model dataset_name are different'
    assert corpus_type == chkpoint['corpus_type'], \
        'corpus_type and resume model corpus_type are different'
    sent_senti_cls = SentenceSentimentClassifier(chkpoint['idx2word'], chkpoint['sentiment_categories'],
                                                 chkpoint['settings'])
    sent_senti_cls.to(opt.device)
    sent_senti_cls.load_state_dict(chkpoint['model'])
    sent_senti_cls.eval()

    word2idx = {}
    for i, w in enumerate(idx2word):
        word2idx[w] = i

    senti_label2idx = {}
    for i, w in enumerate(opt.sentiment_categories):
        senti_label2idx[w] = i

    print('====> process image captions begin')
    captions_id = {}
    for split, caps in img_captions.items():
        print('convert %s captions to index' % split)
        captions_id[split] = {}
        for fn, seqs in tqdm.tqdm(caps.items(), ncols=100):
            tmp = []
            for seq, senti in seqs:
                tmp.append([[captioner.sos_id] +
                            [word2idx.get(w, None) or word2idx['<UNK>'] for w in seq] +
                            [captioner.eos_id], senti_label2idx[senti]])
            captions_id[split][fn] = tmp
    img_captions = captions_id
    print('====> process image captions end')

    print('====> process vis_sentiments begin')
    vis_sentiments_di = {}
    for fn, senti in tqdm.tqdm(vis_sentiments.items(), ncols=100):
        vis_sentiments_di[fn] = senti_label2idx[senti]
    vis_sentiments = vis_sentiments_di
    print('====> process vis_sentiments end')

    print('====> process image det_concepts begin')
    det_concepts_id = {}
    for fn, cpts in tqdm.tqdm(img_det_concepts.items(), ncols=100):
        det_concepts_id[fn] = [word2idx[w] for w in cpts]
    img_det_concepts = det_concepts_id
    print('====> process image det_concepts end')

    print('====> process image det_sentiments begin')
    det_sentiments_id = {}
    for fn, sentis in tqdm.tqdm(img_det_sentiments.items(), ncols=100):
        det_sentiments_id[fn] = [word2idx[w] for w in sentis if w in word2idx]
    img_det_sentiments = det_sentiments_id
    print('====> process image det_concepts end')

    senti_label2idx = {}
    for i, w in enumerate(opt.sentiment_categories):
        senti_label2idx[w] = i
    print('====> process senti corpus begin')
    senti_captions['positive'] = senti_captions['positive'] * int(
        len(senti_captions['neutral']) / len(senti_captions['positive']))
    senti_captions['negative'] = senti_captions['negative'] * int(
        len(senti_captions['neutral']) / len(senti_captions['negative']))
    # del senti_captions['neutral']
    senti_captions_id = []
    for senti, caps in senti_captions.items():
        print('convert %s corpus to index' % senti)
        senti_id = senti_label2idx[senti]
        for cap, cpts, sentis in tqdm.tqdm(caps, ncols=100):
            cap = [captioner.sos_id] + \
                  [word2idx.get(w, None) or word2idx['<UNK>'] for w in cap] + \
                  [captioner.eos_id]
            cpts = [word2idx[w] for w in cpts if w in word2idx]
            sentis = [word2idx[w] for w in sentis if w in word2idx]
            senti_captions_id.append([cap, cpts, sentis, senti_id])
    random.shuffle(senti_captions_id)
    senti_captions = senti_captions_id
    print('====> process senti corpus end')

    region_feats = os.path.join(opt.feats_dir, dataset_name, '%s_36_att.h5' % dataset_name)
    spatial_feats = os.path.join(opt.feats_dir, dataset_name, '%s_att.h5' % dataset_name)
    train_data = get_caption_dataloader(region_feats, spatial_feats, img_captions['train'], vis_sentiments,
                                        img_det_concepts, img_det_sentiments, idx2word.index('<PAD>'),
                                        opt.max_seq_len, opt.num_concepts, opt.num_sentiments,
                                        opt.rl_bs, opt.rl_num_works, mode='rl')
    val_data = get_caption_dataloader(region_feats, spatial_feats, img_captions['val'], vis_sentiments,
                                      img_det_concepts, img_det_sentiments, idx2word.index('<PAD>'),
                                      opt.max_seq_len, opt.num_concepts, opt.num_sentiments, opt.rl_bs,
                                      opt.rl_num_works, shuffle=False, mode='rl')
    scs_data = get_senti_corpus_with_sentis_dataloader(
        senti_captions, idx2word.index('<PAD>'), opt.max_seq_len,
        opt.num_concepts, opt.num_sentiments, opt.rl_bs, opt.rl_num_works)

    test_captions = {}
    for fn in img_captions['test']:
        test_captions[fn] = [[[], -1]]
    test_data = get_caption_dataloader(region_feats, spatial_feats, test_captions, vis_sentiments,
                                       img_det_concepts, img_det_sentiments, idx2word.index('<PAD>'),
                                       opt.max_seq_len, opt.num_concepts, opt.num_sentiments, opt.rl_bs,
                                       opt.rl_num_works, shuffle=False, mode='rl')

    model = Detector(captioner, optimizer, sent_senti_cls)
    model.set_ciderd_scorer(img_captions)

    lms = {}
    lm_dir = os.path.join(opt.captions_dir, dataset_name, corpus_type, 'lm')
    for senti, i in senti_label2idx.items():
        lms[i] = kenlm.LanguageModel(os.path.join(lm_dir, '%s_id.kenlm.arpa' % senti))
    model.set_lms(lms)

    tmp_dir = f'fuse_scores/{opt.settings["N_enc"]}_{opt.settings["N_dec"]}_{model.min_iter_cnt}_{model.cls_flag}_{model.lm_flag}_{model.seq_flag}_{model.xe_flag}'
    checkpoint = os.path.join(opt.checkpoint, 'rl', dataset_name, corpus_type, tmp_dir)
    # if not os.path.exists(checkpoint):
    #     os.makedirs(checkpoint)
    result_dir = os.path.join(opt.result_dir, 'rl', dataset_name, corpus_type, tmp_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for epoch in range(opt.rl_epochs):
        print('--------------------epoch: %d' % epoch)
        print('tmp_dir:', tmp_dir, 'cls_flag:', model.cls_flag, 'seq_flag:', model.seq_flag)
        torch.cuda.empty_cache()
        train_loss = model.forward((train_data, scs_data), training=True)
        print('train_loss: %s' % dict(train_loss))

        with torch.no_grad():
            torch.cuda.empty_cache()
            val_loss = model.forward((val_data,), training=False)
            print('val_loss:', dict(val_loss))

            # test
            results = defaultdict(list)
            det_sentis = {}
            for fns, vis_sentis, region_feats, spatial_feats, _, _, cpts_tensor, sentis_tensor, _ in tqdm.tqdm(
                    test_data, ncols=100):
                vis_sentis = vis_sentis.to(opt.device)
                region_feats = region_feats.to(opt.device)
                spatial_feats = spatial_feats.to(opt.device)
                cpts_tensor = cpts_tensor.to(opt.device)
                sentis_tensor = sentis_tensor.to(opt.device)
                for i, fn in enumerate(fns):
                    caption, (fuse_scores, _) = model.captioner.sample(
                        region_feats[i], spatial_feats[i], vis_sentis[i:i + 1], cpts_tensor[i], sentis_tensor[i],
                        beam_size=opt.beam_size)
                    det_img_senti = opt.sentiment_categories[int(vis_sentis[i])]
                    results[det_img_senti].append({'image_id': fn, 'caption': caption, 'fuse_scores': fuse_scores})
                    det_sentis[fn] = det_img_senti

            for senti in results:
                json.dump(results[senti],
                          open(os.path.join(result_dir, 'result_%d_%s.json' % (epoch, senti)), 'w'))

            sents = defaultdict(str)
            sents_w = defaultdict(str)
            sent_scores_w = defaultdict(str)
            for senti in results:
                ress = results[senti]
                for res in ress:
                    caption = res['caption']
                    fuse_scores = res['fuse_scores']
                    sents_w[senti] += caption + '\n'
                    sent_scores_w[senti] += \
                        caption + '\n' + \
                        '\n'.join(
                            [f'{s_name}: ' + ' '.join([f'{val}' for val in s_vals]) + f', sum: {sum(s_vals)}'
                             for s_name, s_vals in fuse_scores.items()]) + \
                        '\n' + '\n'
                    caption = [str(word2idx[w]) for w in caption.split()] + [str(word2idx['<EOS>'])]
                    caption = ' '.join(caption) + '\n'
                    sents[senti] += caption
            for senti in sents:
                with open(os.path.join(result_dir, 'result_%d_%s.txt' % (epoch, senti)), 'w') as f:
                    f.write(sents[senti])
                with open(os.path.join(result_dir, 'result_%d_%s_w.txt' % (epoch, senti)), 'w') as f:
                    f.write(sents_w[senti])
                with open(os.path.join(result_dir, 'result_%d_%s_scores_w.txt' % (epoch, senti)), 'w') as f:
                    f.write(sent_scores_w[senti])

        if epoch < -1:
            chkpoint = {
                'epoch': epoch,
                'model': captioner.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': opt.settings,
                'idx2word': idx2word,
                'max_seq_len': opt.max_seq_len,
                'sentiment_categories': opt.sentiment_categories,
                'dataset_name': dataset_name,
                'corpus_type': corpus_type,
            }
            checkpoint_path = os.path.join(
                checkpoint, 'model_%d_%s.pth' % (
                    epoch, time.strftime('%m%d-%H%M')))
            torch.save(chkpoint, checkpoint_path)


if __name__ == '__main__':
    try:
        opt = parse_opt()
        train()
    except (BdbQuit, torch.cuda.memory_allocated()):
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)
