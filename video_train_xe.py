# coding:utf8
import tqdm
import os
import time
import json
from collections import defaultdict
import sys
import pdb
import traceback
from bdb import BdbQuit
import torch

from opts import parse_opt
from models.video_captioner import Captioner
from dataloader import get_vid_caption_dataloader, get_senti_corpus_with_sentis_dataloader


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train():
    dataset_name = opt.dataset_name
    corpus_type = opt.corpus_type

    idx2word = json.load(open(os.path.join(opt.captions_dir, dataset_name, corpus_type, 'idx2word.json'), 'r'))
    vid_captions = json.load(open(os.path.join(opt.captions_dir, dataset_name, corpus_type, 'vid_captions_senti.json'), 'r'))
    vid_det_concepts = json.load(open(os.path.join(opt.captions_dir, dataset_name, 'vid_det_concepts.json'), 'r'))
    vid_det_sentiments = json.load(open(os.path.join(opt.captions_dir, dataset_name, corpus_type, 'vid_det_sentiments.json'), 'r'))
    senti_captions = json.load(open(os.path.join(opt.captions_dir, dataset_name, corpus_type, 'senti_captions.json'), 'r'))

    captioner = Captioner(idx2word, opt.sentiment_categories, opt.settings)
    captioner.to(opt.device)
    lr = opt.xe_lr
    optimizer, xe_crit, da_crit = captioner.get_optim_criterion(lr)
    if opt.xe_resume:
        print("====> loading checkpoint '{}'".format(opt.xe_resume))
        chkpoint = torch.load(opt.xe_resume, map_location=lambda s, l: s)
        assert idx2word == chkpoint['idx2word'], \
            'idx2word and resume model idx2word are different'
        assert opt.sentiment_categories == chkpoint['sentiment_categories'], \
            'sentiment_categories and resume model sentiment_categories are different'
        assert dataset_name == chkpoint['dataset_name'], \
            'dataset_name and resume model dataset_name are different'
        assert corpus_type == chkpoint['corpus_type'], \
            'corpus_type and resume model corpus_type are different'
        captioner.load_state_dict(chkpoint['model'])
        optimizer.load_state_dict(chkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(opt.xe_resume, chkpoint['epoch']))

    word2idx = {}
    for i, w in enumerate(idx2word):
        word2idx[w] = i

    senti_label2idx = {}
    for i, w in enumerate(opt.sentiment_categories):
        senti_label2idx[w] = i

    print('====> process image captions begin')
    captions_id = {}
    for split, caps in vid_captions.items():
        print('convert %s captions to index' % split)
        captions_id[split] = {}
        for fn, seqs in tqdm.tqdm(caps.items(), ncols=100):
            tmp = []
            for seq, senti in seqs:
                tmp.append([[captioner.sos_id] +
                           [word2idx.get(w, None) or word2idx['<UNK>'] for w in seq] +
                           [captioner.eos_id], senti_label2idx[senti]])
            captions_id[split][fn] = tmp
    vid_captions = captions_id
    print('====> process image captions end')

    print('====> process image det_concepts begin')
    det_concepts_id = {}
    for fn, cpts in tqdm.tqdm(vid_det_concepts.items(), ncols=100):
        det_concepts_id[fn] = [word2idx[w] for w in cpts]
    vid_det_concepts = det_concepts_id
    print('====> process image det_concepts end')

    print('====> process image det_sentiments begin')
    det_sentiments_id = {}
    for fn, sentis in tqdm.tqdm(vid_det_sentiments.items(), ncols=100):
        det_sentiments_id[fn] = [word2idx[w] for w in sentis if w in word2idx]
    vid_det_sentiments = det_sentiments_id
    print('====> process image det_concepts end')

    print('====> process senti corpus begin')
    senti_captions['positive'] = senti_captions['positive'] * int(len(senti_captions['neutral']) / len(senti_captions['positive']))
    senti_captions['negative'] = senti_captions['negative'] * int(len(senti_captions['neutral']) / len(senti_captions['negative']))
    # del senti_captions['neutral']
    senti_captions_id = []
    for senti, caps in senti_captions.items():
        print('convert %s corpus to index' % senti)
        senti_id = senti_label2idx[senti]
        for cap, cpts, sentis in tqdm.tqdm(caps, ncols=100):
            cap = [captioner.sos_id] +\
                  [word2idx.get(w, None) or word2idx['<UNK>'] for w in cap] +\
                  [captioner.eos_id]
            cpts = [word2idx[w] for w in cpts if w in word2idx]
            sentis = [word2idx[w] for w in sentis if w in word2idx]
            senti_captions_id.append([cap, cpts, sentis, senti_id])
    senti_captions = senti_captions_id
    print('====> process senti corpus end')

    two_d_feature_file = os.path.join(opt.feats_dir, dataset_name, '%s_ResNet101.h5' % dataset_name)
    three_d_feature_file = os.path.join(opt.feats_dir, dataset_name, '%s_3DResNext101.h5' % dataset_name)
    audio_feature_file = os.path.join(opt.feats_dir, dataset_name, '%s_audio_VGGish.pickle' % dataset_name)
    train_data = get_vid_caption_dataloader(two_d_feature_file, three_d_feature_file, audio_feature_file,
                                            vid_captions['train'], {},
                                            vid_det_concepts, vid_det_sentiments, idx2word.index('<PAD>'),
                                            opt.max_seq_len, opt.num_concepts, opt.num_sentiments,
                                            opt.xe_bs, opt.xe_num_works)
    val_data = get_vid_caption_dataloader(two_d_feature_file, three_d_feature_file, audio_feature_file,
                                          vid_captions['val'], {},
                                          vid_det_concepts, vid_det_sentiments, idx2word.index('<PAD>'),
                                          opt.max_seq_len, opt.num_concepts, opt.num_sentiments, opt.xe_bs,
                                          opt.xe_num_works, shuffle=False)
    scs_data = get_senti_corpus_with_sentis_dataloader(
        senti_captions, idx2word.index('<PAD>'), opt.max_seq_len,
        opt.num_concepts, opt.num_sentiments, opt.xe_bs*4, opt.xe_num_works)

    test_captions = {}
    for fn in vid_captions['test']:
        test_captions[fn] = [[[], -1]]
    test_data = get_vid_caption_dataloader(two_d_feature_file, three_d_feature_file, audio_feature_file,
                                           test_captions, {},
                                           vid_det_concepts, vid_det_sentiments, idx2word.index('<PAD>'),
                                           opt.max_seq_len, opt.num_concepts, opt.num_sentiments, opt.xe_bs,
                                           opt.xe_num_works, shuffle=False)

    def forward(data, training=True):
        captioner.train(training)
        if training:
            seq2seq_data = iter(scs_data)
        loss_val = defaultdict(float)
        for _, _, (two_d_feats_tensor, two_d_feats_lengths), (three_d_feats_tensor, three_d_feats_lengths), (audio_feats_tensor, audio_feats_lengths), (caps_tensor, lengths), xe_senti_labels, cpts_tensor, sentis_tensor, _ in tqdm.tqdm(data, ncols=100):
            two_d_feats_tensor = two_d_feats_tensor.to(opt.device)
            three_d_feats_tensor = three_d_feats_tensor.to(opt.device)
            audio_feats_tensor = audio_feats_tensor.to(opt.device)
            caps_tensor = caps_tensor.to(opt.device)
            xe_senti_labels = xe_senti_labels.to(opt.device)
            cpts_tensor = cpts_tensor.to(opt.device)
            sentis_tensor = sentis_tensor.to(opt.device)

            pred = captioner(two_d_feats_tensor, two_d_feats_lengths, three_d_feats_tensor, three_d_feats_lengths,
                             audio_feats_tensor, audio_feats_lengths, xe_senti_labels, cpts_tensor, sentis_tensor,
                             caps_tensor, lengths, mode='xe')
            cap_loss = xe_crit(pred, caps_tensor[:, 1:], lengths)
            loss_val['cap_loss'] += float(cap_loss)

            seq2seq_loss = 0.0
            if training:
                try:
                    (caps_tensor, lengths), cpts_tensor, sentis_tensor, senti_labels = next(seq2seq_data)
                except:
                    seq2seq_data = iter(scs_data)
                    (caps_tensor, lengths), cpts_tensor, sentis_tensor, senti_labels = next(seq2seq_data)
                caps_tensor = caps_tensor.to(opt.device)
                cpts_tensor = cpts_tensor.to(opt.device)
                sentis_tensor = sentis_tensor.to(opt.device)
                senti_labels = senti_labels.to(opt.device)
                pred = captioner(senti_labels, cpts_tensor, sentis_tensor, caps_tensor, lengths,
                                 mode='seq2seq')
                seq2seq_loss = xe_crit(pred, caps_tensor[:, 1:], lengths)
                loss_val['seq2seq_loss'] += float(seq2seq_loss)

            all_loss = cap_loss + seq2seq_loss
            loss_val['all_loss'] += float(all_loss)

            if training:
                optimizer.zero_grad()
                all_loss.backward()
                clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()

        for k, v in loss_val.items():
            loss_val[k] = v / len(data)
        return loss_val

    tmp_dir = 'fuse_scores'
    checkpoint = os.path.join(opt.checkpoint, 'xe', dataset_name, corpus_type, tmp_dir)
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    result_dir = os.path.join(opt.result_dir, 'xe', dataset_name, corpus_type, tmp_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    previous_loss = None
    for epoch in range(opt.xe_epochs):
        print('--------------------epoch: %d, tmp_dir: %s' % (epoch, tmp_dir))
        # torch.cuda.empty_cache()
        train_loss = forward(train_data)
        with torch.no_grad():
            val_loss = forward(val_data, training=False)

        if previous_loss is not None and val_loss['all_loss'] > previous_loss:
            lr = lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = val_loss['all_loss']

        print('train_loss: %s, val_loss: %s' % (dict(train_loss), dict(val_loss)))
        if epoch > -1:
            with torch.no_grad():
                senti_label = torch.LongTensor([captioner.neu_idx]).to(opt.device)
                results = []
                fact_txt = ''
                fact_scores_txt = ''
                for fns, _, (two_d_feats_tensor, two_d_feats_lengths), (three_d_feats_tensor, three_d_feats_lengths), (audio_feats_tensor, audio_feats_lengths), _, _, cpts_tensor, sentis_tensor, _ in tqdm.tqdm(test_data, ncols=100):
                    two_d_feats_tensor = two_d_feats_tensor.to(opt.device)
                    three_d_feats_tensor = three_d_feats_tensor.to(opt.device)
                    audio_feats_tensor = audio_feats_tensor.to(opt.device)
                    cpts_tensor = cpts_tensor.to(opt.device)
                    sentis_tensor = sentis_tensor.to(opt.device)
                    for i, fn in enumerate(fns):
                        caption, (fuse_scores, _) = captioner.sample(
                            two_d_feats_tensor[i, :two_d_feats_lengths[i]],
                            three_d_feats_tensor[i, :three_d_feats_lengths[i]],
                            audio_feats_tensor[i, :audio_feats_lengths[i]],
                            senti_label, cpts_tensor[i], sentis_tensor[i],
                            beam_size=opt.beam_size)
                        results.append({'image_id': fn, 'caption': caption, 'fuse_scores': fuse_scores})
                        fact_txt += caption + '\n'
                        fact_scores_txt += \
                            caption + '\n' + \
                            '\n'.join(
                                [f'{s_name}: ' + ' '.join([f'{val}' for val in s_vals]) + f', sum: {sum(s_vals)}'
                                 for s_name, s_vals in fuse_scores.items()]) + \
                            '\n' + '\n'
                json.dump(results, open(os.path.join(result_dir, 'result_%d.json' % epoch), 'w'))
                with open(os.path.join(result_dir, 'result_%d.txt' % epoch), 'w') as f:
                    f.write(fact_txt)
                with open(os.path.join(result_dir, 'result_%d_scores.txt' % epoch), 'w') as f:
                    f.write(fact_scores_txt)

            chkpoint = {
                'epoch': epoch,
                'model': captioner.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': opt.settings,
                'idx2word': idx2word,
                'sentiment_categories': opt.sentiment_categories,
                'dataset_name': dataset_name,
                'corpus_type': corpus_type,
            }
            checkpoint_path = os.path.join(checkpoint, 'model_%d_%.4f_%.4f_%s.pth' % (
                epoch, train_loss['all_loss'], val_loss['all_loss'], time.strftime('%m%d-%H%M')))
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
