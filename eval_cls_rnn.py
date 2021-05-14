import sys
import torch
import tqdm
import numpy as np
import os

from models.sent_senti_cls import SentenceSentimentClassifier
from dataloader import get_senti_sents_dataloader

device = torch.device('cuda:0')
max_seq_len = 16

result_dir = './result/eval_cls'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


def compute_cls(captions_file_prefix):
    dataset_name = 'coco'
    if 'flickr30k' in captions_file_prefix:
        dataset_name = 'flickr30k'
    corpus_type = 'part'
    if 'full' in captions_file_prefix:
        corpus_type = 'full'

    ss_cls_file = os.path.join('./checkpoint', 'sent_senti_cls', dataset_name, corpus_type, 'model-best.pth')
    print("====> loading checkpoint '{}'".format(ss_cls_file))
    chkpoint = torch.load(ss_cls_file, map_location=lambda s, l: s)
    settings = chkpoint['settings']
    idx2word = chkpoint['idx2word']
    sentiment_categories = chkpoint['sentiment_categories']
    assert dataset_name == chkpoint['dataset_name'], \
        'dataset_name and resume model dataset_name are different'
    assert corpus_type == chkpoint['corpus_type'], \
        'corpus_type and resume model corpus_type are different'
    model = SentenceSentimentClassifier(idx2word, sentiment_categories, settings)
    model.load_state_dict(chkpoint['model'])
    model.eval()
    model.to(device)

    val_sets = {}
    val_sets['all'] = []
    for senti_id, senti in enumerate(sentiment_categories):
        val_sets[senti] = []
        fn = '%s_%s.txt' % (captions_file_prefix, senti)
        with open(fn, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split()
            line = [int(l) for l in line]
            val_sets[senti].append([senti_id, line])
            val_sets['all'].append([senti_id, line])

    val_datas = {}
    for senti in val_sets:
        val_datas[senti] = get_senti_sents_dataloader(val_sets[senti], idx2word.index('<PAD>'), max_seq_len,
                                                      shuffle=False)

    for senti, val_data in val_datas.items():
        all_num = 0
        wrong_num = 0
        cor_res = ''
        wro_res = ''
        with torch.no_grad():
            for sentis, (caps_tensor, lengths) in tqdm.tqdm(val_data, ncols=100):
                sentis = sentis.to(device)
                caps_tensor = caps_tensor.to(device)

                rest, _, scores, att_weights = model.sample(caps_tensor, lengths)
                rest = torch.LongTensor(np.array(rest)).to(device)
                all_num += int(sentis.size(0))
                wrong_num += int((sentis != rest).sum())

                for cap_idx, cap_len, senti_idx, rest_idx, score, att_weight in zip(caps_tensor, lengths, sentis, rest, scores, att_weights):
                    cap = ' '.join([idx2word[idx] for idx in cap_idx[:cap_len]])
                    real_senti = sentiment_categories[int(senti_idx)]
                    pred_senti = sentiment_categories[int(rest_idx)]
                    att_weight = att_weight.detach().cpu().numpy().tolist()
                    att_weight = ' '.join(['%.3f' % w for w in att_weight])
                    cap = '  |  '.join([cap, real_senti, pred_senti, '%.4f' % score, att_weight]) + '\n'
                    if real_senti == pred_senti:
                        cor_res += cap
                    else:
                        wro_res += cap
        wrong_rate = wrong_num / all_num
        print('%s acc_rate: %.6f' % (senti, 1 - wrong_rate))
        with open(os.path.join(result_dir, senti+'_cor.txt'), 'w') as f:
            f.write(cor_res)
        with open(os.path.join(result_dir, senti+'_wro.txt'), 'w') as f:
            f.write(wro_res)


if __name__ == "__main__":
    compute_cls(sys.argv[1])
