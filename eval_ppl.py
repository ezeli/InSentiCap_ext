import json
import sys
import numpy as np
import os
# import kenlm

sentis = ['positive', 'negative', 'neutral']
lm_cmd = 'ngram -ppl %s -lm ./data/captions/%s/%s/lm/%s_w.sri'


def compute_ppl(captions_file_prefix):
    dataset_name = 'coco'
    if 'flickr30k' in captions_file_prefix:
        dataset_name = 'flickr30k'
    elif 'msrvtt' in captions_file_prefix:
        dataset_name = 'msrvtt'
    corpus_type = 'part'
    if 'full' in captions_file_prefix:
        corpus_type = 'full'
    train_mode = 'rl'
    if '/xe/' in captions_file_prefix:
        train_mode = 'xe'

    lm_cmds = {}
    for senti in sentis:
        if train_mode == 'rl':
            lm_cmds[senti] = lm_cmd % ('%s_%s_w.txt' % (captions_file_prefix, senti), dataset_name, corpus_type, senti)
        else:
            lm_cmds[senti] = lm_cmd % ('%s.txt' % captions_file_prefix, dataset_name, corpus_type, senti)
    # print('lm cms:', lm_cmds)
    scores = {}
    for senti, cmd in lm_cmds.items():
        out = os.popen(cmd).read().split()
        try:
            scores[senti] = round(float(out[out.index('ppl=') + 1]), 1)
        except Exception:
            scores[senti] = 0

    print('ppl scores:', scores)
    print('ppl scores sum:', sum(scores.values()))
    return scores


if __name__ == "__main__":
    compute_ppl(sys.argv[1])
