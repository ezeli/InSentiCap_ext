import argparse
import json
import torch


def parse_opt():
    parser = argparse.ArgumentParser()

    # train settings
    # train concept detector
    parser.add_argument('--concept_lr', type=float, default=4e-5)  # 4e-5 for video, 4e-4 for image
    parser.add_argument('--concept_bs', type=int, default=80)
    parser.add_argument('--concept_resume', type=str, default='')
    parser.add_argument('--concept_epochs', type=int, default=100)
    parser.add_argument('--concept_num_works', type=int, default=2)

    # train sentiment detector
    parser.add_argument('--senti_lr', type=float, default=4e-4)
    parser.add_argument('--senti_bs', type=int, default=80)
    parser.add_argument('--senti_resume', type=str, default='')
    parser.add_argument('--senti_epochs', type=int, default=30)
    parser.add_argument('--senti_num_works', type=int, default=0)

    parser.add_argument('--img_senti_labels', type=str, default='./data/captions/img_senti_labels.json')
    parser.add_argument('--vid_senti_labels', type=str, default='./data/captions/vid_senti_labels.json')
    parser.add_argument('--sentiment_categories', type=list, default=['positive', 'negative', 'neutral'])

    # train full model
    # xe
    parser.add_argument('--xe_lr', type=float, default=4e-4)
    parser.add_argument('--xe_bs', type=int, default=20)
    parser.add_argument('--xe_resume', type=str, default='')
    parser.add_argument('--xe_epochs', type=int, default=40)
    parser.add_argument('--xe_num_works', type=int, default=2)

    parser.add_argument('--scheduled_sampling_start', type=int, default=0)
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=4)
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05)
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25)

    # rl
    parser.add_argument('--rl_lr', type=float, default=4e-5)
    parser.add_argument('--rl_bs', type=int, default=20)
    parser.add_argument('--rl_num_works', type=int, default=2)
    parser.add_argument('--rl_resume', type=str, default='')
    parser.add_argument('--rl_senti_resume', type=str, default='checkpoint/sentiment/model-10.pth')
    parser.add_argument('--rl_epochs', type=int, default=40)

    # common
    parser.add_argument('--dataset_name', type=str, default='msrvtt', choices=['coco', 'flickr30k', 'msrvtt'])
    parser.add_argument('--corpus_type', type=str, default='part', choices=['part', 'full'])
    parser.add_argument('--captions_dir', type=str, default='./data/captions')
    parser.add_argument('--feats_dir', type=str, default='./data/features')
    parser.add_argument('--corpus_dir', type=str, default='./data/corpus')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/')
    parser.add_argument('--result_dir', type=str, default='./result/')
    # parser.add_argument('--sentence_sentiment_classifier_rnn', type=str, default='')
    parser.add_argument('--max_seq_len', type=int, default=16)
    parser.add_argument('--num_concepts', type=int, default=5)
    parser.add_argument('--num_sentiments', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=0.1)

    # eval settings
    parser.add_argument('-e', '--eval_model', type=str, default='')
    parser.add_argument('-r', '--result_file', type=str, default='')
    parser.add_argument('--beam_size', type=int, default=3)

    # test settings
    parser.add_argument('-t', '--test_model', type=str, default='')
    parser.add_argument('-i', '--image_file', type=str, default='')
    # encoder settings
    parser.add_argument('--resnet101_file', type=str, default='./data/pre_models/resnet101.pth',
                        help='Pre-trained resnet101 network for extracting image features')

    args = parser.parse_args()

    # network settings
    settings = dict()
    settings['att_feat_dim'] = 2048
    settings['fc_feat_dim'] = 2048
    settings['audio_feat_dim'] = 128
    settings['d_model'] = 512  # model dim
    settings['d_ff'] = 2048  # feed forward dim
    settings['h'] = 8  # multi heads num
    settings['N_enc'] = 4  # encoder layers num
    settings['N_dec'] = 4  # decoder layers num
    settings['dropout_p'] = 0.5
    settings['max_seq_len'] = args.max_seq_len

    settings['word_emb_dim'] = settings['d_model']
    settings['feat_emb_dim'] = settings['d_model']
    settings['rnn_hid_dim'] = settings['d_model']
    settings['att_hid_dim'] = settings['d_model']

    settings['concept_mid_him'] = 1024
    settings['sentiment_convs_num'] = 2
    # settings['num_kernels_per_sentiment'] = 4
    settings['sentiment_fcs_num'] = 2
    settings['text_cnn_filters'] = (3, 4, 5)
    settings['text_cnn_out_dim'] = 256

    args.settings = settings
    args.use_gpu = torch.cuda.is_available()
    args.device = torch.device('cuda:0') if args.use_gpu else torch.device('cpu')
    return args
