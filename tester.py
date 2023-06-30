import os
import sys
import json
import torch
import pickle
import logging
import argparse

import evaluation
from model import get_model
from validate import norm_score, cal_perf

import util.tag_data_provider as data
from util.text2vec import get_text_encoder
import util.metrics as metrics
from util.vocab import Vocabulary

from basic.util import read_dict, log_config
from basic.constant import ROOT_PATH
from basic.bigfile import BigFile
from basic.common import makedirsforfile, checkToSkip

def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('--testCollection', type=str, help='test collection')
    parser.add_argument('--collectionStrt', type=str, default='single', help='collection structure (single|multiple)')
    parser.add_argument('--split', default='test', type=str, help='split, only for single-folder collection structure (val|test)')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],  help='overwrite existed file. (default: 0)')
    parser.add_argument('--log_step', default=100, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=5, type=int, help='Number of data loader workers.')
    parser.add_argument('--logger_name', default='runs', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth.tar', type=str, help='name of checkpoint (default: model_best.pth.tar)')

    args = parser.parse_args()
    return args


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    collectionStrt = opt.collectionStrt
    resume = os.path.join(opt.logger_name, opt.checkpoint_name)

    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)

    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_rsum = checkpoint['best_rsum']
    print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
          .format(resume, start_epoch, best_rsum))
    options = checkpoint['opt']
    
    # collection setting
    testCollection = opt.testCollection
    collections_pathname = options.collections_pathname
    collections_pathname['test'] = testCollection

    trainCollection = options.trainCollection
    output_dir = resume.replace(trainCollection, testCollection)
    if 'checkpoints' in output_dir:
        output_dir = output_dir.replace('/checkpoints/', '/results/')
    else:
        output_dir = output_dir.replace('/%s/' % options.cv_name, '/results/%s/%s/' % (options.cv_name, trainCollection))
    result_pred_sents = os.path.join(output_dir, 'id.sent.score.txt')
    pred_error_matrix_file = os.path.join(output_dir, 'pred_errors_matrix.pth.tar')
    if checkToSkip(pred_error_matrix_file, opt.overwrite):
        sys.exit(0)
    makedirsforfile(pred_error_matrix_file)

    log_config(output_dir)
    logging.info(json.dumps(vars(opt), indent=2))

    opt.split='train'

    # data loader prepare
    test_cap = os.path.join(rootpath, collections_pathname['test'], 'TextData', '%s.caption.txt'%testCollection)
    if collectionStrt == 'single':
        test_cap = os.path.join(rootpath, collections_pathname['test'], 'TextData', '%s%s.caption.txt'%(testCollection, opt.split))
    elif collectionStrt == 'multiple':
        test_cap = os.path.join(rootpath, collections_pathname['test'], 'TextData', '%s.caption.txt'%testCollection)
    else:
        raise NotImplementedError('collection structure %s not implemented' % collectionStrt)

    caption_files = {'test': test_cap}
    img_feat_path = os.path.join(rootpath, collections_pathname['test'], 'FeatureData', options.visual_feature)
    visual_feats = {'test': BigFile(img_feat_path)}
    assert options.visual_feat_dim == visual_feats['test'].ndims
    video2frames = {'test': read_dict(os.path.join(rootpath, collections_pathname['test'], 'FeatureData', options.visual_feature, 'video2frames.txt'))}

    # set bow vocabulary and encoding
    bow_vocab_file = os.path.join(rootpath, collections_pathname['train'], 'TextData', 'vocabulary', 'bow', options.vocab+'.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)
    options.bow_vocab_size = len(bow_vocab)

    # set rnn vocabulary 
    rnn_vocab_file = os.path.join(rootpath, collections_pathname['train'], 'TextData', 'vocabulary', 'rnn', options.vocab+'.pkl')
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    options.vocab_size = len(rnn_vocab)

    # Construct the model
    model = get_model(options.model)(options)
    model.load_state_dict(checkpoint['model'])
    model.Eiters = checkpoint['Eiters']
    model.val_start()

    # set data loader
    video_ids_list = data.read_video_ids(caption_files['test'])
    vid_data_loader = data.get_vis_data_loader(visual_feats['test'], options.support_set_number_V, opt.batch_size, opt.workers, video2frames['test'], video_ids=video_ids_list)
    if not hasattr(options, 'support_set_number'):
        options.support_set_number = 20
    text_data_loader = data.get_txt_data_loader(caption_files['test'], rnn_vocab, bow2vec, options.support_set_number_T, opt.batch_size, opt.workers, video2frames['test'], visual_feats['test'])

    # mapping
    video_embs, video_ids = evaluation.encode_vid(model.embed_vis, vid_data_loader)
    cap_embs, caption_ids = evaluation.encode_text(model.embed_txt, text_data_loader)
    # if options.style == 'distill_from_best_model' and options.student_model == 'text+video':
    #     video_embs, video_ids = evaluation.encode_vid(model.embed_vis_distill, vid_data_loader)
    # else:
    #     video_embs, video_ids = evaluation.encode_vid(model.embed_vis, vid_data_loader)

    # if options.style == 'distill_from_best_model':
    #     cap_embs, caption_ids = evaluation.encode_text(model.embed_txt_distill, text_data_loader, options.style)
    # elif options.style == 'GT':
    #     cap_embs, caption_ids = evaluation.encode_text(model.embed_txt_GT, text_data_loader, options.style)


    # v2t_gt, t2v_gt = metrics.get_gt(video_ids, caption_ids)

    logging.info("write into: %s" % output_dir)
 
    t2v_all_errors_1 = evaluation.cal_error(video_embs, cap_embs, options.measure)

    # cal_perf(t2v_all_errors_1, v2t_gt, t2v_gt)
    # torch.save({'errors': t2v_all_errors_1, 'videos': video_ids, 'captions': caption_ids}, pred_error_matrix_file)    
    torch.save({'errors': t2v_all_errors_1, 'videos': video_ids, 'captions': caption_ids}, 'retrieval.pth')    
    logging.info("write into: %s" % pred_error_matrix_file)



if __name__ == '__main__':
    main()
