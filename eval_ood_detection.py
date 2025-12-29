import os
import argparse
import numpy as np
import torch
from scipy import stats
import config
from utils.common import setup_seed, get_and_print_results, print_measures
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.dataloaders_utils import set_few_shot_loader, set_val_loader, set_ood_loader_ImageNet, set_train_loader

from trainers.id_like import get_idlike_prompts, get_idlike_result, load_model
from trainers.coop import get_coop_prompts
from trainers.zsclip import get_zsclip_prompts, get_zsclip_result
from trainers.zsclip_TAD import get_zsclipTAD_prompts, get_zsclipTAD_result

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def process_args():
    parser = argparse.ArgumentParser(description='Evaluates OOD for CLIP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root-dir', default="./datasets", type=str,
                        help='root dir of datasets')
    parser.add_argument('--in_dataset', default='ImageNet', type=str, help='in-distribution dataset')
    parser.add_argument('--ood_dataset', default='near_semantic', type=str, help='out-of-distribution dataset')
    parser.add_argument('--seed', default=1, type=int, help="random seed")
    parser.add_argument('--method', type=str, default='zsclip', help='the ood detection methods')
    parser.add_argument('--score', default='id-like', choices=['logitgap_topN', 'MCM', 'energy', 'max-logit', 'id-like', 'logitgap_softmax', 'logitgap_sqrt', 'logitgap_square', 'logitgap_idlike', 'logitgap_exp',  'gen', 'energy_topN', 'MCM_maxlogit', 'MCM_topN', 'margin',], type=str)
    parser.add_argument('--aug', action="store_true", default=False, help='TAG augmenation')
    parser.add_argument('--T', default=1.0, type=float, help='tempreature of softmax')
    parser.add_argument('--MCM_topN', type=int, default=5, help='the hyper-parameter topN for MCM_topN score')

    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50', 'RN101'],
                        help='which pretrained img encoder to use')
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')
    parser.add_argument('--test_batch_size', default=512, type=int, help='mini-batch size')
    
    # for id-like
    parser.add_argument('--n_shot', default=0, type=int,
                        help="how many samples are used to estimate classwise mean and precision matrix")
    parser.add_argument('--n_crop', default=256, type=int, help='crop num')
    parser.add_argument('--n_selection', default=32, type=int, help='selection num')
    # parser.add_argument('--selection_p', default=0.2, type=float, help='confidence selection percentile')
    parser.add_argument('--n_ex_prompts', default=100, type=int, help='number of extra prompts')
    parser.add_argument('--n_epoch', default=3, type=int, help='number of epoch')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float, metavar='LR', help='initial learning rate',
                        dest='lr')
    parser.add_argument('--lam_in', default=1.0, type=float, help='lambda of id loss')
    parser.add_argument('--lam_out', default=0.3, type=float, help='lambda of ood loss')
    parser.add_argument('--lam_diff', default=0.2, type=float, help='lambda of difference')

    args = parser.parse_args()

    arch_map = {'ViT-B/32':'vit-b32', 'ViT-B/16':'vit-b16', 'ViT-L/14':'vit-l14', 'RN50':'rn50', 'RN101':'rn101'}
    arch = arch_map[args.CLIP_ckpt]

    args.n_cls = config.data_info[args.in_dataset]['n_cls']

    if 'zsclip' in args.method:
        args.model_directory = f"results_{arch}/{args.in_dataset}_{args.ood_dataset}/{args.method}/{args.aug}/{args.score}/{args.T}_T"   # save path
        args.log_directory = args.model_directory  # save path
    elif args.method == 'id-like':
        args.model_directory = f"results_{arch}/{args.in_dataset}_{args.ood_dataset}/{args.method}/{args.n_shot}_shot_{args.n_epoch}epoch/id-like/{args.T}_T"   # save path
        args.log_directory = f"results_{arch}/{args.in_dataset}_{args.ood_dataset}/{args.method}/{args.n_shot}_shot_{args.n_epoch}epoch/{args.score}/{args.T}_T"   # save path
    elif args.method == 'coop':
        args.model_directory = f"results_{arch}/{args.in_dataset}_{args.ood_dataset}/{args.method}/{args.n_shot}_shot_{args.n_epoch}epoch/MCM/{args.T}_T"   # save path
        args.log_directory = f"results_{arch}/{args.in_dataset}_{args.ood_dataset}/{args.method}/{args.n_shot}_shot_{args.n_epoch}epoch/{args.score}/{args.T}_T"   # save path
    
    if 'topN' in args.score:
        args.model_directory = args.model_directory.replace(f'{args.T}_T', f'{args.MCM_topN}/{args.T}_T')
        args.log_directory = args.log_directory.replace(f'{args.T}_T', f'{args.MCM_topN}/{args.T}_T')
    

    os.makedirs(args.log_directory, exist_ok=True)
    setup_seed(args.seed)
    return args


def train():
    args = process_args()

    log = setup_log(args)

    log.info("Command line arguments:")
    for arg, value in vars(args).items():
        log.info(f"{arg}: {value}")
    
    if args.ood_dataset == 'near_semantic':
        out_datasets = ['ninco', 'ImageNet-O', 'ImageNetOOD']
    elif args.ood_dataset == 'far_semantic':
        out_datasets = ['iNaturalist', 'dtd', 'openimage_o']
    elif args.ood_dataset == 'covariate':
        out_datasets = ['ImageNet-R', 'ImageNet-A', 'ImageNet-Sketch']
    else:
        out_datasets = [args.ood_dataset]

    test_labels = config.data_info[args.in_dataset]['labels']

    if args.method == 'id-like':
        ex_labels = ['X'] * args.n_ex_prompts
    else:
        ex_labels = []

    log.info(f"ex_labels: {ex_labels}")
    log.info(f"len(ex_labels): {len(ex_labels)}")
    
    model_checkpoint_save_path = os.path.join(args.model_directory, 'model_checkpoint.pth')

    if args.method in ['id-like', 'coop'] and os.path.exists(model_checkpoint_save_path):
        log.info(f"Loading model from {model_checkpoint_save_path}")
        model = load_model(args, test_labels, ex_labels)
    else:
        if args.method == 'zsclip':
            model = get_zsclip_prompts(args, test_labels)
        elif args.method == 'zsclipTAD':
            model = get_zsclipTAD_prompts(args, test_labels)
        elif args.method == 'coop':
            few_shot_loader = set_train_loader(args, subset=True, max_count=args.n_shot)
            model = get_coop_prompts(args, few_shot_loader, test_labels)
        elif args.method == 'id-like':
            few_shot_loader = set_few_shot_loader(args)
            model = get_idlike_prompts(args, few_shot_loader, test_labels, ex_labels)
        
        
        
    test_loader = set_val_loader(args)

    
    if args.method == 'zsclip':
        result_in = get_zsclip_result(args, model, test_loader, test_labels, ex_labels, if_acc=True)
    elif args.method == 'zsclipTAD':
        result_in = get_zsclipTAD_result(args, model, test_loader, test_labels, ex_labels, if_acc=True)
    elif args.method == 'id-like' or args.method == 'coop':
        result_in = get_idlike_result(args, model, test_loader, test_labels, ex_labels, if_acc=True)

    score_in = result_in['scores']
    acc = result_in['acc']
    log.debug(f"Acc: {acc}")

    auroc_list, aupr_list, fpr_list = [], [], []
    for out_dataset in out_datasets:
        log.debug(f"Evaluting OOD dataset {out_dataset}")
        if args.method == 'finetune_rn50':
            ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess=data_transform)
        else:
            ood_loader = set_ood_loader_ImageNet(args, out_dataset)
        if args.method == 'zsclip':
            result_out = get_zsclip_result(args, model, ood_loader, test_labels, ex_labels)
        elif args.method == 'zsclipTAD':
            result_out = get_zsclipTAD_result(args, model, ood_loader, test_labels, ex_labels)
        elif args.method == 'id-like' or args.method == 'coop':
            result_out = get_idlike_result(args, model, ood_loader, test_labels, ex_labels)

        score_out = result_out['scores']
        log.debug(f"in scores: {stats.describe(score_in)}")
        log.debug(f"out scores: {stats.describe(score_out)}")

        # normalize
        min_score = min(np.min(score_in), np.min(score_out))
        max_score = max(np.max(score_in), np.max(score_out))
        score_in_norm = (score_in - min_score) / (max_score - min_score) -1
        score_out_norm = (score_out - min_score) / (max_score - min_score) -1
        log.debug("normalized in scores and out scores")

        plot_distribution(args, score_in_norm, score_out_norm, out_dataset)
        auroc, aupr, fpr, fpr_threshold = get_and_print_results(args, log, score_in, score_out,
                              auroc_list, aupr_list, fpr_list)
    log.debug('\n\nMean Test Results')
    print_measures(log, np.mean(auroc_list), np.mean(aupr_list),
                   np.mean(fpr_list), method_name=args.score)
    save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)


if __name__ == "__main__":
    train()
