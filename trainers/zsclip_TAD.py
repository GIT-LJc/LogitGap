import math
from typing import List, Tuple
import os
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from transformers import CLIPTokenizer
from tqdm import tqdm

from clip import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils.common import AverageMeter, accuracy

from utils.compute_utils import cal_ood_score
import config

_tokenizer = _Tokenizer()
DOWNLOAD_ROOT = config.DOWNLOAD_ROOT

from utils.id_like_utils import TextEncoder
from .zsclip import CLIP, get_zsclip_prompts

class CLIP_TAD(CLIP):
    
    def __init__(self, args,
                 classnames, arch="ViT-B/16", device='cuda',
                 n_ctx=16, ctx_init=None, ctx_position='end'):
        super().__init__(args,
                 classnames, arch="ViT-B/16", device='cuda',
                 n_ctx=16, ctx_init=None, ctx_position='end')
            
    
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits_list = []

        for text_features in self.text_features:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.t()
            logits_list.append(logits)
        
        logits_list = torch.cat(logits_list)
        logits_list = logits_list.reshape(len(self.text_features), logits.shape[0],logits.shape[1])
        # logits = torch.mean(logits_list, dim=0)

        return logits_list

def get_zsclipTAD_prompts(args, labels):        
    model = CLIP_TAD(args,
                 classnames=labels,
                 arch="ViT-B/16", device='cuda',
                 n_ctx=16, ctx_init=config.ctx_init, ctx_position=config.ctx_position)
    
    return model



def get_zsclipTAD_result(args, model, loader, labels, ex_labels, if_acc=False):
    tqdm_object = tqdm(loader, total=len(loader))
    outputs = []
    all_targets = []

    result = {
        'scores': None,
        'acc': None,
        'id_indexs_dict': None,
    }

    with torch.no_grad():
        
        for batch_idx, (images, targets) in enumerate(tqdm_object):
            images = images.cuda()
            targets = targets.long().cuda()
            output = model(images)
            output = output.detach().cuda()
            outputs.append(output)
            all_targets.append(targets)
        outputs = torch.cat(outputs, dim=1)   # [M,B,C]
        all_targets = torch.cat(all_targets, dim=0)

    # scores
    outputs = outputs / args.T
    scores_sum = 0
    for i in range(outputs.shape[0]):
        output_i = outputs[i]
        outputs_softmax = F.softmax(output_i, dim=1)
        scores = cal_ood_score(args, output_i, outputs_softmax)
        scores_sum += scores
    
    scores = scores_sum / outputs.shape[0]
    outputs = outputs.mean(dim=0)  # [B,C]

    result['scores'] = scores
    # acc
    if if_acc:
        res = accuracy(outputs[:, :args.n_cls].cpu(), all_targets.detach().cpu())
        result['acc'] = [acc.item() for acc in res]

    return result

