import math
from typing import List, Tuple
import os
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cpu.amp import autocast, GradScaler
import numpy as np

from transformers import CLIPTokenizer
from tqdm import tqdm

from clip import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils.common import AverageMeter, accuracy
from utils.imagenet_templates import shuffled_template

from utils.compute_utils import cal_ood_score
import config

_tokenizer = _Tokenizer()
DOWNLOAD_ROOT = config.DOWNLOAD_ROOT

from utils.id_like_utils import TextEncoder

class CLIP(nn.Module):
    
    def __init__(self, args,
                 classnames, arch="ViT-B/16", device='cuda',
                 n_ctx=16, ctx_init=None, ctx_position='end'):
        super().__init__()
        arch = args.CLIP_ckpt
        print('model arch:', arch)
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)

        self.image_encoder = clip.visual
        self.clip = clip

        # self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        
        self.aug = args.aug
        if args.aug:
            self.template = shuffled_template
        else:
            self.template = [lambda c: f'a photo of a {c}.']

        self.classnames = classnames
        
        self.get_text_features()

   
        
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

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
        logits = torch.mean(logits_list, dim=0)

        return logits
    

    
    def get_text_features(self):
        text_lists = []

        prompts = [c.replace("_", " ") for c in self.classnames]
        prompts = torch.cat([tokenize(p) for p in prompts])
        prompts = prompts.cuda()
        # prompts = prompts.cpu()
        with torch.no_grad():
            self.clsname_features = self.clip.encode_text(prompts)

        for i, template in enumerate(self.template):
            prompts = [template(c.replace("_", " ")) for c in self.classnames]
            prompts = torch.cat([tokenize(p) for p in prompts])
            prompts = prompts.cuda()
            with torch.no_grad():
                text_features = self.clip.encode_text(prompts)
                text_lists.append(text_features)
        text_lists = torch.stack(text_lists)
        
        self.text_features = text_lists

    def get_image_features(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        return image_features



def get_zsclip_prompts(args, labels):        
    model = CLIP(args,
                 classnames=labels,
                 arch="ViT-B/16", device='cuda',
                 n_ctx=16, ctx_init=config.ctx_init, ctx_position=config.ctx_position)
    
    return model


def get_zsclip_result(args, model, loader, labels, ex_labels, if_acc=False):
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
        outputs = torch.cat(outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

    # scores
    outputs = outputs / args.T
    outputs_softmax = F.softmax(outputs, dim=1)
    scores = cal_ood_score(args, outputs, outputs_softmax)

    result['scores'] = scores
    # acc
    if if_acc:
        res = accuracy(outputs[:, :args.n_cls].cpu(), all_targets.detach().cpu())
        result['acc'] = [acc.item() for acc in res]

    return result

