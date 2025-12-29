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

import config

_tokenizer = _Tokenizer()
DOWNLOAD_ROOT = config.DOWNLOAD_ROOT

from utils.id_like_utils import PromptLearner, TextEncoder

class CoOp(nn.Module):
    
    def __init__(self, args,
                 classnames, arch="ViT-B/16", device='cuda',
                 n_ctx=16, ctx_init=None, ctx_position='end'):
        super().__init__()

        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)

        self.image_encoder = clip.visual

        self.text_encoder = TextEncoder(clip)
        self.text_encoder = nn.parallel.DataParallel(self.text_encoder).to(torch.device("cuda"))  # for mutil GPU
        self.logit_scale = clip.logit_scale.data
        # prompt
        self.prompt_learner = PromptLearner(clip, classnames, n_ctx,
                                            ctx_init, ctx_position, learned_cls=False)
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        text_features = self.get_text_features()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


    def get_text_features(self):
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        prompts = prompts.cuda(non_blocking=True)  # for mutil GPU
        tokenized_prompts = tokenized_prompts.cuda(non_blocking=True)  # for mutil GPU
        t_features = self.text_encoder(prompts, tokenized_prompts)
        return t_features

    def get_image_features(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        return image_features


def get_coop_loss(args, prompt_features, image_features_in, targets_in, logit_scale):
    # loss_in
    logit_in = logit_scale * image_features_in @ prompt_features.t()
    loss_in = F.cross_entropy(logit_in, targets_in)

    # loss
    loss = loss_in 
    loss_str = f'Loss_now:{loss.detach().cpu().item():.6f}\t' \
               f'Loss_in:{loss_in.detach().cpu().item():.6f}\t' 

    return loss, loss_str




def get_coop_prompts(args, loader, labels):
    model = CoOp(args,
                 classnames=labels,
                 arch="ViT-B/16", device='cuda',
                 n_ctx=16, ctx_init=config.ctx_init, ctx_position=config.ctx_position)
    loss_meter = AverageMeter()
    optimizer = torch.optim.AdamW([{'params': model.prompt_learner.parameters()},], args.lr)

    for epoch in range(args.n_epoch):

        tqdm.write(f'Train epoch:{epoch + 1}/{args.n_epoch}')
        for batch_idx, (images, targets) in enumerate(tqdm(loader)):
            with torch.no_grad():
                images = images.cuda()
                targets = targets.cuda()
                image_features = model.get_image_features(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # train
            # get prompts
            logit_scale = model.logit_scale.exp()
            prompt_features = model.get_text_features()
            prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)
            loss, loss_str = get_coop_loss(args, prompt_features,
                                      image_features,
                                      targets, logit_scale)

            # update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_meter.update(loss.detach().cpu().item())
            tqdm.write(f'Train epoch:{epoch + 1}/{args.n_epoch}\t'
                       f'Loss_avg:{loss_meter.avg:.6f}\t' + loss_str)

        if epoch+1 == args.n_epoch:
            model_save_dir = args.log_directory
            os.makedirs(model_save_dir, exist_ok=True)
            model_checkpoint_save_path = os.path.join(model_save_dir, 'model_checkpoint.pth')
            model_checkpoint = {
                'prompt_learner_state_dict': model.prompt_learner.state_dict()
            }
            torch.save(model_checkpoint, model_checkpoint_save_path)

    return model
