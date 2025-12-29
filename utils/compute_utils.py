import numpy as np
import torch
import torch.nn.functional as F


def cal_ood_score(args, output, output_softmax, others_num=None):
    if args.score == 'logitgap_topN':
        score = -1*compute_k_largest_diff_topn(output, k=1, topn=args.MCM_topN)
    elif args.score == 'logitgap_idlike':
        score = -1*(compute_k_largest_diff_topn(output[:, :args.n_cls], k=1, topn=args.MCM_topN))

    elif args.score == 'MCM':
        score = -1*torch.max(output_softmax, dim=1)[0].detach().cpu().squeeze().numpy()
    elif args.score == 'max-logit':
        score = -1*torch.max(output, dim=1)[0].detach().cpu().squeeze().numpy()
    elif args.score == 'energy':
        score = -1*torch.logsumexp(output, dim=1).detach().cpu().squeeze().numpy() 
    elif args.score == 'id-like':
        score = torch.sum(output_softmax[:, args.n_cls:], dim=1).detach().cpu().squeeze().numpy() - 1

    elif args.score == 'logitgap_exp':
        output_exp = torch.exp(output)
        score = -1*compute_k_largest_diff_topn(output_exp, k=1, topn=args.MCM_topN)
    elif args.score == 'logitgap_square':
        output_square = torch.square(output)
        score = -1*compute_k_largest_diff_topn(output_square, k=1, topn=args.MCM_topN)
    elif args.score == 'logitgap_sqrt':
        score = -1*compute_k_largest_diff_sqrt(output, k=1, topn=args.MCM_topN)
    elif args.score == 'logitgap_softmax':   # equal to MCM
        # score = -1*compute_k_largest_diff(output_softmax, k=1)
        score = -1*compute_k_largest_diff_topn(output_softmax, k=1, topn=args.n_cls-1)

    elif args.score =="gen":
        score = -1 * generalized_entropy(output_softmax)
    elif args.score == 'energy_topN':
        score = -1*entropy_topN(output)
    elif args.score == 'margin':
        top2 = torch.topk(output_softmax,2)[0]
        score = -1*(top2[:,0] - top2[:,1]).detach().cpu().squeeze().numpy()
    elif args.score == 'MCM_maxlogit':
        mcm_score = torch.max(output_softmax, dim=1)[0].detach().cpu().squeeze().numpy()
        maxlogit_score = torch.max(output, dim=1)[0].detach().cpu().squeeze().numpy()
        score = -1*(mcm_score * maxlogit_score)
    elif args.score == 'MCM_topN':   # use topN to calculate the softmax score
        output_topN = torch.topk(output, args.MCM_topN, dim=1)[0]
        output_softmax_topN = F.softmax(output_topN, dim=1)
        score = -1*torch.max(output_softmax_topN, dim=1)[0].detach().cpu().squeeze().numpy()

    else:
        raise NotImplementedError

    return score


def compute_k_largest_diff_topn(logits, k=1, topn=0):
    logits = logits.cpu().detach().numpy()
    result = []
    for logit in logits:
        sorted_logit = np.sort(logit)
        kth_largest = sorted_logit[-k]
        if topn > 0:
            diff_mean = np.mean(kth_largest - sorted_logit[-k-topn:-k])
        elif topn==0:
            diff_mean = kth_largest    
        result.append(diff_mean)
    return np.array(result)


def compute_k_largest_diff_sqrt(logits, k=1, topn=0):
    logits = logits.cpu().detach().numpy()
    result = []
    for logit in logits:
        sorted_logit = np.sort(logit)
        kth_largest = sorted_logit[-k]
        diff_mean = (kth_largest - sorted_logit[-k-topn:-k]) ** 2
        diff_mean = np.mean(diff_mean)
        result.append(diff_mean)
    return np.array(result)



def generalized_entropy(output_softmax, gamma=0.1, M=100):
    probs =  output_softmax.cpu().detach().numpy()
    probs_sorted = np.sort(probs, axis=1)[:,-M:]
    scores = np.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma), axis=1)
    return -scores 

def entropy_topN(output):
    topN = int(output.shape[1] * 0.1)
    output_sorted = torch.sort(output, dim=1)[0][:,-topN:]
    score = torch.logsumexp(output_sorted, dim=1).detach().cpu().squeeze().numpy()
    return score