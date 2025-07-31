
import torch
import pdb

def recall(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / labels.sum(1).float()).mean().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(n, k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()

def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}
    
    answer_count = labels.sum(dim=1).float()
    labels_float = labels.float()
    _, rank_indices = scores.sort(dim=1, descending=True)

    for k in sorted(ks, reverse=True):
        topk_indices = rank_indices[:, :k]
        hits = labels_float.gather(dim=1, index=topk_indices)

        recall = hits.sum(dim=1) / answer_count.clamp(min=1)
        metrics['Recall@%d' % k] = recall.mean().item()

        position = torch.arange(2, 2 + k, device=scores.device).float()
        weights = 1.0 / torch.log2(position)
        dcg = (hits * weights[:k]).sum(dim=1)
        max_weights = weights.cumsum(dim=0)
        idcg = max_weights[torch.min(answer_count.long(), torch.tensor(k - 1, device=scores.device))]
        # idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count], device=scores.device)
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg.item()
        dcg = (hits * weights[:k]).sum(dim=1)
        effective_rel = torch.min(answer_count.long(), torch.tensor(k, device=scores.device))
        idcg_weights = weights[:k].cumsum(dim=0)
        idcg = idcg_weights[effective_rel - 1]
        ndcg = (dcg / idcg).mean()
        # ndcg = dcg.mean()
        metrics['NDCG@%d' % k] = ndcg.item()
    return metrics

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]
