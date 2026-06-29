import numpy as np
import torch



def ndcg_k(scores, targets, k):
    """nDCG@k

    Args:
        scores (np.ndarray): Score matrix with shape (N_queries, N_docs).
        targets (list[int] or list[list[int]]): Relevant document index or indices for each query.
        k (int): Number of top-ranked documents to evaluate.

    Returns:
        float: Mean nDCG@k over all queries.
    """
    ndcg_scores = []
    n_queries = scores.shape[0]

    for i in range(n_queries):
        target_indices = targets[i]
        if isinstance(target_indices, (int, np.integer)):
            target_indices = [target_indices]

        target_set = set(target_indices)

        if not target_set:
            ndcg_scores.append(0.0)
            continue

        top_k_indices = np.argsort(scores[i], kind="mergesort")[::-1][:k]

        dcg = 0.0
        for rank, doc_idx in enumerate(top_k_indices):
            if doc_idx in target_set:
                dcg += 1.0 / np.log2(rank + 2)

        idcg = 0.0
        num_relevant = len(target_set)
        for rank in range(min(num_relevant, k)):
            idcg += 1.0 / np.log2(rank + 2)

        if idcg == 0.0:
            ndcg_scores.append(0.0)
        else:
            ndcg_scores.append(dcg / idcg)

    return np.mean(ndcg_scores)


def recall_at_k(scores, targets, k):
    """Recall@k

    Args:
        scores (np.ndarray): Score matrix with shape (N_queries, N_docs).
        targets (list[int]): Relevant document index for each query.
        k (int): Number of top-ranked documents to evaluate.

    Returns:
        float: Mean Recall@k over all queries.
    """
    hits = 0
    n_queries = len(scores)

    for i in range(n_queries):
        target_idx = targets[i]
        top_k_indices = np.argsort(scores[i], kind="mergesort")[::-1][:k]

        if target_idx in top_k_indices:
            hits += 1

    return hits / n_queries


def ndcg_k_graded_matrix(scores, targets_dicts, k=5):
    """Graded nDCG@k

    Args:
        scores (np.ndarray): Score matrix with shape (N_queries, N_docs).
        targets_dicts (list[dict[int, float]]): Relevance dictionary for each query,
            where keys are document indices and values are relevance scores.
        k (int): Number of top-ranked documents to evaluate.

    Returns:
        float: Mean graded nDCG@k over all queries.
    """
    ndcgs = []

    for i in range(scores.shape[0]):
        rel_dict = targets_dicts[i]

        if rel_dict is None or len(rel_dict) == 0:
            ndcgs.append(0.0)
            continue

        top_k = np.argsort(scores[i], kind="mergesort")[::-1][:k]

        dcg = 0.0
        for rank, doc_idx in enumerate(top_k):
            rel = rel_dict.get(int(doc_idx), 0)
            gain = float(rel)
            dcg += gain / np.log2(rank + 2)

        ideal_rels = sorted(rel_dict.values(), reverse=True)[:k]

        idcg = 0.0
        for rank, rel in enumerate(ideal_rels):
            gain = float(rel)
            idcg += gain / np.log2(rank + 2)

        ndcgs.append(0.0 if idcg == 0.0 else dcg / idcg)

    return float(np.mean(ndcgs))


def recall_at_k_multi(scores, targets, k):
    """Recall@k for multiple relevant documents

    Args:
        scores (np.ndarray): Score matrix with shape (N_queries, N_docs).
        targets (list[int], list[list[int]], list[set[int]], or list[dict[int, float]]):
            Relevant document indices for each query.
        k (int): Number of top-ranked documents to evaluate.

    Returns:
        float: Mean Recall@k over all queries.
    """
    hits = 0
    n_queries = scores.shape[0]

    for i in range(n_queries):
        target = targets[i]

        if isinstance(target, (int, np.integer)):
            target_set = {int(target)}
        elif isinstance(target, dict):
            target_set = set(int(doc_idx) for doc_idx in target.keys())
        else:
            target_set = set(int(doc_idx) for doc_idx in target)

        top_k = np.argsort(scores[i], kind="mergesort")[::-1][:k]

        if any(doc_idx in target_set for doc_idx in top_k):
            hits += 1

    return hits / n_queries

def maxsim_scores(q_embeds, d_embeds, q_mask, d_mask):
    """Compute ColBERT-style MaxSim scores with padded inputs and masks.

    Args:
        q_embeds (torch.Tensor): Query embeddings with shape
            (num_queries, max_query_len, dim).
        d_embeds (torch.Tensor): Document embeddings with shape
            (num_docs, max_doc_len, dim).
        q_mask (torch.Tensor): Query padding mask with shape
            (num_queries, max_query_len), where 1 indicates a valid token and
            0 indicates padding.
        d_mask (torch.Tensor): Document padding mask with shape
            (num_docs, max_doc_len), where 1 indicates a valid token and
            0 indicates padding.

    Returns:
        torch.Tensor: Score matrix with shape (num_queries, num_docs).
    """
    d_embeds_T = d_embeds.transpose(1, 2)

    scores_list = []

    for i in range(len(q_embeds)):
        q = q_embeds[i].unsqueeze(0)
        q_m = q_mask[i].unsqueeze(0)

        sim = torch.matmul(q, d_embeds_T)
        sim = sim.masked_fill(d_mask.unsqueeze(1) == 0, -1e9)

        max_sim, _ = sim.max(dim=2)
        max_sim = max_sim.masked_fill(q_m == 0, 0.0)

        score = max_sim.sum(dim=1)
        scores_list.append(score)

    return torch.stack(scores_list)
