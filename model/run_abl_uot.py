import argparse
import os
import time
import numpy as np
import torch

from utils import (
    ndcg_k_graded_matrix,
    recall_at_k_multi,
    maxsim_scores,
    recall_at_k,
    ndcg_k,
)
from abl_uot import ot_ablation_baselines
from logger_abl import ExperimentLogger


def get_args():
    parser = argparse.ArgumentParser(description="OT/UOT ablation experiment runner")

    parser.add_argument("--model", type=str, default="colqwen", help="Model name")
    parser.add_argument("--data_dir", type=str, default="data", help="Input data directory")
    parser.add_argument("--exp_dir", type=str, default="experiments", help="Experiment log directory")
    parser.add_argument("--exp_name", type=str, default="abl_uot", help="Experiment name")

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "arxivqa",
            "docvqa",
            "infovqa",
            "tabfquad",
            "tatdqa",
            "shiftproject",
            "artificial_intelligence",
            "energy",
            "government_reports",
            "healthcare_industry",
        ],
        help="List of datasets to evaluate",
    )

    parser.add_argument("--mfs", nargs="+", type=int, default=[5, 10, 25, 50], help="Merging factors")
    parser.add_argument("--gamma", type=float, default=0.04, help="Sinkhorn entropy regularization")
    parser.add_argument("--rho", type=float, default=10.0, help="Unbalanced Sinkhorn regularization")
    parser.add_argument("--n_iter", type=int, default=20, help="Number of EM iterations")
    parser.add_argument("--sinkhorn_iter", type=int, default=3, help="Number of Sinkhorn iterations")

    parser.add_argument(
        "--ablation_type",
        type=int,
        default=5,
        choices=[4, 5, 6],
        help=(
            "4 = Balanced OT + standard K-Means++ initialization, "
            "5 = Unbalanced OT + standard K-Means++ initialization, "
            "6 = additional ablation variant"
        ),
    )

    parser.add_argument("--ndcg_k", type=int, default=5, help="Cutoff for nDCG@k")
    parser.add_argument("--recall_k", type=int, default=1, help="Cutoff for Recall@k")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use")

    return parser.parse_args()


def get_targets_and_eval_mode(queries_data):
    """Return relevance targets and evaluation mode from preprocessed query data."""
    if "relevant_doc_rels" in queries_data:
        return list(queries_data["relevant_doc_rels"]), "graded_or_multi"

    if "relevant_doc_indices" in queries_data:
        return list(queries_data["relevant_doc_indices"]), "single"

    raise KeyError(
        "Neither 'relevant_doc_rels' nor 'relevant_doc_indices' was found in queries_data."
    )


def evaluate_retrieval(scores, targets, eval_mode, ndcg_cutoff, recall_cutoff):
    """Evaluate retrieval scores using the appropriate relevance format."""
    if eval_mode == "graded_or_multi":
        ndcg10 = ndcg_k_graded_matrix(scores, targets, k=10)
        ndcg_score = ndcg_k_graded_matrix(scores, targets, k=ndcg_cutoff)
        recall_score = recall_at_k_multi(scores, targets, k=recall_cutoff)
    elif eval_mode == "single":
        ndcg10 = ndcg_k(scores, targets, k=10)
        ndcg_score = ndcg_k(scores, targets, k=ndcg_cutoff)
        recall_score = recall_at_k(scores, targets, k=recall_cutoff)
    else:
        raise ValueError(f"Unsupported evaluation mode: {eval_mode}")

    return ndcg10, ndcg_score, recall_score


def apply_mmlongbench_candidate_mask(scores, q_ids, doc_ids):
    """Restrict retrieval candidates to documents from the same PDF for MMLongBench."""
    num_queries, num_docs = scores.shape
    candidate_mask = np.zeros((num_queries, num_docs), dtype=bool)

    for qi, qid in enumerate(q_ids):
        query_pdf = str(qid).rsplit("::", 1)[0]

        for di, docid in enumerate(doc_ids):
            doc_pdf = str(docid).rsplit("::", 1)[0]

            if doc_pdf == query_pdf:
                candidate_mask[qi, di] = True

    masked_scores = scores.copy()
    masked_scores[~candidate_mask] = -1e9

    return masked_scores


def main():
    args = get_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    config = vars(args)
    config["device"] = device

    logger = ExperimentLogger(
        experiment_name=(
            f"{args.exp_name}"
            f"_abl{args.ablation_type}"
            f"_g{args.gamma:.3f}"
            f"_r{args.rho:.1f}"
            f"_n{args.n_iter}"
            f"_s{args.sinkhorn_iter}"
        ),
        config=config,
        base_dir=args.exp_dir,
    )

    for dataset_name in args.datasets:
        file_path = f"{args.data_dir}/{args.model}_{dataset_name}.npz"

        if not os.path.exists(file_path):
            print(f"Skipping {dataset_name}: file not found at {file_path}")
            continue

        data = np.load(file_path, allow_pickle=True)

        docs_data = data["documents"].item()
        queries_data = data["queries"].item()

        raw_d_embeds = torch.from_numpy(docs_data["features"]).to(device)
        raw_d_masks = torch.from_numpy(docs_data["pad_masks"]).to(device)

        if "attention" in docs_data:
            raw_d_attn = torch.from_numpy(docs_data["attention"]).to(device)
        else:
            raw_d_attn = torch.ones_like(raw_d_masks, dtype=raw_d_embeds.dtype)

        q_embeds = torch.from_numpy(queries_data["features"]).to(device)
        q_masks = torch.from_numpy(queries_data["pad_masks"]).to(device)

        q_embeds = torch.nn.functional.normalize(q_embeds, p=2, dim=-1)

        targets, eval_mode = get_targets_and_eval_mode(queries_data)

        doc_ids = data["docid"] if "docid" in data else data["doc_ids"] if "doc_ids" in data else None
        q_ids = data["qid"] if "qid" in data else data["query_ids"] if "query_ids" in data else None

        num_docs = raw_d_embeds.shape[0]
        max_doc_len = raw_d_embeds.shape[1]
        embed_dim = raw_d_embeds.shape[2]

        for mf in args.mfs:
            num_clusters = max(1, int(max_doc_len // mf))

            start_time = time.time()

            compressed_d_embeds = torch.zeros(
                (num_docs, num_clusters, embed_dim),
                device=device,
                dtype=raw_d_embeds.dtype,
            )

            compressed_d_masks = torch.zeros(
                (num_docs, num_clusters),
                device=device,
                dtype=raw_d_masks.dtype,
            )

            for i in range(num_docs):
                centroids = ot_ablation_baselines(
                    x=raw_d_embeds[i],
                    attention=raw_d_attn[i],
                    mask=raw_d_masks[i],
                    num_clusters=num_clusters,
                    ablation_type=args.ablation_type,
                    gamma=args.gamma,
                    rho=args.rho,
                    n_iter=args.n_iter,
                    sinkhorn_iter=args.sinkhorn_iter,
                )

                k_actual = centroids.shape[0]

                if k_actual > num_clusters:
                    centroids = centroids[:num_clusters]
                    k_actual = num_clusters

                compressed_d_embeds[i, :k_actual, :] = centroids
                compressed_d_masks[i, :k_actual] = 1.0

            compress_duration = time.time() - start_time

            score_start = time.time()

            with torch.no_grad():
                norm_compressed = torch.nn.functional.normalize(
                    compressed_d_embeds,
                    p=2,
                    dim=-1,
                )

                scores_tensor = maxsim_scores(
                    q_embeds,
                    norm_compressed,
                    q_masks,
                    compressed_d_masks,
                )

            scores_numpy = scores_tensor.cpu().numpy()
            latency_ms = (time.time() - score_start) * 1000 / q_embeds.shape[0]

            if "mmlongbench" in dataset_name.lower():
                if q_ids is None or doc_ids is None:
                    raise KeyError(
                        "MMLongBench evaluation requires query IDs and document IDs."
                    )

                scores_numpy = apply_mmlongbench_candidate_mask(
                    scores=scores_numpy,
                    q_ids=q_ids,
                    doc_ids=doc_ids,
                )

            ndcg10, ndcg_score, recall_score = evaluate_retrieval(
                scores=scores_numpy,
                targets=targets,
                eval_mode=eval_mode,
                ndcg_cutoff=args.ndcg_k,
                recall_cutoff=args.recall_k,
            )

            logger.log_metric(
                dataset=dataset_name,
                mf=mf,
                num_clusters=num_clusters,
                ablation_type=args.ablation_type,
                ndcg10=ndcg10,
                ndcg=ndcg_score,
                recall=recall_score,
                latency=latency_ms,
                compress_time=compress_duration,
                ndcg_k=args.ndcg_k,
                recall_k=args.recall_k,
            )


    logger.save_summary()


if __name__ == "__main__":
    main()