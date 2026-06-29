import json
import os
from datetime import datetime
import pandas as pd


class ExperimentLogger:
    """Minimal experiment logger for saving configuration, results, and summaries."""
    def __init__(self, experiment_name, config, base_dir="experiments"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)
        self.config = config
        self.results = []
        self.config_path = os.path.join(self.exp_dir, "config.json")
        self.csv_path = os.path.join(self.exp_dir, "results.csv")
        self.summary_path = os.path.join(self.exp_dir, "summary_avg.csv")
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    def log_metric(
        self,
        dataset,
        mf=None,
        num_clusters=None,
        ndcg10=None,
        ndcg=None,
        recall=None,
        latency=0.0,
        compress_time=0.0,
        **kwargs,
    ):
        """Append one evaluation result and save it to results.csv."""
        ndcg_col = f"ndcg@{kwargs.pop('ndcg_k', 5)}"
        recall_col = f"recall@{kwargs.pop('recall_k', 1)}"

        entry = {
            **self.config,
            "dataset": dataset,
            "mf": mf,
            "num_clusters": num_clusters,
            **kwargs,
            "ndcg@10": ndcg10,
            ndcg_col: ndcg,
            recall_col: recall,
            "latency_ms": latency,
            "compress_s": compress_time,
        }
        self.results.append(entry)
        df = pd.DataFrame([entry])
        write_header = not os.path.exists(self.csv_path)
        df.to_csv(
            self.csv_path,
            index=False,
            mode="a",
            header=write_header,
            encoding="utf-8-sig",
        )
    def save_summary(self):
        """Save averaged results to summary_avg.csv."""
        if not self.results:
            return
        df = pd.DataFrame(self.results)
        numeric_candidates = [
            "ndcg@10",
            "ndcg@5",
            "recall@1",
            "latency_ms",
            "compress_s",
            "num_clusters",
        ]
        group_candidates = [
            "dataset",
            "ablation_type",
            "gamma",
            "rho",
            "n_iter",
            "sinkhorn_iter",
            "mf",
        ]
        numeric_cols = [
            col for col in numeric_candidates
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        group_cols = [
            col for col in group_candidates
            if col in df.columns
        ]
        if group_cols:
            summary = (
                df.groupby(group_cols, dropna=False)[numeric_cols]
                .mean()
                .reset_index()
            )
        else:
            summary = df[numeric_cols].mean().to_frame().T
        summary.to_csv(self.summary_path, index=False, encoding="utf-8-sig")