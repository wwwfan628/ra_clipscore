"""Calculate ranking agreement across saved prompt-template experiments.

This script reads:
    RESULTS_DIR/{prompt_id}/experiment_results_dict.pkl

for each requested prompt id. Rankings are ordered by decreasing forward KL.
"""

import argparse
import os
import pickle

import numpy as np


METRIC_NAMES = ("forward_KL_single", "forward_KL_pair")
TOP_K_VALUES = (1, 3, 5, 10)


def load_prompt_results(results_dir, prompt_ids):
    results = []
    for prompt_id in prompt_ids:
        result_path = os.path.join(
            results_dir, str(prompt_id), "experiment_results_dict.pkl"
        )
        if not os.path.isfile(result_path):
            raise FileNotFoundError(
                f"Missing result for prompt_id={prompt_id}: {result_path}"
            )
        with open(result_path, "rb") as file:
            result = pickle.load(file)
        missing = [name for name in METRIC_NAMES if name not in result]
        if missing:
            raise KeyError(f"{result_path} is missing fields: {missing}")
        results.append(result)
    return results


def rank_biased_overlap(ranking_a, ranking_b, p):
    """Extrapolated RBO for equal-length complete rankings."""
    seen_a, seen_b = set(), set()
    weighted_overlap = 0.0
    depth = len(ranking_a)
    if depth == 0:
        return np.nan

    overlap_at_depth = 0.0
    for d, (item_a, item_b) in enumerate(zip(ranking_a, ranking_b), start=1):
        seen_a.add(int(item_a))
        seen_b.add(int(item_b))
        overlap_at_depth = len(seen_a.intersection(seen_b)) / d
        weighted_overlap += overlap_at_depth * p ** (d - 1)
    return (1 - p) * weighted_overlap + overlap_at_depth * p ** depth


def rankings_to_positions(rankings):
    positions = np.empty_like(rankings)
    for index, ranking in enumerate(rankings):
        positions[index, ranking] = np.arange(len(ranking))
    return positions


def kendalls_w(rankings):
    """Kendall's W for complete rankings (argsort resolves any score ties)."""
    n_judges, n_items = rankings.shape
    if n_items < 2:
        return np.nan
    positions = rankings_to_positions(rankings) + 1
    rank_sums = positions.sum(axis=0)
    rank_sum_mean = n_judges * (n_items + 1) / 2
    return (
        12 * np.square(rank_sums - rank_sum_mean).sum()
        / (n_judges ** 2 * (n_items ** 3 - n_items))
    )


def kendalls_tau(position_a, position_b):
    """Kendall's tau-a for two complete rankings without ties."""
    n_items = len(position_a)
    if n_items < 2:
        return np.nan
    concordance = 0
    for i in range(n_items - 1):
        concordance += np.sign(
            (position_a[i] - position_a[i + 1:])
            * (position_b[i] - position_b[i + 1:])
        ).sum()
    return concordance / (n_items * (n_items - 1) / 2)


def spearman_rank_correlation(position_a, position_b):
    """Spearman's rho for two complete rankings without ties."""
    n_items = len(position_a)
    if n_items < 2:
        return np.nan
    squared_difference = np.square(position_a - position_b).sum()
    return 1 - 6 * squared_difference / (n_items * (n_items ** 2 - 1))


def pairwise_matrix(values, metric):
    n_values = len(values)
    matrix = np.eye(n_values, dtype=float)
    for i in range(n_values):
        for j in range(i + 1, n_values):
            score = metric(values[i], values[j])
            matrix[i, j] = matrix[j, i] = score
    return matrix


def top_k_overlap_matrix(rankings, k):
    """Pairwise |top-k intersection| / k; k is clipped to ranking length."""
    effective_k = min(k, rankings.shape[1])
    if effective_k == 0:
        return np.full((len(rankings), len(rankings)), np.nan)
    return pairwise_matrix(
        rankings,
        lambda a, b: (
            len(set(a[:effective_k]).intersection(b[:effective_k])) / effective_k
        ),
    )


def calculate_agreement(score_lists, rbo_p):
    score_arrays = [np.asarray(scores).reshape(-1) for scores in score_lists]
    lengths = {scores.size for scores in score_arrays}
    if len(lengths) != 1:
        raise ValueError(f"All prompt results must have equal lengths, got {lengths}")

    # Higher forward KL means a larger discrepancy, so it is ranked first.
    rankings = np.stack(
        [np.argsort(-scores, kind="stable") for scores in score_arrays]
    )
    positions = rankings_to_positions(rankings)

    agreement = {
        "forward_KL_values": np.stack(score_arrays),
        "ranking_order": "descending_forward_KL",
        "argsort_indices": rankings,
        "kendalls_w": kendalls_w(rankings),
        "kendalls_tau_matrix": pairwise_matrix(positions, kendalls_tau),
        "spearman_rank_correlation_matrix": pairwise_matrix(
            positions, spearman_rank_correlation
        ),
        "rbo_p": rbo_p,
        "rbo_matrix": pairwise_matrix(
            rankings, lambda a, b: rank_biased_overlap(a, b, rbo_p)
        ),
    }
    for k in TOP_K_VALUES:
        agreement[f"top_{k}_overlap_matrix"] = top_k_overlap_matrix(rankings, k)
    return agreement


def format_report(summary):
    lines = [
        f"Prompt IDs: {summary['prompt_ids']}",
        "Ranking order: descending forward KL",
        f"RBO p: {summary['rbo_p']} (smaller p emphasizes the first ranks)",
    ]
    for metric_name in METRIC_NAMES:
        result = summary[metric_name]
        lines.extend([
            "",
            f"===== {metric_name} =====",
            "Five forward-KL arrays (rows follow Prompt IDs):",
            np.array2string(
                result["forward_KL_values"], precision=6, suppress_small=True
            ),
            "Descending argsort indices (rows follow Prompt IDs):",
            np.array2string(result["argsort_indices"]),
            f"Kendall's W: {result['kendalls_w']:.6f}",
        ])
        for field, label in (
            ("kendalls_tau_matrix", "Kendall's tau matrix"),
            ("spearman_rank_correlation_matrix", "Spearman rank correlation matrix"),
            ("rbo_matrix", "Rank-Biased Overlap matrix"),
            ("top_1_overlap_matrix", "Top-1 overlap matrix"),
            ("top_3_overlap_matrix", "Top-3 overlap matrix"),
            ("top_5_overlap_matrix", "Top-5 overlap matrix"),
            ("top_10_overlap_matrix", "Top-10 overlap matrix"),
        ):
            lines.append(f"{label}:")
            lines.append(np.array2string(
                result[field], precision=6, suppress_small=True
            ))
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare forward-KL rankings from different prompt IDs."
    )
    parser.add_argument(
        "--results_dir",
        default='/Users/yifanlu/Desktop/results_dir/prompt_20_EvalCS_local_rn/A_vs_C',
        help="Directory containing prompt-id subdirectories (1/, 2/, ...).",
    )
    parser.add_argument(
        "--prompt_ids", type=int, nargs="+", default=[1, 2, 3, 4]
    )
    parser.add_argument(
        "--rbo_p", type=float, default=0.8,
        help="RBO persistence; p=0.8 emphasizes approximately the first 5 ranks.",
    )
    parser.add_argument(
        "--output_name", default="prompt_ranking_agreement",
        help="Output basename; .pkl and .txt are written under results_dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.rbo_p < 1:
        raise ValueError(f"--rbo_p must be between 0 and 1, got {args.rbo_p}")
    if len(args.prompt_ids) < 2:
        raise ValueError("At least two prompt IDs are required")

    prompt_results = load_prompt_results(args.results_dir, args.prompt_ids)
    summary = {
        "prompt_ids": args.prompt_ids,
        "rbo_p": args.rbo_p,
    }
    for metric_name in METRIC_NAMES:
        summary[metric_name] = calculate_agreement(
            [result[metric_name] for result in prompt_results], args.rbo_p
        )

    report = format_report(summary)
    print(report)

    pickle_path = os.path.join(args.results_dir, f"{args.output_name}.pkl")
    text_path = os.path.join(args.results_dir, f"{args.output_name}.txt")
    with open(pickle_path, "wb") as file:
        pickle.dump(summary, file)
    with open(text_path, "w", encoding="utf-8") as file:
        file.write(report + "\n")
    print(f"\nSaved pickle: {pickle_path}")
    print(f"Saved report: {text_path}")


if __name__ == "__main__":
    main()
