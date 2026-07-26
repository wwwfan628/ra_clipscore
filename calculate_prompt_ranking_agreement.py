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


def scores_to_tied_ranks(scores, atol, rtol):
    """Descending average ranks, grouping values indistinguishable by tolerance."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        anchor = scores[order[start]]
        while end < len(order):
            value = scores[order[end]]
            tolerance = atol + rtol * max(abs(anchor), abs(value))
            if abs(anchor - value) > tolerance:
                break
            end += 1
        # Ranks are one-based; all members of a tie receive their average rank.
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


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


def kendalls_w_with_ties(rank_vectors):
    """Tie-corrected Kendall's W for rows of one-based average ranks."""
    n_judges, n_items = rank_vectors.shape
    if n_items < 2:
        return np.nan
    rank_sums = rank_vectors.sum(axis=0)
    rank_sum_mean = n_judges * (n_items + 1) / 2
    squared_deviations = np.square(rank_sums - rank_sum_mean).sum()
    tie_correction = 0
    for ranks in rank_vectors:
        _, counts = np.unique(ranks, return_counts=True)
        tie_correction += np.sum(counts ** 3 - counts)
    denominator = (
        n_judges ** 2 * (n_items ** 3 - n_items)
        - n_judges * tie_correction
    )
    return 12 * squared_deviations / denominator if denominator else np.nan


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


def kendalls_tau_b(rank_a, rank_b):
    """Kendall's tau-b, including ties introduced by the KL tolerance."""
    products = []
    non_ties_a = 0
    non_ties_b = 0
    for i in range(len(rank_a) - 1):
        signs_a = np.sign(rank_a[i] - rank_a[i + 1:])
        signs_b = np.sign(rank_b[i] - rank_b[i + 1:])
        products.append(np.sum(signs_a * signs_b))
        non_ties_a += np.square(signs_a).sum()
        non_ties_b += np.square(signs_b).sum()
    denominator = np.sqrt(non_ties_a * non_ties_b)
    return np.sum(products) / denominator if denominator else np.nan


def spearman_rank_correlation(position_a, position_b):
    """Spearman's rho for two complete rankings without ties."""
    n_items = len(position_a)
    if n_items < 2:
        return np.nan
    squared_difference = np.square(position_a - position_b).sum()
    return 1 - 6 * squared_difference / (n_items * (n_items ** 2 - 1))


def pearson_correlation(values_a, values_b):
    centered_a = values_a - np.mean(values_a)
    centered_b = values_b - np.mean(values_b)
    denominator = np.sqrt(
        np.square(centered_a).sum() * np.square(centered_b).sum()
    )
    return (
        np.dot(centered_a, centered_b) / denominator
        if denominator else np.nan
    )


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


def calculate_agreement(score_lists, rbo_p, tie_atol, tie_rtol):
    score_arrays = [np.asarray(scores).reshape(-1) for scores in score_lists]
    lengths = {scores.size for scores in score_arrays}
    if len(lengths) != 1:
        raise ValueError(f"All prompt results must have equal lengths, got {lengths}")

    # Higher forward KL means a larger discrepancy, so it is ranked first.
    rankings = np.stack(
        [np.argsort(-scores, kind="stable") for scores in score_arrays]
    )
    positions = rankings_to_positions(rankings)
    tied_ranks = np.stack([
        scores_to_tied_ranks(scores, tie_atol, tie_rtol)
        for scores in score_arrays
    ])

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
        "raw_KL_pearson_correlation_matrix": pairwise_matrix(
            score_arrays, pearson_correlation
        ),
        "tie_analysis": {
            "absolute_tolerance": tie_atol,
            "relative_tolerance": tie_rtol,
            "tied_rank_vectors": tied_ranks,
            "kendalls_w": kendalls_w_with_ties(tied_ranks),
            "kendalls_tau_b_matrix": pairwise_matrix(
                tied_ranks, kendalls_tau_b
            ),
            "spearman_rank_correlation_matrix": pairwise_matrix(
                tied_ranks, pearson_correlation
            ),
            "number_of_tied_items_per_prompt": np.asarray([
                sum(
                    count for count in np.unique(ranks, return_counts=True)[1]
                    if count > 1
                )
                for ranks in tied_ranks
            ]),
        },
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
            ("raw_KL_pearson_correlation_matrix", "Raw KL Pearson correlation matrix"),
        ):
            lines.append(f"{label}:")
            lines.append(np.array2string(
                result[field], precision=6, suppress_small=True
            ))
        tie_result = result["tie_analysis"]
        lines.extend([
            (
                "Tie-aware analysis "
                f"(atol={tie_result['absolute_tolerance']}, "
                f"rtol={tie_result['relative_tolerance']}):"
            ),
            (
                "Number of items assigned a tied rank per prompt: "
                f"{tie_result['number_of_tied_items_per_prompt']}"
            ),
            f"Tie-corrected Kendall's W: {tie_result['kendalls_w']:.6f}",
            "Tie-aware Kendall's tau-b matrix:",
            np.array2string(
                tie_result["kendalls_tau_b_matrix"],
                precision=6, suppress_small=True,
            ),
            "Tie-aware Spearman rank correlation matrix:",
            np.array2string(
                tie_result["spearman_rank_correlation_matrix"],
                precision=6, suppress_small=True,
            ),
        ])
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
        "--tie_atol", type=float, default=1e-5,
        help="Absolute KL difference treated as a tie (default: 1e-5).",
    )
    parser.add_argument(
        "--tie_rtol", type=float, default=0.0,
        help="Relative KL difference treated as a tie (default: 0).",
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
    if args.tie_atol < 0 or args.tie_rtol < 0:
        raise ValueError("--tie_atol and --tie_rtol must be non-negative")

    prompt_results = load_prompt_results(args.results_dir, args.prompt_ids)
    summary = {
        "prompt_ids": args.prompt_ids,
        "rbo_p": args.rbo_p,
    }
    for metric_name in METRIC_NAMES:
        summary[metric_name] = calculate_agreement(
            [result[metric_name] for result in prompt_results],
            args.rbo_p,
            args.tie_atol,
            args.tie_rtol,
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
