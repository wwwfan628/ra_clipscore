import argparse
import os.path
import pickle
import time

import core
import numpy as np
import utils


PROMPT_TEMPLATES = {
    1: ("This is a photo without {}.", "This is a photo of {}."),
    2: ("There is no {} in the scene.", "There is {} in the scene."),
    3: ("The image does not contain {}.", "The image contains {}."),
    4: ("It is not {} in the picture.", "It is {} in the picture."),
    5: ("{} is absent from the image.", "{} is present in the image."),
}


def make_prompts(text_list, prompt_id):
    negative_template, positive_template = PROMPT_TEMPLATES[prompt_id]
    return ([negative_template.format(item) for item in text_list]
            + [positive_template.format(item) for item in text_list])


def rank_biased_overlap(rank_a, rank_b, p=0.8):
    """Extrapolated RBO for two complete rankings; larger p looks deeper."""
    seen_a, seen_b = set(), set()
    weighted_overlap = 0.0
    depth = len(rank_a)
    if depth == 0:
        return np.nan
    for d, (item_a, item_b) in enumerate(zip(rank_a, rank_b), start=1):
        seen_a.add(int(item_a))
        seen_b.add(int(item_b))
        overlap = len(seen_a.intersection(seen_b)) / d
        weighted_overlap += overlap * p ** (d - 1)
    return (1 - p) * weighted_overlap + overlap * p ** depth


def kendalls_w(rankings):
    """Kendall's coefficient of concordance for complete, untied rankings."""
    n_judges, n_items = rankings.shape
    positions = np.empty_like(rankings, dtype=float)
    for judge, ranking in enumerate(rankings):
        positions[judge, ranking] = np.arange(1, n_items + 1)
    rank_sums = positions.sum(axis=0)
    squared_deviations = np.sum(
        (rank_sums - n_judges * (n_items + 1) / 2) ** 2
    )
    return 12 * squared_deviations / (
        n_judges ** 2 * (n_items ** 3 - n_items)
    )


def pairwise_matrix(rankings, metric):
    n_rankings = len(rankings)
    matrix = np.eye(n_rankings, dtype=float)
    for i in range(n_rankings):
        for j in range(i + 1, n_rankings):
            value = metric(rankings[i], rankings[j])
            matrix[i, j] = matrix[j, i] = value
    return matrix


def kendalls_tau(rank_a, rank_b):
    """Kendall's tau-a for two complete rankings without ties."""
    n_items = len(rank_a)
    if n_items < 2:
        return np.nan
    concordance = 0
    for i in range(n_items - 1):
        concordance += np.sign(
            (rank_a[i] - rank_a[i + 1:]) * (rank_b[i] - rank_b[i + 1:])
        ).sum()
    return concordance / (n_items * (n_items - 1) / 2)


def spearman_rank_correlation(rank_a, rank_b):
    """Spearman's rho for two complete rankings without ties."""
    n_items = len(rank_a)
    if n_items < 2:
        return np.nan
    squared_rank_difference = np.square(rank_a - rank_b).sum()
    return 1 - 6 * squared_rank_difference / (n_items * (n_items ** 2 - 1))


def ranking_agreement(score_lists, rbo_p=0.8):
    if not 0 < rbo_p < 1:
        raise ValueError(f'rbo_p must be between 0 and 1, got {rbo_p}')
    # KL divergence is ranked from largest (most discrepant) to smallest.
    rankings = np.stack([
        np.argsort(-np.asarray(scores), kind='stable') for scores in score_lists
    ])
    rank_positions = np.empty_like(rankings)
    for i, ranking in enumerate(rankings):
        rank_positions[i, ranking] = np.arange(len(ranking))

    def top_overlap(k):
        def metric(rank_a, rank_b):
            effective_k = min(k, len(rank_a))
            if effective_k == 0:
                return np.nan
            return len(set(rank_a[:effective_k]).intersection(rank_b[:effective_k])) / effective_k
        return pairwise_matrix(rankings, metric)

    return {
        'argsort_indices': rankings,
        'ranking_order': 'descending',
        'kendalls_w': kendalls_w(rankings),
        'kendalls_tau_matrix': pairwise_matrix(
            rank_positions, kendalls_tau
        ),
        'spearman_rank_correlation_matrix': pairwise_matrix(
            rank_positions, spearman_rank_correlation
        ),
        'rbo_p': rbo_p,
        'rbo_matrix': pairwise_matrix(
            rankings, lambda a, b: rank_biased_overlap(a, b, p=rbo_p)
        ),
        'top_1_overlap_matrix': top_overlap(1),
        'top_3_overlap_matrix': top_overlap(3),
        'top_5_overlap_matrix': top_overlap(5),
        'top_10_overlap_matrix': top_overlap(10),
    }


def run_prompt(args, text_list, prompt_id, save_dir):
    text_list_prompt = make_prompts(text_list, prompt_id)

    if 'vit' in args.score_type:
        scores_real, scores_per_patch_real = core.get_evalcs_local_vit(
            img_dir=args.real_dir, text_list=text_list_prompt, batch_size=args.batch_size
        )
        scores_gen, scores_per_patch_gen = core.get_evalcs_local_vit(
            img_dir=args.gen_dir, text_list=text_list_prompt, batch_size=args.batch_size
        )
    elif 'rn' in args.score_type:
        scores_real = core.get_evalcs_local_rn(
            img_dir=args.real_dir, text_list=text_list_prompt, batch_size=args.batch_size
        )
        scores_gen = core.get_evalcs_local_rn(
            img_dir=args.gen_dir, text_list=text_list_prompt, batch_size=args.batch_size
        )
    else:
        raise ValueError(f'Unsupported score_type: {args.score_type}')

    scores_mean_real = scores_real.mean(axis=0)
    scores_mean_gen = scores_gen.mean(axis=0)
    scores_cov_real = np.cov(scores_real.T)
    scores_cov_gen = np.cov(scores_gen.T)

    single_attribute_results = core.compute_single(
        scores_real, scores_gen, scores_mean_real, scores_mean_gen,
        scores_cov_real, scores_cov_gen, text_list
    )
    pair_attributes_results = core.compute_pair(
        scores_real, scores_gen, scores_mean_real, scores_mean_gen,
        scores_cov_real, scores_cov_gen, text_list
    )
    all_attributes_results = core.compute_all(
        scores_real, scores_gen, scores_mean_real, scores_mean_gen,
        scores_cov_real, scores_cov_gen, text_list
    )

    results_dict = {
        'prompt_id': prompt_id,
        'attr_list': text_list,
        'text_list_prompt': text_list_prompt,
        'scores_real': scores_real, 'scores_gen': scores_gen,
        'scores_mean_real': scores_mean_real, 'scores_mean_gen': scores_mean_gen,
        'scores_cov_real': scores_cov_real, 'scores_cov_gen': scores_cov_gen,
        'forward_KL_single': single_attribute_results[0],
        'reverse_KL_single': single_attribute_results[1],
        'mean_difference_single': single_attribute_results[2],
        'f_distance_single': single_attribute_results[3],
        'mmd_polynomial_single': single_attribute_results[4],
        'mmd_rbf_single': single_attribute_results[5],
        'forward_KL_pair': pair_attributes_results[0],
        'reverse_KL_pair': pair_attributes_results[1],
        'mean_difference_pair': pair_attributes_results[2],
        'f_distance_pair': pair_attributes_results[3],
        'mmd_polynomial_pair': pair_attributes_results[4],
        'mmd_rbf_pair': pair_attributes_results[5],
        'forward_KL_all': all_attributes_results[0],
        'reverse_KL_all': all_attributes_results[1],
        'mean_difference_all': all_attributes_results[2],
        'f_distance_all': all_attributes_results[3],
        'mmd_polynomial_all': all_attributes_results[4],
        'mmd_rbf_all': all_attributes_results[5],
    }

    prompt_save_dir = os.path.join(save_dir, str(prompt_id))
    os.makedirs(prompt_save_dir, exist_ok=True)
    with open(os.path.join(prompt_save_dir, 'experiment_results_dict.pkl'), 'wb') as f:
        pickle.dump(results_dict, f)
    return results_dict


def main(args):
    start_time = time.time()
    experiment_name = f"{args.exp}_{args.n_attr}_{args.score_type}"
    text_list = utils.load_candidate(
        experiment_name=f"{args.exp}_{args.n_attr}", n_attr=args.n_attr
    )
    dataset_real = os.path.basename(os.path.normpath(args.real_dir))
    dataset_gen = os.path.basename(os.path.normpath(args.gen_dir))
    save_dir = f"results_dir/{experiment_name}/{dataset_real}_vs_{dataset_gen}"
    os.makedirs(save_dir, exist_ok=True)

    prompt_results = []
    for prompt_id in PROMPT_TEMPLATES:
        print(f'Running prompt template {prompt_id}/{len(PROMPT_TEMPLATES)}...')
        prompt_results.append(run_prompt(args, text_list, prompt_id, save_dir))

    agreement_results = {
        'prompt_ids': list(PROMPT_TEMPLATES),
        'forward_KL_single': ranking_agreement(
            [result['forward_KL_single'] for result in prompt_results],
            rbo_p=args.rbo_p,
        ),
        'forward_KL_pair': ranking_agreement(
            [result['forward_KL_pair'] for result in prompt_results],
            rbo_p=args.rbo_p,
        ),
    }
    with open(os.path.join(save_dir, 'prompt_ranking_agreement.pkl'), 'wb') as f:
        pickle.dump(agreement_results, f)

    for name in ('forward_KL_single', 'forward_KL_pair'):
        print(f"{name} Kendall's W: {agreement_results[name]['kendalls_w']:.6f}")
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'Execution Time: {elapsed}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='prompt', help='experiment name, to find the corresponding attribute list')
    parser.add_argument('--real_dir', type=str, default='/home/x_yiflu/dataset/Datasets/Dataset_Compare/A', help='the directory of real images')
    parser.add_argument('--gen_dir', type=str, default='/home/x_yiflu/dataset/Datasets/Dataset_Compare/B', help='the directory of generated images')
    parser.add_argument('--n_attr', type=int, default=20, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--score_type', type=str, default='EvalCS_local_rn', help='score type, e.g. EvalCS_local_rn or EvalCS_local_vit')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size to extract image features')
    parser.add_argument('--rbo_p', type=float, default=0.8, help='RBO persistence; 0.8 emphasizes approximately the first five ranks')

    args = parser.parse_args()
    main(args)
