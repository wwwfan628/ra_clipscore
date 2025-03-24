from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from utils.attribute_list import *
import scipy


def load_attributes_list(attr_list_name, n_attr):
    return attr_dict[attr_list_name][:n_attr]


def plot_bar(result, labels, name, save_path=None):
    # check if results has the same length as labels
    assert len(result) == len(labels)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.bar(labels, result, width=0.8)

    plt.title(f"{name} \n mean={np.sum(result)/(result.shape[0])}, var={np.var(result, dtype=np.float32)}")
    plt.ylabel('each attribute difference between 2 datasets', fontsize=5)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{name}.png", dpi=2000)
    # plt.show()


def plot_single(results, attr_list, save_path):
    print('###### Plotting results of single attribute ######')
    results_contents = ['forward_KL', 'reverse_KL', 'mean_differences', 'frechet_distance', 'kernel_distance']
    save_path = os.path.join(save_path, 'figs_single_attribute')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, result in enumerate(results):
        attr_list_sort_by_result = [attr_list[j] for j in result.argsort()[::-1]]
        result[::-1].sort()
        plot_bar(result, attr_list_sort_by_result, results_contents[i], save_path)
    return


def plot_single_reduced(results, attr_list, save_path):
    print('###### Plotting results of single attribute ######')
    attr_list = attr_list[:10]  # only consider human face related attributes
    results_contents = ['forward_KL', 'reverse_KL', 'mean_differences', 'frechet_distance', 'kernel_distance']
    save_path = os.path.join(save_path, 'figs_single_attribute')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, result in enumerate(results):
        attr_list_sort_by_result = [attr_list[j] for j in result.argsort()[::-1]]
        result[::-1].sort()
        plot_bar(result, attr_list_sort_by_result, results_contents[i], save_path)
    return


def plot_pair(results, attr_list, save_path, max_bar=20):
    print('###### Plotting results of pair attributes ######')
    results_contents = ['forward_KL', 'reverse_KL', 'mean_differences', 'frechet_distance', 'kernel_distance']
    save_path = os.path.join(save_path, 'figs_pair_attributes')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pair_attr_list = [pair for pair in combinations(attr_list, 2)]
    max_bar = min(max_bar, len(pair_attr_list))

    for i, result in enumerate(results):
        pair_attr_list_sort_by_result = [str(pair_attr_list[j]) for j in result.argsort()[::-1]]
        result[::-1].sort()
        plot_bar(result[:max_bar], pair_attr_list_sort_by_result[:max_bar], results_contents[i], save_path)
    return


def plot_pair_reduced(results, attr_list, save_path, max_bar=20):
    print('###### Plotting results of pair attributes ######')
    attr_list = attr_list[:10]  # only consider human face related attributes
    results_contents = ['forward_KL', 'reverse_KL', 'mean_differences', 'frechet_distance', 'kernel_distance']
    save_path = os.path.join(save_path, 'figs_pair_attributes')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pair_attr_list = [pair for pair in combinations(attr_list, 2)]
    max_bar = min(max_bar, len(pair_attr_list))

    for i, result in enumerate(results):
        pair_attr_list_sort_by_result = [str(pair_attr_list[j]) for j in result.argsort()[::-1]]
        result[::-1].sort()
        plot_bar(result[:max_bar], pair_attr_list_sort_by_result[:max_bar], results_contents[i], save_path)
    return


def compute_single(real_score, gen_score, real_mean, gen_mean, real_cov, gen_cov, attr_list):
    print('###### Computing for single attribute ######')
    # single attribute
    difference_single = []
    KL_real_gen_single = []
    KL_gen_real_single = []
    f_distance_single = []
    kernel_distance_single = []

    for n, i in tqdm(enumerate(np.arange(len(attr_list))), total=len(attr_list)):
        real_mean_i = np.expand_dims(real_mean, 1)[i]
        gen_mean_i = np.expand_dims(gen_mean, 1)[i]
        # variance: diagonal elements of covariance matrix
        real_var_i = np.expand_dims(real_cov, [2, 3])[i, i]
        gen_var_i = np.expand_dims(gen_cov, [2, 3])[i, i]
        real_score_i = np.expand_dims(real_score[:, i], 1)
        gen_score_i = np.expand_dims(gen_score[:, i], 1)

        # difference of mean values
        difference_single.append(float(abs(gen_mean_i - real_mean_i)))

        KL_real_gen_single.append(compute_KL_divergence_1d_gaussian_pdf(real_mean_i, gen_mean_i, real_var_i, gen_var_i))

        KL_gen_real_single.append(compute_KL_divergence_1d_gaussian_pdf(gen_mean_i, real_mean_i, gen_var_i, real_var_i))

        # Frechet Distance
        f_distance_single.append(compute_frechet_distance(real_mean_i, gen_mean_i, real_var_i, gen_var_i))

        # Kernel Distance
        kernel_distance_single.append(compute_kernel_distance(real_score_i, gen_score_i))

    return [np.array(KL_real_gen_single), np.array(KL_gen_real_single), np.array(difference_single), np.array(f_distance_single), np.array(kernel_distance_single)]


def compute_single_reduced(real_score, gen_score, real_mean, gen_mean, real_cov, gen_cov, attr_list):
    print('###### Computing for single attribute ######')
    attr_list = attr_list[:10]  # only consider human face related attributes
    # single attribute
    difference_single = []
    KL_real_gen_single = []
    KL_gen_real_single = []
    f_distance_single = []
    kernel_distance_single = []

    for n, i in tqdm(enumerate(np.arange(len(attr_list))), total=len(attr_list)):
        real_mean_i = np.expand_dims(real_mean, 1)[i]
        gen_mean_i = np.expand_dims(gen_mean, 1)[i]
        # variance: diagonal elements of covariance matrix
        real_var_i = np.expand_dims(real_cov, [2, 3])[i, i]
        gen_var_i = np.expand_dims(gen_cov, [2, 3])[i, i]
        real_score_i = np.expand_dims(real_score[:, i], 1)
        gen_score_i = np.expand_dims(gen_score[:, i], 1)

        # difference of mean values
        difference_single.append(float(abs(gen_mean_i - real_mean_i)))

        KL_real_gen_single.append(compute_KL_divergence_1d_gaussian_pdf(real_mean_i, gen_mean_i, real_var_i, gen_var_i))

        KL_gen_real_single.append(compute_KL_divergence_1d_gaussian_pdf(gen_mean_i, real_mean_i, gen_var_i, real_var_i))

        # Frechet Distance
        f_distance_single.append(compute_frechet_distance(real_mean_i, gen_mean_i, real_var_i, gen_var_i))

        # Kernel Distance
        kernel_distance_single.append(compute_kernel_distance(real_score_i, gen_score_i))

    return [np.array(KL_real_gen_single), np.array(KL_gen_real_single), np.array(difference_single), np.array(f_distance_single), np.array(kernel_distance_single)]



def compute_pair(real_score, gen_score, real_mean, gen_mean, real_cov, gen_cov, attr_list):
    print('###### Computing for pair attributes ######')
    # pair attributes
    difference_pair = []
    KL_real_gen_pair = []
    KL_gen_real_pair = []
    f_distance_pair = []
    kernel_distance_pair = []

    for n, (i, j) in tqdm(enumerate(combinations(np.arange(len(attr_list)), 2)), total=len(list(combinations(attr_list, 2)))):
        real_mean_ij = real_mean[[i, j]]
        gen_mean_ij = gen_mean[[i, j]]
        real_cov_ij = real_cov[[i, i, j, j], [i, j, i, j]].reshape([2,2])
        gen_cov_ij = gen_cov[[i, i, j, j], [i, j, i, j]].reshape([2,2])
        real_score_ij = real_score[:, [i, j]]
        gen_score_ij = gen_score[:, [i, j]]

        # difference of mean values
        difference_pair.append(float(abs(real_mean_ij - gen_mean_ij).sum()))

        KL_real_gen_pair.append(compute_KL_divergence_high_dim_gaussian_pdf(real_mean_ij, gen_mean_ij, real_cov_ij, gen_cov_ij))

        KL_gen_real_pair.append(compute_KL_divergence_high_dim_gaussian_pdf(gen_mean_ij, real_mean_ij, gen_cov_ij, real_cov_ij))

        # Frechet Distance
        f_distance_pair.append(compute_frechet_distance(real_mean_ij, gen_mean_ij, real_cov_ij, gen_cov_ij))

        # Kernel Distance
        kernel_distance_pair.append(compute_kernel_distance(real_score_ij, gen_score_ij))

    return [np.array(KL_real_gen_pair), np.array(KL_gen_real_pair), np.array(difference_pair), np.array(f_distance_pair), np.array(kernel_distance_pair)]


def compute_pair_reduced(real_score, gen_score, real_mean, gen_mean, real_cov, gen_cov, attr_list):
    print('###### Computing for pair attributes ######')
    attr_list = attr_list[:10]  # only consider human face related attributes
    # pair attributes
    difference_pair = []
    KL_real_gen_pair = []
    KL_gen_real_pair = []
    f_distance_pair = []
    kernel_distance_pair = []

    for n, (i, j) in tqdm(enumerate(combinations(np.arange(len(attr_list)), 2)), total=len(list(combinations(attr_list, 2)))):
        real_mean_ij = real_mean[[i, j]]
        gen_mean_ij = gen_mean[[i, j]]
        real_cov_ij = real_cov[[i, i, j, j], [i, j, i, j]].reshape([2,2])
        gen_cov_ij = gen_cov[[i, i, j, j], [i, j, i, j]].reshape([2,2])
        real_score_ij = real_score[:, [i, j]]
        gen_score_ij = gen_score[:, [i, j]]

        # difference of mean values
        difference_pair.append(float(abs(real_mean_ij - gen_mean_ij).sum()))

        KL_real_gen_pair.append(compute_KL_divergence_high_dim_gaussian_pdf(real_mean_ij, gen_mean_ij, real_cov_ij, gen_cov_ij))

        KL_gen_real_pair.append(compute_KL_divergence_high_dim_gaussian_pdf(gen_mean_ij, real_mean_ij, gen_cov_ij, real_cov_ij))

        # Frechet Distance
        f_distance_pair.append(compute_frechet_distance(real_mean_ij, gen_mean_ij, real_cov_ij, gen_cov_ij))

        # Kernel Distance
        kernel_distance_pair.append(compute_kernel_distance(real_score_ij, gen_score_ij))

    return [np.array(KL_real_gen_pair), np.array(KL_gen_real_pair), np.array(difference_pair), np.array(f_distance_pair), np.array(kernel_distance_pair)]


def compute_all(real_score, gen_score, real_mean, gen_mean, real_cov, gen_cov, attr_list):
    print('###### Computing for all attributes ######')
    # all attributes

    # difference of mean values
    difference_all = float(abs(real_mean - gen_mean).sum())

    KL_real_gen_all = compute_KL_divergence_high_dim_gaussian_pdf(real_mean, gen_mean, real_cov, gen_cov)

    KL_gen_real_all = compute_KL_divergence_high_dim_gaussian_pdf(gen_mean, real_mean, gen_cov, real_cov)

    # Frechet Distance
    f_distance_all = compute_frechet_distance(real_mean, gen_mean, real_cov, gen_cov)

    # Kernel Distance
    kernel_distance_all = compute_kernel_distance(real_score, gen_score)

    return [KL_real_gen_all, KL_gen_real_all, difference_all, f_distance_all, kernel_distance_all]


def compute_KL_divergence_1d_gaussian_pdf(p_mean, q_mean, p_var, q_var):
    # KL(p,q) = log(phi_q/phi_p) + ((phi_p² + (mean_p - mean_q)²)/2phi_q²) - 0.5   p represents original, q generated
    KL_div = np.log(np.sqrt(q_var) / np.sqrt(p_var)) + (
            (p_var + np.square((p_mean - q_mean))) / (2 * q_var)) - 0.5
    return float(KL_div)


def compute_KL_divergence_high_dim_gaussian_pdf(p_mean, q_mean, p_cov, q_cov):
    # KL(p,q) = 0.5 * [ log(det(phi_q)/det(phi_p)) - d + tr(cov_q^(-1) @ cov_p) + ((mean_q - mean_p).T @ cov_q^(-1) @ (mean_q - mean_p))]  p represents original, q generated

    part_1 = np.log(np.linalg.det(q_cov) / np.linalg.det(p_cov)) - 2
    part_2 = np.trace(np.linalg.inv(q_cov) @ p_cov)
    part_3 = (q_mean - p_mean).T @ np.linalg.inv(q_cov) @ (q_mean - p_mean)

    KL_div = 0.5 * (part_1 + part_2 + part_3)
    return float(KL_div)


def compute_frechet_distance(real_mean, gen_mean, real_cov, gen_cov):
    m = np.square(gen_mean - real_mean).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(gen_cov, real_cov), disp=False)  # pylint: disable=no-member
    frechet_distance = np.real(m + np.trace(gen_cov + real_cov - s * 2))
    return float(frechet_distance)


def compute_kernel_distance(real_features, gen_features, num_subsets=100, max_subset_size=1000):
    n = real_features.shape[1]
    m = min(min(real_features.shape[0], gen_features.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kernel_distance = t / num_subsets / m
    return float(kernel_distance)