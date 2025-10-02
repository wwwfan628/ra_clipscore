from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from utils.attribute_list import *
import scipy
import seaborn as sns


def load_attributes_list(attr_list_name, n_attr):
    return attr_dict[attr_list_name][:n_attr]


def plot_bar(result, labels, name, save_path=None):
    # check if results has the same length as labels
    assert len(result) == len(labels)

    # 设置 seaborn 风格
    sns.set_theme(style="whitegrid")  # 白色背景+网格

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=labels, y=result, color="skyblue")

    # 设置标题和标签
    plt.title(f"{name}\nmean={np.mean(result):.4f}, var={np.var(result, dtype=np.float32):.4f}")
    plt.ylabel('each attribute difference between 2 datasets', fontsize=8)
    plt.xticks(rotation=90)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}/{name}.png", dpi=300)
    else:
        plt.show()


def plot_single(results, attr_list, save_path):
    print('###### Plotting results of single attribute ######')
    results_contents = ['forward_KL', 'reverse_KL', 'mean_differences', 'frechet_distance', 'mmd_polynomial', 'mmd_rbf' ]
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
    mmd_polynomial_single = []
    mmd_rbf_single = []

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

        # MMD with polynomial kernel
        mmd_polynomial_single.append(compute_mmd_polynomial(real_score_i, gen_score_i))

        # MMD with RBF kernel
        mmd_rbf_single.append(compute_mmd_rbf(real_score_i, gen_score_i))

    return [np.array(KL_real_gen_single), np.array(KL_gen_real_single), np.array(difference_single), np.array(f_distance_single), np.array(mmd_polynomial_single), np.array(mmd_rbf_single)]


def compute_single_reduced(real_score, gen_score, real_mean, gen_mean, real_cov, gen_cov, attr_list):
    print('###### Computing for single attribute ######')
    attr_list = attr_list[:10]  # only consider human face related attributes
    # single attribute
    difference_single = []
    KL_real_gen_single = []
    KL_gen_real_single = []
    f_distance_single = []
    mmd_polynomial_single = []
    mmd_rbf_single = []

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

        # MMD with polynomial kernel
        mmd_polynomial_single.append(compute_mmd_polynomial(real_score_i, gen_score_i))

        # MMD with RBF kernel
        mmd_rbf_single.append(compute_mmd_rbf(real_score_i, gen_score_i))

    return [np.array(KL_real_gen_single), np.array(KL_gen_real_single), np.array(difference_single), np.array(f_distance_single), np.array(mmd_polynomial_single), np.array(mmd_rbf_single)]



def compute_pair(real_score, gen_score, real_mean, gen_mean, real_cov, gen_cov, attr_list):
    print('###### Computing for pair attributes ######')
    # pair attributes
    difference_pair = []
    KL_real_gen_pair = []
    KL_gen_real_pair = []
    f_distance_pair = []
    mmd_polynomial_pair = []
    mmd_rbf_pair = []

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

        # MMD with polynomial kernel
        mmd_polynomial_pair.append(compute_mmd_polynomial(real_score_ij, gen_score_ij))

        # MMD with RBF kernel
        mmd_rbf_pair.append(compute_mmd_rbf(real_score_ij, gen_score_ij))

    return [np.array(KL_real_gen_pair), np.array(KL_gen_real_pair), np.array(difference_pair), np.array(f_distance_pair), np.array(mmd_polynomial_pair), np.array(mmd_rbf_pair)]


def compute_pair_reduced(real_score, gen_score, real_mean, gen_mean, real_cov, gen_cov, attr_list):
    print('###### Computing for pair attributes ######')
    attr_list = attr_list[:10]  # only consider human face related attributes
    # pair attributes
    difference_pair = []
    KL_real_gen_pair = []
    KL_gen_real_pair = []
    f_distance_pair = []
    mmd_polynomial_pair = []
    mmd_rbf_pair = []

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

        # MMD with polynomial kernel
        mmd_polynomial_pair.append(compute_mmd_polynomial(real_score_ij, gen_score_ij))

        # MMD with RBF kernel
        mmd_rbf_pair.append(compute_mmd_rbf(real_score_ij, gen_score_ij))

    return [np.array(KL_real_gen_pair), np.array(KL_gen_real_pair), np.array(difference_pair), np.array(f_distance_pair), np.array(mmd_polynomial_pair), np.array(mmd_rbf_pair)]


def compute_all(real_score, gen_score, real_mean, gen_mean, real_cov, gen_cov, attr_list):
    print('###### Computing for all attributes ######')
    # all attributes

    # difference of mean values
    difference_all = float(abs(real_mean - gen_mean).sum())

    KL_real_gen_all = compute_KL_divergence_high_dim_gaussian_pdf(real_mean, gen_mean, real_cov, gen_cov)

    KL_gen_real_all = compute_KL_divergence_high_dim_gaussian_pdf(gen_mean, real_mean, gen_cov, real_cov)

    # Frechet Distance
    f_distance_all = compute_frechet_distance(real_mean, gen_mean, real_cov, gen_cov)

    # MMD with polynomial kernel
    mmd_polynomial_all = compute_mmd_polynomial(real_score, gen_score)

    # MMD with RBF kernel
    mmd_rbf_all = compute_mmd_rbf(real_score, gen_score)

    return [KL_real_gen_all, KL_gen_real_all, difference_all, f_distance_all, mmd_polynomial_all, mmd_rbf_all]


def compute_KL_divergence_1d_gaussian_pdf(p_mean, q_mean, p_var, q_var):
    # KL(p,q) = log(phi_q/phi_p) + ((phi_p² + (mean_p - mean_q)²)/2phi_q²) - 0.5   p represents original, q generated
    KL_div = np.log(np.sqrt(q_var) / np.sqrt(p_var)) + (
            (p_var + np.square((p_mean - q_mean))) / (2 * q_var)) - 0.5
    return float(KL_div)


def compute_KL_divergence_high_dim_gaussian_pdf(p_mean, q_mean, p_cov, q_cov):
    # KL(p,q) = 0.5 * [ log(det(phi_q)/det(phi_p)) - d + tr(cov_q^(-1) @ cov_p) + ((mean_q - mean_p).T @ cov_q^(-1) @ (mean_q - mean_p))]  p represents original, q generated

    part_1 = np.log(np.linalg.det(q_cov) / np.linalg.det(p_cov)) - p_cov.shape[0]
    part_2 = np.trace(np.linalg.inv(q_cov) @ p_cov)
    part_3 = (q_mean - p_mean).T @ np.linalg.inv(q_cov) @ (q_mean - p_mean)

    KL_div = 0.5 * (part_1 + part_2 + part_3)
    return float(KL_div)


def compute_frechet_distance(real_mean, gen_mean, real_cov, gen_cov):
    m = np.square(gen_mean - real_mean).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(gen_cov, real_cov), disp=False)  # pylint: disable=no-member
    frechet_distance = np.real(m + np.trace(gen_cov + real_cov - s * 2))
    return float(frechet_distance)


def compute_mmd_polynomial(real_features, gen_features, num_subsets=100, max_subset_size=1000):
    # real_features: [N_r, D], gen_features: [N_g, D]
    n = real_features.shape[1]  # embedding dim for the scaling inside kernel
    m = min(min(real_features.shape[0], gen_features.shape[0]), max_subset_size)

    def poly_kernel(A, B, n):
        return (A @ B.T / n + 1.0) ** 3

    acc = 0.0
    for _ in range(num_subsets):
        # Subsample m examples from each set to construct subsets
        x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]

        K_xx = poly_kernel(x, x, n)   # (m, m)  including the diagonal
        K_yy = poly_kernel(y, y, n)   # (m, m)  including the diagonal
        K_xy = poly_kernel(x, y, n)   # (m, m)  including the diagonal

        # Biased / minimum-variance estimator: use the full kernel matrix mean (including the diagonal)
        mmd2 = K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()
        acc += mmd2

    return float(acc / num_subsets)


_SIGMA = 10.0   # RBF bandwidth
_SCALE = 1000.0 # readability scale

def compute_mmd_rbf(x: np.ndarray, y: np.ndarray,
                              sigma: float = 1.0,
                              num_subsets: int = 50,
                              subset_size: int = 1000,
                              rng: np.random.Generator = None):
    """
    Unbiased MMD^2 with Gaussian RBF kernel using random subsets (KID-style).

    Args:
        x: array (n, d)
        y: array (m, d)
        sigma: RBF bandwidth (if None -> median heuristic可加)
        num_subsets: how many random subsets to average
        subset_size: number of samples per subset from each distribution
        rng: np.random.Generator for reproducibility

    Returns:
        mean, std of MMD^2 estimates
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = x.shape[0], x.shape[1]
    m = y.shape[0]

    rng = np.random.default_rng() if rng is None else rng
    gamma = 1.0 / (2.0 * sigma * sigma)

    scores = []
    for _ in range(num_subsets):
        # 随机子集
        idx_x = rng.choice(n, subset_size, replace=False)
        idx_y = rng.choice(m, subset_size, replace=False)
        X = x[idx_x]
        Y = y[idx_y]

        # pairwise 距离
        D_xx = np.sum((X[:, None, :] - X[None, :, :])**2, axis=-1)
        D_yy = np.sum((Y[:, None, :] - Y[None, :, :])**2, axis=-1)
        D_xy = np.sum((X[:, None, :] - Y[None, :, :])**2, axis=-1)

        K_xx = np.exp(-gamma * D_xx)
        K_yy = np.exp(-gamma * D_yy)
        K_xy = np.exp(-gamma * D_xy)

        # 去掉对角 (无偏估计)
        np.fill_diagonal(K_xx, 0.0)
        np.fill_diagonal(K_yy, 0.0)

        term_x = K_xx.sum() / (subset_size * (subset_size - 1))
        term_y = K_yy.sum() / (subset_size * (subset_size - 1))
        term_xy = K_xy.mean()

        mmd2 = term_x + term_y - 2.0 * term_xy
        scores.append(mmd2)

    return float(np.mean(scores))       # float(np.std(scores))
