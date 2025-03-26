from tqdm import tqdm
import numpy as np


def compute_single_empirical(real_score, gen_score, real_mean, gen_mean, real_cov, gen_cov, attr_list):
    print('###### Computing for single attribute ######')
    # single attribute
    difference_single = []
    KL_real_gen_single = []
    KL_gen_real_single = []

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

        KL_real_gen_single.append(compute_empirical_KL_divergence_1d(real_mean_i, gen_mean_i, real_var_i, gen_var_i))

        KL_gen_real_single.append(compute_empirical_KL_divergence_1d(gen_mean_i, real_mean_i, gen_var_i, real_var_i))


    return [np.array(KL_real_gen_single), np.array(KL_gen_real_single), np.array(difference_single)]


def compute_single_empirical_reduced(real_score, gen_score, real_mean, gen_mean, real_cov, gen_cov, attr_list):
    print('###### Computing for single attribute ######')
    attr_list = attr_list[:10]  # only consider human face related attributes
    # single attribute
    difference_single = []
    KL_real_gen_single = []
    KL_gen_real_single = []

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

        KL_real_gen_single.append(compute_empirical_KL_divergence_1d(real_mean_i, gen_mean_i, real_var_i, gen_var_i))

        KL_gen_real_single.append(compute_empirical_KL_divergence_1d(gen_mean_i, real_mean_i, gen_var_i, real_var_i))

    return [np.array(KL_real_gen_single), np.array(KL_gen_real_single), np.array(difference_single)]


def compute_empirical_KL_divergence_1d(p_mean, q_mean, p_var, q_var):
    # KL(p,q) = Sigma p(x)log(p(x)/q(x))   use mean value, ignore variance here
    KL_div = p_mean * np.log(p_mean / q_mean)
    return float(KL_div)
