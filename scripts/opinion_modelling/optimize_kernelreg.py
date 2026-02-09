import concurrent.futures
import datetime

import matplotlib.pyplot as plt
import numba
import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg

# from stancemining.estimate import _get_time_series_data


def gaussian_bootstrap_vectorized(h, Xi, x):
    """
    Vectorized gaussian kernel.
    Xi: shape (n_bootstrap, nobs, k_vars)
    x: shape (N_predict, k_vars)
    Returns: shape (n_bootstrap, N_predict, nobs)
    """
    # Reshape for broadcasting: (N_predict, 1, k_vars) - (1, nobs, k_vars)
    diff = x[np.newaxis, :, np.newaxis, :] - Xi[:, np.newaxis, ...]
    return (1. / np.sqrt(2 * np.pi)) * np.exp(-np.sum(diff**2, axis=3) / (h**2 * 2.))


def _est_loc_linear_bootstrap_vectorized(bw, endog, exog, data_predict):
    """
    Fully vectorized local linear estimation for multiple prediction points.
    data_predict: shape (N_predict, k_vars)
    Returns: mean (n_bootstrap, N_predict,), mfx (n_bootstrap, N_predict, k_vars)
    """
    n_bootstrap, nobs, k_vars = exog.shape
    N_predict = data_predict.shape[0]
    
    # Compute kernels for all prediction points at once
    # ker shape: (N_predict, nobs)
    ker = gaussian_bootstrap_vectorized(bw[0], exog, data_predict) / (bw[0] * float(nobs))
    
    M12 = exog[:, np.newaxis, :, :] - data_predict[np.newaxis, :, np.newaxis, :] # shape (n_bootstrap, N_predict, nobs, k_vars)
    ker_weighted = ker[..., np.newaxis] # shape (n_bootstrap, N_predict, nobs, 1)
    
    # M22: (N_predict, k_vars, k_vars)
    # For each i: M12[i].T @ (M12[i] * ker[i])
    M22 = np.einsum('bpnk,bpnj->bpkj', M12 * ker_weighted, M12) # shape (n_bootstrap, N_predict, k_vars, k_vars)
    M12_sum = (M12 * ker_weighted).sum(axis=-2) # shape (n_bootstrap, N_predict, k_vars)
    
    # Build M matrix: (N_predict, k_vars+1, k_vars+1)
    M = np.zeros((n_bootstrap, N_predict, k_vars + 1, k_vars + 1))
    M[..., 0, 0] = ker.sum(axis=-1)
    M[..., 0, 1:] = M12_sum
    M[..., 1:, 0] = M12_sum
    M[..., 1:, 1:] = M22

    # ker_endog: (N_predict, nobs, 1)
    ker_endog = ker_weighted * endog[:, np.newaxis, :, :] # shape (n_bootstrap, N_predict, nobs, 1)
    
    # Build V vector: (n_bootstrap, N_predict, k_vars+1, 1)
    V = np.zeros((n_bootstrap, N_predict, k_vars + 1, 1))
    V[..., 0, 0] = ker_endog.sum(axis=(-2, -1))
    V[..., 1:, 0] = (M12 * ker_endog).sum(axis=-2)

    # Solve all linear systems at once
    # (N_predict, k_vars+1, k_vars+1) @ (N_predict, k_vars+1, 1)
    mean_mfx = np.linalg.pinv(M) @ V

    means = mean_mfx[..., 0, 0]
    mfx_all = mean_mfx[..., 1:, 0]

    return means, mfx_all


def kernel_reg_fit_bootstrap_vectorized(endog, exog, data_predict, bw):
    k_vars = 1
    endog = endog[..., np.newaxis]  # shape (n_bootstrap, nobs, 1)
    exog = exog[..., np.newaxis]  # shape (n_bootstrap, nobs, 1)
    data_predict = data_predict[:, np.newaxis]  # shape (N_predict, 1)
    bw = np.asarray(bw)

    mean, _ = _est_loc_linear_bootstrap_vectorized(
        bw, 
        endog, 
        exog,
        data_predict
    )

    return mean


def gaussian_vectorized(h, Xi, x):
    """
    Vectorized gaussian kernel.
    Xi: shape (nobs, k_vars)
    x: shape (N_predict, k_vars)
    Returns: shape (N_predict, nobs)
    """
    # Reshape for broadcasting: (N_predict, 1, k_vars) - (1, nobs, k_vars)
    diff = x[:, np.newaxis, :] - Xi[np.newaxis, :, :]
    return (1. / np.sqrt(2 * np.pi)) * np.exp(-np.sum(diff**2, axis=2) / (h**2 * 2.))

def _est_loc_linear_vectorized(bw, endog, exog, data_predict):
    """
    Fully vectorized local linear estimation for multiple prediction points.
    data_predict: shape (N_predict, k_vars)
    Returns: mean (N_predict,), mfx (N_predict, k_vars)
    """
    nobs, k_vars = exog.shape
    N_predict = data_predict.shape[0]
    
    # Compute kernels for all prediction points at once
    # ker shape: (N_predict, nobs)
    ker = gaussian_vectorized(bw[0], exog, data_predict) / (bw[0] * float(nobs))
    
    # M12: (N_predict, nobs, k_vars)
    M12 = exog[np.newaxis, :, :] - data_predict[:, np.newaxis, :]
    
    # ker_weighted: (N_predict, nobs, 1)
    ker_weighted = ker[:, :, np.newaxis]
    
    # M22: (N_predict, k_vars, k_vars)
    # For each i: M12[i].T @ (M12[i] * ker[i])
    M22 = np.einsum('pnk,pnj->pkj', M12 * ker_weighted, M12)
    
    M12_sum = (M12 * ker_weighted).sum(axis=1) # shape (N_predict, k_vars)
    
    # Build M matrix: (N_predict, k_vars+1, k_vars+1)
    M = np.zeros((N_predict, k_vars + 1, k_vars + 1))
    M[:, 0, 0] = ker.sum(axis=1)
    M[:, 0, 1:] = M12_sum
    M[:, 1:, 0] = M12_sum
    M[:, 1:, 1:] = M22
    
    # ker_endog: (N_predict, nobs, 1)
    ker_endog = ker_weighted * endog[np.newaxis, :, :] # shape (N_predict, nobs, 1)
    
    # Build V vector: (N_predict, k_vars+1, 1)
    V = np.zeros((N_predict, k_vars + 1, 1))
    V[:, 0, 0] = ker_endog.sum(axis=(-2,-1))
    V[:, 1:, 0] = (M12 * ker_endog).sum(axis=-2)

    # Solve all linear systems at once
    # (N_predict, k_vars+1, k_vars+1) @ (N_predict, k_vars+1, 1)
    mean_mfx = np.linalg.pinv(M) @ V
    
    means = mean_mfx[:, 0, 0]
    mfx_all = mean_mfx[:, 1:, 0]
    
    return means, mfx_all

def kernel_reg_fit_vectorized(endog, exog, data_predict, bw):
    k_vars = 1
    endog = _adjust_shape(endog, 1)
    exog = _adjust_shape(exog, k_vars)
    data_predict = _adjust_shape(data_predict, k_vars)
    bw = np.asarray(bw)

    mean, _ = _est_loc_linear_vectorized(
        bw, 
        endog, 
        exog,
        data_predict
    )

    return mean

# @numba.njit
def gaussian(h, Xi, x):
    return (1. / np.sqrt(2 * np.pi)) * np.exp(-(Xi - x)**2 / (h**2 * 2.))



def gpke(bw, data, data_predict):
    return gaussian(bw[0], data[:, 0], data_predict[0]) / bw[0]

# @numba.njit
def _adjust_shape(dat, k_vars):
    """ Returns an array of shape (nobs, k_vars) for use with `gpke`."""
    dat = np.asarray(dat)
    nobs = len(dat)
    dat = np.reshape(dat, (nobs, k_vars))
    return dat



def _est_loc_linear(bw, endog, exog, data_predict):
    nobs, k_vars = exog.shape
    ker = gpke(bw, data=exog, data_predict=data_predict) / float(nobs)

    # Convert ker to a 2-D array to make matrix operations below work
    ker = ker[:, np.newaxis]

    M12 = exog - data_predict # shape (nobs, k_vars)
    M22 = np.dot(M12.T, M12 * ker) # shape (k_vars, k_vars)
    M12 = (M12 * ker).sum(axis=0) # shape (k_vars,)
    M = np.empty((k_vars + 1, k_vars + 1))
    M[0, 0] = ker.sum()
    M[0, 1:] = M12
    M[1:, 0] = M12
    M[1:, 1:] = M22

    ker_endog = ker * endog # shape (nobs, k_vars)
    V = np.empty((k_vars + 1, 1))
    V[0, 0] = ker_endog.sum()
    V[1:, 0] = ((exog - data_predict) * ker_endog).sum(axis=0)

    mean_mfx = np.dot(np.linalg.pinv(M), V)
    mean = mean_mfx[0]
    mfx = mean_mfx[1:, :]
    return mean, mfx



# @numba.njit
def kernel_reg_fit(endog, exog, data_predict, bw):
    k_vars = 1
    endog = _adjust_shape(endog, 1)
    exog = _adjust_shape(exog, k_vars)
    data_predict = _adjust_shape(data_predict, k_vars)
    bw = np.asarray(bw)

    N_data_predict = np.shape(data_predict)[0]
    mean = np.empty((N_data_predict,))
    for i in range(N_data_predict):
        i_mean, _ = _est_loc_linear(
            bw, 
            endog, 
            exog,
            data_predict[i, :]
        )
        mean[i] = i_mean[0]

    return mean

# @numba.njit
def bootstrap_kernelreg(stance, timestamps, test_x, bandwidth, n_samples, n_bootstrap=100):
    all_preds = np.zeros((n_bootstrap, len(test_x)))
    for j in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_endog = stance[indices]
        boot_exog = timestamps[indices]

        # Fit kernel regression on bootstrap sample
        sample_pred = kernel_reg_fit_vectorized(boot_endog, boot_exog, test_x, [bandwidth])
        all_preds[j] = sample_pred

    all_preds = np.clip(all_preds, -1, 1)
    return all_preds

def bootstrap_kernelreg_vectorized(stance, timestamps, test_x, bandwidth, n_samples, n_bootstrap=100):
    indices = np.random.choice(n_samples, size=(n_bootstrap, n_samples), replace=True)

    # Resample with replacement
    boot_endog = stance[indices]
    boot_exog = timestamps[indices]

    # Fit kernel regression on bootstrap sample
    all_preds = kernel_reg_fit_bootstrap_vectorized(boot_endog, boot_exog, test_x, [bandwidth])

    all_preds = np.clip(all_preds, -1, 1)
    return all_preds

def main():
    np.random.seed(42)

    test_x = np.linspace(0, 365, 365)
    timestamps = np.random.choice(test_x, size=30)
    stance = np.random.uniform(-1, 1, size=30)
    stance = np.round(stance).astype(int)

    bandwidth = 10.0

    start_time = datetime.datetime.now()
    kr = KernelReg(stance, timestamps, var_type='c', bw=[bandwidth])
    sm_pred, _ = kr.fit(test_x)
    end_time = datetime.datetime.now()
    print(f"Statsmodels kernel regression took {end_time - start_time}")

    start_time = datetime.datetime.now()
    np_pred = kernel_reg_fit(stance, timestamps, test_x, [bandwidth])
    end_time = datetime.datetime.now()
    print(f"Custom kernel regression took {end_time - start_time}")

    start_time = datetime.datetime.now()
    op_pred = kernel_reg_fit_vectorized(stance, timestamps, test_x, [bandwidth])
    end_time = datetime.datetime.now()
    print(f"Optimized kernel regression took {end_time - start_time}")

    assert np.allclose(np_pred, sm_pred)
    assert np.allclose(op_pred, sm_pred)

    n_samples = stance.shape[0]
    indices = np.random.choice(n_samples, size=(1, n_samples), replace=True)

    # Resample with replacement
    boot_endog = stance[indices]
    boot_exog = timestamps[indices]

    # Fit kernel regression on bootstrap sample
    vec_boot_preds = kernel_reg_fit_bootstrap_vectorized(boot_endog, boot_exog, test_x, [bandwidth])
    boot_preds = kernel_reg_fit_vectorized(boot_endog[0], boot_exog[0], test_x, [bandwidth])
    assert np.allclose(vec_boot_preds[0], boot_preds)

    n_bootstrap = 100
    n_samples = len(stance)
    all_preds = np.zeros((n_bootstrap, len(test_x)))

    start_time = datetime.datetime.now()

    all_preds = bootstrap_kernelreg(stance, timestamps, test_x, bandwidth, n_samples, n_bootstrap=n_bootstrap)

    end_time = datetime.datetime.now()
    print(f"Kernel regression with bootstrapping took {end_time - start_time}")

    start_time = datetime.datetime.now()

    v_all_preds = bootstrap_kernelreg_vectorized(stance, timestamps, test_x, bandwidth, n_samples, n_bootstrap=n_bootstrap)
    # mean_pred = np.mean(all_preds, axis=0)
    # lower_pred = np.percentile(all_preds, 5, axis=0)
    # upper_pred = np.percentile(all_preds, 95, axis=0)

    end_time = datetime.datetime.now()
    print(f"Kernel regression with vectorized bootstrapping took {end_time - start_time}")

    mean_pred = np.mean(all_preds, axis=0)
    lower_pred = np.percentile(all_preds, 5, axis=0)
    upper_pred = np.percentile(all_preds, 95, axis=0)

    v_mean_pred = np.mean(v_all_preds, axis=0)
    v_lower_pred = np.percentile(v_all_preds, 5, axis=0)
    v_upper_pred = np.percentile(v_all_preds, 95, axis=0)

    fig, ax = plt.subplots()
    ax.scatter(timestamps, stance, label='Posts', alpha=0.3)
    ax.plot(test_x, mean_pred, label=f'Custom KR', color='orange')
    ax.fill_between(test_x, lower_pred, upper_pred, color='orange', alpha=0.2)

    ax.plot(test_x, v_mean_pred, label=f'Vectorized KR', color='green')
    ax.fill_between(test_x, v_lower_pred, v_upper_pred, color='green', alpha=0.2)

    ax.legend()
    fig.savefig('./figs/opt_kernelreg.png')


if __name__ == "__main__":
    main()
