import numpy as np
from numba import njit

# numba

@njit
def calc_wap_njit(bp1, ap1, bs1, as1):
    return (bp1 * as1 + ap1 * bs1) / (bs1 + as1)

@njit
def prod_njit(a, b):
    return np.multiply(a, b)

@njit
def rv_njit(values):
    return np.sqrt(np.sum(np.square(values))) if len(values) > 0 else 0

@njit
def sum_njit(a): 
    return np.sum(a)

@njit
def mean_njit(values):
    return np.mean(values) if len(values) > 0 else 0

@njit
def mean_abs_njit(values):
    return np.mean(np.abs(values)) if len(values) > 0 else 0

@njit
def std_njit(values):
    return np.std(values) if len(values) > 0 else 0

@njit
def skew_njit(values):
    return np.skew(values) if len(values) > 0 else 0

@njit
def min_njit(values):
    return np.min(values) if len(values) > 0 else 0

@njit
def max_njit(values):
    return np.max(values) if len(values) > 0 else 0

@njit
def q1_njit(values):
    return np.quantile(values, q=0.25) if len(values) > 0 else 0

@njit
def q2_njit(values):
    return np.quantile(values, q=0.50) if len(values) > 0 else 0

@njit
def q3_njit(values):
    return np.quantile(values, q=0.75) if len(values) > 0 else 0


# for pandas

def rv_numba(values, index):
    return np.sqrt(np.sum(np.square(values))) if len(values) > 0 else 0

def rvp_numba(values, index):
    return np.sqrt(np.sum(np.square(np.maximum(values, 0)))) if len(values) > 0 else 0

def rvn_numba(values, index):
    return np.sqrt(np.sum(np.square(np.minimum(values, 0)))) if len(values) > 0 else 0

def bpv_numba(values, index):
    mu1_sq = 2 / np.pi
    return 1 / mu1_sq * np.sqrt(np.sum(np.abs(values[1:] * values[:-1]))) if len(values) > 1 else 0

def jv_numba(values, index):
    mu1_sq = 2 / np.pi
    rv = np.sqrt(np.sum(np.square(values))) if len(values) > 0 else 0
    bpv = 1 / mu1_sq * np.sqrt(np.sum(np.abs(values[1:] * values[:-1]))) if len(values) > 1 else 0
    return max(rv - bpv, 0)

def rq_numba(values, index):
    scaler = len(values) / 3
    return np.sqrt(np.sqrt(scaler * np.sum(np.power(values, 4)))) if len(values) > 0 else 0

def count_numba(values, index):
    return len(values)

def sqrt_inv_count_numba(values, index):
    return np.sqrt(1 / (1 + len(values)))

def sum_numba(values, index):
    return np.sum(values) if len(values) > 0 else 0

def sqrt_inv_sum_numba(values, index):
    return np.sqrt(1 / (1 + np.sum(values))) if len(values) > 0 else 0

def mean_numba(values, index):
    return np.mean(values) if len(values) > 0 else 0

def mean_abs_numba(values, index):
    return np.mean(np.abs(values)) if len(values) > 0 else 0

def std_numba(values, index):
    return np.std(values) if len(values) > 0 else 0

def skew_numba(values, index):
    return np.skew(values) if len(values) > 0 else 0

def min_numba(values, index):
    return np.min(values) if len(values) > 0 else 0

def max_numba(values, index):
    return np.max(values) if len(values) > 0 else 0

def q1_numba(values, index):
    return np.quantile(values, q=0.25) if len(values) > 0 else 0

def q2_numba(values, index):
    return np.quantile(values, q=0.50) if len(values) > 0 else 0

def q3_numba(values, index):
    return np.quantile(values, q=0.75) if len(values) > 0 else 0

def ptp_numba(values, index):
    return np.ptp(values) if len(values) > 0 else 0

def last_numba(values, index):
    return values[-1] if len(values) > 0 else 0

def sum_sq_numba(values, index):
    return np.sum(np.square(values)) if len(values) > 0 else 0

def iqr_numba(values, index):
    return np.percentile(values, 75) - np.percentile(values, 25) if len(values) > 0 else 0
