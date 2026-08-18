"""
This is the code for solving the $M=2$ two-industry version of the model
"""
# Import packages
import os
import pickle
import time
import numpy as np
import scipy.optimize as opt
from scipy.optimize import elementwise as eopt
import matplotlib.pyplot as plt

# Designate data and image directories
cur_dir = os.path.dirname(os.path.abspath(__file__))
# designate the data directory as "data/m1" and create those two directories
# if they do not exist
data_dir = os.path.join(cur_dir, "..", "data", "m2b")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
images_dir = os.path.join(cur_dir, "..", "images", "m2b")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)


# Set parameters
years_per_period = 30
T = 15  # Guess for number of periods to steady-state
n1 = 1.0
n2 = 0.3
beta_annual = 0.96
beta = beta_annual ** (years_per_period)
print("Beta (per period):", beta)
sigma = 1.7
alpha1 = 0.5
alpha2 = 0.5
alpha_i_vec = np.array([alpha1, alpha2])
Z1 = 1.0
Z2 = 1.2
gamma1 = 0.35
gamma2 = 0.36
delta1_annual = 0.05
delta1 = 1 - (1 - delta1_annual) ** (years_per_period)
print("Delta1 (per period):", delta1)
delta2_annual = 0.045
delta2 = 1 - (1 - delta2_annual) ** (years_per_period)
print("Delta2 (per period):", delta2)
p1 = 1.0
c_min1 = 0.01
c_min2 = 0.02
SS_tol_stage1 = 1e-10
SS_max_iter_stage1 = 1000
p2_scan_min = 1e-2  # Lower end of p2 range scanned for the steady state
p2_scan_max = 1e1  # Upper end of p2 range scanned for the steady state
n_scan_ss = 400  # Number of grid points in the steady-state p2 scan
TPI_tol_stage1 = 1e-2
TPI_max_iter_stage1 = 300
TPI_tol_root = 1e-10
xi_ss = 0.01  # Damping parameter for SS iteration
xi_tpi = 0.0001  # Damping parameter for TPI iteration
b2_epsilon_ss = 1e-5  # Error term for steady-state boundaries on b2
L1_epsilon_ss = 1e-3  # Error term for steady-state boundaries on L1

# Define functions


def display_time(seconds):
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.5f}s"


def get_p_ss(pi_vec, alpha_i_vec):
    """
    Solve for steady-state composite consumption good price from differentiated
    goods prices and parameters

    Args:
        pi_vec (np.array): Vector of differentiated goods prices. Should have
            dimension of number of industries I. In teh steady-state, dimension
            should be (,I) or (1,I), If in the transition path, dimension
            should be (T,I). The first element or first column should always be
            1.
        alpha_i_vec (np.array): Vector of alpha_i parameters of dimension (,I)

    Returns:
        p (float): Composite price
    """
    p = np.prod((pi_vec / alpha_i_vec) ** alpha_i_vec)

    return p


def get_p_tp(p1_path, p2_path, alpha_i_vec):
    """
    Solve for transition path composite consumption good price from
    differentiated goods prices and parameters

    Args:
        p1_path ((T,) vector): Vector of numeraire good prices. Should be a vector
            of ones of length T.
        p2_path ((T,) vector): Vector of differentiated good prices.
        alpha_i_vec (I,): Vector of alpha_i parameters of dimension (,I)

    Returns:
        p (float): Composite price
    """
    p = (
        ((p1_path / alpha_i_vec[0]) ** alpha_i_vec[0]) *
        ((p2_path / alpha_i_vec[1]) ** alpha_i_vec[1])
    )

    return p


def ss_KL1_zerofunc(KLratio, *args):
    p2, gamma1, Z1, delta1, gamma2, Z2, delta2 = args
    MPK1 = gamma1 * Z1 * (KLratio ** (gamma1 - 1)) - delta1
    MPK2 = (
        gamma2 * Z2 *
        (
            (
                ((1 - gamma2) * p2 * Z2) /
                ((1 - gamma1) * Z1 * (KLratio ** gamma1))
            ) ** ((1 - gamma2) / gamma2)
        ) - delta2
    )
    zerofunc = MPK1 - MPK2

    return zerofunc


def get_KL2ratio(KL1ratio, p2_init, gamma1, Z1, gamma2, Z2):
    KL2ratio = (
        ((1 - gamma1) * Z1 * (KL1ratio ** gamma1)) /
        ((1 - gamma2) * p2_init * Z2)
    ) ** (1 / gamma2)

    return KL2ratio


def get_KL1_max_ss(gamma1, Z1, delta1, delta2):
    """
    Largest steady-state KL1 ratio for which industry 2's capital-labor ratio is
    defined.

    The steady-state no-arbitrage condition is r = MPK1 - delta1 = MPK2 -
    delta2, and MPK2 > 0 requires r + delta2 > 0. Because MPK1 is strictly
    decreasing in KL1, this puts a finite upper bound on KL1 whenever
    delta1 > delta2:

        gamma1 * Z1 * KL1 ** (gamma1 - 1) - delta1 + delta2 > 0
        =>  KL1 < (gamma1 * Z1 / (delta1 - delta2)) ** (1 / (1 - gamma1))

    Args:
        gamma1 (float): Capital share in industry 1
        Z1 (float): Total factor productivity in industry 1
        delta1 (float): Depreciation rate in industry 1
        delta2 (float): Depreciation rate in industry 2

    Returns:
        KL1_max (float): Upper bound of the admissible KL1 interval (np.inf if
            delta1 <= delta2)
    """
    if delta1 <= delta2:
        return np.inf

    return (gamma1 * Z1 / (delta1 - delta2)) ** (1 / (1 - gamma1))


def get_p2_of_KL1_ss(KL1ratio, gamma1, Z1, delta1, gamma2, Z2, delta2):
    """
    Closed-form steady-state price p2 implied by a given KL1 ratio.

    Rather than solving ss_KL1_zerofunc() numerically for KL1 given p2, this
    inverts the same two equilibrium conditions analytically in the other
    direction. Given KL1:

        r    = gamma1 * Z1 * KL1 ** (gamma1 - 1) - delta1     (firm 1 FOC_K)
        KL2  = ((r + delta2) / (gamma2 * Z2)) ** (1 / (gamma2 - 1))
                                                              (firm 2 FOC_K)
        p2   = (1 - gamma1) * Z1 * KL1 ** gamma1 /
               ((1 - gamma2) * Z2 * KL2 ** gamma2)             (w1 = w2)

    Args:
        KL1ratio (float): Capital-labor ratio in industry 1, must lie in
            (0, get_KL1_max_ss(...))
        gamma1, Z1, delta1 (float): Industry 1 parameters
        gamma2, Z2, delta2 (float): Industry 2 parameters

    Returns:
        p2 (float): Implied price of good 2
        KL2ratio (float): Implied capital-labor ratio in industry 2
        r (float): Implied interest rate
    """
    r = gamma1 * Z1 * (KL1ratio ** (gamma1 - 1)) - delta1
    if r + delta2 <= 0:
        raise ValueError(
            f"KL1ratio={KL1ratio} is above KL1_max: MPK2 = r + delta2 = "
            f"{r + delta2} <= 0, so KL2 is undefined."
        )
    KL2ratio = ((r + delta2) / (gamma2 * Z2)) ** (1 / (gamma2 - 1))
    p2 = (
        ((1 - gamma1) * Z1 * (KL1ratio ** gamma1)) /
        ((1 - gamma2) * Z2 * (KL2ratio ** gamma2))
    )

    return p2, KL2ratio, r


def get_KL1_turnpoint_ss(gamma1, Z1, delta1, gamma2, Z2, delta2):
    """
    Interior minimizer of p2(KL1), which exists only when delta1 <= delta2.

    When delta1 > delta2 the domain of KL1 is bounded above by KL1_max and
    p2(KL1) falls monotonically from +infinity to 0 across it. When
    delta1 <= delta2 the domain is unbounded, r stays above -delta2 for every
    KL1, and p2(KL1) is U shaped: it goes to +infinity at both ends, so a given
    p2 corresponds to either zero or two KL1 values. This returns the KL1 at
    the bottom of that U, which splits the domain into two monotone branches.

    Args:
        gamma1, Z1, delta1 (float): Industry 1 parameters
        gamma2, Z2, delta2 (float): Industry 2 parameters

    Returns:
        KL1_turn (float): Minimizer of p2(KL1), or np.inf when delta1 > delta2
            (no interior turning point on the admissible domain)
    """
    if delta1 > delta2:
        return np.inf

    def neg_log_p2(log_KL1):
        return np.log(get_p2_of_KL1_ss(
            np.exp(log_KL1), gamma1, Z1, delta1, gamma2, Z2, delta2
        )[0])

    sol_turn = opt.minimize_scalar(
        neg_log_p2, bounds=(np.log(1e-250), np.log(1e250)), method="bounded",
        options={"xatol": 1e-12}
    )

    return np.exp(sol_turn.x)


def solve_KL1_given_p2(
    p2, gamma1, Z1, delta1, gamma2, Z2, delta2, KL1_floor=1e-300,
    branch="low", max_expand=400
):
    """
    Solve for the steady-state KL1 ratio implied by p2 > 0.

    This replaces a fixed bracket on ss_KL1_zerofunc(), which only brackets a
    root for a narrow window of p2 values. Two things make this robust for any
    p2 > 0:

    1. The search interval is the analytic domain (0, KL1_max) from
       get_KL1_max_ss(), on which p2(KL1) from get_p2_of_KL1_ss() is strictly
       decreasing from +infinity down to 0. A root therefore always exists and
       is unique.
    2. The zero function is written in logs of both KL1 and p2. The map is very
       stiff when gamma1 is close to gamma2 (here KL1 moves by ~60 orders of
       magnitude as p2 moves over [0.01, 5]), so a linearly scaled bracket
       cannot resolve it, while a log-scaled one bisects it easily.

    When delta1 <= delta2 the map is not one-to-one (see
    get_KL1_turnpoint_ss()). The search is then confined to one of the two
    monotone branches, selected by the branch argument.

    Args:
        p2 (float): Price of good 2, must be strictly positive
        gamma1, Z1, delta1 (float): Industry 1 parameters
        gamma2, Z2, delta2 (float): Industry 2 parameters
        KL1_floor (float): Smallest KL1 the search will consider
        branch (str): "low" or "high". Only used when delta1 <= delta2, where
            p2 maps to two KL1 values. "low" takes the low KL1 (high r) branch,
            which is the one that connects continuously to the delta1 > delta2
            case.
        max_expand (int): Cap on bracket expansion steps, so an unbracketable
            p2 raises instead of looping forever

    Returns:
        KL1ratio (float): Capital-labor ratio in industry 1 consistent with p2
    """
    if p2 <= 0:
        raise ValueError(f"p2 must be strictly positive. Got p2={p2}.")
    if branch not in ("low", "high"):
        raise ValueError(f"branch must be 'low' or 'high'. Got '{branch}'.")
    log_p2 = np.log(p2)

    def log_p2_zerofunc(log_KL1):
        p2_implied = get_p2_of_KL1_ss(
            np.exp(log_KL1), gamma1, Z1, delta1, gamma2, Z2, delta2
        )[0]

        return np.log(p2_implied) - log_p2

    KL1_max = get_KL1_max_ss(gamma1, Z1, delta1, delta2)
    log_KL1_floor = np.log(KL1_floor)
    if np.isfinite(KL1_max):
        # p2(KL1) decreases from +infinity to 0 on (0, KL1_max). The upper
        # endpoint itself is not admissible (MPK2 = 0 there), so step inside.
        log_KL1_hi = np.log(KL1_max) - 1e-10
        log_KL1_lo = log_KL1_hi - 1.0
        n_expand = 0
        while log_p2_zerofunc(log_KL1_lo) < 0:
            log_KL1_lo -= 5.0
            n_expand += 1
            if log_KL1_lo < log_KL1_floor or n_expand > max_expand:
                raise ValueError(
                    f"No KL1 in ({KL1_floor}, {KL1_max}) is consistent with "
                    f"p2={p2}. Either p2 is implausibly large or the industry "
                    "parameters are inconsistent."
                )
    else:
        # delta1 <= delta2: p2(KL1) is U shaped. Confine the search to one
        # monotone branch on either side of the turning point.
        KL1_turn = get_KL1_turnpoint_ss(
            gamma1, Z1, delta1, gamma2, Z2, delta2
        )
        log_KL1_turn = np.log(KL1_turn)
        if log_p2_zerofunc(log_KL1_turn) > 0:
            p2_turn = get_p2_of_KL1_ss(
                KL1_turn, gamma1, Z1, delta1, gamma2, Z2, delta2
            )[0]
            raise ValueError(
                f"p2={p2} is below the minimum attainable price {p2_turn} "
                f"(reached at KL1={KL1_turn}). No KL1 is consistent with it."
            )
        n_expand = 0
        if branch == "low":
            log_KL1_hi = log_KL1_turn
            log_KL1_lo = log_KL1_turn - 1.0
            while log_p2_zerofunc(log_KL1_lo) < 0:
                log_KL1_lo -= 5.0
                n_expand += 1
                if log_KL1_lo < log_KL1_floor or n_expand > max_expand:
                    raise ValueError(
                        f"Could not bracket KL1 on the low branch for p2={p2}."
                    )
        else:
            log_KL1_lo = log_KL1_turn
            log_KL1_hi = log_KL1_turn + 1.0
            while log_p2_zerofunc(log_KL1_hi) < 0:
                log_KL1_hi += 5.0
                n_expand += 1
                if n_expand > max_expand:
                    raise ValueError(
                        f"Could not bracket KL1 on the high branch for "
                        f"p2={p2}."
                    )
    sol_KL1 = opt.root_scalar(
        log_p2_zerofunc, bracket=[log_KL1_lo, log_KL1_hi], method="brentq",
        xtol=1e-15, rtol=8.9e-16
    )
    if not sol_KL1.converged:
        raise ValueError(
            f"KL1 root finding did not converge for p2={p2}. "
            f"Flag: {sol_KL1.flag}"
        )

    return np.exp(sol_KL1.root)


def tp_KL1_zerofunc(KL1rat_path, *args):
    p2t_path, p2tm1_path, gamma1, Z1, delta1, gamma2, Z2, delta2 = args
    MPK1 = gamma1 * Z1 * (KL1rat_path ** (gamma1 - 1)) - delta1
    MPK2 = (
        (p2t_path / p2tm1_path) * (
            gamma2 * Z2 * (
                (
                    (p2t_path * (1 - gamma2) * Z2) /
                    ((1 - gamma1) * Z1 * (KL1rat_path ** gamma1))
                ) ** ((1 - gamma2) / gamma2)
            ) + 1 - delta2
        ) - 1
    )
    zerofunc = MPK1 - MPK2

    return zerofunc


def get_w_KL(KLratio_i, p_i, Z_i, gamma_i):
    w = p_i * (1 - gamma_i) * Z_i * (KLratio_i ** gamma_i)

    return w


def get_r_KL(KLratio_i, p_it, p_itm1, Z_i, gamma_i, delta_i):
    r = (
        (p_it / p_itm1) * gamma_i * Z_i * (KLratio_i ** (gamma_i - 1)) + 1 -
        delta_i
    ) - 1

    return r


def get_YLi_ratio(KLi_ratio, Z_i, gamma_i):
    YLi_ratio = Z_i * (KLi_ratio ** gamma_i)

    return YLi_ratio


def get_b2_ss(w, r, p, p1, p2, n1, n2, c_min1, c_min2, beta, sigma):
    num_term1 = w * n1 - p1 * c_min1 - p2 * c_min2
    num_term2 = (
        ((beta * (1 + r)) ** (-1 / sigma)) *
        (w * n2 - p1 * c_min1 - p2 * c_min2)
    )
    denom = 1 + ((beta * (1 + r)) ** (-1 / sigma)) * (1 + r)
    b2 = (num_term1 - num_term2) / denom

    return b2

def get_b2tp1_tp(
    wt_path, pt_path, p2t_path, wtp1_path, rtp1_path, ptp1_path, p2tp1_path,
    n1, n2, c_min1, c_min2, beta, sigma
):
    numer = (
        (wt_path / pt_path) * n1 - (1 / pt_path) * c_min1 -
        (p2t_path / pt_path) * c_min2 -
        (((beta * pt_path * (1 + rtp1_path)) / (ptp1_path)) ** (-1 / sigma)) *
        (
            (wtp1_path / ptp1_path) * n2 - (1 / ptp1_path) * c_min1 -
            (p2tp1_path / ptp1_path) * c_min2
        )
    )
    denom = (
        (1 / pt_path) +
        (((beta * pt_path * (1 + rtp1_path)) / (ptp1_path)) ** (-1 / sigma)) *
        ((1 + rtp1_path) / ptp1_path)
    )
    b2tp1_path = numer / denom

    return b2tp1_path


def get_ct_s1(b2tp1, wt, p1t, p2t, pt, n1, c_min1, c_min2):
    """
    Calculate the composite consumption of an young agent (s=1) from the budget
    constraint
    """
    c_s1t = (
        (wt / pt) * n1 - (p1t / pt) * c_min1 - (p2t / pt) * c_min2 -
        (b2tp1 / pt)
    )

    return c_s1t


def get_ct_s2(b2t, wt, rt, p1t, p2t, pt, n2, c_min1, c_min2):
    """
    Calculate the composite consumption of an old agent (s=2) from the budget
    constraint
    """
    c_s2t = (
        ((1 + rt) / pt) * b2t + (wt / pt) * n2 - (p1t / pt) * c_min1 -
        (p2t / pt) * c_min2
    )

    return c_s2t


def get_c_is(alpha_i, p_i, p, c_s, c_min_i):
    c_is = alpha_i * (p / p_i) * c_s + c_min_i

    return c_is


def get_L1_path(b2t, p2tm1, KL1rat, KL2rat, n1, n2, L1_epsilon):
    L1_path_uncstr = (
        (b2t - p2tm1 * (n1 + n2) * KL2rat) / (KL1rat - p2tm1 * KL2rat)
    )
    L1_min = L1_epsilon
    L1_max = n1 + n2 - L1_epsilon
    L1_path = np.minimum(np.maximum(L1_path_uncstr, L1_min), L1_max)
    # Make a boolean vector that equals True if any element of L1_path is equal
    # to L1_min or L1_max
    L1_path_cstr = np.logical_or(L1_path == L1_min, L1_path == L1_max)

    return L1_path, L1_path_cstr


def get_p2_new_ss(L1, K2, w, n1, n2, Z2, gamma2):
    p2_new = (
        (w * ((n1 + n2 - L1) ** gamma2)) / ((1-gamma2) * Z2 * (K2 ** gamma2))
    )

    return p2_new


def gen_ssvals_given_p2(p2, param_args, return_cstr=False):
    """
    Steady-state values implied by a price p2. Thin wrapper that recovers the
    KL1 ratio consistent with p2 and defers to gen_ssvals_given_KL1().
    """
    (
        p1, n1, n2, c_min1, c_min2, beta, sigma, alpha_i_vec, gamma1, Z1,
        delta1, gamma2, Z2, delta2, epsilon_ss
    ) = param_args
    KL1ratio = solve_KL1_given_p2(
        p2, gamma1, Z1, delta1, gamma2, Z2, delta2
    )

    return gen_ssvals_given_KL1(KL1ratio, param_args, return_cstr=return_cstr)


def gen_ssvals_given_KL1(KL1ratio, param_args, return_cstr=False):
    """
    Steady-state values implied by a capital-labor ratio KL1 in industry 1.

    KL1 rather than p2 is the natural state variable for the steady state: it
    lives on a bounded interval (0, KL1_max), every admissible p2 is the image
    of some KL1 under the closed form in get_p2_of_KL1_ss(), and no root
    finding is needed to go in this direction. gen_ssvals_given_p2() is the
    same computation preceded by an inversion.
    """
    (
        p1, n1, n2, c_min1, c_min2, beta, sigma, alpha_i_vec, gamma1, Z1,
        delta1, gamma2, Z2, delta2, epsilon_ss
    ) = param_args
    alpha1 = alpha_i_vec[0]
    alpha2 = alpha_i_vec[1]
    p2, KL2ratio, _ = get_p2_of_KL1_ss(
        KL1ratio, gamma1, Z1, delta1, gamma2, Z2, delta2
    )
    pi_vec = np.array([1.0, p2])
    p = get_p_ss(pi_vec, alpha_i_vec)
    w = get_w_KL(KL1ratio, p1, Z1, gamma1)
    r = get_r_KL(KL1ratio, p1, p1, Z1, gamma1, delta1)
    YL1ratio = get_YLi_ratio(KL1ratio, Z1, gamma1)
    YL2ratio = get_YLi_ratio(KL2ratio, Z2, gamma2)
    IL1ratio = delta1 * KL1ratio
    IL2ratio = delta2 * KL2ratio
    b2_min = (p1 * c_min1 + p2 * c_min2 - w * n2) / (1 + r) + epsilon_ss
    b2_max = (
        (w / p) * n1 - (p1 / p) * c_min1 - (p2 / p) * c_min2 - epsilon_ss
    )
    if b2_max <= b2_min:
        raise ValueError("No feasible b2 exists: b2_max <= b2_min.")
    b2_uncstr = get_b2_ss(
        w, r, p, p1, p2, n1, n2, c_min1, c_min2, beta, sigma
    )
    if b2_uncstr < b2_min:
        b2 = b2_min
    elif b2_uncstr > b2_max:
        b2 = b2_max
    elif b2_uncstr >= b2_min and b2_uncstr <= b2_max:
        b2 = b2_uncstr
    c1 = get_ct_s1(b2, w, p1, p2, p, n1, c_min1, c_min2)
    c2 = get_ct_s2(b2, w, r, p1, p2, p, n2, c_min1, c_min2)
    c_11 = get_c_is(alpha1, p1, p, c1, c_min1)
    c_12 = get_c_is(alpha1, p1, p, c2, c_min1)
    c_21 = get_c_is(alpha2, p2, p, c1, c_min2)
    c_22 = get_c_is(alpha2, p2, p, c2, c_min2)
    C1 = c_11 + c_12
    C2 = c_21 + c_22
    L1_min = epsilon_ss
    L1_max = n1 + n2 - epsilon_ss
    L1_uncstr = C1 / (YL1ratio - IL1ratio)
    if L1_max <= L1_min:
        raise ValueError("No feasible L1 exists: L1_max <= L1_min.")
    if L1_uncstr < L1_min:
        L1 = L1_min
    elif L1_uncstr > L1_max:
        L1 = L1_max
    elif L1_uncstr >= L1_min and L1_uncstr <= L1_max:
        L1 = L1_uncstr
    K1 = KL1ratio * L1
    I1 = IL1ratio * L1
    Y1 = YL1ratio * L1
    L2 = n1 + n2 - L1
    K2 = KL2ratio * L2
    I2 = IL2ratio * L2
    Y2 = YL2ratio * L2

    ss_vals = (
        b2, c1, c2, c_11, c_12, c_21, c_22, C1, C2, p1, p2, p, w, r, L1, L2,
        K1, K2, Y1, Y2, I1, I2
    )
    if return_cstr:
        # True when b2 or L1 had to be clipped to a boundary. Clipped points
        # are not equilibria of the model, so the steady-state solver must not
        # accept a market-clearing "root" that sits in a clipped region.
        b2_cstr = (b2_uncstr < b2_min) or (b2_uncstr > b2_max)
        L1_cstr = (L1_uncstr < L1_min) or (L1_uncstr > L1_max)

        return ss_vals, b2_cstr, L1_cstr

    return ss_vals


def gen_paths_given_p2tm1_tp(p2tm1_path, param_args):
    (
        T, p1_path, n1, n2, c_min1, c_min2, beta, sigma, alpha_i_vec, gamma1,
        Z1, delta1, gamma2, Z2, delta2, b2_1, epsilon_ss, w_ss, r_ss, p_ss,
        p2_ss, K1_ss, K2_ss
    ) = param_args
    alpha1 = alpha_i_vec[0]
    alpha2 = alpha_i_vec[1]
    p2t_path = np.append(p2tm1_path[1:], p2_ss)
    p_path = get_p_tp(p1_path, p2t_path, alpha_i_vec)
    KL1_args = (
        p2t_path, p2tm1_path, gamma1, Z1, delta1, gamma2, Z2, delta2
    )
    sol_KL1_bracket = eopt.bracket_root(
        tp_KL1_zerofunc, xl0=1e-12 * np.ones(T), args=KL1_args
    )
    # If any of the elements of sol_KL1_bracket.success are False, raise an
    # error that the bracket was not found
    if not np.all(sol_KL1_bracket.success):
        print("sol_KL1_bracket.success:")
        print(sol_KL1_bracket.success)
        err_msg = "KL1 bracket finding did not succeed in every period."
        raise ValueError(err_msg)
    sol_KL1_tp = eopt.find_root(
        tp_KL1_zerofunc, sol_KL1_bracket.bracket, args=KL1_args
    )
    if not np.all(sol_KL1_tp.success):
        print("sol_KL1_tp.success:")
        print(sol_KL1_tp.success)
        err_msg = "KL1 root finding did not succeed in every period."
        raise ValueError(err_msg)
    KL1rat_path = sol_KL1_tp.x
    KL2rat_path = get_KL2ratio(KL1rat_path, p2t_path, gamma1, Z1, gamma2, Z2)
    w_path = get_w_KL(KL1rat_path, p1_path, Z1, gamma1)
    r_path = get_r_KL(KL1rat_path, p1_path, p1_path, Z1, gamma1, delta1)
    YL1rat_path = get_YLi_ratio(KL1rat_path, Z1, gamma1)
    YL2rat_path = get_YLi_ratio(KL2rat_path, Z2, gamma2)
    wtp1_path = np.append(w_path[1:], w_ss)
    rtp1_path = np.append(r_path[1:], r_ss)
    ptp1_path = np.append(p_path[1:], p_ss)
    p2tp1_path = np.append(p2t_path[1:], p2_ss)
    b2tp1_path = get_b2tp1_tp(
        w_path, p_path, p2t_path, wtp1_path, rtp1_path, ptp1_path,
        p2tp1_path, n1, n2, c_min1, c_min2, beta, sigma
    )
    b2t_path = np.append(b2_1, b2tp1_path[:-1])
    c_s1_path = get_ct_s1(
        b2tp1_path, w_path, p1_path, p2t_path, p_path, n1, c_min1, c_min2
    )
    c_s2_path = get_ct_s2(
        b2t_path, w_path, r_path, p1_path, p2t_path, p_path, n2, c_min1, c_min2
    )
    c_11_path = get_c_is(alpha1, p1_path, p_path, c_s1_path, c_min1)
    c_12_path = get_c_is(alpha1, p1_path, p_path, c_s2_path, c_min1)
    c_21_path = get_c_is(alpha2, p2t_path, p_path, c_s1_path, c_min2)
    c_22_path = get_c_is(alpha2, p2t_path, p_path, c_s2_path, c_min2)
    C1_path = c_11_path + c_12_path
    C2_path = c_21_path + c_22_path
    L1_path, L1_path_cstr = get_L1_path(
        b2t_path, p2tm1_path, KL1rat_path, KL2rat_path, n1, n2, epsilon_ss
    )
    L2_path = n1 + n2 - L1_path
    K1_path = KL1rat_path * L1_path
    K2_path = KL2rat_path * L2_path
    K1tp1_path = np.append(K1_path[1:], K1_ss)
    K2tp1_path = np.append(K2_path[1:], K2_ss)
    I1_path = K1tp1_path - (1 - delta1) * K1_path
    I2_path = K2tp1_path - (1 - delta2) * K2_path
    Y1_path = YL1rat_path * L1_path
    Y2_path = YL2rat_path * L2_path
    p2tm1_path_long = np.append(p2tm1_path, p2_ss)

    return (
        b2t_path, c_s1_path, c_s2_path, c_11_path, c_12_path, c_21_path,
        c_22_path, C1_path, C2_path, p1_path, p2tm1_path_long, p_path, w_path,
        r_path, L1_path, L2_path, K1_path, K2_path, Y1_path, Y2_path, I1_path,
        I2_path
    )


def get_final_constraint_vec_error_ss(p2, *args):
    (
        b2, c1, c2, c_11, c_12, c_21, c_22, C1, C2, p1, p2, p, w, r, L1, L2,
        K1, K2, Y1, Y2, I1, I2
    ) = gen_ssvals_given_p2(p2, args)
    KMC_err = K1 + p2 * K2 - b2

    return KMC_err


def get_KMC_err_ss(KL1ratio, param_args):
    """
    Capital market clearing error K1 + p2 * K2 - b2 at a given KL1 ratio,
    together with a flag for whether any household or labor constraint was
    clipped there.
    """
    ss_vals, b2_cstr, L1_cstr = gen_ssvals_given_KL1(
        KL1ratio, param_args, return_cstr=True
    )
    b2 = ss_vals[0]
    p2 = ss_vals[10]
    K1, K2 = ss_vals[16], ss_vals[17]
    KMC_err = K1 + p2 * K2 - b2

    return KMC_err, (b2_cstr or L1_cstr)


def solve_p2_ss(
    param_args, p2_guess=None, p2_scan_min=1e-2, p2_scan_max=1e1, n_scan=400,
    p2_tol=1e-13, KL1_scan_floor=1e-40, KL1_scan_cap=1e12,
    print_diagnostics=True
):
    """
    Solve for the steady-state price p2_ss of good 2.

    This replaces the damped fixed-point iteration
    p2 <- p2 * (1 + xi_ss * pct_error), which only converged from starting
    values already close to p2_ss. Three properties make this version
    insensitive to the initial guess:

    1. It searches over KL1 rather than p2, and it scans rather than steps.
       KL1 lives on the bounded interval (0, KL1_max) from get_KL1_max_ss(),
       every admissible p2 is the image of some KL1 under the closed form in
       get_p2_of_KL1_ss(), and no inversion is needed in this direction. A
       log-uniform grid of KL1 therefore sweeps the whole equilibrium space
       with no risk of walking off the domain, whatever the ordering of
       gamma1 vs gamma2 or delta1 vs delta2. It also adapts automatically to
       the stiffness of the p2 <-> KL1 map: with gamma1 close to gamma2, KL1
       moves ~60 orders of magnitude as p2 crosses [0.01, 10], so a grid that
       is uniform in p2 resolves the interesting region very poorly.
    2. It only accepts sign changes between two adjacent grid points at which
       neither b2 nor L1 was clipped. The clipping in gen_ssvals_given_KL1()
       introduces kinks that can flip the sign of the market clearing error at
       a point that is not an equilibrium; with this model's calibration there
       is one such spurious crossing near p2 = 0.86, where b2 sits at b2_min.
    3. It finishes with Brent's method on a verified bracket in log(KL1),
       which enforces KL1 > 0 (hence p2 > 0) and converges to machine
       precision in a few iterations.

    p2_guess is no longer needed for convergence. It is used only to pick
    among multiple interior roots if the model admits more than one.

    Args:
        param_args (tuple): The ss_args tuple passed to gen_ssvals_given_p2()
        p2_guess (float or None): Optional hint used only to select among
            multiple roots
        p2_scan_min (float): Lower end of the p2 range to scan. Soft request:
            if it is outside the attainable range of p2(KL1), the scan falls
            back to the corresponding end of the KL1 domain.
        p2_scan_max (float): Upper end of the p2 range to scan, same caveat
        n_scan (int): Number of grid points in the scan
        p2_tol (float): Relative tolerance in the final Brent solve
        KL1_scan_floor (float): Lower end of the KL1 domain to scan
        KL1_scan_cap (float): Upper end of the KL1 domain to scan, used when
            KL1_max is infinite (delta1 <= delta2)
        print_diagnostics (bool): Print the scan summary

    Returns:
        p2_ss (float): Steady-state price of good 2
        KL1_ss (float): Steady-state capital-labor ratio in industry 1. Pass
            this to gen_ssvals_given_KL1() rather than round tripping p2_ss
            through gen_ssvals_given_p2(): when gamma1 > gamma2 or
            delta1 <= delta2, p2(KL1) is not one-to-one, so the inversion can
            land on a different KL1 than the one solved for.
    """
    (
        p1, n1, n2, c_min1, c_min2, beta, sigma, alpha_i_vec, gamma1, Z1,
        delta1, gamma2, Z2, delta2, epsilon_ss
    ) = param_args
    ind_args = (gamma1, Z1, delta1, gamma2, Z2, delta2)

    # Scan interval in log(KL1). The requested p2 range is a soft request: an
    # endpoint outside the attainable range of p2(KL1) falls back to the end
    # of the KL1 domain rather than raising. p2(KL1) need not be monotone here
    # (it is hump shaped when gamma1 > gamma2 and U shaped when
    # delta1 <= delta2), which is exactly why the search runs over KL1.
    KL1_max = get_KL1_max_ss(gamma1, Z1, delta1, delta2)
    log_KL1_domain_hi = np.log(min(KL1_max, KL1_scan_cap)) - 1e-10
    log_KL1_domain_lo = np.log(KL1_scan_floor)
    if log_KL1_domain_lo >= log_KL1_domain_hi:
        raise ValueError(
            f"Empty KL1 domain: KL1_scan_floor={KL1_scan_floor} is not below "
            f"min(KL1_max, KL1_scan_cap)={min(KL1_max, KL1_scan_cap)}."
        )

    def get_log_KL1_endpoint(p2_endpoint, log_KL1_fallback):
        try:
            return np.clip(
                np.log(solve_KL1_given_p2(p2_endpoint, *ind_args)),
                log_KL1_domain_lo, log_KL1_domain_hi
            )
        except ValueError:
            return log_KL1_fallback

    log_KL1_hi = get_log_KL1_endpoint(p2_scan_min, log_KL1_domain_hi)
    log_KL1_lo = get_log_KL1_endpoint(p2_scan_max, log_KL1_domain_lo)
    if log_KL1_lo >= log_KL1_hi:
        raise ValueError(
            f"Empty scan interval: p2_scan_min={p2_scan_min} and "
            f"p2_scan_max={p2_scan_max} map to KL1 bounds "
            f"[{np.exp(log_KL1_lo)}, {np.exp(log_KL1_hi)}]."
        )
    log_KL1_grid = np.linspace(log_KL1_lo, log_KL1_hi, n_scan)

    p2_grid = np.full(n_scan, np.nan)
    KMC_errs = np.full(n_scan, np.nan)
    cstrs = np.ones(n_scan, dtype=bool)
    for ind, log_KL1 in enumerate(log_KL1_grid):
        try:
            p2_grid[ind] = get_p2_of_KL1_ss(np.exp(log_KL1), *ind_args)[0]
            KMC_errs[ind], cstrs[ind] = get_KMC_err_ss(
                np.exp(log_KL1), param_args
            )
        except ValueError:
            # KL1 outside its domain, or b2 or L1 has no feasible range here
            continue

    def KMC_err_of_log_KL1(log_KL1):
        return get_KMC_err_ss(np.exp(log_KL1), param_args)[0]

    roots = []
    for ind in range(n_scan - 1):
        if cstrs[ind] or cstrs[ind + 1]:
            continue
        if not (np.isfinite(KMC_errs[ind]) and np.isfinite(KMC_errs[ind + 1])):
            continue
        if KMC_errs[ind] * KMC_errs[ind + 1] > 0:
            continue
        sol_KL1 = opt.root_scalar(
            KMC_err_of_log_KL1,
            bracket=[log_KL1_grid[ind], log_KL1_grid[ind + 1]],
            method="brentq", xtol=1e-15, rtol=max(p2_tol, 8.9e-16)
        )
        if sol_KL1.converged:
            KL1_root = np.exp(sol_KL1.root)
            roots.append(
                (get_p2_of_KL1_ss(KL1_root, *ind_args)[0], KL1_root)
            )

    p2_scanned_min = np.nanmin(p2_grid)
    p2_scanned_max = np.nanmax(p2_grid)
    if len(roots) == 0:
        n_interior = int(np.sum(~cstrs))
        raise ValueError(
            "No steady-state p2 found. The scan covered p2 in "
            f"[{p2_scanned_min:.6g}, {p2_scanned_max:.6g}] with {n_interior} "
            f"of {n_scan} grid points free of binding constraints, and found "
            "no sign change in the capital market clearing error between "
            "adjacent unconstrained points. Try widening "
            "[p2_scan_min, p2_scan_max] or increasing n_scan."
        )
    if len(roots) > 1 and p2_guess is not None:
        # More than one interior equilibrium: take the one nearest the guess
        p2_roots = np.array([root[0] for root in roots])
        p2_ss, KL1_ss = roots[
            int(np.argmin(np.abs(np.log(p2_roots) - np.log(p2_guess))))
        ]
    else:
        p2_ss, KL1_ss = roots[0]

    if print_diagnostics:
        print(f"  Scanned {n_scan} points over p2 in "
              f"[{p2_scanned_min:.6g}, {p2_scanned_max:.6g}], "
              f"{int(np.sum(~cstrs))} unconstrained.")
        print(f"  Interior root(s) found (p2, KL1): {roots}")
        if len(roots) > 1:
            print(f"  Selected p2_ss = {p2_ss} (nearest to p2_guess).")

    return p2_ss, KL1_ss


def get_final_constraint_vec_error_tp(p2tm1_path, *args):
    (
        b2t_path, c_s1_path, c_s2_path, c_11_path, c_12_path, c_21_path,
        c_22_path, C1_path, C2_path, p1_path, p2tm1_path_long, p_path, w_path,
        r_path, L1_path, L2_path, K1_path, K2_path, Y1_path, Y2_path, I1_path,
        I2_path
    ) = gen_paths_given_p2tm1_tp(p2tm1_path, args)
    Y2MC_err_vec = Y2_path - C2_path - I2_path

    return Y2MC_err_vec


def series_plot(
    series_lst, legend_lst, start_per_lst, y_label, title=None, filename=None
):
    plt.figure()
    for ind, series in enumerate(series_lst):
        T = len(series)
        per_vec = np.arange(start_per_lst[ind], T + start_per_lst[ind])
        plt.plot(per_vec, series, label=legend_lst[ind])
    plt.grid(True)
    # Add minor tick marks every 1 period only on the x-axis
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
    plt.ylabel(y_label)
    plt.xlabel(r"Period $t$")
    plt.legend(loc="best")
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig(os.path.join(images_dir, filename))
    plt.close()


# Solve for steady state equilibrium
start_time_ss = time.time()
# Guess initial values for \bar{p}_2 and \bar{K}_1
p2_guess = 1.0

ss_args = (
    p1, n1, n2, c_min1, c_min2, beta, sigma, alpha_i_vec, gamma1, Z1, delta1,
    gamma2, Z2, delta2, epsilon_ss
)

print("")
print("Starting steady-state equilibrium computation.")
p2_ss_sol, KL1_ss_sol = solve_p2_ss(
    ss_args, p2_guess=p2_guess, p2_scan_min=p2_scan_min,
    p2_scan_max=p2_scan_max, n_scan=n_scan_ss
)
(
    b2_ss, c1_ss, c2_ss, c_11_ss, c_12_ss, c_21_ss, c_22_ss, C1_ss, C2_ss,
    p1_ss, p2_ss, p_ss, w_ss, r_ss, L1_ss, L2_ss, K1_ss, K2_ss, Y1_ss,
    Y2_ss, I1_ss, I2_ss
) = gen_ssvals_given_KL1(KL1_ss_sol, ss_args)
SS_dist = abs(K1_ss + p2_ss * K2_ss - b2_ss)
if SS_dist > SS_tol_stage1:
    raise ValueError(
        f"Steady-state solution p2_ss={p2_ss} has capital market clearing "
        f"error {SS_dist} > SS_tol_stage1={SS_tol_stage1}."
    )

compute_time_ss = time.time() - start_time_ss
print("Steady state compute time:", display_time(compute_time_ss))
print(f"Steady-state p2_ss = {p2_ss}, SS_dist = {SS_dist}")
resource_constraint_error_ss = Y2_ss - C2_ss - I2_ss
print(
    "Steady-state resource constraint " +
    f"error: {resource_constraint_error_ss}"
)
# b2_ss = b2
# if b2_ss == b2_min:
#     print("Steady-state b2:", b2_ss, "<-- b2_ss constrained to b2_min")
# elif b2_ss == b2_max:
#     print("Steady-state b2:", b2_ss, "<-- b2_ss constrained to b2_max")
# elif b2_ss > b2_min and b2_ss < b2_max:
#     print("Steady-state b2:", b2_ss, "<-- b2_ss unconstrained")
# else:
#     print("Steady-state b2:", b2_ss, "<-- b2_ss outside feasible range")
# c_s1_ss = c1
# print("Steady state c_s1:", c_s1_ss)
# c_s2_ss = c2
# print("Steady state c_s2:", c_s2_ss)
# c1_1_ss = c1_1
# print("Steady state c1_1:", c1_1_ss)
# c1_2_ss = c1_2
# print("Steady state c1_2:", c1_2_ss)
# c2_1_ss = c2_1
# print("Steady state c2_1:", c2_1_ss)
# c2_2_ss = c2_2
# print("Steady state c2_2:", c2_2_ss)
# C1_ss = C1
# print("Steady state C1:", C1_ss)
# C2_ss = C2
# print("Steady state C2:", C2_ss)
# p1_ss = p1
# print("Steady state p1:", p1_ss)
# p2_ss = p2_init
# print("Steady state p2:", p2_ss)
# p_ss = p
# print("Steady state p:", p_ss)
# w_ss = w
# print("Steady state w:", w_ss)
# r_ss = r
# print("Steady state r:", r_ss)
# L1_ss = L1
# if L1_ss == L1_min:
#     print("Steady-state L1:", L1_ss, "<-- L1_ss constrained to L1_min")
# elif L1_ss == L1_max:
#     print("Steady-state L1:", L1_ss, "<-- L1_ss constrained to L1_max")
# elif L1_ss > L1_min and L1_ss < L1_max:
#     print("Steady-state L1:", L1_ss, "<-- L1_ss unconstrained")
# else:
#     print("Steady-state L1:", L1_ss, "<-- L1_ss outside feasible range")
# L2_ss = L2
# if L2_ss == L1_min:
#     print("Steady-state L2:", L2_ss, "<-- L2_ss constrained to L2_min")
# elif L2_ss == L1_max:
#     print("Steady-state L2:", L2_ss, "<-- L2_ss constrained to L2_max")
# elif L2_ss > L1_min and L2_ss < L1_max:
#     print("Steady-state L2:", L2_ss, "<-- L2_ss unconstrained")
# else:
#     print("Steady-state L2:", L2_ss, "<-- L2_ss outside feasible range")
# K1_ss = K1
# print("Steady state K1:", K1_ss)
# K2_ss = K2
# print("Steady state K2:", K2_ss)
# I1_ss = I1
# print("Steady state I1:", I1_ss)
# I2_ss = I2
# print("Steady state I2:", I2_ss)
# Y1_ss = Y1
# print("Steady state Y1:", Y1_ss)
# Y2_ss = Y2
# print("Steady state Y2:", Y2_ss)

# ss_output = {
#     "b2_ss": b2_ss,
#     "c1_1_ss": c1_1_ss,
#     "c1_2_ss": c1_2_ss,
#     "c2_1_ss": c2_1_ss,
#     "c2_2_ss": c2_2_ss,
#     "c_s1_ss": c_s1_ss,
#     "c_s2_ss": c_s2_ss,
#     "C1_ss": C1_ss,
#     "C2_ss": C2_ss,
#     "p1_ss": p1_ss,
#     "p2_ss": p2_ss,
#     "p_ss": p_ss,
#     "w_ss": w_ss,
#     "r_ss": r_ss,
#     "L1_ss": L1_ss,
#     "L2_ss": L2_ss,
#     "K1_ss": K1_ss,
#     "K2_ss": K2_ss,
#     "Y1_ss": Y1_ss,
#     "Y2_ss": Y2_ss,
#     "I1_ss": I1_ss,
#     "I2_ss": I2_ss,
#     "resource_constraint_error_ss": resource_constraint_error_ss,
#     "compute_time_ss": compute_time_ss
# }
# # Save ss_output dictionary as a pickle file "ss_output.pkl" in the data_dir
# ss_output_file = os.path.join(data_dir, "ss_output.pkl")
# with open(ss_output_file, "wb") as f:
#     pickle.dump(ss_output, f)

# # Solve for the transition path equilibrium
# print("")
# print("Starting transition path equilibrium computation.")
# start_time_tpi = time.time()
# # Start with total household savings being 90% of the steady-state
# b2_1 = 0.96 * b2_ss
# print(r"Initial total household savings $b_{2,t=1}$ :", b2_1)

# # We know the equilibrium time path of the prices of the numeraire good
# p1_path = np.ones(T)

# # Guess a transition path for industry-2 prices
# p2tm1_path_init_guess = p2_ss * np.ones(T)
# p2tm1_path_init = p2tm1_path_init_guess.copy()

# tp_args = (
#     T, p1_path, n1, n2, c_min1, c_min2, beta, sigma, alpha_i_vec, gamma1, Z1,
#     delta1, gamma2, Z2, delta2, b2_1, epsilon_ss, w_ss, r_ss, p_ss, p2_ss,
#     K1_ss, K2_ss
# )

# # Run the time path algorithm until the maximum absolute error of the
# # Y2MC_err_path is less and or equal to TPI_tol_stage1
# TPI_dist = 100.0
# TPI_iter = 0
# print("")
# print("Starting stage 1, time path iteration.")
# while TPI_dist > TPI_tol_stage1 and TPI_iter < TPI_max_iter_stage1:
#     TPI_iter += 1
#     (
#         b2t_path, c_s1_path, c_s2_path, c_11_path, c_12_path, c_21_path,
#         c_22_path, C1_path, C2_path, p1_path, p2tm1_path_long, p_path, w_path,
#         r_path, L1_path, L2_path, K1_path, K2_path, Y1_path, Y2_path, I1_path,
#         I2_path
#     ) = gen_paths_given_p2tm1_tp(p2tm1_path_init, tp_args)
#     Y2MC_err_path = Y2_path - C2_path - I2_path
#     TPI_dist = np.max(np.absolute(Y2_path - C2_path - I2_path))
#     pct_diff = (
#         (Y2_path - C2_path - I2_path) / np.maximum(Y2_path, C2_path + I2_path)
#     )
#     print(f"Iteration {TPI_iter}, TPI_dist = {TPI_dist}")
#     p2tm1_path_new = (
#         (1 + xi_tpi * pct_diff) * p2tm1_path_init
#     )
#     if TPI_dist > TPI_tol_stage1:
#         p2tm1_path_init = p2tm1_path_new.copy()
# print("")
# print("Starting stage 2, root finder over the time path.")
# sol_p2tm1_path = opt.root(
#     get_final_constraint_vec_error_tp, p2tm1_path_init, tol=TPI_tol_root,
#     args=tp_args
# )
# p2tm1_path = sol_p2tm1_path.x
# TPI_dist = np.max(np.absolute(sol_p2tm1_path.fun))
# print("Transition path root finding success:", sol_p2tm1_path.success)
# print(
#     "Transition path root finding number of function evaluations:",
#     sol_p2tm1_path.nfev
# )
# print("Transition path root finding TPI_dist:", TPI_dist)

# (
#     b2t_path, c_s1_path, c_s2_path, c_11_path, c_12_path, c_21_path, c_22_path,
#     C1_path, C2_path, p1_path, p2tm1_path_long, p_path, w_path, r_path,
#     L1_path, L2_path, K1_path, K2_path, Y1_path, Y2_path, I1_path, I2_path
# ) = gen_paths_given_p2tm1_tp(p2tm1_path, tp_args)

# resource_constraint_error_path_tpi = Y1_path - C1_path - I1_path
# print(
#     "Maximum absolute resource constraint error:",
#     np.max(np.absolute(resource_constraint_error_path_tpi))
# )
# compute_time_tpi = time.time() - start_time_tpi
# print(
#     "Transition path equilibrium compute time:",
#     display_time(compute_time_tpi)
# )

# tpi_output = {
#     "b2t_path": b2t_path,
#     "c_s1_path": c_s1_path,
#     "c_s2_path": c_s2_path,
#     "c_11_path": c_11_path,
#     "c_12_path": c_12_path,
#     "c_21_path": c_21_path,
#     "c_22_path": c_22_path,
#     "C1_path": C1_path,
#     "C2_path": C2_path,
#     "p1_path": p1_path,
#     "p2tm1_path_long": p2tm1_path_long,
#     "p_path": p_path,
#     "w_path": w_path,
#     "r_path": r_path,
#     "L1_path": L1_path,
#     "L2_path": L2_path,
#     "K1_path": K1_path,
#     "K2_path": K2_path,
#     "Y1_path": Y1_path,
#     "Y2_path": Y2_path,
#     "I1_path": I1_path,
#     "I2_path": I2_path,
#     "resource_constraint_error_path_tpi": resource_constraint_error_path_tpi,
#     "compute_time_tpi": compute_time_tpi
# }

# # Save tpi_output dictionary as a pickle file "tpi_output.pkl" in the data_dir
# tpi_output_file = os.path.join(data_dir, "tpi_output.pkl")
# with open(tpi_output_file, "wb") as f:
#     pickle.dump(tpi_output, f)

# # Plot the following series
# series_to_plot = [
#     (
#         [b2t_path], [r"$b_{2,t}$"], [1], r"$b_{2,t}$", "Household savings",
#         "b2t_path.png"
#     ),
#     (
#         [c_s1_path, c_s2_path], [r"$c_{s=1,t}$", r"$c_{s=2,t}$"], [1, 1],
#         r"$c_{s,t}$", r"Household age $s$ composite consumption",
#         "c_st_path.png"
#     ),
#     (
#         [c_11_path, c_12_path, c_21_path, c_22_path],
#         [r"$c_{1,1,t}$", r"$c_{1,2,t}$", r"$c_{2,1,t}$", r"$c_{2,2,t}$" ],
#         [1, 1, 1, 1], r"c_{i,s,t}$",
#         "Household differentiated good consumption", "c_ist_path.png"
#     ),
#     (
#         [C1_path, C2_path], [r"$C_{1,t}$", r"$C_{2,t}$"], [1, 1], r"$C_{i,t}$",
#         r"Aggregate consumption of differentiated good $i$", "Ci_path.png"
#     ),
#     (
#         [p1_path, p2tm1_path_long, p_path],
#         [r"$p_{1,t}$", r"$p_{2,t}$", r"$p_t$"], [1, 0, 1],
#         r"Composite and differentiated good prices",
#         "Prices of composite and differentiated goods",
#         "p_pi_path.png"
#     ),
#     ([w_path], [r"$w_t$"], [1], r"$w_t$", "Wage", "w_path.png"),
#     ([r_path], [r"$r_t$"], [1], r"$r_t$", "Interest rate", "r_path.png"),
#     (
#         [K1_path, K2_path], [r"$K_{1,t}$", r"$K_{2,t}$"], [1, 1], r"$K_{i,t}$",
#         r"Aggregate capital in industries", "Ki_path.png"
#     ),
#     (
#         [L1_path, L2_path], [r"$L_{1,t}$", r"$L_{2,t}$"], [1, 1], r"$L_{i,t}$",
#         r"Aggregate labor in industries", "Li_path.png"
#     ),
#     (
#         [Y1_path, Y2_path], [r"$Y_{1,t}$", r"$Y_{2,t}$"], [1, 1], r"$Y_{i,t}$",
#         r"Aggregate output in industries", "Yi_path.png"
#     ),
#     (
#         [I1_path, I2_path], [r"$I_{1,t}$", r"$I_{2,t}$"], [1, 1], r"$I_{i,t}$",
#         r"Aggregate investment in industries", "Ii_path.png"
#     ),
#     (
#         [resource_constraint_error_path_tpi],
#         [r"$Y_{1,t} - C_{1,t} - I_{1,t}$"], [1],
#         r"$Y_{1,t} - C_{1,t} - I_{1,t}$", "Resource constraint error",
#         "rc_error_path_tpi.png"
#     )
# ]
# for inputs_tup in series_to_plot:
#     (
#         series_lst, legend_lst, y_label, title, filename, start_per_lst
#     ) = inputs_tup
#     series_plot(
#         series_lst, legend_lst, y_label, title, filename, start_per_lst
#     )
