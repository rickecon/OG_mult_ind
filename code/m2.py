"""
This is the code for solving the $M=2$ two-industry version of the model
"""
# Import packages
import os
import pickle
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Designate data and image directories
cur_dir = os.path.dirname(os.path.abspath(__file__))
# designate the data directory as "data/m1" and create those two directories
# if they do not exist
data_dir = os.path.join(cur_dir, "..", "data", "m2")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
images_dir = os.path.join(cur_dir, "..", "images", "m2")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)


# Set parameters
years_per_period = 30
T = 20  # Guess for number of periods to steady-state
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
TPI_tol = 1e-9
TPI_max_iter = 1000
SS_tol = 1e-10
SS_max_iter = 1000
xi_ss = 0.01  # Damping parameter for SS iteration
xi_tpi = 0.5  # Damping parameter for TPI iteration
epsilon_ss = 1e-5  # Error term for steady-state boundaries

# Define functions


def display_time(seconds):
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.5f}s"


def get_p(pi_vec, alpha_i_vec):
    """
    Solve for composite consumption good price from differentiated goods prices
    and parameters

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


def ss_KL2_zerofunc(KLratio, *args):
    p2, gamma1, Z1, delta1, gamma2, Z2, delta2 = args
    MPK1 = (
        gamma1 * Z1 *
        (
            (
                ((1 - gamma1) * Z1) /
                ((1 - gamma2) * p2 * Z2 * (KLratio ** gamma2))
            ) ** ((1 - gamma1) / gamma1)
        ) - delta1
    )
    MPK2 = gamma2 * Z2 * (KLratio ** (gamma2 - 1)) - delta2
    zerofunc = MPK1 - MPK2

    return zerofunc


def get_w_KL(KLratio_i, p_i, Z_i, gamma_i):
    w = (1 - gamma_i) * p_i * Z_i * (KLratio_i ** gamma_i)

    return w


def get_r_KL(KLratio_i, Z_i, gamma_i, delta_i):
    r = gamma_i * Z_i * (KLratio_i ** (gamma_i - 1)) - delta_i

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


def get_c_s1(b2, w, p1, p2, p, n1, c_min1, c_min2):
    """
    Calculate the composite consumption of an young agent (s=1) from the budget
    constraint
    """
    c_s1 = (w / p) * n1 - (p1 / p) * c_min1 - (p2 / p) * c_min2 - (b2 / p)

    return c_s1


def get_c_s2(b2, w, r, p1, p2, p, n2, c_min1, c_min2):
    """
    Calculate the composite consumption of an old agent (s=2) from the budget
    constraint
    """
    c_s2 = (
        ((1 + r) / p) * b2 + (w / p) * n2 - (p1 / p) * c_min1 -
        (p2 / p) * c_min2
    )

    return c_s2


def get_c_is(alpha_i, p_i, p, c_s, c_min_i):
    c_is = alpha_i * (p / p_i) * c_s + c_min_i

    return c_is


def get_p2_new_ss(L1, K2, w, n1, n2, Z2, gamma2):
    p2_new = (
        (w * ((n1 + n2 - L1) ** gamma2)) / ((1-gamma2) * Z2 * (K2 ** gamma2))
    )

    return p2_new


# def get_K1_new_ss(Y1, L1, Z1, gamma1):
#     K1_new = (Y1 / (Z1 * (L1 ** (1 - gamma1)))) ** (1 / gamma1)

#     return K1_new


# def get_Y(K, L, Z, gamma):
#     Y = Z * (K ** gamma) * (L ** (1 - gamma))

#     return Y


# def get_L(n1, n2):
#     L = n1 + n2

#     return L


# def get_w(K, L, p, Z, gamma):
#     w = (1 - gamma) * p * Z * (K / L) ** gamma

#     return w


# def get_r(K, L, p, Z, gamma, delta):
#     r = gamma * p * Z * (L / K) ** (1 - gamma) - delta

#     return r


# def get_b2_ss(w, r, n1, n2, c_min1, beta, sigma):
#     numerator = (
#         w * n1 - c_min1 -
#         ((beta * (1 + r)) ** (-1 / sigma)) * (w * n2 - c_min1)
#     )
#     denominator = 1 + ((beta * (1 + r)) ** (-1 / sigma)) * (1 + r)
#     b2_ss = numerator / denominator

#     return b2_ss


# def b2_ss_zerofunc(b2, *args):
#     p, n1, n2, c_min1, Z, beta, sigma, gamma, delta = args
#     L = get_L(n1, n2)
#     K = b2
#     w = get_w(K, L, p, Z, gamma)
#     r = get_r(K, L, p, Z, gamma, delta)
#     zerofunc = b2 - get_b2_ss(w, r, n1, n2, c_min1, beta, sigma)

#     return zerofunc


# def get_b2_tp(
#     b2_0, w_path, r_path, p1_path, p_path, n1, n2, c_min1, beta, sigma
# ):
#     T = len(w_path)
#     b2_path = np.zeros(T)
#     b2_path[0] = b2_0
#     w_path_t = w_path[:-1]
#     w_path_tp1 = w_path[1:]
#     r_path_tp1 = r_path[1:]
#     p1_path_t = p1_path[:-1]
#     p1_path_tp1 = p1_path[1:]
#     p_path_t = p_path[:-1]
#     p_path_tp1 = p_path[1:]
#     num_term1 = (w_path_t / p_path_t) * n1 - (p1_path_t / p_path_t) * c_min1
#     num_term2 = (
#         (
#             (
#                 (beta * p_path_t * (1 + r_path_tp1)) / p_path_tp1
#             ) ** (-1 / sigma)
#         ) *
#         ((w_path_tp1 / p_path_tp1) * n2 - (p1_path_tp1 / p_path_tp1) * c_min1)
#     )
#     denom = (
#         (1 / p_path_t) + (
#             (
#                 ((beta * p_path_t * (1 + r_path_tp1)) / p_path_tp1) **
#                 (-1 / sigma)
#             ) * ((1 + r_path_tp1) / p_path_tp1)
#         )
#     )
#     b2_path[1:] = (num_term1 - num_term2) / denom

#     return b2_path


# def series_plot(series, y_label, title=None, filename=None):
#     plt.figure()
#     T = len(series)
#     per_vec = np.arange(1, T + 1)
#     plt.plot(per_vec, series)
#     plt.grid(True)
#     # Add minor tick marks every 1 period only on the x-axis
#     plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
#     plt.ylabel(y_label)
#     plt.xlabel(r"Period $t$")
#     if title is not None:
#         plt.title(title)
#     if filename is not None:
#         plt.savefig(os.path.join(images_dir, filename))
#     plt.close()


# Solve for steady state equilibrium
start_time_ss = time.time()
# Guess initial values for \bar{p}_2 and \bar{K}_1
p2_guess = 0.8
p2_init = p2_guess

SS_dist = 100.0
SS_iter = 0
print("")
print("Starting steady-state equilibrium computation.")
while SS_dist > SS_tol and SS_iter < SS_max_iter:
    SS_iter += 1
    # print("p2_init:", p2_init)
    pi_vec = np.array([1.0, p2_init])
    p = get_p(pi_vec, alpha_i_vec)
    # print("p:", p)
    sol_KL1_ss = opt.root_scalar(
        ss_KL1_zerofunc, bracket=[1e-9, 20.0], method="brentq", args=(
            p2_init, gamma1, Z1, delta1, gamma2, Z2, delta2
        )
    )
    if not sol_KL1_ss.converged:
        raise ValueError("KL1 root finding did not converge.")
        print("Converged:", sol_KL1_ss.converged)
        print("Flag:", sol_KL1_ss.flag)
        print("Function calls:", sol_KL1_ss.function_calls)
        print("Iterations:", sol_KL1_ss.iterations)
        print("Method:", sol_KL1_ss.method)
    KL1ratio = sol_KL1_ss.root
    # print("KL1ratio:", KL1ratio)
    sol_KL2_ss = opt.root_scalar(
        ss_KL2_zerofunc, bracket=[1e-9, 20.0], method="brentq", args=(
            p2_init, gamma1, Z1, delta1, gamma2, Z2, delta2
        )
    )
    if not sol_KL2_ss.converged:
        raise ValueError("KL2 root finding did not converge.")
        print("Converged:", sol_KL2_ss.converged)
        print("Flag:", sol_KL2_ss.flag)
        print("Function calls:", sol_KL2_ss.function_calls)
        print("Iterations:", sol_KL2_ss.iterations)
        print("Method:", sol_KL2_ss.method)
    KL2ratio = sol_KL2_ss.root
    # print("KL2ratio:", KL2ratio)
    w = get_w_KL(KL1ratio, p1, Z1, gamma1)
    # print("w:", w)
    r = get_r_KL(KL1ratio, Z1, gamma1, delta1)
    # print("r:", r)
    YL1ratio = get_YLi_ratio(KL1ratio, Z1, gamma1)
    # print("YL1ratio:", YL1ratio)
    YL2ratio = get_YLi_ratio(KL2ratio, Z2, gamma2)
    # print("YL2ratio:", YL2ratio)
    IL1ratio = delta1 * KL1ratio
    # print("IL1ratio:", IL1ratio)
    IL2ratio = delta2 * KL2ratio
    # print("IL2ratio:", IL2ratio)
    b2_min = (p1 * c_min1 + p2_init * c_min2 - w * n2) / (1 + r) + epsilon_ss
    b2_max = (
        (w / p) * n1 - (p1 / p) * c_min1 - (p2_init / p) * c_min2 - epsilon_ss
    )
    if b2_max <= b2_min:
        raise ValueError("No feasible b2 exists: b2_max <= b2_min.")
    b2_uncstr = get_b2_ss(
        w, r, p, p1, p2_init, n1, n2, c_min1, c_min2, beta, sigma
    )
    if b2_uncstr < b2_min:
        b2 = b2_min
        # print("b2:", b2, "<-- b2 constrained to b2_min")
    elif b2_uncstr > b2_max:
        b2 = b2_max
        # print("b2:", b2, "<-- b2 constrained to b2_max")
    elif b2_uncstr >= b2_min and b2_uncstr <= b2_max:
        b2 = b2_uncstr
        # print("b2:", b2, "<-- b2 unconstrained")
    c1 = get_c_s1(b2, w, p1, p2_init, p, n1, c_min1, c_min2)
    # print("c1:", c1)
    c2 = get_c_s2(b2, w, r, p1, p2_init, p, n2, c_min1, c_min2)
    # print("c2:", c2)
    c1_1 = get_c_is(alpha1, p1, p, c1, c_min1)
    # print("c1_1:", c1_1)
    c1_2 = get_c_is(alpha1, p1, p, c2, c_min1)
    # print("c1_2:", c1_2)
    c2_1 = get_c_is(alpha2, p2_init, p, c1, c_min2)
    # print("c2_1:", c2_1)
    c2_2 = get_c_is(alpha2, p2_init, p, c2, c_min2)
    # print("c2_2:", c2_2)
    C1 = c1_1 + c1_2
    # print("C1:", C1)
    C2 = c2_1 + c2_2
    # print("C2:", C2)
    L1_min = epsilon_ss
    L1_max = n1 + n2 - epsilon_ss
    L1_uncstr = C1 / (YL1ratio - IL1ratio)
    if L1_max <= L1_min:
        raise ValueError("No feasible L1 exists: L1_max <= L1_min.")
    if L1_uncstr < L1_min:
        L1 = L1_min
        # print("L1:", L1, "<-- L1 constrained to L1_min")
    elif L1_uncstr > L1_max:
        L1 = L1_max
        # print("L1:", L1, "<-- L1 constrained to L1_max")
    elif L1_uncstr >= L1_min and L1_uncstr <= L1_max:
        L1 = L1_uncstr
        # print("L1:", L1, "<-- L1 unconstrained")
    K1 = KL1ratio * L1
    # print("K1:", K1)
    I1 = IL1ratio * L1
    # print("I1:", I1)
    Y1 = YL1ratio * L1
    # print("Y1:", Y1)
    L2 = n1 + n2 - L1
    # print("L2:", L2)
    K2 = KL2ratio * L2
    # print("K2:", K2)
    I2 = IL2ratio * L2
    # print("I2:", I2)
    Y2 = YL2ratio * L2
    # print("Y2:", Y2)
    # print("Goods market clearing Y2 - C2 - I2:", Y2 - C2 - I2)
    SS_dist = abs(K1 + p2_init * K2 - b2)
    # SS_dist = abs(Y2 - C2 - I2)
    # print(
    #     f"Iteration {SS_iter}: p2 = {p2_init}, SS_dist = {SS_dist}"
    # )
    dif_pct = (K1 + p2_init * K2 - b2) / max(K1 + p2_init * K2, b2)
    # dif_pct = (Y2 - C2 - I2) / max(Y2, C2 + I2)
    if SS_dist > SS_tol:
        p2_init = p2_init * (1 + xi_ss * dif_pct)

compute_time_ss = time.time() - start_time_ss
print("Steady state compute time:", display_time(compute_time_ss))
print(f"Steady-state iterations = {SS_iter}, SS_dist = {SS_dist}")
resource_constraint_error_ss = Y2 - C2 - I2
# resource_constraint_error_ss = K1 + p2_init * K2 - b2
print(
    "Steady-state resource constraint " +
    f"error: {resource_constraint_error_ss}"
)
b2_ss = b2
if b2_ss == b2_min:
    print("Steady-state b2:", b2_ss, "<-- b2_ss constrained to b2_min")
elif b2_ss == b2_max:
    print("Steady-state b2:", b2_ss, "<-- b2_ss constrained to b2_max")
elif b2_ss > b2_min and b2_ss < b2_max:
    print("Steady-state b2:", b2_ss, "<-- b2_ss unconstrained")
else:
    print("Steady-state b2:", b2_ss, "<-- b2_ss outside feasible range")
c1_1_ss = c1_1
print("Steady state c1_1:", c1_1_ss)
c1_2_ss = c1_2
print("Steady state c1_2:", c1_2_ss)
c2_1_ss = c2_1
print("Steady state c2_1:", c2_1_ss)
c2_2_ss = c2_2
print("Steady state c2_2:", c2_2_ss)
c_s1_ss = c1
print("Steady state c_s1:", c_s1_ss)
c_s2_ss = c2
print("Steady state c_s2:", c_s2_ss)
C1_ss = C1
print("Steady state C1:", C1_ss)
C2_ss = C2
print("Steady state C2:", C2_ss)
p1_ss = p1
print("Steady state p1:", p1_ss)
p2_ss = p2_init
print("Steady state p2:", p2_ss)
p_ss = p
print("Steady state p:", p_ss)
w_ss = w
print("Steady state w:", w_ss)
r_ss = r
print("Steady state r:", r_ss)
L1_ss = L1
if L1_ss == L1_min:
    print("Steady-state L1:", L1_ss, "<-- L1_ss constrained to L1_min")
elif L1_ss == L1_max:
    print("Steady-state L1:", L1_ss, "<-- L1_ss constrained to L1_max")
elif L1_ss > L1_min and L1_ss < L1_max:
    print("Steady-state L1:", L1_ss, "<-- L1_ss unconstrained")
else:
    print("Steady-state L1:", L1_ss, "<-- L1_ss outside feasible range")
L2_ss = L2
if L2_ss == L1_min:
    print("Steady-state L2:", L2_ss, "<-- L2_ss constrained to L2_min")
elif L2_ss == L1_max:
    print("Steady-state L2:", L2_ss, "<-- L2_ss constrained to L2_max")
elif L2_ss > L1_min and L2_ss < L1_max:
    print("Steady-state L2:", L2_ss, "<-- L2_ss unconstrained")
else:
    print("Steady-state L2:", L2_ss, "<-- L2_ss outside feasible range")
K1_ss = K1
print("Steady state K1:", K1_ss)
K2_ss = K2
print("Steady state K2:", K2_ss)
I1_ss = I1
print("Steady state I1:", I1_ss)
I2_ss = I2
print("Steady state I2:", I2_ss)
Y1_ss = Y1
print("Steady state Y1:", Y1_ss)
Y2_ss = Y2
print("Steady state Y2:", Y2_ss)

ss_output = {
    "b2_ss": b2_ss,
    "c1_1_ss": c1_1_ss,
    "c1_2_ss": c1_2_ss,
    "c2_1_ss": c2_1_ss,
    "c2_2_ss": c2_2_ss,
    "c_s1_ss": c_s1_ss,
    "c_s2_ss": c_s2_ss,
    "C1_ss": C1_ss,
    "C2_ss": C2_ss,
    "p1_ss": p1_ss,
    "p2_ss": p2_ss,
    "p_ss": p_ss,
    "w_ss": w_ss,
    "r_ss": r_ss,
    "L1_ss": L1_ss,
    "L2_ss": L2_ss,
    "K1_ss": K1_ss,
    "K2_ss": K2_ss,
    "Y1_ss": Y1_ss,
    "Y2_ss": Y2_ss,
    "I1_ss": I1_ss,
    "I2_ss": I2_ss,
    "resource_constraint_error_ss": resource_constraint_error_ss,
    "compute_time_ss": compute_time_ss
}
# Save ss_output dictionary as a pickle file "ss_output.pkl" in the data_dir
ss_output_file = os.path.join(data_dir, "ss_output.pkl")
with open(ss_output_file, "wb") as f:
    pickle.dump(ss_output, f)




# sol_ss = opt.root_scalar(
#     b2_ss_zerofunc, x0=0.01, args=(
#         p1, n1, n2, c_min1, Z1, beta, sigma, gamma1, delta1
#     )
# )
# compute_time_ss = time.time() - start_time_ss
# print("Steady state compute time:", display_time(compute_time_ss))
# print("sol_ss output:")
# print(sol_ss)
# b2_ss = sol_ss.root
# print("Steady state b2:", b2_ss)
# L1_ss = get_L(n1, n2)
# print("Steady state L:", L1_ss)
# K1_ss = b2_ss
# print("Steady state K:", K1_ss)
# Y1_ss = get_Y(K1_ss, L1_ss, Z1, gamma1)
# w_ss = get_w(K1_ss, L1_ss, p1, Z1, gamma1)
# print("Steady state w:", w_ss)
# r_ss = get_r(K1_ss, L1_ss, p1, Z1, gamma1, delta1)
# print("Steady state r:", r_ss)
# c_s1_ss = get_c_s1(b2_ss, w_ss, p1, p1, n1, c_min1)
# print("Steady state c1_1:", c_s1_ss)
# c_s2_ss = get_c_s2(b2_ss, w_ss, r_ss, p1, p1, n2, c_min1)
# print("Steady state c1_2:", c_s2_ss)
# c1_1_ss = get_c_is(alpha1, p1, p1, c_s1_ss, c_min1)
# print("Steady state c1_1:", c1_1_ss)
# c1_2_ss = get_c_is(alpha1, p1, p1, c_s2_ss, c_min1)
# print("Steady state c1_2:", c1_2_ss)
# C1_ss = c1_1_ss + c1_2_ss
# print("Steady state C1:", C1_ss)
# I1_ss = delta1 * K1_ss
# print("Steady state I1:", I1_ss)
# resource_constraint_error_ss = Y1_ss - C1_ss - I1_ss
# print("Steady state resource constraint error:", resource_constraint_error_ss)

# ss_output = {
#     "sol_ss": sol_ss,
#     "b2_ss": b2_ss,
#     "L1_ss": L1_ss,
#     "K1_ss": K1_ss,
#     "Y1_ss": Y1_ss,
#     "w_ss": w_ss,
#     "r_ss": r_ss,
#     "c_s1_ss": c_s1_ss,
#     "c_s2_ss": c_s2_ss,
#     "c1_1_ss": c1_1_ss,
#     "c1_2_ss": c1_2_ss,
#     "C1_ss": C1_ss,
#     "I1_ss": I1_ss,
#     "resource_constraint_error_ss": resource_constraint_error_ss,
#     "compute_time_ss": compute_time_ss
# }
# # Save ss_output dictionary as a pickle file "ss_output.pkl" in the data_dir
# ss_output_file = os.path.join(data_dir, "ss_output.pkl")
# with open(ss_output_file, "wb") as f:
#     pickle.dump(ss_output, f)

# # Solve for the transition path equilibrium
# start_time_tpi = time.time()
# K1_0 = 0.9 * K1_ss # Start with capital being 90% of the steady-state
# b2_0 = 0.9 * b2_ss

# # We know the equilibrium time path of aggregate labor and prices
# L1_path = np.ones(T) * L1_ss
# p1_path = np.ones(T)
# p_path = np.ones(T)

# # Guess a transition path for the aggregate capital stock
# K1_path_init_guess = np.linspace(K1_0, K1_ss, T)
# K1_path_init = K1_path_init_guess.copy()

# TPI_dist = 100.0
# TPI_iter = 0
# print("")
# print("Starting transition path equilibrium computation.")
# while TPI_dist > TPI_tol and TPI_iter < TPI_max_iter:
#     TPI_iter += 1
#     w_path = get_w(K1_path_init, L1_path, p1_path, Z1, gamma1)
#     r_path = get_r(K1_path_init, L1_path, p1_path, Z1, gamma1, delta1)
#     b2_path = get_b2_tp(
#         b2_0, w_path, r_path, p1_path, p_path, n1, n2, c_min1, beta, sigma
#     )
#     K1_path_new = b2_path.copy()
#     TPI_dist = np.max(np.absolute(K1_path_new - K1_path_init))
#     K1_path_init = xi_tpi * K1_path_new + (1 - xi_tpi) * K1_path_init
#     print(f"Iteration {TPI_iter}: TPI_dist = {TPI_dist}")

# K1_path = K1_path_init.copy()
# b2_path = K1_path.copy()
# w_path = get_w(K1_path, L1_path, p1_path, Z1, gamma1)
# r_path = get_r(K1_path, L1_path, p1_path, Z1, gamma1, delta1)
# I1_path = np.append(K1_path[1:], K1_ss) - (1 - delta1) * K1_path
# Y1_path = get_Y(K1_path, L1_path, Z1, gamma1)
# b2_path_tp1 = np.append(b2_path[1:], b2_ss)
# c_s1_path = get_c_s1(b2_path_tp1, w_path, p1_path, p_path, n1, c_min1)
# c_s2_path = get_c_s2(b2_path, w_path, r_path, p1_path, p_path, n2, c_min1)
# c1_1_path = get_c_is(alpha1, p1_path, p_path, c_s1_path, c_min1)
# c1_2_path = get_c_is(alpha1, p1_path, p_path, c_s2_path, c_min1)
# C1_path = c1_1_path + c1_2_path

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
#     "b2_path": b2_path,
#     "L1_path": L1_path,
#     "K1_path": K1_path,
#     "Y1_path": Y1_path,
#     "w_path": w_path,
#     "r_path": r_path,
#     "c_s1_path": c_s1_path,
#     "c_s2_path": c_s2_path,
#     "c1_1_path": c1_1_path,
#     "c1_2_path": c1_2_path,
#     "C1_path": C1_path,
#     "I1_path": I1_path,
#     "resource_constraint_error_path_tpi": resource_constraint_error_path_tpi,
#     "compute_time_tpi": compute_time_tpi
# }

# # Save tpi_output dictionary as a pickle file "tpi_output.pkl" in the data_dir
# tpi_output_file = os.path.join(data_dir, "tpi_output.pkl")
# with open(tpi_output_file, "wb") as f:
#     pickle.dump(tpi_output, f)

# # Plot the following series
# series_to_plot = [
#     (b2_path, r"$b_{2,t+1}$", "Household savings", "b2_path.png"),
#     (K1_path, r"$K_{1,t}$", "Aggregate capital", "K1_path.png"),
#     (Y1_path, r"$Y_{1,t}$", "Output", "Y1_path.png"),
#     (w_path, r"$w_t$", "Wage rate", "w_path.png"),
#     (r_path, r"$r_t$", "Interest rate", "r_path.png"),
#     (
#         c_s1_path, r"$c_{s=1,t}$", "Household age 1 composite consumption",
#         "c_s1_path.png"
#     ),
#     (
#         c_s2_path, r"$c_{s=2,t}$", "Household age 2 composite consumption",
#         "c_s2_path.png"
#     ),
#     (
#         c1_1_path, r"$c_{1,1,t}$", "Household age 1 good 1 consumption",
#         "c1_1_path.png"
#     ),
#     (
#         c1_2_path, r"$c_{1,2,t}$", "Household age 2 good 1 consumption",
#         "c1_2_path.png"
#     ),
#     (C1_path, r"$C_{1,t}$", "Aggregate consumption", "C1_path.png"),
#     (I1_path, r"$I_{1,t}$", "Investment", "I1_path.png"),
#     (
#         resource_constraint_error_path_tpi, r"$Y_{1,t} - C_{1,t} - I_{1,t}$",
#         "Resource constraint error", "resource_constraint_error_path_tpi.png"
#     )
# ]
# for inputs_tup in series_to_plot:
#     series, y_label, title, filename = inputs_tup
#     series_plot(series, y_label, title, filename)
