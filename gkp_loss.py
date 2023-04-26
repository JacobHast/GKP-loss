import numpy as np
from scipy.linalg import det, inv
import matplotlib.pyplot as plt
import qutip

qutip.settings.auto_tidyup = False


def Gaussian(mu, sigma, x):
    # Eq. (20)
    if np.shape(sigma) == ():
        G = 1 / np.sqrt(sigma * 2 * np.pi) * np.exp(-0.5 * (x - mu) ** 2 / sigma)
    else:
        n = np.shape(sigma)[0]
        sigma_inv = inv(sigma)
        prefactor = 1 / np.sqrt(det(sigma) * (2 * np.pi) ** n)
        if len(np.shape(x)) == 1:
            G = prefactor * np.exp(-0.5 * (x - mu).T @ sigma_inv @ (x - mu))
        else:
            x_minus_mu = np.array([x[i] - mu[i] for i in range(2)])
            G = prefactor * np.exp(
                -0.5 * np.einsum("kij,km,mij->ij", x_minus_mu, sigma_inv, x_minus_mu)
            )
    return G


def Sigma_eps(eps):
    # Eq. (21)
    return 0.5 * np.tanh(eps) * np.identity(2)


def mu_eps(eps, m):
    # Eq. (21)
    return 1 / np.cosh(eps) * np.sqrt(np.pi) / 2 * m


def Sigma_p(eps, eta, G):
    # Eq. (36)
    return 1 / 4 * (np.tanh(eps) * (1 + eta * G) + eta * (G - 1) + 1 - eta)


def mu_p(eps, eta, G, m):
    # Eq. (37)
    return (
        1
        / np.cosh(eps)
        * np.sqrt(np.pi)
        / (2 * np.sqrt(2))
        * (np.sqrt(eta * G) * m[0] + m[1])
    )


def M_set(L, m_max):
    # Eq. (22)
    m1_list = np.arange(-m_max, m_max)
    m2_list = m1_list
    if L == 0:
        m1_list = m1_list[m1_list % 2 == 0]
        m2_list = m2_list[m2_list % 2 == 0]
    if L == 1:
        m1_list = m1_list[m1_list % 2 == 1]
        m2_list = m2_list[m2_list % 2 == 0]
    if L == 2:
        m1_list = m1_list[m1_list % 2 == 1]
        m2_list = m2_list[m2_list % 2 == 1]
    if L == 3:
        m1_list = m1_list[m1_list % 2 == 0]
        m2_list = m2_list[m2_list % 2 == 1]

    pairs = []
    for m1 in m1_list:
        for m2 in m2_list:
            pairs.append(np.array([m1, m2], "complex"))
    return pairs


def sign_fun(k, m):
    # Eq. (23)
    if k == 0:
        return 1
    if k == 1:
        return (-1) ** (m[1] / 2)
    if k == 2:
        return (-1) ** ((m[0] + m[1]) / 2)
    if k == 3:
        return (-1) ** (m[0] / 2)
    raise ValueError(f"{k} must be 0, 1, 2 or 3")


def coef(eps, m):
    # Eq. (24)
    return np.exp(-np.tanh(eps) * np.pi / 4 * np.linalg.norm(m) ** 2)


def k_to_l(k1, k2):
    # Eq. (38)
    K = [k1, k2]
    if K == [0, 0]:
        return 0, 0
    if K == [0, 1]:
        return 3, 0
    if K == [0, 2]:
        return 3, 3
    if K == [0, 3]:
        return 0, 3
    if K == [1, 0]:
        return 1, 0
    if K == [1, 1]:
        return 2, 0
    if K == [1, 2]:
        return 2, 3
    if K == [1, 3]:
        return 1, 3
    if K == [2, 0]:
        return 1, 1
    if K == [2, 1]:
        return 2, 1
    if K == [2, 2]:
        return 2, 2
    if K == [2, 3]:
        return 1, 2
    if K == [3, 0]:
        return 0, 1
    if K == [3, 1]:
        return 3, 1
    if K == [3, 2]:
        return 3, 2
    if K == [3, 3]:
        return 0, 2


def k_to_lp(k1, k2):
    # Eq. (40)
    K = [k1, k2]
    if K == [0, 0]:
        return 0, 0
    if K == [0, 1]:
        return 0, 1
    if K == [0, 2]:
        return 1, 1
    if K == [0, 3]:
        return 1, 0
    if K == [1, 0]:
        return 0, 3
    if K == [1, 1]:
        return 0, 2
    if K == [1, 2]:
        return 1, 2
    if K == [1, 3]:
        return 1, 3
    if K == [2, 0]:
        return 3, 3
    if K == [2, 1]:
        return 3, 2
    if K == [2, 2]:
        return 2, 2
    if K == [2, 3]:
        return 2, 3
    if K == [3, 0]:
        return 3, 0
    if K == [3, 1]:
        return 3, 1
    if K == [3, 2]:
        return 2, 1
    if K == [3, 3]:
        return 2, 0


def g_coef(eps, l, lp, x, eta, G, m_max):
    # Eq. (43)
    n_set = M_set(l, m_max)
    g = 0
    for n in n_set:
        g += (
            coef(eps, n)
            * sign_fun(lp, n)
            * Gaussian(mu_p(eps, eta, G, n), Sigma_p(eps, eta, G), x)
        )
    return g


def g_product(k1, k2, qm, pm, eps, eta, G, m_max):
    # product of the g's in Eq. (44)
    l1, l2 = k_to_l(k1, k2)
    l1p, l2p = k_to_lp(k1, k2)
    g1 = g_coef(eps, l1, l1p, qm, eta, G, m_max)
    g2 = g_coef(eps, l2, l2p, pm, eta, G, m_max)
    return np.outer(g1, g2)


def g_coef_precomputed(eps, l, lp, m_max, gaussians):
    # Eq. (43)
    n_set = M_set(l, m_max)
    g = 0
    for n in n_set:
        g += (
            coef(eps, n)
            * sign_fun(lp, n)
            * gaussians[int(n[0] + m_max)][int(n[1] + m_max)]
        )
    return g


def g_product_precomputed(k1, k2, eps, m_max, gaussians):
    # product of the g's in Eq. (44)
    l1, l2 = k_to_l(k1, k2)
    l1p, l2p = k_to_lp(k1, k2)
    g1 = g_coef_precomputed(eps, l1, l1p, m_max, gaussians)
    g2 = g_coef_precomputed(eps, l2, l2p, m_max, gaussians)
    return np.outer(g1, g2)


def lambda_coef(k2, qm, pm, eps, eta, G, a_vec, m_max):
    # Eq. (44)
    lam = 0
    for k1 in range(4):
        l1, l2 = k_to_l(k1, k2)
        l1p, l2p = k_to_lp(k1, k2)
        g1 = g_coef(eps, l1, l1p, qm, eta, G, m_max)
        g2 = g_coef(eps, l2, l2p, pm, eta, G, m_max)
        lam += a_vec[k1] * np.outer(g1, g2)
    return lam


def get_m_max(eps, exp_cutoff=23):
    return np.ceil(np.sqrt(exp_cutoff * 4 / np.pi * np.tanh(eps) ** -1))


def GKP_pauli_normalization(eps, k):
    m_max = get_m_max(eps)
    m_set = M_set(k, m_max)
    N = 0
    for m in m_set:
        c = coef(eps, m)
        s = sign_fun(k, m)
        N += s * c
    if k == 2:
        N *= -1
    return abs(N)


def db_to_eps(rdB):
    return np.arctanh(10 ** (-rdB / 10))


def eps_to_cutoff(eps, coef=4, min_cutoff=10):
    return int(max(coef / eps, min_cutoff))


def x_axis(eps, points_per_peak=10):
    m_max = get_m_max(eps)
    peak_variance = 0.5 * np.tanh(eps)
    dx = np.sqrt(peak_variance) / points_per_peak
    x_max = 1.5 * np.sqrt(np.pi) / (2 * np.sqrt(2)) * m_max
    x = np.arange(-x_max, x_max, dx)
    x -= np.mean(x)
    return x, dx


def normalization(eps, a_vec):
    # Eq. (5)
    N_pauli = [GKP_pauli_normalization(eps, k) for k in range(4)]
    N = sum([a_vec[k] * N_pauli[k] for k in range(4)])
    return N


def GKP_bell_normalization(eps):
    return sum([GKP_pauli_normalization(eps, k) ** 2 for k in [0, 1, 3]])


def GKP_pauli_wigner(eps, k, q, p):
    m_max = get_m_max(eps)
    m_set = M_set(k, m_max)
    W = 0
    sigma = Sigma_eps(eps)
    for m in m_set:
        c = coef(eps, m)
        if c > 1e-10:
            s = sign_fun(k, m)
            mu = mu_eps(eps, m)
            W += s * c * Gaussian(mu, sigma, np.meshgrid(q, p))
    if k == 2:
        W *= -1
    return W


##################################################################
## Fock-space functions for Uhlmann fidelity calculation
##################################################################


def hermite_functions(n_max, x):
    """
    Subrutine to calculate Hermite functions used for quadrature distribution
    Parameters
    ----------
    n_max : Integer
        Index of largest Hermite function.
    x : array
        x-axis.
    Returns
    -------
    Hlist : list
        List of Hermite functions.
    """
    Hlist = []
    Hlist.append(np.pi ** (-1 / 4) * np.exp(-1 / 2 * x**2))
    if n_max > 0:
        Hlist.append(np.sqrt(2) * x * np.pi ** (-1 / 4) * np.exp(-1 / 2 * x**2))
        for n in range(1, n_max):
            Hlist.append(
                (x * Hlist[n] - np.sqrt(n / 2) * Hlist[n - 1]) / np.sqrt((n + 1) / 2)
            )
    return Hlist


def quadrature_eigenstate(cutoff, x, angle=0):
    """
    Quadrature eigenstate |x>
    Parameters
    ----------
    cutoff : integer
        Fock space cutoff dimension.
    x : real number
        eigenvalue of the state.
    angle : real number or 'q' or 'p', optional
        quadrature direction. The default is 0.
    Returns
    -------
    qutip Qobj
        quadrature eigenstate in the Fock basis.
    """
    psi = np.zeros(cutoff)
    if angle == "p":
        angle = np.pi / 2
    if angle == "q":
        angle = 0
    Hlist = hermite_functions(cutoff - 1, x)
    psi = qutip.Qobj(np.array(Hlist))
    return qutip.Qobj(psi)


def get_gkp_ideal(cutoff, alpha=np.sqrt(2 * np.pi)):
    # Eq. (1)
    gkp0 = qutip.Qobj(np.zeros(cutoff))
    gkp1 = qutip.Qobj(np.zeros(cutoff))

    for m in range(-25, 25):
        gkp0 += quadrature_eigenstate(cutoff, x=m * alpha * np.sqrt(2))
        gkp1 += quadrature_eigenstate(cutoff, x=(m + 1 / 2) * alpha * np.sqrt(2))

    return gkp0, gkp1


def damping_operator(cutoff, eps):
    n = np.arange(cutoff)
    return qutip.Qobj(np.diag(np.exp(-eps * n)))


def get_gkp_finite(cutoff, eps, gkp_ideal=None):
    # Eq. (2) Unormalized gkp state with finite squeezing
    if gkp_ideal is None:
        gkp_ideal = get_gkp_ideal(cutoff)
    gkp0, gkp1 = gkp_ideal
    damping = damping_operator(cutoff, eps)
    return damping * gkp0, damping * gkp1


def get_gkp_paulis(gkp_finite):
    # Eq. (3)
    sigma_0 = gkp_finite[0].proj() + gkp_finite[1].proj()
    sigma_1 = gkp_finite[1] * gkp_finite[0].dag() + gkp_finite[0] * gkp_finite[1].dag()
    sigma_2 = 1j * (
        gkp_finite[1] * gkp_finite[0].dag() - gkp_finite[0] * gkp_finite[1].dag()
    )
    sigma_3 = gkp_finite[0].proj() - gkp_finite[1].proj()
    return sigma_0, sigma_1, sigma_2, sigma_3


def gkp_dm(a_vec, gkp_paulis):
    # Eq. (4)
    state = sum([a_vec[k] * gkp_paulis[k] for k in range(4)])
    return state / state.tr() if state.tr() > 0 else state


def gkp_ket(a_vec, gkp_finite):
    # Defining target output states as kets makes the fidelity calculation more numerically stable
    if a_vec[1] == 1:
        return (gkp_finite[0] + gkp_finite[1]).unit()
    elif a_vec[1] == -1:
        return (gkp_finite[0] - gkp_finite[1]).unit()
    elif a_vec[2] == 1:
        return (gkp_finite[0] + 1j * gkp_finite[1]).unit()
    elif a_vec[2] == -1:
        return (gkp_finite[0] - 1j * gkp_finite[1]).unit()
    elif a_vec[3] == 1:
        return gkp_finite[0].unit()
    elif a_vec[3] == -1:
        return gkp_finite[1].unit()
    else:
        raise NotImplementedError("a_vec must correspond to Pauli eigenstate")


if __name__ == "__main__":
    rdB_small = np.linspace(0.01, 3, 21)
    rdB_large = np.arange(3, 17, 0.5)
    rdB_axis = np.append(rdB_small[:-1], rdB_large)
    # rdB_axis = np.arange(15, 17, 0.5)
    # rdB_axis = [15]

    s_vecs = np.array([[1, 1, 1], [1, -1, -1], [-1, -1, 1], [-1, 1, -1]])

    eta = 1
    G = 1
    a_sets = [
        [1, 0, 0, 1],
        [1, 0, 0, -1],
        [1, 0, 1, 0],
        [1, 0, -1, 0],
        [1, 1, 0, 0],
        [1, -1, 0, 0],
    ]

    fidelity_logical_avg = np.zeros(len(rdB_axis)) * np.nan
    fidelity_uhlmann_avg = np.zeros(len(rdB_axis)) * np.nan
    for i_rdB, rdB in enumerate(rdB_axis):
        gkp_ideal = get_gkp_ideal(cutoff)
        gkp_finite = get_gkp_finite(cutoff, eps)
        gkp_paulis = get_gkp_paulis(gkp_finite)

        print(rdB)
        eps = db_to_eps(rdB)
        m_max = get_m_max(eps)
        x, dx = x_axis(eps)
        cutoff = eps_to_cutoff(eps)

        N_bell = GKP_bell_normalization(eps)
        N_pauli = [GKP_pauli_normalization(eps, i) for i in range(4)]
        g_product_matrix = np.array(
            [
                [g_product(k1, k2, x, x, eps, eta, G, m_max) for k1 in range(4)]
                for k2 in range(4)
            ]
        )

        F = [np.zeros((len(x), len(x))) for _ in range(4)]
        probability_total = np.zeros((len(x), len(x)))
        a_vec_out = []
        probability = []
        for a_vec in a_sets:
            N = normalization(eps, a_vec)
            lam_vec = np.einsum("abcd,a->bcd", g_product_matrix, a_vec) / (N * N_bell)

            a_vec_out.append([lam_vec[i] / lam_vec[0] for i in range(4)])

            probability.append(
                np.real(np.einsum("acd,a->cd", lam_vec, N_pauli) * dx**2)
            )

            probability_total += probability[-1] / len(a_sets)

            for i in range(4):
                F_post = 0.5 * (
                    1 + np.einsum("abc,a->bc", a_vec_out[-1][1:], s_vecs[i] * a_vec[1:])
                )
                F[i] += np.real(probability[-1]) * np.real(F_post) / len(a_sets)

        optimal = np.max(F / probability_total, axis=0)
        optimal = np.nan_to_num(optimal)
        optimal_frame = np.argmax(F / probability_total, axis=0)

        fidelity_uhlmann = np.zeros(len(a_sets))
        for i_a_vec, a_vec in enumerate(a_sets):
            probability_frame = [
                sum(probability[i_a_vec][optimal_frame == k]) for k in range(4)
            ]
            weighted_a_vec_out = probability[i_a_vec] * a_vec_out[i_a_vec]
            avg_a_vec_out = [
                np.sum(weighted_a_vec_out[:, optimal_frame == k], axis=1)
                for k in range(4)
            ]
            avg_a_vec_out = [avg_a_vec_out[k] / probability_frame[k] for k in range(4)]
            avg_a_vec_out = np.nan_to_num(avg_a_vec_out)

            states = [gkp_dm(a, gkp_paulis) for a in avg_a_vec_out]
            states_ideal = [
                gkp_ket(np.append(1, s_vecs[i] * a_vec[1:]), gkp_finite)
                for i in range(4)
            ]
            fidelity_uhlmann[i_a_vec] = np.sum(
                [
                    probability_frame[i]
                    * (states_ideal[i].dag() * states[i] * states_ideal[i]).full()
                    for i in range(4)
                ]
            )

        print(1 - fidelity_uhlmann)

        fidelity_uhlmann_avg[i_rdB] = np.mean(fidelity_uhlmann)
        fidelity_logical_avg[i_rdB] = np.sum(probability_total * optimal)

        fig, ax = plt.subplots()
        ax.semilogy(rdB_axis, 1 - fidelity_uhlmann_avg, "o-", label="Uhlmann")
        ax.semilogy(rdB_axis, 1 - fidelity_logical_avg, "o-", label="Logical")

        ax.set_ylim(1e-8, 1)
        ax.set_xlim(0, 25)
        ax.set_xlabel("Squeezing (dB)")
        ax.set_ylabel("Infidelity, 1 - F")
        ax.grid()
        ax.legend()
        fig.show()
