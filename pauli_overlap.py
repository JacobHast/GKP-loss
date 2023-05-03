import numpy as np
from gkp_loss import *
import matplotlib.pyplot as plt


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


def GKP_pauli_overlap(eps, k1, k2):
    m_max = get_m_max(eps)
    m_set_1 = M_set(k1, m_max)
    m_set_2 = M_set(k2, m_max)
    overlap = 0
    sigma = Sigma_eps(eps)
    for m1 in m_set_1:
        for m2 in m_set_2:
            c1 = coef(eps, m1)
            c2 = coef(eps, m2)
            if c1 > 1e-10 and c2 > 1e-10:
                s1 = sign_fun(k1, m1)
                s2 = sign_fun(k2, m2)
                if np.sqrt(sum((m1 - m2) ** 2)) > 10:
                    continue
                mu1 = mu_eps(eps, m1)
                mu2 = mu_eps(eps, m2)
                overlap += s1 * s2 * c1 * c2 * Gaussian(0, 2 * sigma, mu2 - mu1)
    return overlap


# Gaussian(0, 2*sigma, np.array(mu1)-np.array(mu2))


r = 3
eps = db_to_eps(r)
k = 0
x, dx = np.linspace(-6, 6, 300, retstep=True)
q, p = np.meshgrid(x, x)
# N = normalization(eps, [1, 0, 0, 0])

# W = GKP_pauli_wigner(eps, k, x, x)
cutoff = 100
gkp_paulis = [GKP_pauli_qutip(cutoff, r, k) for k in range(4)]


print(2 * np.pi * GKP_pauli_overlap(eps, 0, 0) / N**2)

# plt.contourf(x, x, W)


# print(2*np.pi*np.sum(W*W)*dx**2/N**2)
