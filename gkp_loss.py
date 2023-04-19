import numpy as np
from scipy.linalg import det, inv
import matplotlib.pyplot as plt
import qutip



def Gaussian(mu, sigma, x):
    # Eq. (20)
    if np.shape(sigma) == ():
        G = 1/np.sqrt(sigma*2*np.pi)*np.exp(-0.5*(x - mu)**2/sigma)
    else:
        n = np.shape(sigma)[0]
        sigma_inv = inv(sigma)
        prefactor = 1/np.sqrt(det(sigma)*(2*np.pi)**n)
        if len(np.shape(x)) == 1:
            G = prefactor*np.exp(-0.5*(x - mu).T@sigma_inv@(x - mu))
        else:
            x_minus_mu = np.array([x[i] - mu[i] for i in range(2)])
            G = prefactor*np.exp(-0.5*np.einsum('kij,km,mij->ij', x_minus_mu, sigma_inv, x_minus_mu))
    return G


def Sigma_eps(eps):
    # Eq. (21)
    return 0.5*np.tanh(eps)*np.identity(2)

def mu_eps(eps, m):
    # Eq. (21)
    return 1/np.cosh(eps)*np.sqrt(np.pi)/2*m

def Sigma_p(eps, eta, G):
    # Eq. (36)
    return 1/4*(np.tanh(eps)*(1 + eta*G) + eta*(G - 1) + 1 - eta)

def mu_p(eps, eta, G, m):
    # Eq. (37)
    return 1/np.cosh(eps)*np.sqrt(np.pi)/(2*np.sqrt(2))*(np.sqrt(eta*G)*m[0] + m[1])


def M_set(L, m_max):
    # Eq. (22)
    m1_list = np.arange(-m_max, m_max)
    m2_list = m1_list
    if L == 0:
        m1_list = m1_list[m1_list%2 == 0]
        m2_list = m2_list[m2_list%2 == 0]
    if L == 1:
        m1_list = m1_list[m1_list%2 == 1]
        m2_list = m2_list[m2_list%2 == 0]
    if L == 2:
        m1_list = m1_list[m1_list%2 == 1]
        m2_list = m2_list[m2_list%2 == 1]
    if L == 3:
        m1_list = m1_list[m1_list%2 == 0]
        m2_list = m2_list[m2_list%2 == 1]

    pairs = []
    for m1 in m1_list:
        for m2 in m2_list:
            pairs.append(np.array([m1, m2], 'complex'))
    return pairs



def sign_fun(k, m):
    # Eq. (23)
    if k == 0:
        return 1
    if k == 1:
        
        return (-1)**(m[1]/2)
    if k == 2:
        return (-1)**((m[0] + m[1])/2)
    if k == 3:
        return (-1)**(m[0]/2)
    raise ValueError(f"{k} must be 0, 1, 2 or 3")


def coef(eps, m):
    # Eq. (24)
    return np.exp(-np.tanh(eps)*np.pi/4*np.linalg.norm(m)**2)


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
        g += coef(eps, n)*sign_fun(lp, n)*Gaussian(mu_p(eps, eta, G, n), Sigma_p(eps, eta, G), x)
    return g


def g_product(k1, k2, qm, pm, eps, eta, G, m_max):
    # product of the g's in Eq. (44)
    l1, l2 = k_to_l(k1, k2)
    l1p, l2p = k_to_lp(k1, k2)
    g1 = g_coef(eps, l1, l1p, qm, eta, G, m_max)
    g2 = g_coef(eps, l2, l2p, pm, eta, G, m_max)
    return np.outer(g1, g2)


def lambda_coef(k2, qm, pm, eps, eta, G, a_vec, m_max):
    # Eq. (44)
    lam = 0
    for k1 in range(4):
        l1, l2 = k_to_l(k1, k2)
        l1p, l2p = k_to_lp(k1, k2)
        g1 = g_coef(eps, l1, l1p, qm, eta, G, m_max)
        g2 = g_coef(eps, l2, l2p, pm, eta, G, m_max)
        lam += a_vec[k1]*np.outer(g1, g2)
    return lam

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
    Hlist.append(np.pi**(-1/4)*np.exp(-1/2*x**2))
    if n_max > 0:
        Hlist.append(np.sqrt(2)*x*np.pi**(-1/4)*np.exp(-1/2*x**2))
        for n in range(1, n_max):
            Hlist.append((x*Hlist[n] - np.sqrt(n/2)*Hlist[n-1])/np.sqrt((n + 1)/2))
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
    if angle == 'p':
        angle = np.pi/2
    if angle == 'q':
        angle = 0
    Hlist = hermite_functions(cutoff - 1, x)
    psi = qutip.Qobj(np.array(Hlist))
    return qutip.Qobj(psi)

def GKP(cutoff, squeezing_dB=10, alpha=np.sqrt(2*np.pi), L=0):
    """
    Finite-squeezing rectangular GKP state calculated as exp(-epsilon*n)|GKP> where
    epsilon = atan(Delta) with Delta = 10^(-dB/10)
    Parameters
    ----------
    cutoff : integer
        Fock space cutoff dimension.
    squeezing_dB : real number, optional
        The squeezing of the GKP state in dB. The default is 10.
    alpha : real number, optional
        GKP lattice spacing in the q-quadrature divided by sqrt(2). The default is np.sqrt(2*np.pi).
    L : 0 or 1, optional
        logical state of the GKP qubit. The default is 0.
    Returns
    -------
    state : qutip Qobj
        Finitely squeezed GKP state.
    """
    state = qutip.Qobj(np.zeros(cutoff))
    if squeezing_dB == 0:
        return qutip.fock(cutoff, 0)

    for m in range(-15, 15):
        q = (m + L/2)*alpha*np.sqrt(2)
        state += quadrature_eigenstate(cutoff, q)
    n = np.arange(cutoff)
    eps = np.arctanh(10**(-squeezing_dB/10))
    E = qutip.Qobj(np.diag(np.exp(-eps*n)))
    state = E*state
    
    # state = state.unit()
    return state


def get_m_max(eps, exp_cutoff=23):
    return np.ceil(np.sqrt(exp_cutoff*4/np.pi*np.tanh(eps)**-1))


def GKP_pauli_normalization(eps, k):
    m_max = get_m_max(eps)
    m_set = M_set(k, m_max)
    N = 0
    for m in m_set:
        c = coef(eps, m)
        s = sign_fun(k, m)
        N += s*c
    if k == 2:
        N *= -1
    return abs(N)

def db_to_eps(rdB):
    return np.arctanh(10**(-rdB/10))

def normalization(eps, a_vec):
    # Eq. (5)
    N_pauli = [GKP_pauli_normalization(eps, k) for k in range(4)]
    N = sum([a_vec[k]*N_pauli[k] for k in range(4)])
    return N

def GKP_bell_normalization(eps):
    return sum([GKP_pauli_normalization(eps, k)**2 for k in [0, 1, 3]])

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
            W += s*c*Gaussian(mu, sigma, np.meshgrid(q, p))
    if k == 2:
        W *= -1
    return W


rdB_axis = np.linspace(0.001, 25, 25)

# rdB_axis = np.linspace(10, 15, 6)

eta = 0.95
G = 1
dx = 0.1
x = np.arange(-40, 40, dx)
# x, dx = np.linspace(-30, 30, 1001, retstep=True)
a_sets = [
    [1, 1, 0, 0],
    [1, -1, 0, 0],
    [1, 0, 1, 0],
    [1, 0, -1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, -1],
]

FC = np.zeros(len(rdB_axis))*np.nan
for i_rdB, rdB in enumerate(rdB_axis):
    print(rdB)
    eps = db_to_eps(rdB)
    m_max = get_m_max(eps)


    N_bell = GKP_bell_normalization(eps)
    N_pauli = [GKP_pauli_normalization(eps, i) for i in range(4)]
    g_product_matrix = np.array([[g_product(k1, k2, x, x, eps, eta, G, m_max) for k1 in range(4)] for k2 in range(4)])

    s_vecs = np.array([[1, 1, 1], [1, -1, -1], [-1, -1, 1], [-1, 1, -1]])


    F = [np.zeros((len(x), len(x))) for i in range(4)]
    probability_total = np.zeros((len(x), len(x)))
    for a_vec in a_sets:
        N = normalization(eps, a_vec)
        lam_vec = np.einsum('abcd,a->bcd', g_product_matrix, a_vec)/(N*N_bell)

        a_vec_out = [lam_vec[i]/lam_vec[0] for i in range(4)]
        probability = np.real(np.einsum('acd,a->cd', lam_vec, N_pauli)*dx**2)
        probability_total += probability/len(a_sets)
        # print(np.sum(probability))

        for i in range(4):
            F_post = 0.5*(1 + np.einsum('abc,a->bc', a_vec_out[1:], s_vecs[i]*a_vec[1:]))
            F[i] += np.real(probability)*np.real(F_post)/len(a_sets)

    optimal = np.max(F/probability_total, axis=0)
    optimal = np.nan_to_num(optimal)
    # plt.contourf(x, x, optimal, 100, vmin=0, vmax=1, cmap='seismic')
    # plt.colorbar()

    FC[i_rdB] = np.sum(probability_total*optimal)
    plt.semilogy(rdB_axis, 1-FC)
    plt.ylim(1e-8, 1)
    plt.xlim(0, 25)
    plt.xlabel('Squeezing (dB)')
    plt.ylabel('Fidelity')
    plt.grid()
    plt.show()


# weights = np.einsum('ijk->i', lam_vec_2)*dx**2
# print(sum([weights[i]*N_pauli[i] for i in range(4)])



# plt.contourf(np.real(lam_vec[0] - lam_vec_2[0]), 100, vmin=-1, vmax=1, cmap='seismic')
# plt.colorbar()

# lam_vec_2 = [np.einsum('abcd,a->bcd', g_product_matrix, a_vec)*dx**2 for a_vec in a_sets]




# plt.contourf(x, x, np.real(lam), 100)


# G = 1
# eta = 0.95
# l = 0
# lp = 1

# x = 1

# k2 = 0
# qm = 2
# pm = 4

# m_max = np.ceil(np.sqrt(13*4/np.pi*np.tanh(eps)**-1))
# a = [1,0.5,0.1,-0.2]

# lambda_coef(k2, qm, pm, eps, eta, G, a, m_max)
# sigma = 0.5*np.identity(2)
# mu = np.array([1,2])


# xvec = np.linspace(-5, 5, 101)
# yvec = xvec


# F = [[Gaussian(mu, sigma, np.array([x, y])) for x in xvec] for y in yvec]

# # def g(l, lp, x, nmax):


# plt.pcolor(xvec, yvec, F)
# plt.grid()

# Gaussian(1, 0, 2)


# M(1, 5)