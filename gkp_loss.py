import numpy as np
from scipy.linalg import det, inv
import matplotlib.pyplot as plt
import qutip



def Gaussian(mu, sigma, x):
    if np.shape(sigma) == ():
        G = 1/np.sqrt(sigma*2*np.pi)*np.exp(-0.5*(x - mu)**2/sigma)
    else:
        n = np.shape(sigma)[0]
        G = 1/np.sqrt(det(sigma)*(2*np.pi)**n)*np.exp(-0.5*(x - mu).T@inv(sigma)@(x - mu))
    return G


def Sigma_eps(eps):
    return 0.5*np.tanh(eps)*np.identity(2)

def mu_eps(eps, m):
    return 1/np.cosh(eps)*np.sqrt(np.pi)/2*m

def Sigma_p(eps, eta, G):
    return 1/4*(np.tanh(eps)*(1 + eta*G) + eta*(G - 1) + 1 - eta)

def mu_p(eps, eta, G, m):
    return 1/np.cosh(eps)*np.sqrt(np.pi)/(2*np.sqrt(2))*(np.sqrt(eta*G)*m[0] + m[1])


def M_set(L, m_max):
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
    return np.exp(-np.tanh(eps)*np.pi/4*np.linalg.norm(m)**2)


def k_to_l(k1, k2):
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
        return 0, 0


def k_to_lp(k1, k2):
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
    n_set = M_set(l, m_max)
    g = 0
    for n in n_set:
        g += coef(eps, n)*sign_fun(lp, n)*Gaussian(mu_p(eps, eta, G, n), Sigma_p(eps, eta, G), x)
    return g

def lambda_coef(k2, qm, pm, eps, eta, G, a, m_max):
    lam = 0
    for k1 in range(4):
        l1, l2 = k_to_l(k1, k2)
        l1p, l2p = k_to_lp(k1, k2)
        g1 = g_coef(eps, l1, l1p, qm, eta, G, m_max)
        g2 = g_coef(eps, l2, l2p, pm, eta, G, m_max)
        lam += a[k1]*g1*g2
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


def GKP_normalization(eps, k):
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




GKP_normalization(5, 0)



rdB = 0.001
eps = db_to_eps(rdB)
N = [GKP_normalization(eps, k) for k in range(4)]


# print(N1/N2)

# print((GKP0*GKP0.dag() - GKP1*GKP1.dag()).tr())

# print((GKP0*GKP1.dag() + GKP1*GKP0.dag()).tr())

# print((1j*GKP0*GKP1.dag() - 1j*GKP1*GKP0.dag()).tr())


GKP0 = GKP(500, rdB)
GKP1 = GKP(500, rdB, L=1)

N2 = [(GKP0.proj() + GKP1.proj()).tr(), (GKP0*GKP1.dag() + GKP1*GKP0.dag()).tr()]

print(N[0]/N2[0])


print(N2[0]/N2[1],N[0]/N[1])

# plt.plot(abs(GKP0.full()))


# eps = np.arctanh(10**(-rdB/10))


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