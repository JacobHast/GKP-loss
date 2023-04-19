import numpy as np
from scipy.linalg import det, inv

def Gaussian(mu, sigma, x):
    # Eq. (20)
    if np.shape(sigma) == ():
        G = 1/np.sqrt(sigma*2*np.pi)*np.exp(-0.5*(x - mu)**2/sigma)
    else:
        n = np.shape(sigma)[0]
        G = 1/np.sqrt(det(sigma)*(2*np.pi)**n)*np.exp(-0.5*(x - mu).T@inv(sigma)@(x - mu))
    return G

