import numpy as np


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


sigma = 0.07
x, dx = np.linspace(-10, 10, 1000, retstep=True)
mu1 = 2
mu2 = 2

g1 = Gaussian(mu1, sigma, x)
g2 = Gaussian(mu2, sigma, x)

print(sum(g1) * dx)
print(sum(g2) * dx)
print(sum(g1 * g2) * dx)
print(Gaussian(0, 2 * sigma, mu1 - mu2))


sigma = np.array([[0.7, 0.2], [0.2, 0.7]])
x, dx = np.linspace(-10, 10, 1000, retstep=True)
mu1 = [2, 2]
mu2 = [1, 1]

g1 = Gaussian(mu1, sigma, np.meshgrid(x, x))
g2 = Gaussian(mu2, sigma, np.meshgrid(x, x))


np.sum(g1) * dx**2
np.sum(g2) * dx**2

print(np.sum(g1 * g2) * dx**2)

print(Gaussian(0, 2 * sigma, np.array(mu1) - np.array(mu2)))
