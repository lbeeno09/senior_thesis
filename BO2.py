#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(0)

# GP Regression
def GPR(xtest, xtrain, ytrain, kernel, TV=False, *param):
    n0 = len(ytrain)
    N = len(xtest)
    noise, epsilon = param

    K = np.zeros([n0, n0])
    for j, y in enumerate(xtrain):
        for i, x in enumerate(xtrain):
            K[i][j] = kernel(x, y)
    K = K + noise * noise * np.eye(n0)

    k_star = np.zeros([n0, N])
    for j, y in enumerate(xtest):
        for i, x in enumerate(xtrain):
            k_star[i][j] = kernel(x, y)

    k_starstar = np.zeros([N, N])
    for j, y in enumerate(xtest):
        for i, x in enumerate(xtest):
            k_starstar[i][j] = kernel(x, y)

    # Calculation of time-varying kernel
    if TV:
        D = np.zeros([n0, n0])
        d_star = np.zeros([n0, N])

        for i in range(n0):
            for j in range(n0):
                D[i][j] = (1 - epsilon) ** (np.abs((i - j) / 2))
        K = K * D # element-wise multiplication

        for i in range(n0):
            for j in range(N):
                d_star[i][j] = (1 - epsilon) ** ((n0 - i) / 2)
        k_star = k_star * d_star

    Kinv = np.linalg.inv(K)
    mu = np.matmul(k_star.T, np.matmul(Kinv, ytrain))
    var = k_starstar - np.matmul(k_star.T, np.matmul(Kinv, k_star))
    
    return np.array(mu), np.array(var)


# calculating p-1 points to query
def ucblist(grid, kernel, ucb, weight, n):
    bestidxs = []
    for i in range(n):
        sumgrid = []
        for z in grid:
            sum = 0
            for j in range(i):
                sum += kernel(z, grid[bestidxs[j]])
            sumgrid.append(sum)
        sumgrid = [weight * s for s in sumgrid]

        bestidx = np.argmax([u - s for u, s in zip(ucb, sumgrid)])
        ucb[bestidx] = -100
        bestidxs.append(bestidx)

    return bestidxs

# TV-GP-UCB algorithm
def TVGPUCB(grid, kernel, f, N, samplepts, noise, lamda, epsilon, singret=False):
    gridsize = len(grid)
    regrets = []
    regret = 0

    f_star = f
    sol = []

    mu = np.zeros(grid.shape[0])
    var = np.eye(grid.shape[0])

    xtrain = []
    ytrain = []
    for iter in range(1, N+1):
        alpha = np.sqrt(0.8 * np.log(4 * iter))
        ucb = mu + alpha * np.sqrt(np.diag(var))
        bestidxs = ucblist(grid, kernel, ucb, lamda, samplepts)
        bestpos = [grid[i] for i in bestidxs]

        xtrain.extend(bestpos)
        ytrain.extend([f_star[i] + np.random.normal(0, noise) for i in bestidxs])
        mu, var = GPR(grid, np.array(xtrain), np.array(ytrain), kernel, True, noise, epsilon)
            
        sol.append(f_star.tolist())

        # branch for single regret
        if singret:
            regret += np.max(f_star) - f_star[math.floor(bestpos[0] * gridsize) - 1]
        else:
            regret += (len(bestpos) * np.max(f_star) - np.sum([f_star[math.floor(i * gridsize) - 1] for i in bestpos])) / len(bestpos)
        regrets.append(regret / iter)

        f_star = np.sqrt(1 - epsilon) * f_star + np.sqrt(epsilon) * np.random.multivariate_normal(np.zeros(gridsize), np.eye(gridsize))

    return regrets


# f_1
def f_random(gridsize):
    return np.random.multivariate_normal(np.zeros(gridsize), np.eye(gridsize))


# SE kernel
def gauss_kern(x, y, gamma: float = 800):
    return np.exp(-gamma * np.linalg.norm(x - y)**2)


# MPTVGPUCB env setup
def TVGPUCBtest(case):
    gridsize:int = 50 
    noise: float = 0.01
    epsilon: float = 0.1
    X_star = np.linspace(0, 1, num=gridsize)
    f_init = f_random(gridsize)

    def plot(l, e):
        if case == 1 or case == 2:
            tot1, pt1 = (200, 1)
            tot2, pt2 = (100, 2)
            tot3, pt3 = (67, 3)
        elif case == 3 or case == 4:
            tot1, pt1 = (200, 1)
            tot2, pt2 = (200, 2)
            tot3, pt3 = (200, 3)
        
        if case == 1 or case == 3:
            regret1 = TVGPUCB(X_star, gauss_kern, f_init, tot1, pt1, noise, l, e)
            regret2 = TVGPUCB(X_star, gauss_kern, f_init, tot2, pt2, noise, l, e)
            regret3 = TVGPUCB(X_star, gauss_kern, f_init, tot3, pt3, noise, l, e)
        elif case == 2 or case == 4:
            regret1 = TVGPUCB(X_star, gauss_kern, f_init, tot1, pt1, noise, l, e, True)
            regret2 = TVGPUCB(X_star, gauss_kern, f_init, tot2, pt2, noise, l, e, True)
            regret3 = TVGPUCB(X_star, gauss_kern, f_init, tot3, pt3, noise, l, e, True)

        print(f"plot l={l}, e={e}")
        plt.xlabel(r"$t$")
        plt.ylabel(r"$R_t / t$")
        plt.title(r"$\varepsilon = $" + f"{e}, " + r"$\lambda = $" + f"{l}")
        plt.plot(np.linspace(0, tot1, num=tot1), regret1, color="black", linestyle="dotted", label="1 point")
        plt.plot(np.linspace(0, tot2, num=tot2), regret2, color="black", linestyle="solid", label="2 points")
        plt.plot(np.linspace(0, tot3, num=tot3), regret3, color="black", linestyle="dashdot", label="3 points")
        plt.legend()
        plt.show()

    for l in [0.5, 0.1, 0.01]:
        for e in [0.3, 0.1, 0.03, 0.01, 0.001]:
            plot(l, e)


def main():
    TVGPUCBtest(1) # case 1
    TVGPUCBtest(2) # case 2
    TVGPUCBtest(3) # case 3
    TVGPUCBtest(4) # case 4

if __name__ == "__main__":
    main()