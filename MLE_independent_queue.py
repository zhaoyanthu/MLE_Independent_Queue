import numpy as np
import copy
from cvxopt import matrix, log, div, spdiag, solvers
from cvxopt.solvers import qp
solvers.options['show_progress'] = False

"""
EM_MLE
    MLE
    for: Mstep, calLikelihood
    MLE
"""

def EM_MLE(obs_queue, L_max, init_p, threshold=1e-10, max_iter=500, 
    show_progress=True, MLE_show_progress=False):
    """ Apply EM algorithm to Maximum Likelihood Estimation
    Args:
        obs_queue: list[np.array], observed partial queues
        L_max: int, maximum queue length
        init_p: float, initial guess of the penetration rate
        threshold: float, threshold of the convergence of the objective function
        max_iter: int, max number of EM iterations
        show_progress: bool, show details of the EM algorithm
        MLE_show_progress: bool, show details of the initial MLE
    Returns:
        p: float, estimated penetration rate
        pi: list[float], estimated queue length distribution
    """
    # Step 1: given the initial p, calculate the initial pi
    try:
        init_pi = MLE(obs_queue, L_max, init_p, show_progress=MLE_show_progress)
    except (TypeError, ValueError, ArithmeticError) as error:
        print('Initial MLE error: ', error)
        init_pi = [1.0 / (L_max + 1)] * (L_max + 1)

    # Step 2: initial likelihood
    init_llh = calLikelihood(obs_queue, init_pi, init_p, L_max)
    if not init_llh: # when there is no observation
        return init_p, init_pi

    # Step 3: EM updates
    p, pi, llh = init_p, init_pi, init_llh
    loop = 0
    while (loop == 0 or abs(llh - old_llh) / abs(old_llh) > threshold) and loop < max_iter:
        old_llh = llh
        loop += 1
        old_p, old_pi = p, copy.deepcopy(pi)
        p, pi = Mstep(obs_queue, L_max, old_p, old_pi)
        llh = calLikelihood(obs_queue, pi, p, L_max)
        if show_progress:
            print('gaps:', p - old_p, max(abs(np.array(pi) - np.array(old_pi))))
            print('p(0):', round(init_p, 4), ', p(' + str(loop) + '):', round(p, 4))
            print('l(0):', round(init_llh, 4), ', l(' + str(loop) + '):', round(llh, 4))
    print('loop', abs(llh - old_llh) / abs(old_llh), threshold, loop)

    # Step 4: update pi again with p fixed
    pi = MLE(obs_queue, L_max, p, show_progress=MLE_show_progress)
    return p, pi


def MLE(obs_queue, L_max, p, max_iter=50, show_progress=True):
    """Given p, estimate P(Q=l) by maximizing the likelihood function
    Args:
        obs_queue: list[np.array], observed partial queues
        L_max: int, maximum queue length
        p: float, the (fixed) value of the penetration rate
        max_iter: int, max number of EM iterations
        show_progress: bool, show details of the MLE
    Returns:
        list(sol['x']): list[float], estimated pi
    """
    solvers.options['show_progress'] = show_progress
    solvers.options['maxiters'] = max_iter
    def F(x = None, z = None):
        x0 = matrix(0.0, (L_max + 1, 1))
        x0[:] = 1.0 / (L_max + 1)
        if x is None:  return 0, x0
        # obj, gradient, Hessian
        f, Df, H = 0, matrix(0.0, (1, L_max + 1)), matrix(0.0, (L_max + 1, L_max + 1))
        for q in obs_queue:
            n_i = sum(q)
            den = sum([x[l] * (1 - p)**(l - n_i) for l in range(len(q), L_max + 1)])
            # Check domain (different from feasible region)
            if den * p**n_i <= 0: return None
            f -= log(den * p**n_i)
            for j in range(len(q), L_max + 1):
                Df[j] -= (1 - p)**(j - n_i)  / den
                for k in range(len(q), L_max + 1):
                    H[j, k] += (1 - p)**(j + k - 2 * n_i) / den**2
        if z is None:  return f, Df
        return f, Df, H
    
    A, b = matrix(1.0, (1, L_max + 1)), matrix(1.0, (1, 1))
    G, h = matrix(0.0, (2 * (L_max + 1), L_max + 1)), matrix(0.0, (2 * (L_max + 1), 1))
    G[:L_max + 1, :] = spdiag(matrix(1.0, (1, L_max + 1)))
    G[L_max + 1:, :] = spdiag(matrix(-1.0, (1, L_max + 1)))
    h[:L_max + 1] = 1
    sol = solvers.cp(F, A=A, b=b, G=G, h=h)
    for i in range(len(sol['x']) - 1): 
        sol['x'][i] = max(0, sol['x'][i]) # to avoid small negative numbers
    return list(sol['x'])


def Mstep(obs_queue, L_max, old_p, old_pi):
    """Analytical solution of the M step
    Args:
        obs_queue: list[np.array], observed partial queues
        L_max: int, maximum queue length
        old_p: float, current guess of p
        old_pi: list[float], current guess of pi
    Returns:
        p: float, new guess of p
        pi: list[float], new guess of pi
    """
    B, D, E = 0, 0, [0] * (L_max + 1)
    for q in obs_queue:
        n_i = sum(q)
        for l in range(len(q), L_max + 1):
            den = sum((1 - old_p)**(w - l) * old_pi[w] for w in range(len(q), L_max + 1))
            A_li = old_pi[l] / den
            B += A_li * n_i
            D += A_li * (l - n_i)
            E[l] += old_pi[l] / den
    p = B / (B + D)
    sum_E = sum(E)
    # sometimes the value of El can be negative (probably for numerical reasons)
    pi = [max(El, 0) / sum_E for El in E]
    return p, pi


def calLikelihood(obs_queue, pi, p, L_max):
    """calculate likelihood given P(Q=l) and p
    Args:
        obs_queue: list[np.array], observed partial queues
        pi: list[float], queue length distribution
        p: float, penetration rate
        L_max: int, maximum queue length
    Returns:
        llh: float, likelihood
    """
    if p == 0:
        return np.nan
    llh = 0
    for q in obs_queue:
        n_i = sum(q)
        llh += log(sum([pi[l] * (1 - p)**(l - n_i) * p**n_i for l in range(len(q), L_max + 1)]))
    return llh


if __name__ == '__main__':
    from QLsimulator import *
    import matplotlib.pyplot as plt
    
    # Hyperparameters
    r = 5
    iteration = 500
    p = 0.5
    seed = 0
    
    # Sample generation and parameter estimation
    simulator = QLsimulator('poisson', iteration, p, r, seed=seed, isOverflow=False)
    simulator.generate_queues()
    real_queue_length, obs_queue, L_max = simulator.real_queue_length, simulator.obs_queue, simulator.L_max
    mle_est_p, mle_dist = EM_MLE(obs_queue, L_max, np.random.random(), threshold=1e-10,
                                 show_progress=False, MLE_show_progress=False)
    
    # Results
    print(p, mle_est_p)
    plt.figure(figsize=(10, 8))
    gt_dist = np.histogram(real_queue_length, bins=np.arange(0, max(real_queue_length)+2), density=1)[0]
    plt.bar(np.array(range(max(real_queue_length) + 1)), height=gt_dist, width=1, color='C0', alpha=0.3)
    plt.bar(np.array(range(L_max + 1)), height=mle_dist, width=1, color='C1', alpha=0.3)
    plt.legend(['Ground truth', 'Estimated distribution'])
    plt.xlabel('Queue length')
    plt.ylabel('Probability')