import scipy as sp
import scipy.optimize as opt
import numpy as np
import numdifftools as nd

np.set_printoptions(precision=1, suppress=True)

distrib = {
    'uniform': np.array([
        [0, 0],
        [1e+8, 0.1],
        [2e+8, 0.2],
        [3e+8, 0.3],
        [4e+8, 0.4],
        [5e+8, 0.5],
        [6e+8, 0.6],
        [7e+8, 0.7],
        [8e+8, 0.8],
        [9e+8, 0.9],
        [1e+09, 1]
    ])
}

def dist_cdf(x, d='uniform'):
    samples = distrib[d]
    return np.interp(x, samples[:,0],samples[:,1])

def cdf_inv(x, d='uniform'):
    samples = distrib[d]
    return np.interp(x, samples[:,1],samples[:,0])

epsilon = 1e-2 # A really small number

def threshold_calc(K=8, load=0.5, dist='uniform'):
    theta = np.random.random_sample(K-1,)
    theta = theta / np.sum(theta)
    avgFlowSize = np.dot(np.insert(np.diff(distrib[dist][:,1]),0,0), distrib[dist][:,0])
    
    def TauObjective(theta):
        mu0 = 1.25e+9 
        lmd0 = load * mu0 / avgFlowSize
        alpha = cdf_inv(np.cumsum(theta), d=dist)
        alpha = np.insert(alpha,0,0)
        diff_alpha = np.diff(alpha)
        EL = np.multiply(diff_alpha, 1-dist_cdf(alpha[:-1], d=dist))
        l = lmd0 * EL # arrival rate at each queue

        mu = np.empty(K-1)
        rho = np.empty(K-1)
        idle_perc = 1

        for i in range(K-1):
            mu[i] = idle_perc * mu0
            rho[i] = l[i]/mu[i]
            idle_perc *= (1 - rho[i])

        T = 1./np.subtract(mu,l)

        Tau = np.dot(theta,np.cumsum(T))
        return Tau

    def TauJac(theta):
        return nd.Jacobian(TauObjective)(theta)

    def cons_1(theta):
        return 1-np.sum(theta)

    def jac_cons1(theta):
        return nd.Jacobian(cons_1)(theta)

    cons = ({'type': 'ineq',
             'fun' : cons_1,
             'jac' : jac_cons1})

    bnds = np.reshape(np.array([0,1]*(K-1)), (K-1,2))

    def accptst(f_new=0, x_new=0, f_old=0, x_old=0):
        nonzero = np.all(np.logical_and(x_new > 0, x_new < 1))
        sumcons_upper = (np.sum(x_new) < 1.01)
        sumcons_lower = (np.sum(x_new) > 0.99)   
        return (nonzero and sumcons_upper and sumcons_lower)

    accptst(f_new=0, x_new=theta)

    res = opt.basinhopping(TauObjective,theta,niter=100, accept_test=accptst, T=np.sqrt(K), #disp=True,
                     minimizer_kwargs={
                         "method" : "SLSQP",
                         "jac" : TauJac,
                         "constraints" : cons,
                         "bounds" : bnds})

    print(cdf_inv(np.cumsum(res.x), d=dist)/1460)

Ks = [2,4,8]
Loads = [0.5, 0.6, 0.7, 0.8, 0.9]

for di in ['uniform']:
    print("Ditribution: {}".format(di))
    for k in Ks:
        for lds in Loads:
            print("{} Queues, Load = {}".format(k, lds))
            threshold_calc(k, lds, di)
    print("="*80)


