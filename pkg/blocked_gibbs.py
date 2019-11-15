import numpy as np
import numpy.random as npr
from .gaussian_process import GP,SampledGrid

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def find_endpoints(a,G,start):
    left,right = G[start],G[start+1]
    index = start
    for g in G[start+1:]:
        if a < right:
            break
        left = right
        right = g
        index += 1
    return left,right,index

def sample_poisson_process(Omega,T):
    samples = []
    i = 0
    N = npr.poisson( lam = Omega*T )
    while (i < N):
        sample = npr.exponential(scale=1./Omega)
        if sample <= T:
            samples.append(sample)
            i += 1
    return np.array(samples)
    
def blocked_gibbs_gp_mod_rp(T,G,G_prev,l,Omega,lam_s,haz):
    """
    -- Input --
    T :: the time interval [0,T]
    G :: fixed set of event times
    G_prev :: previously thinned times
    l :: a Gaussian Process instantiated on G \cup G_prev

    -- Output --
    G_new :: a new set of thinned times
    l :: a Gaussian Process instantiated on G \cup G_new
    ? Maybe points sampled from the 'new' l?
    """


    A = sample_poisson_process(Omega,T)
    l.update_test(A)
    lam_A = lam_s * sigmoid(l.sample_posterior())
    start = 0
    
    G_new = SampledGrid()
    for a,lam_a in zip(A,lam_A):
        G_l,G_r,start = find_endpoints(a,G.t,start)
        acc_prob = 1 - lam_a * haz.l(a - G_l) / Omega
        flip = npr.rand(1)[0]
        if flip > acc_prob: # reject, thin
            G_new.times += [a]
            G_new.vals += [lam_a]
    times = np.vstack([G.t,G_new.t])
    values = np.vstack([G.v,G_new.v])
    l.update_data(times,values)
    sample_values = l.sample_posterior(1)
    sample = SampledGrid(l.x_test,sample_values)
    return sample,G_new.t,l.pmeans,l.psigma
