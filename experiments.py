import uuid
import numpy as np
from pkg.gaussian_process import GP,SampledGrid
from pkg.blocked_gibbs import *
from pkg.utils import *
from pkg.distributions import *
import matplotlib.pyplot as plt

def check_gp():
    
    # parameters
    n = 50
    nsim = 5
    ntest = n
    ndata = 3
    alpha = 1e-13
    param = .1
    xtest = np.linspace(-np.pi-1.,2*np.pi+1.,ntest).reshape(-1,1)
    xdata = np.linspace(-np.pi+.3,np.pi+.5,ndata).reshape(-1,1)

    # plot data
    ydata = np.sin(xdata).reshape(-1,1)
    fig,axs = plt.subplots(3,1)
    axs[0].plot(xdata,ydata,'+-')
    axs[0].set_title("sin(x)")

    # create our gaussian process
    gp = GP(xtest,xdata,ydata,'rqk',alpha,nsim)

    # plot prior   
    f_prior = gp.sprior(nsim)
    axs[1].plot(xtest,f_prior,'+-')
    axs[1].set_title("Prior")

    # plot posterior
    f_post = gp.spost(nsim)
    means = gp.posterior_means()
    stddev = gp.posterior_stddev()
    axs[2].plot(xtest,f_post,'-')
    axs[2].plot(xdata,ydata,'b--',color='blue') # plot the function
    axs[2].set_title("Posterior")
    axs[2].fill_between(xtest.flat,means.flat - 2*stddev, means.flat + 2*stddev,
                        alpha=0.15,color='red',label="$\sigma$")
    plt.show()

    
def synthetic_lambda_1(obs_times,haz):
    return 2*np.exp(obs_times/5) + np.exp( (-(t - 25)/100.)^2 )

def synthetic_lambda_2(obs_times,haz):
    lam = 5*np.sin(obs_times**2) + 6
    sample = sample_prior(lam,haz)
    return sample

def sample_prior(lam,haz):
    """
    This function is to simulate data from
    the "truth";

    We need to sample event times 
    from Equation (4) on page 2 of the paper.

    I don't think there is an (obv) algorithm when
    we can evaluate only evaluate at "t".

    In fact this paper is to do just that.

    Instead, we want to analytically compute this?


    Say we do the following:
    g = homogenousPoissonProcess(constant=)

    event_1 = g.sample()


    """
    pass

def experiment_1():
    
    # experiment parameters
    T = 3.
    lam_s = 5.
    alpha = 1e-1
    number_of_samples = 200
    load_file = False
    uuid_str = "testing" #uuid.uuid4()
    obs_times = np.arange(0.1,T,0.1)
    num_of_obs = len(obs_times)

    # gamma rate prior; tbd

    # the inhomogenous Poisson Process component
    shape = 3
    rate = 1

    haz = Gamma({'shape':shape,'rate':rate})
    # collect observation values
    obs_values = synthetic_lambda_2(obs_times,haz)
    obs_values = np.linspace(0.1,lam_s,num_of_obs) # temporary code-debugging data

    # gaussian process prior init
    G = SampledGrid(obs_times,obs_values)
    W = SampledGrid(npr.rand(10)*T) # init to uniform; random grid
    l = GP(W.t,G.t,G.v,alpha=alpha)

    # mcmc variables
    omega = lam_s * haz.shape
    aggregate = {"W":[],"V":[],"T":[],"means":[],"sigma":[]}
    save_iter = 100
    # mcmc chain begins
    if not load_file:
        for i in range(number_of_samples):
            sample,W,means,sigma = blocked_gibbs_gp_mod_rp(T,G,W,l,omega,lam_s,haz)
            aggregate['W'].append(W)
            aggregate['V'].append(sample.v)
            aggregate['T'].append(sample.t)
            aggregate['means'].append(means)
            aggregate['sigma'].append(sigma)
            #print('posterior',np.c_[sample.v,sample.t])

            # ~~ loop info ~~
            if (i % save_iter) == 0 and ( i > 0 ):
                print("i = {}".format(i))
                print("saving current samples to file.")
                save_samples_in_pickle(aggregate,omega,'gpmodrp',uuid_str,i)

        # save to memory
        p_u_str = 'gpmodrp_{}'.format(uuid_str)
        save_samples_in_pickle(aggregate,omega,'gpmodrp',uuid_str)
    else:
        # load to memory
        fn = "results_541f8ddf-1342-41db-8daa-855a7041081e_final.pkl"
        if filename:
            fn = filename
        aggregate,uuid_str,omega = load_samples_in_pickle(fn)
    print("omega: {}".format(omega))




    return aggregate,uuid_str,omega



