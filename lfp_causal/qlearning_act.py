import numpy as np
# from scipy.special import softmax
import scipy.special as ss
from itertools import product

###############################################################################
#                          Q-LEARNING FUNCTIONS
###############################################################################


def softmax(x, T=1, ax=None):
    # Softmax function algorithm for stochastics selection
    # T value defines the temperature
    if ax is None:
        _x = x - ss.logsumexp(x)
        return np.exp(_x / T) / np.sum(np.exp(_x / T))
    else:
        _x = x - np.expand_dims(ss.logsumexp(x, axis=ax), ax)
        return np.exp(_x / T) / \
            np.expand_dims(np.sum(np.exp(_x / T), axis=ax), ax)


def fit_qlearning(states, actions, rewards, uniq_s, uniq_a):
    """Fit Q-learning model to behavioral data using a grid-search


    Parameters
    ----------
    s : array_like | int
        Sequence of stimuli {1, ..., nStims}
    a : array_like | int
        Sequence of actions {1, ..., nActions}
    r : array_like | int
        Sequence of rewards {0,1}
    Returns
    -------
    LL : float
        Log-likelihood with the data
    """

    # Init free params
    log_likelihood = float('-Inf')
    # Learning rate
    alpha = np.arange(0.1, 1.0, 0.01)
    # inverse temperature softmax
    beta = np.arange(1, 10.0, 0.2)
    # Prior on Q values
    q0 = [0.33, 0.5]

    # Grid search
    # for a, b, q in product(alpha, beta, q0):
    #     # Run Q-learning
    #     regs = qlearning(states, actions, rewards, uniq_s, uniq_a,
    #                      alpha=a, beta=b, q0=q)
    #
    #     # Store maximum log-likelohood
    #     if regs['log_likelihood'] > log_likelihood:
    #         # Add fields
    #         regs['alpha_fit'] = a
    #         regs['beta_fit'] = b
    #         regs['q0_fit'] = q
    #         # Update Log-Likelihood
    #         log_likelihood = regs['log_likelihood']
    #         # Output best fitting regressors
    #         best_regs = regs

    from joblib import Parallel, delayed
    regs = Parallel(n_jobs=-1, verbose=1) \
        (delayed(qlearning)(states, actions, rewards, uniq_s, uniq_a, a, b, q)
         for a, b, q in product(alpha, beta, q0))

    for _rg in regs:
        if _rg['log_likelihood'] > log_likelihood:
            # Add fields
            # _rg['alpha_fit'] = a
            # _rg['beta_fit'] = b
            # _rg['q0_fit'] = q
            # Update Log-Likelihood
            log_likelihood = _rg['log_likelihood']
            # Output best fitting regressors
            best_regs = _rg

    print('Model fitted at', best_regs['log_likelihood'])

    return best_regs

def qlearning(s, a, r, uni_s, uni_a, alpha=0.5, beta=1.0, q0=0.5):
    """Q-learning model

    Parameters
    ----------
    s : array | int
        Sequence of stimuli {1, ..., nStims}
    a : array | int
        Sequence of actions {1, ..., nActions}
    r : array | int
        Sequence of rewards {0,1}
    alpha : float
        Learning rate {0->1}
    beta : float
        Temperature of softmax
    q0 : float
        Initial value of Q
    Returns
    -------
    LL : float
        Log-likelihood with the data
    """

    # Number of stims, actions and trials
    # n_s, n_a, n_t = len(np.unique(s)), len(np.unique(a)), s.shape[0]
    n_s, n_a, n_t = len(uni_s), len(uni_a), s.shape[0]
    # _s, _a = np.unique(s), np.unique(a)
    _s, _a = uni_s, uni_a
    i_s, i_a = np.array(range(n_s)), np.array(range(n_a))

    # Init vars
    log_likelihood = 0.0
    RPE = np.zeros((n_t))
    p_correct = np.zeros((n_t))
    p_incorrect = np.zeros((n_t))
    p_cor_uncho = np.zeros((n_t))
    H = np.zeros((n_t))
    Ho = np.zeros((n_t))
    dP = np.zeros((n_t))
    dPn = np.zeros((n_t))
    shann_surp = np.zeros((n_t))
    bayes_surp = np.zeros((n_t))

    # Init Q-values
    Q = q0 * np.ones(shape=(n_s, n_a))

    # Add the to values
    P = softmax(Q, 1 / beta, ax=1)

    for i in range(n_t):

        # P(A|S): action probabilities given stimulus
        P = softmax(Q, 1/beta, ax=1)

        # Compute regressors
        # P(correct|chosen)
        p_correct[i] = P[i_s[np.where(_s == s[i])], i_a[np.where(_a == a[i])]]

        # P(incorrect|chosen)
        p_incorrect[i] = 1 - p_correct[i]

        # P(correct|unchosen)
        # j = np.setdiff1d(range(0, n_a), i)
        # j = np.setdiff1d(_a, a[i])
        p_cor_uncho[i] = np.mean(P[i_s[np.where(_s == s[i])],
                              i_a[np.where(_a != a[i])]])

        # P(correct|chosen) - P(correct|unchosen)
        dP[i] = p_correct[i] - p_cor_uncho[i]

        # P(correct|chosen) - P(incorrect|chosen)
        dPn[i] = p_correct[i] - p_incorrect[i]

        # Decision Entropy
        H[i] = -np.sum(P[i_s[np.where(_s == s[i])]] *
                       np.log(P[i_s[np.where(_s == s[i])]]))

        # Outcome Entropy
        Ho[i] = -np.sum([p_correct[i], p_incorrect[i]] *
                        np.log([p_correct[i], p_incorrect[i]]))

        # Update Q-values with reward prediction errors
        rpe = alpha * (r[i] - Q[i_s[np.where(_s == s[i])],
                                i_a[np.where(_a == a[i])]])
        Q[i_s[np.where(_s == s[i])], i_a[np.where(_a == a[i])]] += rpe

        # Reward prediction Error
        RPE[i] = rpe

        # Shannon Surprise on observed outcome (information carried by outcome)
        if r[i] == 1:
            shann_surp[i] = -np.log(p_correct[i])
        else:
            shann_surp[i] = -np.log(p_incorrect[i])

        # Bayesian Surprise (update of beliefs)
        Pu = softmax(Q, 1/beta, ax=1) # Updated Probabilities
        bayes_surp[i] = np.sum(P[i_s[np.where(_s == s[i])]] *
                       np.log(P[i_s[np.where(_s == s[i])]] /
                              Pu[i_s[np.where(_s == s[i])]]))

        # Log-likelihood given the observed stim-action pair
        log_likelihood += np.log(p_correct[i])

    # Create ditionary with regressors
    regs = {'log_likelihood': log_likelihood, 'p_correct': p_correct,
            'p_incorrect': p_incorrect, 'p_cor_uncho': p_cor_uncho,
            'dP': dP, 'dPn': dPn, 'H': H, 'Ho': Ho, 'rpe': RPE,
            'absrpe': np.abs(RPE), 'shann_surp': shann_surp,
            'bayes_surprise': bayes_surp}

    return regs


# def qlearning_meg_te(json_fname, sbj, ses):
#     # Performs Q-learning fit on behavioral data from the arbitrary visuomotor learning task
#
#     print('Subject = ' + sbj)
#     print('Session = ' + str(ses))
#
#     # Load directories json
#     raw_dir, prep_dir, trans_dir, mri_dir, src_dir, bem_dir, fwd_dir, hga_dir = read_directories(json_fname)
#
#     # Load behavioral xls file
#     fname_beh = prep_dir.format(sbj, ses) + '/{0}_task_events.xlsx'.format(sbj)
#     beh = load_beh(fname_beh)
#
#     if ses >= '2' or ses <= '5':
#
#         # Fit Q-learning model to behavioral data
#         regs = fit_qlearning(beh['stim'], beh['action'], beh['outcome'])
#
#         # Create dataframe
#         d = pd.DataFrame.from_dict(regs)
#
#         # Write xls file with Q-learning results and regressors
#         fname_qlearning = prep_dir.format(sbj, ses) + '/{0}_qlearning.xlsx'.format(sbj)
#         with pd.ExcelWriter(fname_qlearning) as writer:
#             d.to_excel(writer, sheet_name='Summary')
#
#     return


if __name__ == '__main__':
    s = np.random.randint(0, 1, 20)
    a = np.random.randint(102, 105, 20)
    r = np.random.randint(0, 2, 20)

    import time
    from joblib import Parallel, delayed
    start = time.time()

    res = fit_qlearning(s, a, r)
    # res = Parallel(n_jobs=-1)(delayed(fit_qlearning)(s, a, r))

    end = time.time()
    print(res, '\n Model fitted in', end - start, 'seconds')
