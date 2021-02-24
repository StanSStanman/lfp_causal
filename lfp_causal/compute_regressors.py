import pandas as pd
import os
import os.path as op
import numpy as np
from scipy import special, stats
from lfp_causal.qlearning import fit_qlearning
# from lfp_causal.IO import read_xls


def get_actions(fname):
    csv = pd.read_csv(fname)
    actions = csv['button'].values.copy()
    return actions


def get_outcomes(fname):
    csv = pd.read_csv(fname)
    outcomes = csv['reward'].values.copy()
    return outcomes


def get_behaviour(monkey, condition, session, save_as=None):
    fname_csv = '/media/jerry/TOSHIBA EXT/data/db_behaviour/lfp_causal/' \
                '{0}/{1}/t_events/{2}.csv'.format(monkey, condition, session)
    fname_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
                 '{0}/{1}/files_info.xlsx'.format(monkey, condition)

    actions = get_actions(fname_csv)
    outcomes = get_outcomes(fname_csv)

    infos = pd.read_excel(fname_info.format(monkey, condition),
                          dtype={'file': str, 'sector': str})
    correct_pos = infos[infos['file'] == session]['target_location'].values[0]
    if correct_pos == 'left':
        correct_pos = 104.
    elif correct_pos == 'center':
        correct_pos = 103.
    elif correct_pos == 'right':
        correct_pos = 102.

    data = dict()
    cols = ['Condition', 'Correct', 'Reward',
            'is_R|C', 'is_nR|C', 'is_R|nC', 'is_nR|nC',
            'RnR|C', 'RnR|nC',
            '#R', '#nR', '#R|C', '#nR|C', '#R|nC', '#nR|nC',
            'learn_5t', 'learn_2t', 'early_late_cons',
            'P(R|C)', 'P(R|nC)', 'P(R|Cho)', 'P(R|A)',
            'dP', 'log_dP', 'delta_dP',
            'surprise', 'surprise_bayes', 'act_surp_bayes', 'rpe',
            'q_pcorr', 'q_pincorr', 'q_dP',
            'q_entropy', 'q_rpe', 'q_absrpe',
            'q_shann_surp', 'q_bayes_surp']

    # -------------------------------------------------------------------------
    if condition == 'easy':
        _cond = 0
    elif condition == 'hard':
        _cond = 1
    data['Condition'] = np.full(actions.shape, _cond)

    correct_act = actions.copy()
    correct_act[correct_act != correct_pos] = 0
    correct_act[correct_act == correct_pos] = 1
    data['Correct'] = correct_act.copy()

    data['Reward'] = outcomes.copy()

    # -------------------------------------------------------------------------
    _is_RC = np.logical_and(outcomes == 1, correct_act == 1)
    _is_nRC = np.logical_and(outcomes == 0, correct_act == 1)
    _is_RnC = np.logical_and(outcomes == 1, correct_act == 0)
    _is_nRnC = np.logical_and(outcomes == 0, correct_act == 0)

    data['is_R|C'] = _is_RC.astype(int)
    data['is_nR|C'] = _is_nRC.astype(int)
    data['is_R|nC'] = _is_RnC.astype(int)
    data['is_nR|nC'] = _is_nRnC.astype(int)
    # Group reward /no reward considering one action
    data['RnR|C'] = reward_by_action(data['Reward'], data['Correct'], 1)
    data['RnR|nC'] = reward_by_action(data['Reward'], data['Correct'], 0)

    # -------------------------------------------------------------------------
    # Number of win / lose
    data['#R'] = bincumsum(data['Reward'] == 1)
    data['#nR'] = bincumsum(data['Reward'] == 0)
    # Get the cumulative sum for each condition
    data['#R|C'] = bincumsum(data['is_R|C'])
    data['#nR|C'] = bincumsum(data['is_nR|C'])
    data['#R|nC'] = bincumsum(data['is_R|nC'])
    data['#nR|nC'] = bincumsum(data['is_nR|nC'])

    # -------------------------------------------------------------------------
    data['learn_5t'] = learning_phases(correct_act, [5])
    data['learn_2t'] = learning_phases(correct_act, [3])
    data['early_late_cons'] = learning_phases(correct_act, [3, 7])

    # -------------------------------------------------------------------------
    # Compute P(O|A) and P(O|nA) (Mean of beta distribution)
    # data['P(R|C)'] = data['#R|C'] / (data['#R|C'] + data['#nR|C'])
    # data['P(R|nC)'] = data['#R|nC'] / (data['#R|nC'] + data['#nR|nC'])

    # Compute P(O|A) and P(O|nA) (Mode of beta distribution)
    data['P(R|C)'] = (data['#R|C'] - 1) / ((data['#R|C'] + data['#nR|C']) - 2)
    data['P(R|nC)'] = (data['#R|nC'] - 1) / \
                      ((data['#R|nC'] + data['#nR|nC']) - 2)
    data['P(R|Cho)'] = beh_prob_cho(data['P(R|C)'], data['P(R|nC)'],
                                    data['Correct'])
    data['P(R|A)'] = prob_rew_act(actions, data['Reward'])

    # -------------------------------------------------------------------------
    # Contingency
    data['dP'], data['log_dP'], data['delta_dP'] = beh_contingency(
        data['P(R|C)'], data['P(R|nC)'])

    # -------------------------------------------------------------------------
    # Surprise
    data['surprise'] = beh_surprise(data['P(R|C)'], data['P(R|nC)'],
                                    _is_RC, _is_RnC, _is_nRC, _is_nRnC)

    # Bayesian surprise
    data['surprise_bayes'] = beh_bayes_surprise(data['#R'], data['#nR'])
    data['act_surp_bayes'] = beh_bayes_surp_act(actions, data['Reward'])

    data['rpe'] = np.diff(np.r_[0.5, data['P(R|C)']])

    # -------------------------------------------------------------------------
    # Q learning model values
    model = fit_qlearning(np.zeros_like(actions), actions, outcomes,
                          [0], [102, 103, 104])

    data['q_pcorr'] = model['p_correct']
    data['q_pincorr'] = model['p_incorrect']
    data['q_dP'] = model['dP']
    data['q_entropy'] = model['H']
    data['q_rpe'] = model['rpe']
    data['q_absrpe'] = model['absrpe']
    data['q_shann_surp'] = model['shann_surp']
    data['q_bayes_surp'] = model['bayes_surprise']

    # -------------------------------------------------------------------------
    df = pd.DataFrame(data, columns=cols)
    # data = xr.DataArray(data, coords=[sessions, range(data.shape[1])],
    #                     dims=['sessions', 'trials'])
    if isinstance(save_as, str):
        with pd.ExcelWriter(save_as) as writer:
            df.to_excel(writer, sheet_name='Session %s' % session)

    return df


###############################################################################
###############################################################################
def reward_by_action(reward, action, act_value):
    """Take only the action related reward values

    The other values will be nans

    Parameters
    ----------
    reward : array_like
        Vector array of the reward values
    action : array_like
        Vector array of the action values
    act_value: int
        Value of the considered action (0 = nC, 1 = C)

    Returns
    -------
    _r : array_like
        Vector array of action related reward values
    """
    _r = reward.copy()
    _r[action != act_value] = np.nan
    return _r


def bincumsum(x, prior=1.1):
    """Cumulative sum for entries of 0 and 1.

    This function uses np.cumsum but force the first value to be 1.

    Parameters
    ----------
    x : array_like
        Vector array filled with 0 and 1 of shape (n_trials,)
    prior : float | 1.1
        Prior to use

    Returns
    -------
    cumsum : array_like
        Cumulative sum of shape (n_trials)
    """
    x = np.asarray(x, dtype=int).copy()
    is_valid = (0 <= x.min() <= 1) and (0 <= x.max() <= 1)
    assert is_valid, "x should only contains 0 and 1"
    # x[0] = 1  # ensure first element is 1
    return np.cumsum(x) + prior


def learning_phases(correct_actions, nt=[5]):

    repetitions = np.empty_like(correct_actions)
    _x = 0
    for i, x in enumerate(correct_actions):
        if x == 1:
            repetitions[i] = x + _x
            _x += x
        elif x == 0:
            repetitions[i] = 0
            _x = 0

    ranges = [0] + [nt] + [len(correct_actions)]
    for i, t in enumerate(nt):
        if i == 0:
            repetitions[repetitions <= t] = i
        elif i != 0:
            repetitions[np.logical_and(repetitions > nt[i - 1],
                                       repetitions <= t)] = i
    repetitions[repetitions > nt[-1]] = len(nt)
    return repetitions


def prob_rew_act(actions, rewards):
    """Action dependent probability

    Parameters
    ----------
    actions : array_like
        Array of the actions
    rewards : array_like
        Array of the reward

    Returns
    -------
    probs : array_like
        Array of the probability of receive a reward given an action

    """
    unique, indices = np.unique(actions, return_inverse=True)
    probs = np.zeros_like(rewards, dtype=float)
    for i, u in enumerate(unique):
        is_r = rewards[actions == u]
        is_nr = 1 - is_r
        _r, _nr = bincumsum(is_r), bincumsum(is_nr)
        p = (_r - 1) / ((_r + _nr) - 2)
        probs[indices == i] = p
    # probs = np.hstack(tuple(probs))
    # probs = probs[indices]
    return probs


def beh_prob_cho(p_oa, p_ona, actions):
    """Choice dependent probabily

    Parameters
    ----------
    p_oa : array_like
        Probability of winning when playing
    p_ona : array_like
        Probability of winning when not playing
    actions : array_like
        int array of [0, 1], where 0 = not play; 1 = play

    Returns
    -------
    prob_cho : array_like
        Action based probability

    """
    prob_cho = np.zeros_like(p_oa, dtype=float)
    prob_cho[actions == 1] = p_oa[actions == 1]
    prob_cho[actions == 0] = p_ona[actions == 0]
    return prob_cho


def beh_contingency(p_oa, p_ona):
    """Behavioral contingency.

    Definition :

        * dp = P(O|A) - P(O|nA)
        * log_dp = log(P(O|A)) - log(P(O|nA))
        * delta_dp = dp[t] - dp[t-1]

    Parameters
    ----------
    p_oa : array_like
        Probability of winning when playing
    p_ona : array_like
        Probability of winning when not playing

    Returns
    -------
    dp : array_like
        The contingency
    log_dp : array_like
        The log contingency
    delta_dp : array_like
        The updated contingency
    """
    dp = p_oa - p_ona
    log_dp = np.log(p_oa) - np.log(p_ona)
    delta_dp = np.diff(np.r_[0, dp])
    return dp, log_dp, delta_dp


def beh_surprise(p_wp, p_wnp, is_wp, is_wnp, is_nwp, is_nwnp):
    """Behavioral surprise.

    Definition :

        * surprise[W|P] = -log(P(W|P))
        * surprise[nW|P] = -log(1 - P(W|P))
        * surprise[W|nP] = -log(P(W|nP))
        * surprise[nW|nP] = -log(1 - P(W|nP))

    Parameters
    ----------
    p_wp, p_wnp : array_like
        Probability to win given that the subject is playing (p_wp) or not
        playing (p_wnp)
    is_wp, is_wnp, is_nwp, is_nwnp : array_like
        Indices of the different conditions

    Returns
    -------
    surprise : array_like
        Behavioral surprise
    """
    surprise = np.zeros_like(p_wp, dtype=float)
    # Play
    surprise[is_wp] = -np.log(p_wp[is_wp])
    surprise[is_nwp] = -np.log(1 - p_wp[is_nwp])
    # Not Play
    surprise[is_wnp] = -np.log(p_wnp[is_wnp])
    surprise[is_nwnp] = -np.log(1 - p_wnp[is_nwnp])
    return surprise


def beh_bayes_surprise(n_win, n_lose):
    """Behavioral Bayesian surprise.

    The bayesian surprise is defined as the distance between prior and
    posterior beliefs [1]_

    Definition :

        * prior = beta(n_win[t-1], n_lose[t-1])
        * posterior = beta(n_win[t], n_lose[t])
        * surprise = KLD(prior, posterior)

    Where `KLD` is the kullback-Leibler Divergence

    Parameters
    ----------
    n_win : array_like
        Cumulative number of win
    n_lose : array_like
        Cumulative number of lose

    Returns
    -------
    surprise : array_like
        Bayesian surprise

    References
    ----------
    .. [1] Itti, L., and Baldi, P. F. (2006). Bayesian surprise attracts human
       attention. in Advances in neural information processing systems,
       547–554.
    """
    surprise = np.zeros_like(n_win, dtype=float)
    for num, (w, l) in enumerate(zip(n_win, n_lose)):
        if num == 0:
            _prior = (1.1, 1.1)
        else:
            _prior = (n_win[num - 1], n_lose[num - 1])
        prior = pdf(_prior[0], _prior[1])
        posterior = pdf(w, l)
        surprise[num] = stats.entropy(prior, posterior, base=10)
    return surprise


def beh_bayes_surp_act(actions, rewards):
    unique, indices = np.unique(actions, return_inverse=True)
    act_kl_dist = np.zeros_like(rewards, dtype=float)
    for i, u in enumerate(unique):
        is_r = rewards[actions == u]
        is_nr = 1 - is_r
        _r, _nr = bincumsum(is_r), bincumsum(is_nr)
        klds = np.zeros_like(_r, dtype=float)
        for ti, (tr, tnr) in enumerate(zip(_r, _nr)):
            if ti == 0:
                _prior = (1.1, 1.1)
            else:
                _prior = (_r[ti - 1], _nr[ti - 1])
            prior = pdf(*_prior)
            posterior = pdf(tr, tnr)
            klds[ti] = stats.entropy(prior, posterior, base=10)
        act_kl_dist[indices == i] = klds
    return act_kl_dist


def pdf(a, b, n_pts=1000):
    """Generate a beta distribution from float two inputs.

    Parameters
    ----------
    a : int | float
        Alpha parameter
    b : int | float
        Beta parameter
    n_pts : int | 100
        Number of points composing the beta distribution. Alternatively, you
        can also gives your own vector.

    Returns
    -------
    z : array_like
        The beta distribution of shape (n_pts,)
    """
    a, b = float(a), float(b)
    # Function to generate a beta distribution with a fixed number of steps
    if isinstance(n_pts, int):
        n_pts = np.linspace(0., 1., n_pts, endpoint=True)
    z = special.beta(a, b)
    return (n_pts ** (a - 1.) * ((1. - n_pts) ** (b - 1.)) / z)


if __name__ == '__main__':
    from lfp_causal import MCH, PRJ
    from research.get_dirs import get_dirs
    dirs = get_dirs(MCH, PRJ)

    monkey = 'freddie'
    condition = 'easy'

    print('Calculating regressors for %s, %s' % (monkey, condition))

    csv_dir = dirs['tev'].format(monkey, condition)

    for file in os.listdir(csv_dir):
        # file = '0816.csv'
        if file.endswith('.csv'):
            session = file.replace('.csv', '')
            beh_dir = dirs['reg'].format(monkey, condition)
            os.makedirs(beh_dir, exist_ok=True)
            fname_beh = op.join(beh_dir, '{0}.xlsx'.format(session))
            print('Processing session %s' % session)
            get_behaviour(monkey, condition, session, save_as=fname_beh)
