__author__ = 'zhuchao1'
import datetime
from dateutil.relativedelta import relativedelta
import quandl
import pandas as pd
import os.path
from scipy.optimize import minimize
import numpy as np
import math


def shift_months(cur_month, months):
    '''
    :param cur_month:
    :param months:
    :return: shift months to a datetime string
    '''
    d = datetime.datetime.strptime(cur_month, '%Y-%m-%d')
    return datetime.datetime.strftime(d + relativedelta(months=months), '%Y-%m-%d')


def data_crawler(work_directory, cur_month, months_back, months_forward):
    data = pd.read_csv('{0}.csv'.format(cur_month))
    data = data[data['Asset Class'] == 'Equity']
    data = data[data['Ticker'] != 'AA']
    data['Ticker'].replace({'BF.B': 'BF_B', 'BRKB': 'BRK_B'}, inplace=True)
    start_date = shift_months(cur_month, months_back)
    end_date = shift_months(cur_month, months_forward)
    if os.path.exists(work_directory + '/' + start_date + 'to' + end_date + 'based' + cur_month + '.csv') == True:
        data_frame = pd.read_csv(work_directory + '/' + start_date + 'to' + end_date + 'based' + cur_month + '.csv')
    else:
        query_stock_list = ['WIKI/{0}.11'.format(x) for x in data.Ticker]
        data_frame = quandl.get(query_stock_list, start_date=start_date, end_date=end_date)
        nlist = list(data_frame.columns.values)
        not_found_flag = 0
        for l in nlist:
            if l.split(' - ')[1] == 'Not Found':
                not_found_flag = 1
                print(l.split(' - ')[0], 'not found')
        if not_found_flag == 0:
            print('Completed!')

        data_frame.to_csv(work_directory + '/' + start_date + 'to' + end_date + 'based' + cur_month + '.csv',
                          index=False)
    return data_frame


def cw_weights(data):
    return [x / data['Notional Value'].sum() for x in data['Notional Value']]


def ew_weights(data):
    return [1 / data.shape[0]] * data.shape[0]


def minVar(x, cov_matrix):
    return np.array(x).T.dot(cov_matrix.dot(np.array(x)))


def jac_func(x, cov_matrix):
    return (cov_matrix.T + cov_matrix).dot(x)


def mv_weights(cov_matrix):
    n = cov_matrix.shape[0]
    start_pos = np.ones(n) * (1 / float(n))  # start from ew

    # Says one minus the sum of all variables must be zero
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})

    # Required to have non negative values
    bnds = tuple((0, 1) for x in start_pos)

    res = minimize(minVar, jac=jac_func, args=cov_matrix, x0=start_pos, method='SLSQP', tol=1e-8, bounds=bnds,
                   constraints=cons)
    return res.x


def maxmdp(x, cov_matrix, vol_vec):
    return np.array(x).T.dot(vol_vec) / math.sqrt(np.array(x).T.dot(cov_matrix.dot(np.array(x))))


def mdp_weights(cov_matrix, vol_vec):
    n = cov_matrix.shape[0]
    start_pos = np.ones(n) * (1 / float(n))  # start from ew

    # Says one minus the sum of all variables must be zero
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})

    # Required to have non negative values
    bnds = tuple((0, 1) for x in start_pos)

    res = minimize(maxmdp, args=(cov_matrix, vol_vec), x0=start_pos, method='SLSQP', tol=1e-8, bounds=bnds,
                   constraints=cons)
    return res.x


def minERC(x, cov_matrix):
    x = np.array(x)
    N = cov_matrix.shape[0]
    c = cov_matrix.dot(x) / math.sqrt(x.T.dot(cov_matrix).dot(x))
    M = math.sqrt(x.T.dot(cov_matrix).dot(x)) / float(N)
    return (M - x * c).dot(M - x * c)


def erc_weights(cov_matrix):
    n = cov_matrix.shape[0]
    start_pos = np.ones(n) * (1 / float(n))  # start from ew

    # Says one minus the sum of all variables must be zero
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})

    # Required to have non negative values
    bnds = tuple((0, 1) for x in start_pos)

    res = minimize(minERC, args=cov_matrix, x0=start_pos, method='SLSQP', tol=1e-50, bounds=bnds,
                   constraints=cons)
    return res.x


# a set of measurements

def sigmax(x, cov_matrix):
    x = np.array(x)
    return math.sqrt(x.T.dot(cov_matrix).dot(x))


def sigmax_b(x, b, cov_matrix):
    x_b = (np.array(x) - np.array(b))
    return sigmax(x_b, cov_matrix)


def ratio(x, b, cov_matrix):
    return 1 - sigmax(x, cov_matrix) / sigmax(b, cov_matrix)


def variance(x, b, cov_matrix):
    return np.array(x).T.dot(cov_matrix).dot(np.array(b))


def beta(x, b, cov_matrix):
    return variance(x, b, cov_matrix) / variance(x, x, cov_matrix)


def N_x(x):
    return 1 / np.array(x).dot(np.array(x))


def N_rc(x, cov_matrix):
    x = np.array(x)
    c = cov_matrix.dot(x) / math.sqrt(x.T.dot(cov_matrix).dot(x))
    x1 = np.array([x[i] * c[i] for i in range(len(x))])
    return 1 / x1.dot(x1) * (x.dot(cov_matrix).dot(x))
