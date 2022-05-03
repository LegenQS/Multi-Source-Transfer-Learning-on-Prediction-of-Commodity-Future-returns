import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import wasserstein_distance
import sys

img_path = './img'
csv_path = './csv'
corr_type = 'WD'
corr_threshold = 0

if len(sys.argv) >= 3:
    try:
        data_path = sys.argv[1]
        commodity = sys.argv[2]
        if not os.path.exists(data_path):
            print('data_path not exists!')
            sys.exit()

        img_path = sys.argv[3]
        csv_path = sys.argv[4]
        corr_type = sys.argv[5]
        corr_threshold = int(sys.argv[6])
    except:
        print('values not provided will be set as default values')
else:
    print('you should at least provide data_path and commodity to execute the code!')
    sys.exit()

def pearson_filter(score, rate=0.1, commodity=''):
    column = score.columns
    new_score = score.to_numpy()
    for i in range(new_score.shape[0]):
        if column[i] != commodity:
            continue
        new_column = column[i]
        new_index = []
        data = []
        for j in range(new_score.shape[1]):
            if j != i:
                # print(new_score[i][j])
                if new_score[i][j] >= rate:
                # if float(new_score[i][j].split(',')[0].split(':')[1] ) > rate:
                    # print('{} - {}: {}'.format(column[i], column[j], new_score[i][j]))
                    new_index.append(column[j])
                    data.append(new_score[i][j])
        break
        # if len(new_index) > 2:
        #     result = pd.DataFrame(data, columns=[new_column])
        #     result.index = new_index
        #     # result.columns.name = [new_column]
        #     print(result.sort_values(by=[new_column], ascending=False))
        #     print('---------')
    result = pd.DataFrame(data, columns=[new_column])
    result.index = new_index
    # result.columns.name = [new_column]
    result = result.sort_values(by=[new_column], ascending=False)

    return result, new_index

def pearson_close(start_date='', end_date='', sign=False, save_to_file=False):
    combine = pd.DataFrame()

    for file in os.listdir(data_path):
        # new = pd.DataFrame(pd.read_csv(cur_dir + '/' + file)['Close'])
        new = pd.DataFrame(pd.read_csv(data_path + '/' + file)[['Date', 'Close']])
        new.rename(columns={'Close': file.split('.')[0]}, inplace = True)
        # combine = pd.concat([combine, new], axis=1)
        if start_date:
            new = new[new.Date >= start_date]
        if end_date:
            new = new[new.Date <= end_date]

        if combine.empty:
          combine = new
        else: 
          combine = combine.merge(new, how='outer', on='Date')

    combine = combine.fillna(0)
    combine = combine.sort_values(by=['Date'])
    
    if sign:
        return combine

    # overall_pearson_r = combine.corr()
    labels = list(combine.columns)

    score = pd.DataFrame(index=labels[1:], columns=labels[1:])
    for i in range(1, len(labels)):
        for j in range(1, len(labels)):
            if j != i:
                r, p = stats.pearsonr(combine[labels[i]], combine[labels[j]])
                # score[labels[i]][labels[j]] = "r: %.3f, p: %.3f" % (r, p)
                score[labels[j]][labels[i]] = float(r)

    if save_to_file:
        score.to_csv(csv_path + '/pearson_close.csv')
    return score

def pearson_return(start_date='', end_date='', sign=False, save_to_file=False):
    def cal_return(data, name):
        data.loc[0, name] = 0
        for i in range(1, data.shape[0]):
            if data['Close'][i] == 0 or data['Close'][i-1] == 0:
                data.loc[i, name] = 0
            else:
                data.loc[i, name] = ('%.2f' % ((data['Close'][i] - data['Close'][i-1]) / data['Close'][i] * 100) ) 
        
        data[name] = data[name].astype(float)
        return data[['Date', name]]

    combine = pd.DataFrame()

    for file in os.listdir(data_path):
        # new = pd.DataFrame(pd.read_csv(cur_dir + '/' + file)['Close'])
        new = pd.DataFrame(pd.read_csv(data_path + '/' + file)[['Date', 'Close']])
        
        if start_date:
            new = new[new.Date >= start_date]
        if end_date:
            new = new[new.Date <= end_date]
        # combine = pd.concat([combine, new], axis=1)

        mynew = cal_return(new, file.split('.')[0])
        if combine.empty:
            combine = mynew
        else: 
            combine = combine.merge(mynew, how='outer', on='Date')

    combine = combine.fillna(0)
    combine = combine.sort_values(by=['Date'])
    combine.index = np.arange(0, combine.shape[0])

    if sign:
        return combine
    # overall_pearson_r = combine.corr()
    labels = list(combine.columns)

    score = pd.DataFrame(index=labels[1:], columns=labels[1:])
    for i in range(1, len(labels)):
        for j in range(1, len(labels)):
            if j != i:
                r, p = stats.pearsonr(combine[labels[i]], combine[labels[j]])
                score[labels[j]][labels[i]] = float(r)

    if save_to_file:
        score.to_csv(csv_path + '/pearson_return.csv')
    return score

"""#### Result (`Close Price`)"""
print('------------------------------------------------------------------')
P_score_close = pearson_close(start_date='2000-01-01', end_date='2019-01-01')
pearson_close_result, P_close_cor_list = pearson_filter(P_score_close, rate=corr_threshold, commodity=commodity)
print('Pearson correlation result of {} under close price'.format(commodity))
print(pearson_close_result)

"""#### Result (`Return`)"""

P_score_return = pearson_return(start_date='2000-01-01', end_date='2019-01-01')
pearson_return_result, P_return_cor_list = pearson_filter(P_score_return, rate=corr_threshold, commodity=commodity)
print('------------------------------------------------------------------')
print('Pearson correlation result of {} under return'.format(commodity))
print(pearson_return_result)
print('------------------------------------------------------------------')

"""### 4.2 Wasserstein Distances Correlation

#### Package
"""

def WD_filter(score, rate=0.5, commodity='Crude Oil'):
    column = score.columns
    new_score = score.to_numpy()
    for i in range(new_score.shape[0]):
        if column[i] != commodity:
            continue
        new_column = column[i]
        new_index = []
        data = []
        for j in range(new_score.shape[1]):
            if j != i:
                # print(new_score[i][j])
                if new_score[i][j] >= rate:
                # if float(new_score[i][j].split(',')[0].split(':')[1] ) > rate:
                    # print('{} - {}: {}'.format(column[i], column[j], new_score[i][j]))
                    new_index.append(column[j])
                    data.append(new_score[i][j])
        break
        # if len(new_index) > 2:
        #     result = pd.DataFrame(data, columns=[new_column])
        #     result.index = new_index
        #     # result.columns.name = [new_column]
        #     print(result.sort_values(by=[new_column], ascending=False))
        #     print('---------')
    result = pd.DataFrame(data, columns=[new_column])
    result.index = new_index
    # result.columns.name = [new_column]
    result = result.sort_values(by=[new_column], ascending=False)
    # print(result.sort_values(by=[new_column], ascending=False))
    return result, new_index

def WD_close(start_date="", end_date="", sign=False, save_to_file=False):
    combine = pd.DataFrame()

    for file in os.listdir(data_path):
        # new = pd.DataFrame(pd.read_csv(cur_dir + '/' + file)['Close'])
        new = pd.DataFrame(pd.read_csv(data_path + '/' + file)[['Date', 'Close']])
        if start_date:
            new = new[new.Date >= start_date]
        if end_date:
            new = new[new.Date <= end_date]
        
        new_name = file.split('.')[0]
        new.rename(columns={'Close': new_name}, inplace = True)
        # Scaling to a range
        # new[new_name] = (new[new_name] - new[new_name].min()) / (new[new_name].max() - new[new_name].min())

        if combine.empty:
            combine = new
        else: 
            combine = combine.merge(new, how='outer', on='Date')
    
    combine = combine.fillna(0)
    combine = combine.sort_values(by=['Date'])
    combine.index = np.arange(0, combine.shape[0])

    if sign:
        return combine
    # overall_pearson_r = combine.corr()
    labels = list(combine.columns)

    # # normalize
    # column = list(combine.columns)
    # column.remove('Date')
    # combine[column] /= np.max(combine[column].to_numpy())

    score = pd.DataFrame(index=labels[1:], columns=labels[1:])
    for i in range(1, len(labels)):
        for j in range(1, len(labels)):
            if j != i:
                score[labels[j]][labels[i]] = wasserstein_distance(combine[labels[i]], combine[labels[j]])

    # normalization
    score /= max(score.max())
    score = 1 - score

    if not os.path.exists(csv_path):
        os.mkdir(csv_path)

    if save_to_file:
        score.to_csv(csv_path + '/WD_close.csv')
    return score

def WD_return(start_date='', end_date='', sign=False, save_to_file=False):
    def cal_return(data, name):
        data.loc[0, name] = 0
        for i in range(1, data.shape[0]):
            if data['Close'][i] == 0 or data['Close'][i-1] == 0:
                data.loc[i, name] = 0
            else:
                data.loc[i, name] = ('%.2f' % ((data['Close'][i] - data['Close'][i-1]) / data['Close'][i] * 100) ) 
        
        data[name] = data[name].astype(float)
        return data[['Date', name]]

    combine = pd.DataFrame()

    for file in os.listdir(data_path):
        # new = pd.DataFrame(pd.read_csv(cur_dir + '/' + file)['Close'])
        new = pd.DataFrame(pd.read_csv(data_path + '/' + file)[['Date', 'Close']])
        
        if start_date:
            new = new[new.Date >= start_date]
        if end_date:
            new = new[new.Date <= end_date]
        # combine = pd.concat([combine, new], axis=1)

        mynew = cal_return(new, file.split('.')[0])
        if combine.empty:
            combine = mynew
        else: 
            combine = combine.merge(mynew, how='outer', on='Date')

    combine = combine.fillna(0)
    combine = combine.sort_values(by=['Date'])
    combine.index = np.arange(0, combine.shape[0])

    if sign:
        return combine
    # overall_pearson_r = combine.corr()
    labels = list(combine.columns)

    score = pd.DataFrame(index=labels[1:], columns=labels[1:])
    for i in range(1, len(labels)):
        for j in range(1, len(labels)):
            if j != i:
                score[labels[i]][labels[j]] = wasserstein_distance(combine[labels[i]], combine[labels[j]])

    # normalization
    score /= max(score.max())
    score = 1 - score

    if not os.path.exists(csv_path):
        os.mkdir(csv_path)

    if save_to_file:
        score.to_csv(csv_path + '/WD_return.csv')
    return score

def img_plot(result, cur_path, mytype=''):
    x = np.arange(len(result))
    bar_width = 0.45
    plt.figure(figsize=(26,10))
    plt.bar(x - bar_width / 2, result['WD'], width=bar_width, label='WD')
    plt.bar(x + bar_width / 2, result['Pearson'], width=bar_width, label='Pearson')
    
    plt.xticks([i for i in range(len(result))],np.array((result.index)), rotation='vertical', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=15)
    plt.ylabel('WD/Pearson Correlation Score', fontsize=12)
    
    for i in range(len(result['WD'])):
        if result['WD'][i] > 0:
            plt.text(i - bar_width, 1.01 * result['WD'][i], ('%.3f' % result['WD'][i]), fontsize=12)
    
    for i in range(len(result['Pearson'])):
        if result['Pearson'][i] > 0:
            plt.text(i, 1.01 * result['Pearson'][i], ('%.3f' % result['Pearson'][i]), fontsize=12)
    plt.savefig(cur_path + '/' + 'display_{}.png'.format(mytype))
    plt.close()

"""#### Result of `Close Price` and `Return`


"""

WD_score_close = WD_close(start_date='2000-01-01', end_date='2019-01-01')
WD_score_return = WD_return(start_date='2000-01-01', end_date='2019-01-01')


WD_close_result, WD_close_cor_list = WD_filter(WD_score_close, rate=corr_threshold, commodity=commodity)
WD_return_result, WD_return_cor_list = WD_filter(WD_score_return, rate=corr_threshold, commodity=commodity)
print('Wasserstein distance correlation result of {} under return'.format(commodity))
print(WD_close_result)
print('------------------------------------------------------------------')
print('Wasserstein distance correlation result of {} under return'.format(commodity))
print(WD_return_result)

def test_close_return(commodity=None, start_date='', end_date=''):
    WD_score_close = WD_close(start_date=start_date, end_date=end_date, save_to_file=True)
    WD_score_return = WD_return(start_date=start_date, end_date=end_date, save_to_file=True)
    
    if not commodity:
        mydata_path = os.listdir(data_path)
    else:
        mydata_path = [commodity]

    for file_path in mydata_path:
        files = file_path.split('.')[0]
        pearson_close_result, P_close_cor_list = pearson_filter(P_score_close, rate=corr_threshold, commodity=files)
        pearson_return_result, P_return_cor_list = pearson_filter(P_score_return, rate=corr_threshold, commodity=files)
        WD_close_result, WD_close_cor_list = WD_filter(WD_score_close, rate=corr_threshold, commodity=files)
        WD_return_result, WD_return_cor_list = WD_filter(WD_score_return, rate=corr_threshold, commodity=files)

        cur_img_path = img_path + '/{}'.format(files)
        cur_csv_path = csv_path + '/{}'.format(files)
        if not os.path.exists(cur_img_path):
            os.makedirs(cur_img_path)
        if not os.path.exists(cur_csv_path):
            os.makedirs(cur_csv_path)

        # close result
        result = pearson_close_result.T.merge(WD_close_result.T, how='outer').T
        result.columns = ['Pearson', 'WD']
        result.to_csv(cur_csv_path + '/compare_close.csv')
        result = result.fillna(0)
        img_plot(result, cur_img_path, mytype='close')

        # return result
        result = pearson_return_result.T.merge(WD_return_result.T, how='outer').T
        result.columns = ['Pearson', 'WD']
        result.to_csv(cur_csv_path + '/compare_return.csv')
        result = result.fillna(0)
        img_plot(result, cur_img_path, mytype='return')

test_close_return(start_date='2000-01-01', end_date='2019-01-01')
"""### 4.3 DTW Correlation (Calculation too slow, ignored)"""

# def DTW_filter(score, rate=0.5):
#     column = score.columns
#     new_score = score.to_numpy()
#     for i in range(new_score.shape[0]):
#       new_column = column[i]
#       new_index = []
#       data = []
#       for j in range(i+1, new_score.shape[1]):
#         if new_score[i][j] > rate:
#             new_index.append(column[j])
#             data.append(new_score[i][j])
#       if len(new_index) > 2:
#         result = pd.DataFrame(data, columns=[new_column])
#         result.index = new_index
#         # result.columns.name = [new_column]
#         print(result)
#         print('---------')

# def DTW_close(start_date="", end_date="", sign=False):
#     combine = pd.DataFrame()

#     for file in os.listdir(cur_dir):
#         # new = pd.DataFrame(pd.read_csv(cur_dir + '/' + file)['Close'])
#         new = pd.DataFrame(pd.read_csv(cur_dir + '/' + file)[['Date', 'Close']])
#         if start_date:
#             new = new[new.Date >= start_date]
#         if end_date:
#             new = new[new.Date <= end_date]
#         new.rename(columns={'Close': file.split('.')[0]}, inplace = True)
#         new[file.split('.')[0]] = new[file.split('.')[0]].astype(float)
#         # combine = pd.concat([combine, new], axis=1)
#         if combine.empty:
#           combine = new
#         else: 
#           combine = combine.merge(new, how='outer', on='Date')

#     combine = combine.fillna(0)
#     combine = combine.sort_values(by=['Date'])
#     combine.index = np.arange(0, combine.shape[0])
#     manhattan_distance = lambda x, y: np.abs(x - y)

#     if sign:
#         return combine
#     # overall_pearson_r = combine.corr()
#     labels = list(combine.columns)

#     score = pd.DataFrame(index=labels[1:], columns=labels[1:])
#     for i in range(1, len(labels)):
#         for j in range(1, len(labels)):
#             if j != i:
#                 d, cost_matrix, acc_cost_matrix, path = dtw(combine[labels[i]], combine[labels[j]], dist=manhattan_distance)
#                 score[labels[j]][labels[i]] = d

#     score = (score - score.min()) / (score.max() - score.min())
#     score.to_csv('/content/drive/My Drive/dl/DTW_close.csv')
#     return score

# score = DTW_close(cur_dir)

# DTW_filter(score)