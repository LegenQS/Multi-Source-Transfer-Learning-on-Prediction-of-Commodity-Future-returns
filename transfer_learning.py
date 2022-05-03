import pandas as pd
import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
sns.set_theme()
from datetime import datetime
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,BatchNormalization, CuDNNLSTM,Activation
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import talib as ta

img_path = './img'
csv_path = './csv'
corr_type = 'WD'
repetition = 5
retrained = False
base_repetition = 20
ntl_repetition = 10
label = 'return'

if len(sys.argv) >= 4:
    try:
        data_path = sys.argv[1]
        model_path  = sys.argv[2]
        commodity = sys.argv[3]
        
        if not os.path.exists(data_path):
            print('data_path not exists!')
            sys.exit()

        label = sys.argv[4]
        img_path = sys.argv[5]
        csv_path = sys.argv[6]
        corr_type = sys.argv[7]
        repetition = int(sys.argv[8])
        retrained = bool(int(sys.argv[9]))
        base_repetition = int(sys.argv[10])
        ntl_repetition = int(sys.argv[11])
    except:
        print('values not provided will be set as default values')
else:
    print('you should at least provide four parameters to execute the code!')
    sys.exit()

def load_data(data_path='', label = 'return'):
    try:
        dataset = pd.read_csv(data_path)
    except:
        print('commodity format is not correct!')
        sys.exit()

    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset['return'] = (dataset['Close']/dataset['Close'].shift(1)) - 1
    #predict price or return?
    dataset['y'] = dataset[label].shift(-1)
    #drop na
    dataset = dataset.dropna().reset_index(drop=True)
    dataset = dataset.drop(columns = 'Adj Close')
    dataset['weekday'] = dataset['Date'].apply(lambda x: x.weekday())
    dataset = dataset.rename(columns={'Date':'date', 'Open': 'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume': 'volume', })
    
    return dataset

def produce_factors(dataset):
    # high low ratio
    for n in range(1, 10):
        highlow_name = 'high_low_' + str(n+1) + 'd'
        dataset['temp_high_max'] = dataset['high'].transform( lambda x: pd.DataFrame([x.shift(i) for i in range(n)]).max(axis=0) )
        dataset['temp_low_min'] = dataset['low'].transform( lambda x: pd.DataFrame([x.shift(i) for i in range(n)]).min(axis=0) )
        dataset[highlow_name] = dataset['temp_high_max'] / dataset['temp_low_min']
    
    dataset['temp_high_max'] = dataset['high'].transform( lambda x: pd.DataFrame([x.shift(i) for i in range(15)]).max(axis=0) )
    dataset['temp_low_min'] = dataset['low'].transform( lambda x: pd.DataFrame([x.shift(i) for i in range(15)]).min(axis=0) )
    dataset['high_low_3w'] = dataset['temp_high_max'] / dataset['temp_low_min']
    dataset['temp_high_max'] = dataset['high'].transform( lambda x: pd.DataFrame([x.shift(i) for i in range(20)]).max(axis=0) )
    dataset['temp_low_min'] = dataset['low'].transform( lambda x: pd.DataFrame([x.shift(i) for i in range(20)]).min(axis=0) )
    dataset['high_low_1m'] = dataset['temp_high_max'] / dataset['temp_low_min']
    dataset['temp_high_max'] = dataset['high'].transform( lambda x: pd.DataFrame([x.shift(i) for i in range(40)]).max(axis=0) )
    dataset['temp_low_min'] = dataset['low'].transform( lambda x: pd.DataFrame([x.shift(i) for i in range(40)]).min(axis=0) )
    dataset['high_low_2m'] = dataset['temp_high_max'] / dataset['temp_low_min']
    dataset = dataset.drop(columns = ['temp_high_max','temp_low_min'])

    #std
    dataset['std_3d'] = dataset['return'].transform(lambda x: pd.DataFrame([x.shift(i) for i in range(3)]).std() )
    dataset['std_1w'] = dataset['return'].transform(lambda x: pd.DataFrame([x.shift(i) for i in range(5)]).std() )
    dataset['std_2w'] = dataset['return'].transform(lambda x: pd.DataFrame([x.shift(i) for i in range(10)]).std() )
    dataset['std_3w'] = dataset['return'].transform(lambda x: pd.DataFrame([x.shift(i) for i in range(15)]).std() )
    dataset['std_1m'] = dataset['return'].transform(lambda x: pd.DataFrame([x.shift(i) for i in range(20)]).std() )
    dataset['std_2m'] = dataset['return'].transform(lambda x: pd.DataFrame([x.shift(i) for i in range(40)]).std() )

    #technical indicators
    dataset['psy_3d'] = dataset['return'].transform(lambda x: sum([ x.shift(i)>0 for i in range(3)]) / 3 )
    dataset['psy_1w'] = dataset['return'].transform(lambda x: sum([ x.shift(i)>0 for i in range(5)]) / 5 )
    dataset['psy_2w'] = dataset['return'].transform(lambda x: sum([ x.shift(i)>0 for i in range(10)]) / 10 )
    dataset['cross_2_5'] = dataset['close'].transform(lambda x: sum([x.shift(i) for i in range(2)])/2 - sum([x.shift(i) for i in range(5)])/5)

    # MA
    types=['DEMA','TEMA','TRIMA','KAMA','MAMA','T3']
    for i in range(len(types)):
        dataset[types[i]]=ta.MA(dataset.close,timeperiod=10,matype=i)
    
    dataset['H_line'], dataset['M_line'], dataset['L_line']=ta.BBANDS(dataset.close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    dataset['macd'], dataset['macdsignal'], dataset['macdhist'] = ta.MACD(dataset.close, fastperiod=12, slowperiod=26, signalperiod=9)


    return dataset

# split train, valid and test set
def y_to_3class(y_vec, b1, b2):
    for i in range(len(y_vec)):
        if y_vec[i] <= b1:
            y_vec[i] = 0
        elif y_vec[i] >= b2:
            y_vec[i] = 2
        else:
            y_vec[i] = 1
    return y_vec.astype(int)

def prepare_X(X_data, seq_length):
    # prepare training data for LSTM to train
    # reshape to (num_of_samples, seq_length, feature_dim)
    shape = (X_data.shape[0], seq_length, X_data.shape[1])
    pp_train = np.zeros(shape)

    for i in range(shape[0]):
        if i-seq_length>=0:
            pp_train[i] = X_data[i-seq_length:i, :]

    return pp_train[seq_length:, :, :]

def data_split(data, split_dates, seq_length, crude_oil_data, classify=False, b1=0.4, b2=0.6, start_date = '2015-01-01'):
    scaler = MinMaxScaler(feature_range=(-1, 1))

    var_list = list(data.columns)
    var_list.remove('y')
    var_list.remove('date')

    idx0 = data.loc[data['date']>=start_date].index[0]
    idx1 = data.loc[data['date']>=split_dates[0]].index[0]
    idx2 = data.loc[data['date']>=split_dates[1]].index[0]

    data[(var_list+['y'])] = scaler.fit_transform(crude_oil_data[(var_list+['y'])].values)

    trainset = data.iloc[idx0:idx1]
    validset = data.iloc[idx1-seq_length:idx2]
    testset = data.iloc[idx2-seq_length:]

    train_y = trainset['y'].values[seq_length:]
    train_X = trainset[var_list].values
    train_y_min, train_y_max = np.min(train_y), np.max(train_y)
    train_X_min, train_X_max = np.min(trainset[var_list].values, axis=0), np.max(trainset[var_list].values, axis=0)
    # train_y = (train_y - train_y_min) / (train_y_max - train_y_min)
    # train_X = (train_X - train_X_min) / (train_X_max - train_X_min)

    valid_y = validset['y'].values[seq_length:]
    valid_X = validset[var_list].values
    # valid_y = (valid_y - train_y_min) / (train_y_max - train_y_min)
    # valid_X = (valid_X - train_X_min) / (train_X_max - train_X_min)

    test_y = testset['y'].values[seq_length:]
    test_X = testset[var_list].values
    # test_y = (test_y - train_y_min) / (train_y_max - train_y_min)
    # test_X = (test_X - train_X_min) / (train_X_max - train_X_min)

    train_X, valid_X, test_X = prepare_X(train_X, seq_length), prepare_X(valid_X, seq_length), prepare_X(test_X, seq_length)

    if classify == True:
        train_y, valid_y, test_y = y_to_3class(train_y, b1, b2), y_to_3class(valid_y, b1, b2), y_to_3class(test_y, b1, b2)
    
    return train_y, train_X, valid_y, valid_X, scaler

"""### Model"""

def build_lstm_model(train_X, classify = False):
  
    model = Sequential()
    model.add(CuDNNLSTM(256, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    # model.add(Dropout(0.3))
    model.add(CuDNNLSTM(128, return_sequences=True, ))
    # model.add(CuDNNLSTM(64, return_sequences=True))
    # model.add(Dropout(0.3))
    model.add(CuDNNLSTM(32))
    model.add(Dense(16, activation = 'relu'))
    if classify:
      model.add(Dense(3, activation='softmax') ) #classification
    else:
      model.add(Dense(1)) # regression
    
    return model

def lr_schedule(epoch):

    lr = 1e-3
    if epoch > 30:
        lr *= 0.5e-3
    elif epoch > 20:
        lr *= 1e-3
    elif epoch > 10:
        lr *= 1e-2
    elif epoch > 5:
        lr *= 1e-1
    return lr    

lr_scheduler = LearningRateScheduler(lr_schedule)

def transform_to_real(y, feat_dim, scaler):
    temp = np.zeros((y.shape[0], feat_dim))
    inv_y_train = np.concatenate((temp,y), axis=1)
    inv_y_train = scaler.inverse_transform(inv_y_train)
    y_ = inv_y_train[:, -1]
    return y_

def get_open_high_low(data, split_dates):
    idx1 = data.loc[data['date']>=split_dates[0]].index[0]
    idx2 = data.loc[data['date']>=split_dates[1]].index[0]
    valid_low = data['low'].iloc[idx1+1:idx2+1]
    valid_high = data['high'].iloc[idx1+1:idx2+1]
    valid_open = data['open'].iloc[idx1+1:idx2+1]
    return valid_open.values, valid_high.values, valid_low.values

def backtest(y_pred, y_test, trd_cost, open, high, low, threshold):
    # positive pred -> buy
    # negative pred -> sell
    count = 0
    c = 0
    signal = np.zeros(y_pred.shape)
    for i in range(len(y_pred)):
        if y_pred[i] > 1e-7 :
            signal[i] = 1
            if y_test[i] > 0:
                count+=1
        elif y_pred[i] < -1e-7:
            signal[i] = -1
            if y_test[i] < 0:
                count+=1
        else:
            signal[i] = 0
            c += 1
    
    rets = signal * y_test

    values = np.zeros(y_pred.shape)
    
    for i in range(len(values)):
        if i==0:
            values[i] = 1+rets[i]
        elif i>=1 and signal[i] == signal[i-1]:
            values[i] = values[i-1] * (1+rets[i])
        elif i>=1 and signal[i] != signal[i-1]:
            values[i] = values[i-1] * (1+rets[i])*(1-trd_cost)
        # loss limit
        if signal[i] == 1:
            if (low[i]-open[i])/open[i] <= -threshold:
                values[i] = values[i-1] * (1-threshold) * (1-trd_cost)
        if signal[i] == -1:
            if (high[i]-open[i])/open[i] >= threshold:
                values[i] = values[i-1] * (1-threshold) * (1-trd_cost)

    plt.plot(np.arange(len(y_test))+1, values)
    # plt.savefig(img_path + '/' + fig_name)
    # print('signal0: ', c)
    acc = count/(len(y_test)-c)
    return rets, signal, values, acc

def no_TL(commodity):
    #without transfer learning
    path = data_path + '/' + commodity + '.csv'
    label = 'return' #predict price or return
    crude_oil_data = produce_factors(load_data(path, label)).dropna()

    split_dates = ['2019-01-01', '2021-01-01']
    seq_length = 50
    
    train_y, train_X, test_y, test_X, scaler = data_split(crude_oil_data.copy(), split_dates, seq_length, crude_oil_data.copy(), start_date='2000-01-01')
    # print(train_y.shape, train_X.shape, test_y.shape, test_X.shape ) #valid

    valid_open, valid_high, valid_low = get_open_high_low(crude_oil_data.copy(), split_dates)

    mean_pred = np.zeros(test_y.shape)
    test_y_ori = transform_to_real(test_y.reshape(len(test_y),1), test_X.shape[2], scaler)
    mean_rets = np.zeros(test_y.shape)
    mean_vals = np.zeros(test_y.shape)
    res = []
    val_loss_list = []
    mse_list = []

    tf.random.set_seed(12)
    for i in range(ntl_repetition):
        lstm = build_lstm_model(train_X)
        lstm.compile(loss='mae', optimizer='adam') 
        history = lstm.fit(train_X, train_y, batch_size = 64, epochs = 25, 
                callbacks=[lr_scheduler],
                validation_data = (test_X, test_y), 
                verbose=0)
        val_loss_list.append(history.history['val_loss'][-1])
        
        test_y_pred = lstm.predict(test_X)
        mse = mean_squared_error(test_y, test_y_pred)
        mse_list.append(mse)
        test_y_pred = transform_to_real(test_y_pred, test_X.shape[2], scaler)
        if label == 'Close':
            plt.plot(test_y_pred)
        elif label == 'return':
            rets, signal, values, acc = backtest(y_pred = test_y_pred, y_test = test_y_ori, trd_cost = 3/10000, open=valid_open, high=valid_high, low=valid_low, threshold = 0.05)
            mean_pred += signal #valid
            mean_rets += rets
            mean_vals += values/ntl_repetition
            res.append(acc)
    if label == 'Close':
        plt.plot(test_y_ori, label='truth')
    elif label == 'return':
        plt.plot(np.cumprod(1+test_y_ori), label='truth')
    plt.legend()
    plt.title('backtest')
    plt.xlabel('date index from 20190101 to 20210101')
    plt.ylabel('cumulative return')

    if not os.path.exists(img_path):
        os.mkdir(img_path)
        
    plt.savefig(img_path + '/no_tl_{}'.format(commodity))
    plt.close()
    print('avg val mse: ', np.mean(mse_list))
    print('avg val loss: ', np.mean(val_loss_list))

"""## WAETL

### Package
"""
def TL_test(target, label='return', corr_type='WD', model_rep=1):
    # Load target dataset
    path = data_path + '/'+target+'.csv'
    crude_oil_data = produce_factors(load_data(path, label)).dropna()
    crude_oil_data

    split_dates = ['2019-01-01', '2021-01-01']
    seq_length = 50
    train_y, train_X, test_y, test_X, scaler = data_split(crude_oil_data.copy(), split_dates, seq_length, crude_oil_data.copy(), start_date='2000-01-01')
    # print(train_y.shape, train_X.shape, valid_y.shape, valid_X.shape, test_y.shape, test_X.shape )

    valid_open, valid_high, valid_low = get_open_high_low(crude_oil_data.copy(), split_dates)
    test_y_ori = transform_to_real(test_y.reshape(len(test_y),1), 41, scaler)

    # ensemble model weight
    weight_path = csv_path + '/' + target + '/' + 'compare_{}.csv'.format(label)
    target_weight_file = pd.read_csv(weight_path)
    target_weight_file.index = target_weight_file['Unnamed: 0']
    target_weight_file = target_weight_file.drop(columns='Unnamed: 0').fillna(0)
    similarity = target_weight_file[corr_type]

    ensemble_pattern = 'WAETL'
    source_list = []
    if target == 'Palladium':
        source_list = ['Platinum', 'Gold', 'Copper', 'Silver']
    elif target == 'Crude Oil':
        if model_rep == 1:
            source_list = ['Heating Oil', 'Brent Crude Oil']
        else:
            source_list = ['Heating Oil']       
    else:
        target_weight_file = target_weight_file.sort_values(corr_type,ascending=False)
        source_list = sorted(list(target_weight_file.index))[:5]
  
    model_name_list = []
    if model_rep < 1:
        print('model_rep need to be no less than 1')
        return
    elif model_rep == 1:
        model_rep_left = len(os.listdir(model_path+'/'+source_list[0])) // 2
        model_rep_right =  model_rep_left + 1
    else:
        model_rep_left = len(os.listdir(model_path+'/'+source_list[0])) // 2 - model_rep // 2
        model_rep_right = model_rep_left + model_rep - 1

    for n in source_list:
        for l in sorted(os.listdir(model_path+'/'+n))[model_rep_left : model_rep_right]:
            model_name = model_path+'/'+n +'/'+ l
            model_name_list.append(model_name)
        # print(model_name)

    #step 1: load pre-trained model from different sources
    ensemble_pred = np.zeros(test_y.shape)
    if ensemble_pattern == 'WAETL':
        if model_rep == 1:
            model_weights = []
            for source in source_list:
                w = similarity[source]
                model_weights.append(w)
            model_weights /= np.sum(model_weights)
            print(model_weights)
        else:
            model_weights = [1/len(model_name_list)]*len(model_name_list)
    elif ensemble_pattern == 'AE':
        model_weights = [0.2]*5

    val_loss_list = []
    mse_list = []
    # fintune_lr_scheduler = LearningRateScheduler(fintune_lr_schedule)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    for i in range(len(model_name_list)):
        model_name = model_name_list[i]
        model = build_lstm_model(train_X)
        model.compile(loss='mae', optimizer='adam') 
        model.load_weights(model_name)
        #step 2: fine tune on target training dataset
        history = model.fit(train_X, train_y, batch_size = 64, epochs = 25, 
                callbacks=[lr_scheduler],
                validation_data = (test_X, test_y), 
                verbose=0)
        val_loss_list.append(history.history['val_loss'][-1])
        
        pred = model.predict(test_X).reshape(len(test_y))
        mse = mean_squared_error(test_y, pred)
        mse_list.append(mse)
        test_y_pred = transform_to_real(pred.reshape(len(test_y),1), test_X.shape[2], scaler)
        rets, signal, values, acc = backtest(y_pred = test_y_pred, y_test = test_y_ori, trd_cost = 3/10000, open=valid_open, high=valid_high, low=valid_low, threshold = 0.05)
        
        #step 3: ensemble model output after fine tuning
        ensemble_pred += pred * model_weights[i]

    #step 4: test ensembled model on test set of target dataset
    real_ensemble_pred = transform_to_real(ensemble_pred.reshape(-1,1), 41, scaler)
    ensemble_val_loss = np.mean(np.abs(ensemble_pred-test_y))
    ensemble_mse = mean_squared_error(test_y, ensemble_pred)
    plt.plot(np.cumprod(1+test_y_ori), label='truth')
    plt.legend()
    plt.title('TL backtest of {}'.format(target))
    plt.xlabel('date index from 20190101 to 20210101')
    plt.ylabel('cumulative return')

    if not os.path.exists(img_path):
       os.mkdir(img_path)
    plt.savefig(img_path + '/tl_{}_{}_{}'.format(target, corr_type, model_rep))
    plt.close()
    print('single_transfer_val_loss_mean: ', np.mean(val_loss_list))
    print('ensemble_val_loss: ',ensemble_val_loss)
    print('mse_list_mean: ', np.mean(mse_list))
    print('ensemble_mse: ', ensemble_mse)

# no_TL(commodity=commodity)
# TL_test(cur_dir, target='Crude Oil', index=corr_type)
# TL_test(cur_dir, target='Crude Oil', index='Pearson')

no_TL(commodity=commodity)
TL_test(target=commodity, label=label, corr_type=corr_type)

"""### With Model Repetition"""

TL_test(target=commodity, label=label, corr_type=corr_type, model_rep=repetition)

# TL_test(target='Crude Oil', index='Pearson', model_rep=5)