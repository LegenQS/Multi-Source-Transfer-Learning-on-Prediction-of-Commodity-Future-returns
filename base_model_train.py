import pandas as pd
import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
sns.set_theme()
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from keras.callbacks import LearningRateScheduler
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,BatchNormalization, CuDNNLSTM,Activation
import talib as ta

csv_path = './csv'
corr_type = 'WD'
retrained = False
base_repetition = 20
label = 'return'

if len(sys.argv) >= 4:
    try:
        data_path = sys.argv[1]
        model_path  = sys.argv[2]
        commodity = sys.argv[3]

        if not os.path.exists(data_path):
            print('data_path not exists!')
            sys.exit()
            
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        retrained = bool(int(sys.argv[4]))
        base_repetition = int(sys.argv[5])
        label = sys.argv[6]
        csv_path = sys.argv[7]
        corr_type = sys.argv[8]
    except:
        print('values not provided will be set as default values')
else:
    print('you should at least provide three parameters to execute the code!')
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

def train_basemodel(mymodel_path):
    final_val = []
    for model_name in mymodel_path:
        if model_name in os.listdir(model_path):
            continue

        val_list = []
        crude_oil_data = produce_factors(load_data(data_path + '/' + model_name + '.csv')).dropna()

        split_dates = ['2019-01-01', '2021-01-01']
        seq_length = 50
        train_y, train_X, test_y, test_X, scaler = data_split(crude_oil_data.copy(), split_dates, seq_length, crude_oil_data.copy(), start_date='2000-01-01')
        # print(train_y.shape, train_X.shape, valid_y.shape, valid_X.shape, test_y.shape, test_X.shape)

        # n_models = 20
        tf.random.set_seed(2)
        for i in range(base_repetition):
            lstm = build_lstm_model(train_X)
            lstm.compile(loss='mae', optimizer='adam') 
            lstm.fit(train_X, train_y, batch_size = 64, epochs = 25, 
                    callbacks=[lr_scheduler],
                    # validation_data = (valid_X, valid_y), 
                    verbose=0)

            lstm.save(model_path + '/' + model_name + '/' + datetime.today().strftime('%Y-%m-%d') + (' loss: %.4f' %lstm.history.history['loss'][-1]) + ' No.' + str(i+1) + '.h5')

        #     dim = 41
        #     valid_y_pred = lstm.predict(valid_X)
        #     valid_y_pred = transform_to_real(valid_y_pred, 41, scaler)
        #     val_list.append(valid_y_pred)
        # final_val.append(np.mean(np.array(val_list), axis=0))
        # rets, signal, values= backtest(y_pred = valid_y_pred, y_test = valid_y_ori, trd_cost = 3/10000, open=valid_open, high=valid_high, low=valid_low, threshold = 0.05)
        # mean_pred += signal
        # mean_rets += rets
        # mean_vals += values/n_models
    
    # final = pd.DataFrame(np.array(final_val).T, columns=model_path, index=[i for i in range(len(final_val[0]))])
    # final.to_csv(csv_path + '/validation_result.csv')

    # return val_list

"""### Test"""
if retrained:
    weight = pd.read_csv(csv_path + '/' + commodity + '/' + 'compare_{}.csv'.format(label), index_col=0)
    source_list = weight.sort_values(corr_type,ascending=False)
    train_basemodel(list(source_list.index)[:10])