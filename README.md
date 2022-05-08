# Multi-Source-Transfer-Learning-on-Prediction-of-Commodity-Future-returns
## 1. Get the dataset which we refer to
### 1. Please download dataset from https://www.kaggle.com/datasets/mattiuzc/commodity-futures-price-history and specify your dataset path as `dataset_path`. You can also download the dataset [here](https://github.com/LegenQS/Multi-Source-Transfer-Learning-on-Prediction-of-Commodity-Future-returns/tree/main/Commodity_Data). 

### 2. Details about the dataset:  

| Attributes    | Information |  
| ------------- | ------------- |  
| Total Kinds   | `Feeder Cattle`, `Sugar`, `Brent Crude Oil`, `Wheat`, `Coffee`, `Corn`, `Gold`, `RBOB Gasoline`, `Live Cattle`, `Lumber`, `Soybean`, `Oat`, `Palladium`, `Soybean Meal`, `Heating Oil`, `Crude Oil`, `Silver`, `Natural Gas`, `Platinum`, `Lean Hogs`, `Copper`, `Cocoa`, `Soybean Oil`, `Cotton`|    
| Data Range   | From `2000-01-03` to `2021-06-09` (all data are ranged in this interval)  |  
| Schema       | `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume` |

### 3. Sample format of the file:
![image](https://github.com/LegenQS/Multi-Source-Transfer-Learning-on-Prediction-of-Commodity-Future-returns/blob/main/img/sample.png)  

### 4. For more information about the distribution, here are some indexes to display:   
  - close price of each commodity
![image](https://github.com/LegenQS/Multi-Source-Transfer-Learning-on-Prediction-of-Commodity-Future-returns/blob/main/img/close.png) 
  - colume of each commodity
![image](https://github.com/LegenQS/Multi-Source-Transfer-Learning-on-Prediction-of-Commodity-Future-returns/blob/main/img/volume.png) 
  - return of each commodity
![image](https://github.com/LegenQS/Multi-Source-Transfer-Learning-on-Prediction-of-Commodity-Future-returns/blob/main/img/return.png) 

## 2. Download pretrained model
### 1. If clone the whole project, pretrained models are located in the `model` dir, if not, please download the models from ...
### 2. If you want to train the model by yourself, please specify the parameter `retrained` as 1 when running the project later.

## 3. Get package to import 
### 1. Run the following command to get package `ta-lib` loaded:
```
url = 'https://launchpad.net/~mario-mariomedina/+archive/ubuntu/talib/+files'
ext = '0.4.0-oneiric1_amd64.deb -qO'
!wget $url/libta-lib0_$ext libta.deb
!wget $url/ta-lib0-dev_$ext ta.deb
!dpkg -i libta.deb ta.deb
!pip install ta-lib
```

### 2. Enable GPU to your runtime:  
`For Colab`, connect your runtime to GPU runtime;  
`For local host`, check your gpu availability by:
```
import torch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
```

## 4. Run the project
### 1. Clone the .py to your local dir.

### 2. Here there are 11 parameters for you to define, and three of them are required, specific details and format are shown as follows:

| Parameter       | Information |  
| -------------   | ------------- |     
| `data_path`     |  path of the dataset that downloaded |
| `model_path`    |  the model either you downloaded or the directory that you want the base models to be stored |
| `commodity`     |  the target commodity type that you'd like to see the trend, transfer learning comparison with non-transfer learning etc. |
| `img_path`      |  (optional) the path that you'd like to store all the images plotted in the code, default value is './img' |
| `csv_path`      |  (optional) the path that you'd like to store all the csv files in the code, default value is './csv' |
| `corr_type`     |  (optional) the correlation method that you want to choose, you can select one from `WD` and `Pearson`, which represent the 'Wasserstein Distance' and 'Pearson Distance', default value is 'WD' |
| `corr_threshold`|  (optional) the correlation rate threshold that you want to set of the correlation method you choose, you can select any value from 0 to 1, default value is 0 |
| `repetition`    |  (optional) for transfer learning part and target is 'Crude Oil', choose how many base models to be chosen to reduce the randomness, default value is 5|
| `retrained`     | (optional) set whether you want to retrain the base models for your selected commodity, set '0' for 'False' and any other integer for 'True', default value is 'False' |
| `base_repetition` | (optional) set the repetition times for base models training, default value is 10 |
| `ntl_repetition`  | (optional) for non-transfer learning, set the repetition time for the model training to see the randomness of the model, default value is 10 |

Above first four parameters are required to be provided when running the code, others if not specified will be set as default values.

### 3. Simply run the following codes in the command line or runtime cell to get all the results, several parameters to specify:
#### 1. Preview data close price and return trend by specifying your `data_path`, another parameter `img_path` is optional which is used to store the image results to your local disk.
```
!python data_preview.py data_path (img_path)
```

#### 2. Calculate correlation indexes of data by Pearson method and Wasserstein distance method by specifying the `data_path` and `commodity`, other parameters including `img_path`, `csv_path`, `corr_type` and `corr_threshold` are optional.
```
!python corr_cal.py data_path commodity (img_path) (csv_path) (corr_type) (corr_threshold)
```
#### 3. Train the base model for transfer learning by the result of correlation comparison by specifying `data_path`, `model_path`, `csv_path`, `commodity` and `retrained`. If `retrained` is not provided, please make sure you have downloaded the pretrained model [here](https://github.com/LegenQS/Multi-Source-Transfer-Learning-on-Prediction-of-Commodity-Future-returns/tree/main/model). Other optional parameters are `retrained`, `label`, `corr_type` and `base_repetition`.
```
!python base_model_train.py data_path model_path csv_path commodity (retrained) (label) (corr_type) (base_repetition)
```

#### 4. Use transfer learnning with the base model to predict your estimated return profit by specifying `data_path`, `model_path`, `csv_path` and `commodity`. Other parameters including `label`, `img_path`, `corr_type`, `repetition`, `ntl_repetition`.
```
!python transfer_learning.py data_path model_path csv_path commodity (label) (img_path) (corr_type) (repetition) (ntl_repetition)
```
## 5. Project Description:
For pdf version, you can refer to [here](https://github.com/LegenQS/Multi-Source-Transfer-Learning-on-Prediction-of-Commodity-Future-returns/blob/main/description.pdf)

### Overview

Financial time series prediction is always a difficult task for data scientists to tackle. The difficulty does not only come from the irrational financial market and unpredictable reactions from emotional participants but also the lack of data availability especially when we are considering a low frequency situation such as daily return of commodity futures. Our project mainly focuses on the second problem and aims to utilize a transfer learning method to improve model prediction performance, which is usually present in the Computer Vision field. 

### Motivation

Different commodity future contracts of the same category should have similar time series patterns, which can be learned by the LSTM model. If we first obtained a pre-trained model based on other similar datasets, such a pretrained model should be a good starting point for our target model. Thus, we can try to get different pretrained models from similar datasets and fine tune on the target dataset to see whether transfer learning in this way can help people to predict returns.

### Data Description

Dataset Time Period: train on 2000-2018, test on 2019-2020
Predict Label: daily commodity return
Features: 41 features in total, including price, volume data and other technical indicators, without any fundamental information.


### Result

Fine tuning pre-trained model on target training dataset significantly outperforms the model obtained without transfer learning in terms of out-of-sample Mean Absolute Error ,Mean Squared Error and backtesting strategy return. Results are shown through statistics and plots.

Our trading strategy is simple. Long the future contract when the model predicts positive. Short the future contract when the model predicts negative. Everytime when our position has a 5% loss in a single day, clear the position immediately and stop trading until tomorrow. 

MAE and MSE improvement WTL compared with WAETL testing performance

Target and Source:

G1: 'Palladium' <- ['Platinum', 'Gold', 'Copper', 'Silver']

G2: 'Crude Oil' <- ['Heating Oil', 'Brent Crude Oil']


<img width="579" alt="image" src="https://user-images.githubusercontent.com/93000888/167306103-2f1d1d4a-1633-4641-ad5e-1cf58b360b15.png">

<img width="581" alt="image" src="https://user-images.githubusercontent.com/93000888/167306128-9593e0f7-b6ee-453e-9efd-2624f53aeb05.png">

Bactesting cummulative return for multiple models compared with merely longing the commodity future

LHS: WTL models; RHS: TL models from differen sources

<img width="621" alt="image" src="https://user-images.githubusercontent.com/93000888/167306022-a5039fa0-c9fa-4c23-920f-b8cb5393a0cd.png">

<img width="678" alt="image" src="https://user-images.githubusercontent.com/93000888/167306032-dd35cc0c-b747-4e15-85c4-8b9becd7e468.png">

Bactesting TL model cummulative return based on single-source(heating oil) multiple pretrained models to see influence of randomness in pretraining

<img width="663" alt="image" src="https://user-images.githubusercontent.com/93000888/167306041-364851d3-e037-4ba0-a6d4-14ad9a5f9b1f.png">

We can see in terms of testing loss and backtesting performance, the model was improved significantly.

### Conclusion

Transfer learning indeed works in terms of predicting future contract returns. There are some papers indicating that it also works in predicting stock returns. Basically, when training a neural network, we can view transfer learning as providing a good initializer. Fine tuning is just to do regular training but start with a starting point from another pretrained model. Through our observations, we found by transfer learning, neural network usually performs better than most of the regular learning methods on test set. Normally, initializer brings randomness. Thus, trained models based on the same dataset can be quite different. Models generated through transfer learning on most similar dataset would perform similarly or better as the best model among 10 regular models. So, essentially, transfer learning can seen as providing a really perfect initializer, which helps to finally reach the weights that can make the model perform better maybe not on in-sample data but on out-of-sample data.

### Further Possible Improvement

Based on multi-source pretrained models, we can try other methods to ensemble our fine tuned model outputs.
