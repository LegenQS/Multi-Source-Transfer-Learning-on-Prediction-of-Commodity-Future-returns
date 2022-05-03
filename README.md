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
`For local host`, ...

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
#### 3. Train the base model for transfer learning by the result of correlation comparison by specifying `data_path`, `model_path`, `csv_path`, `commodity` and `retrained`. If `retrained` is not provided, please make sure you have downloaded the pretrained model here[]. Other optional parameters are `retrained`, `base_repetition`, `label` and `corr_type`.
```
!python base_model_train.py data_path model_path csv_path commodity (retrained) (base_repetition) (label) (corr_type)
```

#### 4. Use transfer learnning with the base model to predict your estimated return profit by specifying `data_path`, `model_path`, `csv_path` and `commodity`. Other parameters including `img_path`, `corr_type`, `label`, `retrained`, `repetition`, `ntl_repetition`, `label` and `corr_type`.
```
!python transfer_learning.py data_path model_path csv_path commodity (img_path) (corr_type) (label) (retrained) (repetition) (ntl_repetition) (label)
```

