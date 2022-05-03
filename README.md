# Multi-Source-Transfer-Learning-on-Prediction-of-Commodity-Future-returns
## 1. Get the dataset which we refer to
1. Please download dataset from https://www.kaggle.com/datasets/mattiuzc/commodity-futures-price-history and specify your dataset path as `dataset_path`
2. Details about the dataset:  

| Attributes    | Information |  
| ------------- | ------------- |  
| Total Kinds   | `Feeder Cattle`, `Sugar`, `Brent Crude Oil`, `Wheat`, `Coffee`, `Corn`, `Gold`, `RBOB Gasoline`, `Live Cattle`, `Lumber`, `Soybean`, `Oat`, `Palladium`, `Soybean Meal`, `Heating Oil`, `Crude Oil`, `Silver`, `Natural Gas`, `Platinum`, `Lean Hogs`, `Copper`, `Cocoa`, `Soybean Oil`, `Cotton`|    
| Data Range   | From `2000-01-03` to `2021-06-09` (all data are ranged in this interval)  |  
| Schema       | `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume` |

3. Sample format of the file:
![image](https://github.com/LegenQS/Multi-Source-Transfer-Learning-on-Prediction-of-Commodity-Future-returns/blob/main/img/sample.png)  

4. For more information about the distribution, here are some indexes to display: 
1) close price of each commodity
![image](https://github.com/LegenQS/Multi-Source-Transfer-Learning-on-Prediction-of-Commodity-Future-returns/blob/main/img/close.png) 
2) volume of each commodity
![image](https://github.com/LegenQS/Multi-Source-Transfer-Learning-on-Prediction-of-Commodity-Future-returns/blob/main/img/volume.png) 
3) return of each commodity
![image](https://github.com/LegenQS/Multi-Source-Transfer-Learning-on-Prediction-of-Commodity-Future-returns/blob/main/img/return.png) 

## 2. Download pretrained model
1. If clone the whole project, pretrained models are located in the `model` dir, if not, please download the models from ...
2. If you want to train the model by yourself, please specify the parameter `retrained` as 1 when running the project later.

## 3. Get package to import 
1. Run the following command to get package `ta-lib` loaded:
```
url = 'https://launchpad.net/~mario-mariomedina/+archive/ubuntu/talib/+files'
ext = '0.4.0-oneiric1_amd64.deb -qO'
!wget $url/libta-lib0_$ext libta.deb
!wget $url/ta-lib0-dev_$ext ta.deb
!dpkg -i libta.deb ta.deb
!pip install ta-lib
```

2. Enable GPU to your runtime:  
`For Colab`, connect your runtime to GPU runtime;
`For local host`, ...

## 4. Run the project
1. Clone the .py to your local dir.

2. Here there are 12 parameters for you to define, and four of them are required, specific details and format are shown as follows:

| Parameter       | Information |  
| -------------   | ------------- |  
| `cur_dir `      |  current directory you exceute, also the location that results to be stored by default |    
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

3. Simply run the following codes in the command line or runtime cell to get all the results, several parameters to specify:
```
!python dlFinal.py cur_dir data_path model_path commodity
```
