# Multi-Source-Transfer-Learning-on-Prediction-of-Commodity-Future-returns
## 1. Get the dataset which we refer to
1. Please download dataset from https://www.kaggle.com/datasets/mattiuzc/commodity-futures-price-history and specify your dataset path as `dataset_path`
2. Details about the dataset:  

| Attributes    | Information |  
| ------------- | ------------- |  
| Total Kinds   | `Feeder Cattle`, `Sugar`, `Brent Crude Oil`, `Wheat`, `Coffee`, `Corn`, `Gold`, `RBOB Gasoline`, `Live Cattle`, `Lumber`, `Soybean`, `Oat`, `Palladium`, `Soybean Meal`, `Heating Oil`, `Crude Oil`, `Silver`, `Natural Gas`, `Platinum`, `Lean Hogs`, `Copper`, `Cocoa`, `Soybean Oil`, `Cotton`|    
| Data Range   | From `2000-01-03` to `2021-06-09` (all data are ranged in this interval)  |  
| Schema       | `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume` |

Sample format of the file:
![image]()  
For more information about the distribution, here are some indexes to display: 
![image]() 



## 2. Get package to import 
1. Run the following command to get package `ta-lib` loaded:
```
url = 'https://launchpad.net/~mario-mariomedina/+archive/ubuntu/talib/+files'
ext = '0.4.0-oneiric1_amd64.deb -qO'
!wget $url/libta-lib0_$ext libta.deb
!wget $url/ta-lib0-dev_$ext ta.deb
!dpkg -i libta.deb ta.deb
!pip install ta-lib
```

## 3. Run the project
1. To get the data of the distribution of the whole commodity, run:
