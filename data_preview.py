import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

# default value
img_path = './img'

if len(sys.argv) >= 2:
    try:
        data_path = sys.argv[1]
        img_path = sys.argv[2]
    except:
        print('img_path will be set as default')
else:
    print('you should at least provide data_path to execute the code!')
    sys.exit()

def classify(sign=""):
    means = []
    
    for file_name in os.listdir(data_path):
        raw = pd.read_csv(data_path + '/' + file_name).dropna()
        
        if not sign:
            print('no filter')
            continue
            
        means.append(raw[sign].to_numpy().mean())

    rule = set()
    for val in means:
        for idx in count_bits(val):
            rule.add(idx)
    
    rule = sorted(list(rule))
    group = [dict() for i in range(len(rule))]
    for file_name in os.listdir(data_path):
        raw = pd.read_csv(data_path + '/' + file_name).dropna()
        data = raw[sign].to_numpy()
        label = file_name.split('.')[0]
        mean_val = data.mean()
        
        for val in range(len(rule)):
            if mean_val <= rule[val]:
                group[val][label] = data
                break
    
    return group

def count_bits(num):
    rule_set = []
    count = 0
    while num > 1:
        num /= 10
        count += 1
    if not count:
        return [0]

    return [10**count, 2 * (10**count), 5 * (10**count)]
    
    
def plot(sign="", fig_len=20):
    data_group = classify(data_path, sign)
    for group in data_group:
        if group:
            plt.figure(figsize=(fig_len,8))
            for tag in group.keys():
                plt.plot(group[tag], label=tag + ': %.3f' %(group[tag].mean()))

            plt.legend(fontsize=15)
            plt.xlabel('timeline', fontsize=15)
            plt.ylabel(sign + ' price', fontsize=15)
            # xlim = 2*(group[tag].mean() % 100 * 5)
            # plt.xlim([0, xlim])

def close(save_to_file=False):
    count = 1
    plt.figure(figsize=(24,24))
    for file_name in os.listdir(data_path):
        raw = pd.read_csv(data_path+ '/' + file_name).dropna()
        raw = [int(i) for i in raw['Close'].to_numpy()]
        
        plt.subplot(8, 3, count)
        plt.plot(raw, label=file_name.split('.')[0])
        plt.legend(fontsize=15)
        count += 1
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    if save_to_file:
        plt.savefig(img_path + '/close.png')
    plt.close()

def volume(save_to_file=False):
    count = 1
    plt.figure(figsize=(24,24))
    for file_name in os.listdir(data_path):
        raw = pd.read_csv(data_path + '/' + file_name).dropna()
        raw = [int(i) for i in raw['Volume'].to_numpy()]
        
        plt.subplot(8, 3, count)
        plt.plot(raw, label=file_name.split('.')[0])
        plt.legend(fontsize=15)
        count += 1
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    if save_to_file:
        plt.savefig(img_path + '/volume.png')
    plt.close()

def return_(save_to_file=False):
    count = 1
    plt.figure(figsize=(24,24))
    for file_name in os.listdir(data_path):
        raw = pd.read_csv(data_path + '/' + file_name).dropna()
        raw = [float(i) for i in raw['Close'].to_numpy()]
        percentage = []
        for idx in range(1, len(raw)):
            percentage.append((raw[idx] - raw[idx-1]) / raw[idx-1])
        
        plt.subplot(8, 3, count)
        plt.plot(percentage, label=file_name.split('.')[0])
        plt.legend(fontsize=15)
#         plt.xlabel('timeline', fontsize=15)
#         plt.ylabel('percentage increase', fontsize=15)
#         plt.ylim([-0.55, 0.55])
        count += 1
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    if save_to_file:
        plt.savefig(img_path+ '/return.png')
    plt.close()

"""## 1. Close Price Trend"""

# plot(sign='Close')
close(save_to_file=True)

"""## 2. Volume Trend"""

# plot(sign='Volume', fig_len=25)
volume(save_to_file=True)

"""## 3. Return Trend"""

return_(save_to_file=True)