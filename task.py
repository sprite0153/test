

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:27:11 2019

@author: xuebi
"""

# -*- coding: utf-8 -*-
"""读取数据"""

import pandas as pd
import numpy as np

#encoding ='unicode_escape' 主要是为了解决字典里边有非ascii字符，并且无法对其进行编码/解码。
df = pd.read_csv(r'/Users/xuebi/Documents/datawhale/data.csv',encoding = 'unicode_escape')

#查看数据类型：
print(df.info())

import pandas as pd
from  scipy.stats import kstest


"""单变量统计指标函数,输入变量类型为dataframe"""

def getpercentile(data,percentile):
    try:
        return np.percentile(data.dropna(),percentile)
    except:
        return np.nan

"""获取正态分布的P值"""
def getnorm(data):
    try:
        return kstest(data.dropna(),'norm').pvalue
    except:
        return np.nan

def getwarm(data):
    try:
        IQR = np.percentile(data.dropna(),75) - np.percentile(data.dropna(),25) 
        return "["+str(np.percentile(data.dropna(),25) - 1.5*IQR)+","+str(np.percentile(data.dropna(),75) + 1.5*IQR)+"]"
    except:
        return np.nan
    
def getwarmpct(data):
    try:
        return len(data[(data<data.mean()-1.5*data.std()) | (data>data.mean()+1.5*data.std())])/len(data)
    except:
        return np.nan
    
def getextreme(data):
    try:
        IQR = np.percentile(data.dropna(),75) - np.percentile(data.dropna(),25) 
        return "["+str(np.percentile(data.dropna(),25) - 3*IQR)+","+str(np.percentile(data.dropna(),75) + 3*IQR)+"]"
    except:
        return np.nan

def getextremepct(data):
    try:
        return len(data[(data<data.mean()-3*data.std()) | (data>data.mean()+3*data.std())])/len(data)
    except:
        return np.nan

def getsingalinfo(data):
    df = pd.DataFrame()
    df["label"] = data.columns
    df["count"] = list(data.count())
    df["coverage"] = df["count"]/(df.shape[0])
    
    mean_df = pd.DataFrame(dict(data.mean()),index=["mean"]).T
    mean_df =mean_df.reset_index()
    mean_df.columns = ["label","mean"]
    
    df = pd.merge(df,mean_df,how='left')
    
    std_df = pd.DataFrame(dict(data.std()),index=["std"]).T
    std_df = std_df.reset_index()
    std_df.columns = ["label","std"]
    
    df = pd.merge(df,std_df,how='left')
    
    df["5%"] = [getpercentile(data[column],5) for column in data.columns]
    df["95%"] = [getpercentile(data[column],95) for column in data.columns]
    df["norm"] = [getnorm(data[column]) for column in data.columns]
    df["warm"] = [getwarm(data[column]) for column in data.columns]
    df["warm_pct"] = [getwarmpct(data[column]) for column in data.columns]
    df["extreme"] = [getextreme(data[column]) for column in data.columns]
    df["extreme_pct"] = [getextremepct(data[column]) for column in data.columns]
    return df 
    

#数据单变量统计
normal_label_data_singal_label_info_df = getsingalinfo(df)

#正负比
df.groupby(by='status').count()['custid']


#空值处理
# 去除空值过多的feature
def nan_remove(data, rate_base=0.4):
    
    all_cnt = data.shape[0]
    avaiable_index = []
    # 针对每一列feature统计nan的个数，个数大于全量样本的rate_base的认为是异常feature，进行剔除
    for i in range(data.shape[1]):
        rate = pd.isnull(np.array(data.iloc[:, i])).sum() / all_cnt
        if rate <= rate_base:
            avaiable_index.append(i)
            data_available = data.iloc[:, avaiable_index]
    return data_available, avaiable_index



df_nan_remove, df_index = nan_remove(df ,rate_base=0.4)



"""数值型计算自变量X与Y的相关系数"""
#取出数值型变量

df_int_float = df_nan_remove.select_dtypes(include=['float','int'])

from scipy.stats.stats import spearmanr
def getxycorr(data):
    
    spearmanrCor = pd.DataFrame()
    
    spearmanr_info = spearmanr(data,nan_policy='omit')
    data_cor = pd.DataFrame(spearmanr_info[0],columns=data.columns,index=data.columns)["status"].T
    data_p = pd.DataFrame(spearmanr_info[1],columns=data.columns,index=data.columns)["status"].T
    
    spearmanrCor["label"] = data_cor.index
    spearmanrCor["method"] = 'spearman'
    spearmanrCor["corr"] = list(data_cor)
    spearmanrCor["p"] = list(data_p)
    return spearmanrCor

normal_label_data_spearmanr_info_df = getxycorr(df_int_float)



"""变量分布画图"""
import matplotlib.pyplot as plt
import seaborn as sns

def getplot(data,flag):
    for column in data.columns:
        if column!='custid' and column!= "status":
            try:
                plt.figure()
                plt.title(column)
                plt.hist(data[data[flag]==0][column].dropna(),bins=100,color="red",alpha=0.5)
                plt.hist(data[data[flag]==1][column].dropna(),bins=100,color="green",alpha=0.5)
                plt.legend(["status=0","status=1"])
                #plt.savefig(column+".png")
                plt.show()
            except:
                pass

df_nan_remove.columns
getplot(df_int_float,"status")

plt.hist(normal_label_data["mj_call_contact_pct"].dropna())

'''空值填充'''
# 空feature填充
def nan_fill(data, limit_value=10, countinuous_dealed_method='mean'):
    feature_cnt = data.shape[1]
    normal_index = []
    continuous_feature_index = []
    class_feature_index = []
    continuous_feature_df = pd.DataFrame()
    class_feature_df = pd.DataFrame()
    # 当存在空值且每个feature下独立的样本数小于limit_value，我们认为是class feature采取one_hot_encoding；
    # 当存在空值且每个feature下独立的样本数大于limit_value，我们认为是continuous feature采取mean,min,max方式
    for i in range(feature_cnt):
        print('开始：%r' %i)
        if pd.isnull(np.array(data.iloc[:, i])).sum() > 0:
            print('缺失值大于0的变量：%r'%data.iloc[0, i])
            if len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) >= limit_value:
                print('连续变量：'%i)
                if countinuous_dealed_method == 'mean':
                    continuous_feature_df = pd.concat(
                    [continuous_feature_df, data.iloc[:, i].fillna(data.iloc[:, i].mean())], axis=1)
                    continuous_feature_index.append(i)
                elif countinuous_dealed_method == 'max':
                        continuous_feature_df = pd.concat(
                        [continuous_feature_df, data.iloc[:, i].fillna(data.iloc[:, i].max())], axis=1)
                        continuous_feature_index.append(i)
                elif countinuous_dealed_method == 'min':
                        continuous_feature_df = pd.concat(
                        [continuous_feature_df, data.iloc[:, i].fillna(data.iloc[:, i].min())], axis=1)
                        continuous_feature_index.append(i)
            elif len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) > 0 and len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) < limit_value:
                print('分类变量：%r'%i)
                class_feature_df = pd.concat(
                [class_feature_df, pd.get_dummies(data.iloc[:, i], prefix=data.columns[i])], axis=1)
                class_feature_index.append(i)
        else:
            normal_index.append(i)
            print('其他：%r'%i)
            data_update = pd.concat([data.iloc[:, normal_index], continuous_feature_df, class_feature_df], axis=1)
    return data_update

df_nan_fill  =nan_fill( df_nan_remove)
df_nan_fill  =nan_fill( df_int_float)
df_int_float.data_update


