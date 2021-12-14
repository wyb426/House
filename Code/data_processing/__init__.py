# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 20:38:02 2021
# 房价数据集的基本探索
@author: wyb4
"""


import numpy as np 
import pandas as pd 

df = pd.read_csv('C:/Users/wyb4/Desktop/机器学习/House/input/train.csv')
df.head()
df.describe()

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# 设置 matplotlib 图
plt.figure(figsize=(12,5))
#f, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
plt.subplot(121)
sns.distplot(df['SalePrice'],kde=False)
plt.xlabel('Sale price')
plt.axis([0,800000,0,180])
plt.subplot(122)
sns.distplot(np.log(df['SalePrice']),kde=False)
plt.xlabel('Log (sale price)')
plt.axis([10,14,0,180])

corr = df.select_dtypes(include = ['float64', 'int64']).iloc[:,1:].corr()
#fig = plt.figure()
sns.set(font_scale=1)  
sns.heatmap(corr, vmax=1, square=True)

corr_list = corr['SalePrice'].sort_values(axis=0,ascending=False).iloc[1:]
corr_list

plt.figure(figsize=(18,8))
for i in range(6):
    ii = '23'+str(i+1)
    plt.subplot(ii)
    feature = corr_list.index.values[i]
    plt.scatter(df[feature], df['SalePrice'], facecolors='none',edgecolors='k',s = 75)
    sns.regplot(x = feature, y = 'SalePrice', data = df,scatter=False, color = 'Blue')
    ax=plt.gca() 
    ax.set_ylim([0,800000])
    
plt.figure(figsize = (12, 6))
sns.boxplot(x = 'Neighborhood', y = 'SalePrice',  data = df)
xt = plt.xticks(rotation=45)


score = np.array([0.15702,0.14831,0.15233,0.12856,0.12815,0.12459,0.11696])

plt.figure()
plt.plot(score,'o-')
plt.ylabel('Leaderboard score',fontsize=16)
ax = plt.gca()
ax.set_xlim([-0.5,6.3])
ax.set_xticks([i for i in range(7)])
ax.set_xticklabels(['Untuned Random Forest', 'Tuned Random Forest', 'Tuned Extra Trees', 'Tuned Gradient Boosting'\
                    , 'Tuned XGBoost','Stacking 4 best model','Stacking more models'],fontsize=12)
plt.grid()
plt.xticks(rotation=90)