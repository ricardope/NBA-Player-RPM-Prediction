#!/usr/bin/env python
# coding: utf-8

# In[1]:

#导入所需库函数
import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
color = sns.color_palette()
sns.set_style("whitegrid")
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:

#读物数据
plr = pd.read_csv("nba_2017_nba_players_with_salary.csv")
plr.head(5)


# In[8]:

#查看出场时间和赢球正负值的关系
plt.figure(figsize = (15,7))
sns.regplot(x = "MP",y = "WINS_RPM",data = plr)


# In[9]:

#建立模型并输出参数描述
minutes = smf.ols('WINS_RPM ~ MP', data=plr).fit()
print(minutes.summary())


# In[11]:

#在出场时间和赢球正负值中加入球员位置参数
ax = sns.lmplot(x="MP", y="WINS_RPM", data=plr, hue='POSITION', fit_reg=False, size=6, aspect=2, legend=False, scatter_kws={"s": 200})
ax.set(xlabel='Minutes Played', ylabel='RPM (Real Plus Minus)', title="Minutes Played vs RPM (Real Plus Minus) by Position: 2016-2017 Season")
plt.legend(loc='upper left', title='Position')


# In[15]:


# 查看防守数据和赢球正负值的关系
plr_def = plr[["DRB","STL","BLK","WINS_RPM"]].copy()
plr_def.head()


# In[16]:


#查看进攻数据和赢球正负值的关系
plr_off = plr[["eFG%","FT%","ORB","AST","POINTS","WINS_RPM"]].copy()
plr_off.rename(columns={'eFG%': 'eFG','FT%':'FT'}, inplace=True)
plr_off.head()


# In[17]:


plr_def.corr()


# In[18]:

#绘制防守数据和wins——rpm热图
plt.subplots(figsize=(10,10))
sns.heatmap(plr_def.corr(), xticklabels=plr_def.columns.values, yticklabels=plr_def.columns.values, cmap="Reds")


# In[20]:


plr_def.cov()


# In[21]:


#建立防守数据和wins——rpm的回归模型，并输出参数
defense = smf.ols('WINS_RPM ~ DRB + STL', data=plr_def).fit()
print(defense.summary())


# In[22]:


#建立额外的回归模型
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(defense, fig=fig)


# In[23]:


fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_ccpr_grid(defense, fig=fig)


# In[24]:


fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(defense, "STL", fig=fig)


# In[26]:


sns.jointplot("STL", "WINS_RPM", data=plr_def,size=10, ratio=3, color="r")


# In[27]:


steals = smf.ols('WINS_RPM ~ STL', data=plr_def).fit()
print(steals.summary())


# In[28]:


#Correlation between offensive statistics
plr_off.corr()


# In[29]:

#绘制进攻数据和wins——rpm热图
plt.subplots(figsize=(10,10))
sns.heatmap(plr_off.corr(), xticklabels=plr_off.columns.values, yticklabels=plr_off.columns.values, cmap="Blues")


# In[31]:


plr_off.cov()


# In[32]:


offense = smf.ols('WINS_RPM ~ eFG + ORB + AST + POINTS', data=plr_off).fit()
print(offense.summary())


# In[33]:


#Run some additional regression diagnostics
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(offense, fig=fig)


# In[35]:


fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_ccpr_grid(offense, fig=fig)


# In[36]:


fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(offense, "POINTS", fig=fig)


# In[37]:


sns.jointplot("POINTS", "WINS_RPM", data=plr_off,size=10, ratio=3, color="b")


# In[38]:


eFGs = smf.ols('WINS_RPM ~ POINTS', data=plr_off).fit()
print(eFGs.summary())


# In[40]:


## Final Variables
plr_full = plr[["PLAYER","STL","DRB","eFG%","ORB","AST","POINTS","WINS_RPM"]].copy()
plr_full.rename(columns={'eFG%': 'eFG'}, inplace=True)
plr_full.head()


# In[41]:


combined = smf.ols('WINS_RPM ~ STL + DRB + eFG + AST + POINTS', data=plr_full).fit()
print(combined.summary())


# In[ ]:




