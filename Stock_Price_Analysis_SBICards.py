#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv('SBICARD.NS.csv')


# In[5]:


from scipy import stats


# In[57]:


df.head()


# In[58]:


df.info()


# In[8]:


df.Date = pd.to_datetime(df.Date)


# In[11]:


num_cols = df.select_dtypes(include =['float64','int64']).columns.tolist()


# In[12]:


df['Date'] = pd.to_datetime(df.Date)


# In[14]:


df['Week'] = df['Date'].dt.strftime ('%A')
df['WeekOfYear'] = df['Date'].dt.strftime('%U')


# In[18]:


df.shape[0]


# In[19]:


df['Returns'] = df['Close']-df['Close'].shift(1)


# In[24]:


num_cols.append('Returns')


# In[27]:


df.columns


# In[28]:


df.info()


# In[30]:


#Distribution Analysis
for col in num_cols:
    df[col].plot(kind ='hist')
    plt.title(f'{col} distribution')
    plt.axvline(df[col].mean(), color ='green',linestyle ='--')
    plt.axvline(df[col].median(), color = 'red',linestyle ='-')
    plt.show()


# In[33]:


cor = df.corr(numeric_only = True)


# In[38]:


#Correlation Analysis
plt.figure(figsize = (8,5))
sns.heatmap(cor, annot= True, vmin = -1, vmax = 1, fmt = '.2f', cmap = 'magma')
plt.show()


# In[39]:


cor


# In[43]:


#Trend Analysis
_, ax = plt.subplots(ncols=3, figsize = (15,5))
df.plot(x='WeekOfYear', y = 'Close', kind = 'line', ax=ax[0])
df.plot(x='WeekOfYear', y = 'Returns', kind = 'line', ax=ax[1])
df.plot(x='WeekOfYear', y = 'High', kind = 'line', ax=ax[2])
plt.show()


# In[56]:


_, ax = plt.subplots(ncols=2,nrows=2, figsize = (25,10))
df.plot(x='WeekOfYear', y = 'Close', kind = 'line', ax=ax[0,0])
df.plot(x='WeekOfYear', y = 'Returns', kind = 'line', ax=ax[0,1])
df.plot(x='WeekOfYear', y = 'High', kind = 'line', ax=ax[1,0])
df.plot(x='WeekOfYear', y = 'Low', kind = 'line', ax=ax[1,1])
plt.show()


# In[59]:


df['Log_Returns'] = np.log(df.Close)-np.log(df.Close).shift(1)


# In[60]:


mean = df['Log_Returns'].mean()
std = df['Log_Returns'].std(ddof =1)


# In[61]:


from scipy.stats import norm


# In[62]:


probability_drop_1day = norm.cdf(0.01, loc = mean, scale= std)


# In[70]:


print(probability_drop_1day)


# In[66]:


probability_drop_20day = norm.cdf(0.10, loc = mean, scale= std)


# print(probability_drop_20day)

# In[76]:


conf_level = 0.95
sample_size = df.Close.shape[0]
std_error = np.std(df.Close)/np.sqrt(sample_size)
moe = stats.t.ppf((1+conf_level)/2, df=sample_size-1)*std_error
CI_stock = (np.mean(df.Close)-moe, np.mean(df.Close)+moe)
print(CI_stock)


# In[ ]:




