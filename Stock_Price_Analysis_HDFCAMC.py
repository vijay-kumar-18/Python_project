#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('HDFCAMC.NS.csv')


# In[4]:


df.head()


# In[6]:


df.shape


# In[79]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.Date =pd.to_datetime(df.Date)


# In[12]:


num_cols = df.select_dtypes(include= ["int64","float64"]).columns.tolist()


# In[18]:


df['Week']= df.Date.dt.day_name()
df['WeekOfYear'] = df.Date.dt.weekofyear


# In[19]:


df['Week'] = df['Date'].dt.strftime('%A')  # Day of the week
df['WeekOfYear'] = df['Date'].dt.strftime('%U')


# In[80]:


df['Returns'] = df.Close-df.Close.shift(1)


# In[23]:


num_cols.append('Returns')


# In[27]:


#Distribution Analysis
    for col in num_cols:
        df[col].plot(kind ='hist')
        plt.title(f'{col} distribution')
        plt.axvline(df[col].mean(), color ='green',linestyle ='--')
        plt.axvline(df[col].median(), color = 'red',linestyle ='-')
        plt.show()


# In[37]:


#correlation Analysis
plt.figure(figsize =(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, vmin=-1, vmax=1, fmt='.2f', cmap='magma')
plt.show()


# In[86]:


df.corr(numeric_only=True)


# In[38]:


#Trend Analysis
_,ax = plt.subplots(ncols = 2, figsize = (15,5))
df.plot(x='WeekOfYear', y = 'Close', kind = 'line', ax=ax[0])
df.plot(x='WeekOfYear', y = 'Returns', kind = 'line', ax=ax[1])
plt.show()


# In[39]:


df['Log_Returns']= np.log(df.Close)-np.log(df.Close).shift(1)


# In[63]:


mean = df['Log_Returns'].mean()
std = df['Log_Returns'].std(ddof=1) #delta degree of freedom

print(mean)
print(std)

#When you use ddof=1 (the default is usually ddof=0), 
#it adjusts the divisor by subtracting 1 from the number of observations, 
#which is useful when you are working with a sample rather than the entire population. 
#This correction accounts for the fact that you are
#estimating the population standard deviation based on a sample,
#and it helps to provide an unbiased estimate.


# In[64]:


from scipy import stats
from scipy.stats import norm


# In[69]:


probability_drop_1day = norm.cdf(-0.01, loc=mean, scale =std)
print(probability_drop_1day)


# In[59]:


df.info()


# In[67]:


print(type(mean), mean)
print(type(std), std)


# In[70]:


mean_yearly = mean*251


# In[71]:


std_yearly = std*251


# In[72]:


#99% Confidence Interval for the mean value of the stock price
conf_level = 0.99
sample_size = df.Close.shape[0]
std_error = np.std(df.Close)/np.sqrt(sample_size)


# In[73]:


moe = stats.t.ppf((1+conf_level)/2, df = sample_size -1)*std_error


# In[74]:


CI_stock = (np.mean(df.Close)-moe, np.mean(df.Close) +moe)
print (CI_stock)


# In[ ]:


http://localhost:8888/edit/Projects/HDFCAMC.NS.csv


# In[ ]:




