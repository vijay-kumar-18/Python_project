#!/usr/bin/env python
# coding: utf-8

# # Scope of the Project

# In[ ]:





# In[6]:


import pandas as pd
import numpy as np


# In[12]:


df_c= pd.read_csv('C:/Users/user/Downloads/Music_customers.csv')


# In[13]:


df_c.head()


# In[14]:


df_lh= pd.read_excel('C:/Users/user/Downloads/Music_listening_history.xlsx')


# In[15]:


df_lh.head()


# In[17]:


df_af= pd.read_excel('C:/Users/user/Downloads/Music_listening_history.xlsx', sheet_name= 1)


# In[23]:


df_af


# In[24]:


df_s= pd.read_excel('C:/Users/user/Downloads/Music_listening_history.xlsx', sheet_name= 2)


# In[25]:


df_s.head()


# # Clean Data
# 

# In[19]:


#Missing Data


# In[26]:


# inconsistency text and datatypes


# In[27]:


#Duplicate Rows


# In[22]:


#Outliers


# In[30]:


df_c.dtypes


# In[31]:


df.info()


# In[33]:


df_c['Member Since'] = pd.to_datetime(df_c['Member Since'])


# In[34]:


df_c['Cancellation Date'] = pd.to_datetime(df_c['Cancellation Date'])


# In[ ]:





# In[41]:


df_c.dtypes


# In[40]:


df_c['Subscription Rate'] = pd.to_numeric((df_c['Subscription Rate']).str.replace('$',''))


# In[42]:


df_c.head()


# In[43]:


df_lh.dtypes


# In[44]:


df_af.dtypes


# In[45]:


df_s.dtypes


# In[46]:


df_c.isnull().sum()


# In[52]:


df_c[df_c['Subscription Plan'].isna()]


# In[54]:


df_c[['Subscription Plan', 'Subscription Rate']]


# In[55]:


df_c['Subscription Plan'].fillna('Basic (Ads)', inplace = True)


# In[57]:


df_c['Subscription Plan'].isnull().sum()


# In[59]:


df_c['Discount?'].value_counts(dropna=False)


# In[60]:


df_c['Discount?']=np.where(df_c['Discount?']=='Yes',1,0)


# In[62]:


df_c.describe()


# In[66]:


df_c.columns


# In[67]:


#Outlier detection
df_c[df_c['Subscription Rate']>7.99]


# In[70]:


df_c[df_c.duplicated()]


# In[76]:


df_c.iloc[15,5]=9.99


# In[78]:


df_c[df_c['Subscription Rate']>7.99]


# In[79]:


df_c['cancelled'] = np.where(df_c['Cancellation Date'].notna(),1,0)


# In[80]:


df_c[df_c['Subscription Rate']>7.99]


# In[81]:


df_c['Email']= df_c['Email'].str[6:]


# In[82]:


df_c.head()


# 
# # EDA

# In[85]:


(df_c['Member Since']-df_c['Cancellation Date']).mean()


# In[88]:


cust_discount = df_c[df_c['Discount?']==1]


# In[89]:


cust_discount


# In[90]:


cust_no_discount = df_c[df_c['Discount?']==0]


# In[91]:


cust_no_discount


# In[92]:


cust_discount.cancelled.sum()/cust_discount.cancelled.count()


# In[93]:


cust_no_discount.cancelled.sum()/cust_no_discount.cancelled.count()


# In[94]:


pd.DataFrame([['Had Discount',0.8571428571428571],
             ['No Discount', 0.30434782608695654],
             ], columns = ['Customer_type','cancellation rate']). plot.barh(x='Customer_type', y = 'cancellation rate')


# In[95]:


df_af


# In[98]:


df_af['ID'].str.split('-').to_list()


# In[99]:


pd.DataFrame(df_af['ID'].str.split('-').to_list())


# In[100]:


pd.DataFrame(df_af['ID'].str.split('-').to_list()).rename(columns={0:'Audio Type', 1:'Audio ID'})


# In[101]:


audio_cleaned = pd.DataFrame(df_af['ID'].str.split('-').to_list()).rename(columns={0:'Audio Type', 1:'Audio ID'})


# In[102]:


df_af


# In[104]:


audio_all=pd.concat([audio_cleaned, df_af], axis=1)


# In[105]:


audio_all


# In[117]:


audio_all['Audio ID']=pd.to_numeric(audio_all['Audio ID'])


# In[118]:


audio_all.dtypes


# In[107]:


df_lh


# In[119]:


df_lh['Audio ID'].dtype


# In[122]:


df=df_lh.merge(audio_all, how ='left', on='Audio ID')


# In[123]:


df


# In[125]:


df.groupby('Customer ID')['Session ID'].nunique().plot.hist()


# In[126]:


df.Genre.value_counts()


# In[128]:


df.Genre.value_counts().plot.barh()


# # Preparing Data for Modeling 
# 1. Features (fields)
# 2. Target (predictions)

# In[129]:


df_c.head()


# In[132]:


model_df = df_c[['Customer ID','cancelled', 'Discount?']]


# In[141]:


model_df.head()


# In[138]:


no_of_sessions = df.groupby('Customer ID')['Session ID'].nunique().rename('total_sessions').reset_index()


# In[139]:


no_of_sessions


# In[142]:


model_df = model_df.merge(no_of_sessions, how='left', on='Customer ID')


# In[144]:


model_df.head()


# In[146]:


pd.get_dummies(df.Genre)


# In[147]:


pd.concat([df['Customer ID'], pd.get_dummies(df.Genre)], axis=1)


# In[151]:


genre = pd.concat([df['Customer ID'], pd.get_dummies(df.Genre)], axis=1).groupby('Customer ID').sum().reset_index()


# In[152]:


genre.head()


# In[153]:


df_lh.head()


# In[158]:


audio_all = df_lh.groupby('Customer ID')['Audio ID'].count().rename('total_audio_listened').reset_index()
audio_all


# In[161]:


final_df = genre.merge(audio_all, how='left', on='Customer ID')
final_df.head()


# In[162]:


model_df.head()


# In[170]:


model_df['pop%']= ((final_df['Pop']+final_df['Pop Music'])/final_df['total_audio_listened'])*100


# In[171]:


model_df


# In[172]:


model_df['podcast%']= ((final_df['Comedy']+final_df['True Crime'])/final_df['total_audio_listened'])*100


# In[173]:


# % of people listining to podcast
model_df


# In[174]:


import seaborn as sns


# In[175]:


sns.pairplot(model_df)


# In[179]:


sns.heatmap(model_df.corr(),annot=True, vmin=-1, vmax=1, cmap='coolwarm')


# In[ ]:




