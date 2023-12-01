#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


df_bookings = pd.read_csv("fact_bookings.csv")


# In[3]:


df_bookings.head()


# In[4]:


df_bookings.shape


# In[6]:


df_bookings.room_category.unique()


# In[7]:


df_bookings['room_category'].value_counts()


# In[8]:


df_bookings.booking_platform.unique()


# In[9]:


df_bookings['booking_platform'].value_counts()


# In[10]:


df_bookings['booking_platform'].value_counts().plot(kind='bar')


# In[14]:


df_bookings['booking_platform'].value_counts().plot(kind='pie')


# In[16]:


df_bookings['booking_platform'].value_counts().plot(kind='barh')


# In[17]:


df_bookings.describe()


# In[20]:


df_bookings.revenue_generated.min(), df_bookings.revenue_generated.max()


# In[21]:


df_date = pd.read_csv('dim_date.csv')
df_hotels = pd.read_csv('dim_hotels.csv')
df_rooms = pd.read_csv('dim_rooms.csv')
df_agg_bookings = pd.read_csv('fact_aggregated_bookings.csv')


# In[22]:


df_hotels.shape


# In[23]:


df_hotels.head(4)


# In[24]:


df_hotels.category.value_counts()


# In[26]:


df_hotels.city.value_counts().sort_values()


# In[29]:


df_hotels.city.value_counts().sort_values().plot(kind='bar')


# In[30]:


df_agg_bookings.head(4)


# In[33]:


df_agg_bookings.property_id.unique()


# In[42]:


#Total booking per property id
df_agg_bookings.groupby('property_id')['successful_bookings'].sum()


# In[48]:


#Days on which bookings are greater than capacity
df_agg_bookings[df_agg_bookings.successful_bookings > df_agg_bookings.capacity]


# In[51]:


#Properties that have highest capacity
df_agg_bookings.capacity.max()


# In[57]:


df_agg_bookings[df_agg_bookings.capacity == df_agg_bookings.capacity.max()]


# # Data Cleaning

# In[56]:


df_bookings.head(2)


# In[54]:


df_bookings[df_bookings.no_guests<=0]


# In[58]:


df_bookings = df_bookings[df_bookings.no_guests>0]


# In[59]:


df_bookings.shape


# In[60]:


df_bookings.info()


# In[62]:


df_bookings.revenue_generated.max(),df_bookings.revenue_generated.min()


# In[65]:


avg, std = df_bookings.revenue_generated.mean(), df_bookings.revenue_generated.std()


# In[66]:


avg, std


# In[70]:


higher_limit = avg+3*std
higher_limit


# In[71]:


lower_limit = avg-3*std
lower_limit


# In[73]:


#OUTLIERS
df_bookings[df_bookings.revenue_generated>higher_limit]


# In[75]:


df_bookings = df_bookings[df_bookings.revenue_generated<higher_limit]
df_bookings.shape


# In[76]:


df_bookings.revenue_realized.describe()


# In[77]:


df_bookings.describe()


# In[79]:


higher_limit = df_bookings.revenue_realized.mean()+(df_bookings.revenue_realized.std())*3
higher_limit


# In[81]:


df_bookings[df_bookings.revenue_realized>higher_limit].shape


# In[82]:


df_bookings[df_bookings.revenue_realized>higher_limit]


# In[83]:


df_bookings.room_category.value_counts()


# In[84]:


df_rooms


# In[86]:


#Standard Deviation for only RT4 Rooms

df_bookings[df_bookings.room_category == 'RT4']


# In[87]:


df_bookings[df_bookings.room_category == 'RT4'].revenue_realized.std()


# In[88]:


df_bookings[df_bookings.room_category == 'RT4'].revenue_realized.describe()


# In[90]:


higher_limit_rt4 = 23439.308444 + (3*9048.599076)
higher_limit_rt4


# In[91]:


#Handling NA values
df_bookings.isnull().sum()


# In[93]:


#Total values in our dataframe is 134576. Out of that 77899 rows has null rating. 
#Since there are many rows with null rating, we should not filter these values.
#Also we should not replace this rating with a median or mean rating etc 


# In[94]:


df_agg_bookings.isnull().sum()


# In[95]:


df_agg_bookings.head(3)


# In[96]:


df_agg_bookings.shape


# In[97]:


df_agg_bookings[df_agg_bookings.capacity.isna()]


# In[98]:


df_agg_bookings.capacity.fillna(df_agg_bookings.capacity.median(), inplace = True)


# In[101]:


df_agg_bookings.loc[[8,15]]


# In[110]:


df_agg_bookings.loc[:,['room_category','capacity']]


# In[111]:


df_agg_bookings[df_agg_bookings.successful_bookings >df_agg_bookings.capacity]


# In[112]:


df_agg_bookings.shape


# In[114]:


df_agg_bookings = df_agg_bookings[df_agg_bookings.successful_bookings <= df_agg_bookings.capacity]
df_agg_bookings


# # Data Transformations

# In[115]:


df_agg_bookings['Occupancy_pct'] = df_agg_bookings['successful_bookings']/df_agg_bookings['capacity']
df_agg_bookings


# In[117]:


df_agg_bookings['Occupancy_pct'] = df_agg_bookings['Occupancy_pct'].apply(lambda x: round(x*100, 2))


# In[119]:


df_agg_bookings.head()


# # Insight Generation - Answering Ad hoc questions

# What is the average occupancy rate in each of the room categories?

# In[122]:


df_agg_bookings.groupby('room_category')['Occupancy_pct'].mean().round(2)


# In[123]:


df_rooms.head()


# Performing Joins of agg_booking and rooms table

# In[124]:


df = pd.merge(df_agg_bookings, df_rooms, left_on="room_category", right_on= "room_id")


# In[125]:


df


# In[126]:


df.groupby('room_class')['Occupancy_pct'].mean().round(2)


# In[137]:


df.groupby('room_class').agg({'successful_bookings':'mean', 'capacity':'sum'})


# In[134]:


df.groupby('room_class')['Occupancy_pct'].agg(['sum','mean'])


# In[138]:


df.drop('room_id',axis=1, inplace = True)


# Printing averge occupancy rate per city

# In[139]:


df_hotels.head()


# In[142]:


df1 = pd.merge(df,df_hotels, on ='property_id')
df1


# In[144]:


df1.groupby('city')["Occupancy_pct"].mean()


# In[146]:


df1.groupby('city')["Occupancy_pct"].mean().plot(kind='bar')


# When was the occupancy better? Weekday or weekend

# In[148]:


df_date.head(3)
            


# In[149]:


df3 = pd.merge(df1, df_date, left_on = 'check_in_date', right_on = 'date')
df3.head(3)


# In[150]:


df3.groupby('day_type')["Occupancy_pct"].mean().round(2)


# In the month of June, what was the occupancy for different cities?

# In[154]:


df3['mmm yy'].unique()


# In[166]:


df3_June = df3[df3['mmm yy'] == 'Jun 22']
df3_June.head(3)


# In[170]:


df3_June.groupby('city')['Occupancy_pct'].mean().round(2).sort_values(ascending=False)
                                                                      


# In[173]:


df_Aug = pd.read_csv('new_data_august.csv')


# In[174]:


df_Aug.columns


# In[175]:


df_Aug.shape


# In[180]:


df3.shape


# In[181]:


df.shape


# In[182]:


df1.shape


# In[184]:


latest_df = pd.concat([df3, df_Aug], ignore_index = True, axis = 0)


# In[186]:


latest_df.tail(10)


# In[187]:


latest_df.shape


# Print revenue realized per city

# In[188]:


df_bookings.head(5)


# In[189]:


df_hotels.head(4)


# In[190]:


df_bookings_all = pd.merge(df_bookings, df_hotels, on= 'property_id')


# In[191]:


df_bookings_all


# In[192]:


df_bookings_all.groupby('city')['revenue_realized'].sum()


# Print Month by Month Revenue

# In[193]:


df_date['mmm yy'].unique()


# In[194]:


df_date.head()


# In[195]:


pd.merge(df_bookings_all, df_date, left_on ='check_in_date', right_on  ='date' )


# In[196]:


df_date['date'] = pd.to_datetime(df_date['date'])


# In[197]:


df_date.info()


# In[198]:


df_bookings_all['check_in_date'] = pd.to_datetime(df_bookings_all['check_in_date'])


# In[199]:


df_bookings_all.info()


# In[203]:


df_bookings_all = pd.merge(df_bookings_all, df_date, left_on ='check_in_date', right_on  ='date' )
df_bookings_all.head(4)


# In[204]:


df_bookings_all.groupby("mmm yy")["revenue_realized"].sum()


# Revenue Realized per hotel?

# In[205]:


df_bookings_all.columns


# In[206]:


df_bookings_all.property_name.unique()


# In[208]:


df_bookings_all.groupby('property_name')['revenue_realized'].sum().round(2).sort_values()


# Print Average Rating per city

# In[209]:


df_bookings_all.groupby('city')['ratings_given'].mean().round(2)


# Print Revenue realized per booking platform

# In[210]:


df_bookings_all.groupby('booking_platform')['revenue_realized'].sum().plot(kind='pie')


# In[ ]:




