#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
os.getcwd()


# In[2]:


os.listdir(os.getcwd())


# In[3]:


df = pd.read_csv('pmsm_temperature_data.csv')
df.head()


# In[4]:


#represents number of instances and attributes
df.shape


# In[5]:


df.info()
#Data has only float and integer values
#No variable column has null/missing values


# In[6]:


df.describe()
#since there is not a large diff between 75% percentile and max values of predictors , there are no outliers in dataset


# In[ ]:




