#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os

os.getcwd()


# In[5]:


df = pd.read_csv('emt.csv')
df.head()


# In[5]:


df.shape # representing no. of rows and columns


# In[21]:


df.info() 
#Data has only float and integer values
#No variable column has null/missing values


# In[22]:


df.describe()
#since there is not a large diff between 75% percentile and max values of predictors , there are no outliers in dataset.


# In[17]:


#sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap = 'viridis')
sns.heatmap(df.isnull(), cbar=False)
#no missing values in data


# In[12]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),cmap='Blues',annot=
            True) 


# In[3]:


import matplotlib.pyplot as plt 
  
# x axis values 
x = [-1.222428,-1.2224314,0.52815247,2.0241134,2.0241182] 
# corresponding y axis values 
y = [-2.522071,-2.5226767,-2.520822,-2.4117308,-2.1538434] 
  
# plotting the points  
plt.plot(x, y) 
  
# naming the x axis 
plt.xlabel('motor speed') 
# naming the y axis 
plt.ylabel('rotor temp') 
# function to show the plot 
plt.show() 


# In[7]:


import numpy as np
import matplotlib.pyplot as plt

# Create data
N = 100
x = np.random.rand(N)
y = np.random.rand(N)
colors = (0,0,0)
area = np.pi*3

# Plot
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[18]:


import seaborn as sns
df=pd.read_csv('emt.csv')
df = sns.load_dataset('df')
g = sns.pairplot(df)


# In[ ]:


#random=np.random.choice(df['pm'],1000)
# Seaborn visualization library
import seaborn as sns
# Create the default pairplot
sns.pairplot(df)


# In[8]:


unique_profiles_id = df['profile_id'].unique()

print('Number of unique profiles: %i'%len(unique_profiles_id))


# In[9]:



from random import randint
profile = randint(0,len(unique_profiles_id)) # Get a random index

corr=df[df['profile_id']==unique_profiles_id[profile]].drop('profile_id',axis=1).corr()
print('The correlation heatmap is for the profile ID %i and its is index %i'%(unique_profiles_id[profile], profile))

f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(corr, annot=True, linewidths=.5, fmt='.2f', mask= np.zeros_like(corr,dtype=np.bool), 
            cmap=sns.diverging_palette(100,200,as_cmap=True), square=True, ax=ax)

plt.show()


# In[11]:


num_point_profile = np.zeros(len(unique_profiles_id))
for i in range(len(unique_profiles_id)):
    num_point_profile[i] = df[df['profile_id']==unique_profiles_id[i]].shape[0]

print('Profile ID with minimum number of points %i and it has %i points' %(unique_profiles_id[np.where(num_point_profile == np.amin(num_point_profile))],
                                                                 min(num_point_profile)))    
f, ax = plt.subplots(figsize=(18,5))
plt.bar(unique_profiles_id, (num_point_profile/df.shape[0])*100)
plt.xlabel('Profile ID')
plt.ylabel('Number of points compared to the \n total number of points(%)')


# In[12]:


profile = randint(0,len(unique_profiles_id)) # Get a random index 
print('The profile ID shown is %i and it has %i points' %(unique_profiles_id[profile],num_point_profile[profile]))

sns.pairplot(df[df['profile_id']==unique_profiles_id[profile]][['i_q', 'torque']])


# In[13]:


fig = plt.figure(figsize=(17, 5))
grpd = df.groupby(['profile_id'])
_df = grpd.size().sort_values().rename('samples').reset_index()
ordered_ids = _df.profile_id.values.tolist()
sns.barplot(y='samples', x='profile_id', data=_df, order=ordered_ids)
tcks = plt.yticks(2*3600*np.arange(1, 8), [f'{a} hrs' for a in range(1, 8)]) # 2Hz sample rate

## 
X_train=df['ambient'].values
X_train=X_train[:998000]
X_train=np.reshape(X_train, (998, 1000))
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=1000).fit_transform(X_train)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=36, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

#plt.figure()
plt.figure(figsize=(50,25))
for yi in range(36):

    plt.subplot(6, 6, yi + 1)
    
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")
#####
X_train=df['motor_speed'].values
X_train=X_train[:998000]
X_train=np.reshape(X_train, (998, 1000))
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=1000).fit_transform(X_train)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=36, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

#plt.figure()
plt.figure(figsize=(50,25))
for yi in range(36):

    plt.subplot(6, 6, yi + 1)
    
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")







