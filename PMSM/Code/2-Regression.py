#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
os.getcwd()


# In[2]:


os.listdir(os.getcwd())


# In[9]:


#Load File
df=pd.read_csv('C:\\Users\\Mahima Sharu\\pmsm_temperature_data.csv')


# In[4]:


from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold             #Provides train/test indices to split data in train/test sets
from sklearn import linear_model
from sklearn.metrics import make_scorer

from sklearn import svm           #Support Vector Machine are a set of supervised learning methods
# linear algebra
import numpy as np
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns       #Visualizing dataset structure that can be used to make visualizations with multiple plots
from sklearn.linear_model import LinearRegression

from sklearn import neighbors
from math import sqrt


# In[10]:


#*************************************************Basic Linear Regression**************************************************

import statsmodels.api as sm          #provides classes and functions for the estimation of many different statistical models

#Defining dependent and independent variable
X = df['i_d']
X=sm.add_constant(X)

y = df['motor_speed']

lm=sm.OLS(y,X)               #Leastsquare Minimization using Ordinary Least Square value along the x and y axes
model=lm.fit()                #Data Fitting 

model.summary()


# In[11]:


model.params


# In[12]:


print("f_pvalue:", "%.4f" % model.f_pvalue)


# In[13]:


#mean square value
model.mse_model


# In[14]:


model.rsquared


# In[15]:


model.rsquared_adj


# In[16]:


#Predicted values
model.fittedvalues[0:5]


# In[17]:


#Real values
y[0:5]


# In[18]:


#Model equation
print("Motor speed = " + 
      str("%.3f" % model.params[0]) + ' + i_d' + "*" + 
      str("%.3f" % model.params[1]))


# In[19]:


#Regression Model Visualization 
g=sns.regplot(df['i_d'] , df['motor_speed'], 
              ci=None, scatter_kws={'color': 'r', 's':9})
g.set_title('Model equation: motor_speed = -0.002 + i_d * -0.725')
g.set_ylabel('Motor_speed')
g.set_xlabel('i_d');


# In[20]:


from sklearn.metrics import r2_score,mean_squared_error
mse=mean_squared_error(y, model.fittedvalues)
rmse=np.sqrt(mse)
rmse


# In[21]:


k_t=pd.DataFrame({'Real_values':y[0:50], 
                  'Predicted_values' :model.fittedvalues[0:50]})
k_t['error']=k_t['Real_values']-k_t['Predicted_values']
k_t.head()


# In[22]:


#Easiest way to learn residual
model.resid[0:10]


# In[23]:


plt.plot(model.resid);


# In[24]:


#**************************************Multiple Linear Regression**************************************************
X=df.drop("motor_speed", axis=1)
y=df["motor_speed"]


# In[25]:


from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

training=df.copy()


# In[26]:


lm=sm.OLS(y_train, X_train)

model=lm.fit()
#All coefficients are significant for the model by looking at the p-value. ( P>|t| )
model.summary()


# In[27]:


#Root Mean Squared Error for Train
rmse1=np.sqrt(mean_squared_error(y_train,model.predict(X_train)))
rmse1


# In[28]:


#Root Mean Squared Error for Test
rmse2=np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
rmse2


# In[29]:


#Model Tuning for Multiple Linear Regression
model = LinearRegression().fit(X_train,y_train)
cross_val_score1=cross_val_score(model, X_train, y_train, cv=10, scoring='r2').mean() #verified score value for train model
print('Verified R2 value for Training model: ' + str(cross_val_score1))

cross_val_score2=cross_val_score(model, X_test, y_test, cv=10, scoring='r2').mean() #verified score value for test model
print('Verified R2 value for Testing Model: ' + str(cross_val_score2))


# In[30]:


#For Root Mean square value
RMSE1=np.sqrt(-cross_val_score(model, X_train, y_train, cv=10, 
                               scoring='neg_mean_squared_error')).mean() #verified RMSE score value for train model
print('Verified RMSE value for Training model: ' + str(RMSE1))

RMSE2=np.sqrt(-cross_val_score(model, X_test, y_test, cv=10, 
                               scoring='neg_mean_squared_error')).mean() #verified RMSE score value for test model
print('Verified RMSE value for Testing Model: ' + str(RMSE2))


# In[31]:


#Visualizing for Multiple Linear Regression y values

import seaborn as sns
ax1 = sns.distplot(y_train, hist=False, color="r", label="Actual Value")
sns.distplot(y_test, hist=False, color="b", label="Fitted Values" , ax=ax1);


# In[33]:


#**************************************Principal Component Regression***************************************************
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

pca=PCA()
X_reduced_train=pca.fit_transform(scale(X_train))


# In[34]:


explained_variance_ratio=np.cumsum(np.round(pca.explained_variance_ratio_ , decimals=4)* 100)[0:20]


# In[35]:


plt.bar(x=range(1, len(explained_variance_ratio)+1), height=explained_variance_ratio)
plt.ylabel('percentange of explained variance')
plt.xlabel('principal component')
plt.title('bar plot')
plt.show()
# 7 component is enough for model.


# In[36]:


lm=LinearRegression()
pcr_model=lm.fit(X_reduced_train,y_train)
print('Intercept: ' + str(pcr_model.intercept_))
print('Coefficients: ' + str(pcr_model.coef_))


# In[37]:


#Prediction
y_pred=pcr_model.predict(X_reduced_train)
np.sqrt(mean_squared_error(y_train,y_pred))


# In[38]:


df['motor_speed'].mean()


# In[39]:


#R squared
r2_score(y_train,y_pred)


# In[40]:


# Prediction For testing error 
pca2=PCA()

X_reduced_test=pca2.fit_transform(scale(X_test))
pcr_model2=lm.fit(X_test,y_test)

y_pred=pcr_model2.predict(X_reduced_test)

print('RMSE for test model : ' +str(np.sqrt(mean_squared_error(y_test,y_pred))))


# In[41]:


#Model Tuning for PCR

lm=LinearRegression()
pcr_model=lm.fit(X_reduced_train[:,0:10],y_train)
y_pred=pcr_model.predict(X_reduced_test[:,0:10])

from sklearn import model_selection

cv_10=model_selection.KFold(n_splits=10,
                           shuffle=True,
                           random_state=1)


# In[44]:


lm=LinearRegression()
RMSE=[]

for i in np.arange(1,X_reduced_train.shape[1] + 1):
    score=np.sqrt(-1*model_selection.cross_val_score(lm,
                                                    X_reduced_train[:,:i],
                                                    y_train.ravel(),
                                                    cv=cv_10,
                                                    scoring='neg_mean_squared_error').mean())
    RMSE.append(score)


# In[43]:


plt.plot(RMSE)
plt.xlabel('# of Components')
plt.ylabel('RMSE')
plt.title('PCR Model Tuning for Motor_Speed Prediction'); 


# In[45]:


##10 component is good for the model because RMSE value is the smallest for this component number. 
##That's why there is no need to tune the model.


# In[6]:


#**********************Polynomial Regression****************************************
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
import pandas as pd


# In[7]:


df=pd.read_csv('C:\\Users\\Mahima Sharu\\pmsm_temperature_data.csv')
X=df.drop("motor_speed", axis=1)
y=df["motor_speed"]
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)
training=df.copy()

quad = PolynomialFeatures (degree = 2)
x_quad = quad.fit_transform(X_train)

X_train,X_test,y_train,y_test = train_test_split(x_quad,y_train, random_state = 0)

plr = LinearRegression().fit(X_train,y_train)

Y_train_pred = plr.predict(X_train)
Y_test_pred = plr.predict(X_test)

print('Polynomial Linear Regression:' ,plr.score(X_test,y_test))


# In[10]:


#Plotting Residual in Linear Regression 

import matplotlib.pyplot as plt
from sklearn import linear_model,metrics
#Create linear regression object
reg=linear_model.LinearRegression()

#train the model using the train data sets
reg.fit(X_train,y_train)

#regression coefficients
print("Coefficients: \n", reg.coef_)

#Variance score
print("Variance score: {}".format(reg.score(X_test,y_test)))

plt.style.use('fivethirtyeight')

#plotting residual errors in training data
plt.scatter(reg.predict(X_train),reg.predict(X_train)-y_train, 
            color="green", s=10, label="train data")

#plotting residual errors in test data
plt.scatter(reg.predict(X_test),reg.predict(X_test)-y_test, 
            color="blue", s=10, label="test data")

#plot line for zero residual error
plt.hlines(y=0,xmin=-2, xmax=2, linewidth=2)

#plot legend
plt.legend(loc='upper right')

#plot title
plt.title("residual error")

plt.show()


# In[ ]:




