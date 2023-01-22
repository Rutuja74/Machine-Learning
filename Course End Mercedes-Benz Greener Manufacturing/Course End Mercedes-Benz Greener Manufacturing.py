#!/usr/bin/env python
# coding: utf-8

# # Mercedes-Benz Greener Manufacturing
# 
# DESCRIPTION
# 
# Reduce the time a Mercedes-Benz spends on the test bench.
# 
# Problem Statement Scenario:
# Since the first automobile, the Benz Patent Motor Car in 1886, Mercedes-Benz has stood for important automotive innovations. These include the passenger safety cell with a crumple zone, the airbag, and intelligent assistance systems. Mercedes-Benz applies for nearly 2000 patents per year, making the brand the European leader among premium carmakers. Mercedes-Benz is the leader in the premium car industry. With a huge selection of features and options, customers can choose the customized Mercedes-Benz of their dreams.
# 
# To ensure the safety and reliability of every unique car configuration before they hit the road, the company’s engineers have developed a robust testing system. As one of the world’s biggest manufacturers of premium cars, safety and efficiency are paramount on Mercedes-Benz’s production lines. However, optimizing the speed of their testing system for many possible feature combinations is complex and time-consuming without a powerful algorithmic approach.
# 
# You are required to reduce the time that cars spend on the test bench. Others will work with a dataset representing different permutations of features in a Mercedes-Benz car to predict the time it takes to pass testing. Optimal algorithms will contribute to faster testing, resulting in lower carbon dioxide emissions without reducing Mercedes-Benz’s standards.
# 
# Following actions should be performed:
# 
# If for any column(s), the variance is equal to zero, then you need to remove those variable(s).
# Check for null and unique values for test and train sets.
# Apply label encoder.
# Perform dimensionality reduction.
# Predict your test_df values using XGBoost.

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split,KFold
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


# In[2]:


#importing data
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')


# In[3]:


#checking the shape of dataset
train_data.shape


# In[4]:


test_data.shape


# In[5]:


#checking the categorical columns in dataset
def cat_type(y):
    for x in y.columns:
        if train_data[x].dtype=='object':
            print(x)
        
cat_type(train_data)


# In[6]:


cat_type(test_data)


# # Label Encoding the same in train and test dataset

# In[7]:



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_data['X0'] = le.fit_transform(train_data.X0)
train_data['X1'] = le.fit_transform(train_data.X1)
train_data['X2'] = le.fit_transform(train_data.X2)
train_data['X3'] = le.fit_transform(train_data.X3)
train_data['X4'] = le.fit_transform(train_data.X4)
train_data['X5'] = le.fit_transform(train_data.X5)
train_data['X6'] = le.fit_transform(train_data.X6)
train_data['X8'] = le.fit_transform(train_data.X8)


# In[8]:


train_data.shape


# In[9]:


test_data['X0'] = le.fit_transform(test_data.X0)
test_data['X1'] = le.fit_transform(test_data.X1)
test_data['X2'] = le.fit_transform(test_data.X2)
test_data['X3'] = le.fit_transform(test_data.X3)
test_data['X4'] = le.fit_transform(test_data.X4)
test_data['X5'] = le.fit_transform(test_data.X5)
test_data['X6'] = le.fit_transform(test_data.X6)
test_data['X8'] = le.fit_transform(test_data.X8)


# In[10]:


test_data.shape


# In[11]:


#Visualization part 
train_visual=train_data.set_index('y')

count_final =train_visual[train_visual==1].stack().reset_index().drop(0, axis=1)
count_final


# In[12]:


#undestanding the outliers 
count_final.plot(kind='box')
plt.show()


# In[13]:


#removing the outliers
outlier=train_data[train_data['y'] > 130].index
print(outlier)
train_data.drop(outlier,inplace=True)
test_data.drop(outlier,inplace=True)


# In[14]:


train_data.shape


# In[15]:


test_data.shape


# # If for any column(s), the variance is equal to zero, then need to remove those variable(s).

# In[16]:


train_data.var()


# In[17]:


test_data.var()


# In[18]:


#getting the variance==0 features
var_zero=train_data.var()[train_data.var()==0].index
var_zero


# In[19]:


#droping those features whoes var==0
train_data.drop(var_zero,axis=1,inplace=True)
test_data.drop(var_zero,axis=1,inplace=True)


# In[20]:


train_data.shape


# In[21]:


test_data.shape


# # Check for null and unique values for test and train sets.
# 

# In[22]:


print(train_data.isnull().sum().any())
print(test_data.isnull().sum().any())


# In[23]:


# Checking the unique values for each column in train data
for i in train_data.columns:
    print(i,'**',train_data[i].unique())


# In[24]:


# Checking the unique values for each column in train data
for i in test_data.columns:
    print(i,'**',test_data[i].unique())


# In[25]:


#splitting the data for scaling
xt=train_data.iloc[:,2:]
tx=test_data


# In[26]:


#Scaling the data 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_training= sc.fit_transform(xt)
x_testing=sc.fit_transform(tx)


# In[27]:


x_training.shape


# In[28]:


x_testing.shape


# # Perform dimensionality reduction.

# In[29]:


from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
pca_test=PCA(n_components=146)
x_t=pca.fit_transform(x_training)
t_x=pca_test.fit_transform(x_testing)


# In[30]:


#training dataset after scaling
x_t.shape


# In[31]:


#testing dataset after scaling
t_x.shape


# In[32]:


#creating the dependent feature for our training dataset
y=train_data.y


# In[33]:


y.shape


# # Building XGBoost Regressor

# In[34]:


#creating the instance and building the model
regressor=XGBRegressor(n_estimators=500,learning_rate=0.3,max_depth=5,objective='reg:linear',eval_metric= 'rmse')


# In[35]:


x_train,x_test,y_train,y_test=train_test_split(x_t,y,test_size=.25)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[36]:


#fiting the model
regressor.fit(x_train,y_train)


# In[37]:


#predicting the model
y_predict=regressor.predict(x_test)
y_predict


# In[38]:


#actual y values for our test dataset
y_test


# In[39]:


#Evaluating the model
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
print('RMSE = ',sqrt(mean_squared_error(regressor.predict(x_test),y_test)))


# # Predict test_data values using our regressor xgboost model

# In[40]:


#prediction through our test data
X_new=pd.DataFrame(t_x)
regressor.predict(X_new)


# In[41]:


#this is the RMSE score 
print('RMSE = ',sqrt(mean_squared_error(regressor.predict(X_new),y)))


# # CONCLUSION

# Insights-
# 1.	Removing low variance features contributes in increasing model performance.
# 2.	No null, unique and duplicates value found in the train and test dataset
# 3.	Hyperparameter tuning prevents the overfitting of model.
# 4.	Adding PCA featurization (try to keep 95% important data) which helps in dimensionality reduction of models, which    contributes in decreasing Rmse.
# 5.	Predicted the test_data using XGBoost.
# 

# In[ ]:




