#!/usr/bin/env python
# coding: utf-8

# # Project: Healthcare

# In[1]:


import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#importing the dataset 
df=pd.read_excel('1645792390_cep1_dataset.xlsx')


# # 1.Preliminary analysis:
# 
# a.	Perform preliminary data inspection and report the findings on the structure of the data, missing values, duplicates, etc.
# 
# b.	Based on these findings, remove duplicates (if any) and treat missing values using an appropriate strategy
# 

# In[3]:


#checking the 5 instances of the data 
df.head()


# In[4]:


#checking the shape of the data 
df.shape


# In[5]:


#checking the last five instances of the data 
df.tail()


# In[6]:


#checking the duplicates 
df.duplicated().sum().any()


# In[7]:


#dropping the duplicates 
df.drop_duplicates(inplace=True)


# In[8]:


df.shape


# In[9]:


#checking the null values in dataset 
df.isna().sum().any()


# In[10]:


#checking the unique values in dataset 
df.nunique(axis=0)


# In[11]:


#It appears we have a good balance between the two binary outputs.
df.target.value_counts()


# # 2.	Prepare a report about the data explaining the distribution of the disease and the related factors using the steps listed below:
# 
# a.	Get a preliminary statistical summary of the data and explore the measures of central tendencies and spread of the data
# 
# b.	Identify the data variables which are categorical and describe and explore these variables using the appropriate tools, such as count plot 
# 

# In[12]:


df.describe()


# In[13]:


df.dtypes


# In[14]:


cat = df.loc[:,['sex','cp','fbs','exang','slope','thal']]
cat


# In[15]:


plt.figure(figsize=(5,5))
sns.countplot(cat.cp,palette='Blues_r')
plt.show()
#in the case of cp we have 4 unique values in that 0 is more dominant compare to other values.
sns.countplot(cat.fbs,palette='Accent')
plt.show()
#In fbs we have 2 values in which 0 is more comapre to 1 it means fasting blood sugar is less than 120mg/ml
sns.countplot(cat.exang,palette='viridis')
plt.show()
#In exang hve 2 values in which 0 is more comapre to 1 it means exercise induced angina is no compare to yes.


# In[16]:


cat.describe()


# # c.	Study the occurrence of CVD across the Age category

# In[17]:


sns.boxplot(df.age)
plt.show()


# As we can see that there is no outliers in age column

# In[18]:


#doing some visualization on age feature
plt.figure(figsize=(10,5))
sns.countplot(df.age)
plt.show()


# In[19]:


sns.histplot(df.age)
plt.show()


# In[20]:


df.age.mean()


# # d.	Study the composition of all patients with respect to the Sex category

# In[21]:


sns.countplot(df.sex)
plt.title('Sex Distribution')
#1 = male; 0 = female
plt.show()


# In[ ]:





# In[22]:


df[df['target']==1].groupby('sex')['target'].size().plot(kind='bar',color='orange')
plt.show()
#1 = male; 0 = female


# As we can see that the males are more prone to heartattack/heartdisease compare to women

# In[23]:


#d.Study the composition of all patients with respect to the Sex category
df.groupby('sex').mean()
#we can see that mean age for female is 55 and for male is 53  and chol level is 261 and 239 and major diff is in ca and thal respectively 


# In[24]:


df.groupby('sex').mean().plot(kind='bar')
plt.show()
#1 = male; 0 = female
#we can see that chol level is high in women compare to men.


# # e.	Study if one can detect heart attacks based on anomalies in the resting blood pressure (trestbps) of a patient

# In[25]:


trest_bps=pd.DataFrame(df[df['target']==1].groupby('trestbps')['target'].value_counts().reset_index(name='Target_counts'))
trest_bps.sort_values(by='Target_counts',ascending=False)


# In[26]:


trest_bps_False=pd.DataFrame(df[df['target']==0].groupby('trestbps')['target'].value_counts().reset_index(name='Target_counts'))
sns.lineplot(x=trest_bps['trestbps'],y=trest_bps['Target_counts'],label='True')
sns.lineplot(x='trestbps',y='Target_counts',data=trest_bps_False,label='False')
plt.legend()
plt.show()


# In[27]:


sns.boxplot(x=trest_bps['trestbps'])
plt.title('CVD TRUE')


# In[28]:


sns.boxplot(x=trest_bps_False['trestbps'],color='r')
plt.title('CVD False')


# In[29]:


#as we can see that there are outliers in above 190 so let's drop those rows for better results 
ol=df[df['trestbps']>190].index
ol
df.drop(ol,inplace=True)


# In[30]:


sns.scatterplot(df.trestbps,df.age,hue=df.target)
plt.show()


# ### Overall we can conclude that higer the Resting blood pressure lower the chances of getting heartattack and vice versa.

# ## f.	Describe the relationship between cholesterol levels and a target variable

# In[31]:


plt.figure(figsize=(20,5))
chol_relation=pd.DataFrame(df[df['target']==1].groupby('chol')['target'].value_counts().reset_index(name='Target_counts'))
chol_relation_false=pd.DataFrame(df[df['target']==0].groupby('chol')['target'].value_counts().reset_index(name='Target_counts'))
sns.lineplot(x=chol_relation['chol'],y=chol_relation['Target_counts'],label='True')
sns.lineplot(x=chol_relation_false['chol'],y=chol_relation_false['Target_counts'],label='False')
#sns.lineplot(x='trestbps',y='Target_counts',data=trest_bps_False,label='False')


# In[32]:


sns.boxplot(x=chol_relation['chol'],color='orange')
plt.title('CVD TRUE')
#we can see that there is a outlier above 350 chol level 


# In[33]:


sns.boxplot(x=chol_relation_false['chol'],color='green')
plt.title('CVD False')
#we can see that there is a outlier above 350 chol level 


# In[34]:


#outlier treament 
ol_2=df[df['chol']>350].index
print(ol_2)
df.drop(ol_2,inplace=True)


# We can say that mostly the density of chol level is between 200-350 and the above plots doesn't show the great relationship between target variable but overall we can interpret that chol level between 200-300 has higher chances of getting the heartattack.

# ## g.	State what relationship exists between peak exercising and the occurrence of a heart attack

# In[35]:


df[df['target']==1].groupby('oldpeak')['target'].count().plot()


# In[36]:


sns.boxplot(df.oldpeak)
plt.show()
#above 4 we have outliers 


# In[37]:


df[df['target']==1].groupby('slope')['target'].count().plot()
plt.show()


# In[38]:


sns.boxplot(df.slope,color='k')
plt.show()


# In[39]:


#treatment for outliers 
ol_3=df[df['oldpeak'] > 4].index
ol_3
df.drop(ol_3,inplace=True)


# Overall we can conclude that there are some outliers present in oldpeak data and no outliers in slope data and after looking at the distribution of the data we can say that data is not a normal distribution apart from that we can say that lower the old peak and higher the slope lead to higher risk of heartattack as per data and vice versa.

# ## h.Check if thalassemia is a major cause of CVD

# In[40]:


#let's check though we have any outliers or not in thalassemia 
sns.boxplot(df.thal,color='k')
plt.show()


# So we have the outliers at the lower side so let's considered the minimum threhold as 1.0.

# In[41]:


#Let's check the relationship 
df.groupby('thal')['target'].value_counts().unstack().plot(kind='bar')
plt.show()


# In[42]:


#treatment of outliers 
ol_4=df[df['thal']< 1.0].index
df.drop(ol_4,inplace=True)


# We can see those who are having thal as 2 there is a higher chances that person will suffer Heart disease followed by other.

# ##  i.	List how the other factors determine the occurrence of CVD

# In[43]:


#Let's plot the correlation matrix 
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[44]:


#Correlation with output variable
cor=df.corr()
cor_target = (cor["target"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0]
relevant_features


# In[45]:


#mean distribution of the dataset 
pd.pivot_table(df,index='target')


# # j.	Use a pair plot to understand the relationship between all the given variables

# In[46]:


sns.pairplot(df)
plt.show()


# ### 3.	Build a baseline model to predict the risk of a heart attack using a logistic regression and random forest and exploring the results while using correlation analysis and logistic regression (leveraging standard error and p-values from statsmodels) for feature selection.
# 
# 

# ### LOGISTIC REGRESSION

# In[47]:


#Dividing the data into x and y variable 
x=df.iloc[:,:-1].values
y=df.target.values
print(x)
print(y)


# In[48]:


#scaling the data 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x)
x_scaled=sc.transform(x)


# In[49]:


#RFE for Feature selection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# feature extraction
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=.25)
model = LogisticRegression(solver='lbfgs')


# In[50]:


rfe=RFE(estimator=model)
fit=rfe.fit(x_train,y_train)


# In[51]:


fit.n_features_
#as we can see the RFE has selected top 6 features 


# In[52]:


fit.support_


# In[53]:


df.columns


# In[54]:


fit.ranking_


# In[55]:


#we have selected the top 6 features #logit function  
x_new=df.loc[:,['sex','cp','thalach','oldpeak','ca','thal']].values
y_new=df.target.values
import statsmodels.api as sm
logit_model=sm.Logit(y_new,x_new)
result=logit_model.fit()
print(result.summary2())

#as we can see that the p value is less than our critical error rate that 0.05


# In[56]:


#building our model 
new_df=df.loc[:,['sex','cp','thalach','oldpeak','ca','thal','target']]
x_new_scaled=new_df.drop('target',axis=1).values
y=new_df.target.values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x_new_scaled)
x_scaled_final=sc.transform(x_new_scaled)
x_train,x_test,y_train,y_test=train_test_split(x_scaled_final,y,test_size=.25)


# In[57]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[58]:


#building and training the model
final_model_lr=LogisticRegression()
final_model_lr.fit(x_train,y_train)
y_pred_final=final_model_lr.predict(x_test)


# In[59]:


#Evaluation
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test,y_pred_final))
print('accuracy score :',accuracy_score(y_test,y_pred_final)*100 )


# ### Random Forest Classifier 

# In[77]:


#Using RFE for feature selection
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
rfe=RFE(estimator=clf)

# feature extraction
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=.25)
fit_random=rfe.fit(x_train,y_train)


# In[70]:


fit_random.n_features_


# In[71]:


fit_random.ranking_


# In[72]:


df.columns


#  We can see that in random forest classifier our RFE have selected the 6 different features comapre to Logistic regression model

# In[73]:


#Building and training the model 
#we have selected the top 6 features 
x_random_clf=df.loc[:,['cp','thalach','exang','oldpeak','ca','thal','target']]
x=x_random_clf.drop('target',axis=1).values
y=x_random_clf.target.values
#scaled the new data 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x)
x_scaled_final=sc.transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_scaled_final,y,test_size=0.30)


# In[74]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[78]:


#building and training the model
final_model_random=RandomForestClassifier()
final_model_random.fit(x_train,y_train)
y_pred_final=final_model_random.predict(x_test)


# In[79]:


#Evaluation
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test,y_pred_final))
print('accuracy score :',accuracy_score(y_test,y_pred_final)*100)


# CONCLUSION:
# Performace of both Random Forest is quite better compare to Logistic Regression
# The Features which is a strong representation of target varibale are'cp','thalach','exang','oldpeak','ca','thal' in case of random forest classifier.
# The Features which is a strong representation of target variables are 'sex','cp','thalach','oldpeak','ca','thal' in case of Logistic Regression.
# But more or less than there are some common features in both the models. 
