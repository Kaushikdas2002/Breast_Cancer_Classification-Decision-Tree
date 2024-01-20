#!/usr/bin/env python
# coding: utf-8

# # Predicting whether the Breast cancer is malignant(M) or benign(B) 

# #### Importing the required libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# #### Reading the dataset

# In[2]:


df = pd.read_csv("D://Breast Cancer.csv")
df.head()


# In[3]:


df.tail()


# #### Dataset Analysis

# In[4]:


df['Unnamed: 32'].unique()


# In[5]:


df.drop('Unnamed: 32',axis=1,inplace=True)
df.head()


# In[6]:


df.shape


# In[7]:


duplicated_rows=df[df.duplicated()]
print("The number of duplicate rows are :",duplicated_rows.shape)


# In[8]:


df.isnull().sum()


# In[9]:


df.info()


# In[10]:


df.drop('id',axis=1,inplace=True)
df.head()


# In[11]:


df.describe()


# In[12]:


df.shape


# In[13]:


df['diagnosis'].value_counts()


# In[14]:


df.groupby('diagnosis').mean()


# ### From the above table we can visualize that the value of mean for each attribute of malignant is greater than the mean of benign . 

# #### Assigning the independent variables to 'x' and target variable to 'y'

# In[15]:


x = df.drop('diagnosis',axis=1)
y = df['diagnosis']


# In[16]:


x


# In[17]:


y


# #### Splitting the dataset into training and testing data

# In[18]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)


# #### Training Decision Tree Classifier on training set

# In[19]:


dtreecla = DecisionTreeClassifier()


# In[20]:


dtreecla.fit(x_train,y_train)


# #### Predicting the result

# In[21]:


y_pred = dtreecla.predict(x_test)


# In[22]:


db = pd.DataFrame({'Actual :':y_test,'Predicted :':y_pred})
db


# In[23]:


from sklearn.metrics import confusion_matrix


# In[24]:


confusion_matrix(y_test,y_pred)


# In[25]:


from sklearn.metrics import accuracy_score


# In[26]:


accuracy_score(y_test,y_pred)


# # Our model is approximately 96% accurate .
