#!/usr/bin/env python
# coding: utf-8

# In[127]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[129]:


data = pd.read_csv("C:/Users/maria/True.csv")


# In[130]:


data.head()


# In[131]:


data.describe()


# In[132]:


data.info()


# In[133]:


print(data.columns)


# In[134]:


data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')


# In[135]:


print(data.columns)


# In[136]:


#drops missing values
data = data.dropna()


# In[137]:


#drops duplicates
data = data.drop_duplicates()


# In[138]:


#checks if ther is any missing values
data.isnull().sum()


# In[139]:


# Convert to datetime
data['date'] = pd.to_datetime(data['date'], errors = 'coerce')

# Sort the data by 'date'
data = data.sort_values(by='date', ascending=False)


# In[142]:


# Rename columns
data.rename(columns={
    'title': 'Article Title',
    'text': 'Content',
    'subject': 'Category',
    'date': 'Publication Date'
}, inplace=True)


# In[149]:


print(data.columns)


# In[152]:


data.to_csv('Cleaned_True_Data.csv', index=False)

