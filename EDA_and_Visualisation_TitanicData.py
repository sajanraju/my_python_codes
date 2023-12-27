#!/usr/bin/env python
# coding: utf-8

# # Exploratory data analysis and visualization using seaborn 

# In[64]:


# Exploratory data analysis and visualisation of the famous titanic data


# In[65]:


# import seaborn and matplotlib packages

import seaborn as sns
import matplotlib.pyplot as plt

# command to view the plots within the jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[94]:


# Set the aesthetic style of the plots
sns.set_style('ticks')
sns.set_palette('Set2')


# In[67]:


# load titanic data
titanic = sns.load_dataset('titanic')


# In[68]:


# # Load data using pandas
# import numpy as np
# import pandas as pd
# titanic = pd.read_csv('titanic.csv')


# In[69]:


# check the data and its column names
titanic.head()


# In[74]:


# see summary statistics of the data for the numerical columns
titanic.describe()


# In[95]:


# Check missing informations in the data
plt.figure(figsize=(8,8))
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[118]:


# Make a joint plot with fare and age as x and y resp.
sns.jointplot(x='fare',y='age', data=titanic,dropna=True)


# In[100]:


# Distribution plot
plt.figure(figsize=(5,5))
sns.displot(titanic['fare'],bins=40,kde=False)


# In[101]:


# box plot showing the age distributions with respect to class categories.
sns.boxplot(x='class',y='age',data=titanic)


# In[102]:



# sns.stripplot
sns.stripplot(x="class", y="age", data=titanic, palette='Set2')


# In[104]:


# Scatterplot with non-overlapping points
plt.figure(figsize=(8,6))
sns.swarmplot(x='class',y='age',data=titanic)


# In[105]:


sns.countplot(x='sex',data=titanic, palette='Set2')


# In[106]:


# Plot heatmap 
sns.heatmap(titanic.corr(),cmap='coolwarm')
plt.title('titanic.corr()')


# In[ ]:





# In[107]:



g = sns.FacetGrid(data=titanic,col='sex',height=4)
#plt.figure(figsize=(12,8),layout='constrained')
plt.figure(layout='constrained')
g.map(plt.hist,'age')


# In[ ]:





# 

# In[ ]:





# In[ ]:




