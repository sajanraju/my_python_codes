#!/usr/bin/env python
# coding: utf-8

# ## Linear regression project

# An online clothing store (fake data) provides in-store style and clothing advice sessions. Customers visit the store, meet with personal stylists, and later order clothes they like using either a mobile app or the website.
# 
# The company is deciding where to put more effort: improving their mobile app or their website.

# In[29]:


# Import necessary packages:  pandas, numpy, matplotlib, and seaborn

import pandas as pd ;print("pandas ", pd. __version__)
import numpy as np; print("numpy ",np. __version__)

#Visualisation packages
import matplotlib; print("matplotlib",matplotlib. __version__)
import matplotlib.pyplot as plt; 
import seaborn as sns ;print("seaborn ",sns. __version__)
get_ipython().run_line_magic('matplotlib', 'inline')
# code to view the plots inside the notebook


# In[7]:


# Load the data
ecom_data = pd.read_csv("Ecommerce Customers")


# In[8]:


# check the data

ecom_data.head()


# In[9]:


# Get the statistics of numerical columns
ecom_data.describe()


# In[10]:


# Check the summary of the dataframe
ecom_data.info()


# ### Exploratory data analysis (EDA)

# In[12]:


# Set the aesthetic style of the plots
sns.set_style('whitegrid')
sns.set_palette('Set2')


# In[13]:


# jointplot to see the time on website vs yearly amount spent
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=ecom_data)


# In[14]:


# jointplot to see the time on app vs yearly amount spent
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=ecom_data)


# In[61]:


# check the pairwise relationship in the data
sns.pairplot(data=ecom_data, kind="reg", diag_kind="kde")


# From the above plot, length of membership and time on app seems to be the most correlated feature with Yearly Amount Spent

# In[34]:


# Create a linear model plot (lmplot) of Yearly Amount Spent vs. Length of Membership. 


# In[38]:


plt.figure(figsize=(8,8))
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=ecom_data)
plt.title('Length of Membership vs Yearly Amount Spent')


# In[49]:


plt.figure(figsize=(8,8))
sns.lmplot(x='Time on App',y='Yearly Amount Spent',data=ecom_data)
plt.title('Time on App VS Yearly Amount Spent')


# ## Training and Testing Data
# 

# In[51]:


# let's split the data into training and testing sets. 
# Set a variable X equal to the numerical features of the e_customers and 
# a variable y equal to the "Yearly Amount Spent" column.


# In[52]:


y = ecom_data['Yearly Amount Spent']


# In[53]:


X = ecom_data[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# In[54]:


# Use model_selection.train_test_split from sklearn to split the data into training and testing sets. 
# Set test_size=0.3 and random_state=101 to get the reproducible result


# In[55]:


from sklearn.model_selection import train_test_split


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ### Training the Model

# In[57]:


# Train our model on our training data!
# Import sklearn     
from sklearn.linear_model import LinearRegression


# In[58]:


lm = LinearRegression()


# In[68]:


# fit the linear model on the training data
lm.fit(X_train,y_train)


# In[69]:


print("Intercept for the model is", lm.intercept_, "and the scope is",lm.coef_)


# In[67]:


# print the coefficients
print(lm.intercept_)
print('Coefficients: \n', lm.coef_)


# ### Predicting Test Data
# 

# In[71]:


# let's evaluate our model performance by predicting off the test values!
predictions = lm.predict( X_test)


# In[63]:


# scatterplot of the real test values versus the predicted values
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# ### Evaluating the Model
# 

# Evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
#  Calculate the 
# -  Mean Absolute Error, 
# -  Mean Squared Error, and 
# -  Root Mean Squared Error.

# In[65]:


# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[72]:


# Residuals- Plot a histogram of the residuals and make sure its normally distributed
sns.distplot((y_test-predictions),bins=50);


# In[79]:


coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# In[80]:


# 1 unit increase in variables is associated with an increase of corresponding coeffecient amount of dollars spent.


# Upon analysis, it's clear that the time spent on the website lags behind the performance of the mobile app. **Deciding whether to focus on enhancing the website's performance or further developing the app** depends on additional contextual factors within the company. Exploring the correlation between the length of membership and either the app or the website would be crucial before reaching a definitive conclusion. Understanding these relationships would better guide the decision-making process.
# 
# 
# 
# 
# 

# In[ ]:





# In[ ]:




