#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the package

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import plotly.express as px


# In[3]:


df = pd.read_csv("C:/Users/rosyi/Desktop/Data Coding/Netflix-subs-fee-2021.csv") #import the data from local


# ### Basic Dataframe Info and Basic Data Visualization

# In[4]:


df.head() 


# In[5]:


df.describe()


# In[7]:


df.info()


# In[8]:


# display cost per month (basic, standard, and premium) for every country in descending way
print('Basic')
display(df.sort_values(by='Cost Per Month - Basic ($)', ascending = False))
print('Standard')
display(df.sort_values(by = 'Cost Per Month - Standard ($)', ascending = False))
print('Premium')
display(df.sort_values(by = 'Cost Per Month - Premium ($)', ascending = False))


# In[9]:


# Figure of the cost for basic fee in all countries
fig1 = px.bar(df, x='Country', y='Cost Per Month - Basic ($)', color = "Cost Per Month - Basic ($)")
fig1.show()


# In[10]:


# Comparing subscription prices from each country for Basic, Standard, and Premium plans
from matplotlib import rcParams
rcParams['figure.figsize'] = 20,6
plt.plot(df.Country_code, df['Cost Per Month - Basic ($)'], label = 'Basic')
plt.plot(df.Country_code, df['Cost Per Month - Standard ($)'], label = 'Standard')
plt.plot(df.Country_code, df['Cost Per Month - Premium ($)'], label = 'Premium')
plt.title('Perbandingan Harga Langganan Tiap Negara')
plt.xlabel('Country')
plt.ylabel('Cost Per Month $')
plt.legend(loc = 2)
plt.show


# In[11]:


# Distribution of each variable to country
def plot_hist(variable):
    figure = px.bar(df, x = 'Country', y = df[variable], color = df[variable])
    figure.show()
    
variable = ['Total Library Size','No. of TV Shows','No. of Movies']
for n in variable:
    plot_hist(n)


# In[12]:


# Variable correlation to price
df.plot(x = 'Total Library Size', y = 'Cost Per Month - Basic ($)', kind = 'scatter')
plt.show()


# In[13]:


df_corr = df[['Total Library Size', 'No. of TV Shows', 'No. of Movies', 'Cost Per Month - Basic ($)']]
sns.heatmap(df_corr.corr(), annot = True)

