#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd 


# In[9]:


Nucleus = pd.read_csv('CAKI2_Nucleus_Diameter2.txt', sep='\t',header=None, names = ['Nucleus_Object_Number', 'Nucleus_Diameter'])[2:]
Nucleus.head()


# In[10]:


Cell = pd.read_csv('CAKI2_Cell_Diameter2.txt', sep='\t',header=None, names = ['Cell_Object_Number', 'Cell_Diameter'])[2:]
Cell.head()


# In[11]:


data = pd.concat([Cell,Nucleus],axis = 1 )
data.head()


# In[12]:


data = data.drop(columns=['Nucleus_Object_Number'])
data.head()


# In[13]:


data['Cell_Object_Number'] = data['Cell_Object_Number'].astype('float')
data['Cell_Diameter'] = data['Cell_Diameter'].astype('float')
data['Nucleus_Diameter'] = data['Nucleus_Diameter'].astype('float')
data.head()


# In[14]:


data.columns = ['Object_Number','Cell_Diameter','Nucleus_Diameter']
data.head()


# In[15]:


# Now to clean the data 
# First remove cells that have a nucleus diameter not equal to 0 
# Then remove cells that  have a cell diameter greater then there nucleus diaemter 
data = data[data.Nucleus_Diameter != 0]
data = data[data.Cell_Diameter > data.Nucleus_Diameter]


# In[16]:


data['NC'] = data['Nucleus_Diameter'] / data['Cell_Diameter']
data.head()


# In[17]:


data.to_csv('CAKI2NC.csv',index=False)


# In[18]:


pd.read_csv('CAKI2NC.csv')


# In[ ]:




