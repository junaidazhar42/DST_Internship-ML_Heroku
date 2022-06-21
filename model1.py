#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts

from imblearn.over_sampling import RandomOverSampler

from collections import Counter

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("New_Insurance_Dataset.csv")


# In[3]:


df.shape


# In[4]:


df.dtypes


# In[6]:


# Removing Spaces in Dataset Column Names

df.columns = df.columns.to_series().apply(lambda x: x.replace(' ', '_')).to_list()
df.columns = df.columns.to_series().apply(lambda x: x.replace('/', '_')).to_list()
df.columns.values


# In[7]:


df.rename(columns={'Home_or_self_care,': 'Home_or_self_care'}, inplace=True)


# In[8]:


df.drop_duplicates(inplace=True)


# In[9]:


df.isnull().sum()


# ### Dropping Null Values

# In[10]:


df.dropna(inplace = True)


# In[11]:


df.isnull().sum()


# ### Firstly, we can find unique value in 'Days_spend_hsptl' through the following piece of code. Secondly, the column 'Days_spend_hsptl', Hospital_Id and Mortality_risk are of object datatype so we'll change it to int datatype 

# In[12]:


# After executing the following code we get '120+' as a unique value
for i in df.Days_spend_hsptl.unique():
    try:
        int(i)
    except ValueError:
        print(i)


# In[13]:


# Replacing '120 +' with '120' the numerical value
df.replace({'Days_spend_hsptl': '120 +'}, 120, inplace=True)


# In[15]:


# Converting the columns which are 'object' datatype into 'int' datatype
df.Days_spend_hsptl = df.Days_spend_hsptl.astype('int')
df.Mortality_risk = df.Mortality_risk.astype('int')



# ### The observed dataset has to be balanced and for that, Random Over Sampler technique is used.
# 1) It should be noted that the technique will be applied only on the training dataset and not on the test dataset. 
# 2) For that, the dataset is split into training and test sets.
# 3) Also for simplification, the datatypes of all the columns was changed to 'int' or 'float'.

# In[31]:


le = LabelEncoder()


# In[32]:


le


# In[33]:


var = df.select_dtypes(include = 'object').columns


# In[34]:


var


# In[35]:


for i in var:
    df[i] = le.fit_transform(df[i])


# In[36]:


# The variable x will store the dataset without the 'Result' column
# The variable y will store the dropped column 'Result'
x = df.drop(['Result'], axis=1)
y = df['Result']


# In[37]:


x.head()


# In[38]:


y.head()


# In[39]:


tts


# In[40]:


x_train, x_test, y_train, y_test = tts(x, y, test_size=0.20, random_state=0)


# In[41]:


#Create independent and Dependent Features
columns = df.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Result"]]
# Store the variable we are predicting 
target = "Result"
# Define a random state 
state = np.random.RandomState(42)
X = df[columns]
Y = df[target]
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)


# In[42]:


# Get the Fraud and the normal dataset 

fraud = df[df['Result']==1]

normal = df[df['Result']==0]


# In[43]:


print(fraud.shape,normal.shape)


# In[44]:


#Balancing the dataset
os =  RandomOverSampler(sampling_strategy=1)


# In[45]:


X_train_res, y_train_res = os.fit_sample(x_train, y_train)


# In[46]:


X_train_res.shape,y_train_res.shape


# In[47]:


print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(y_train_res)))


# In[48]:


ax = y_train_res.value_counts().plot.pie(autopct = '%.2f')
_ = ax.set_title("Random Over Sampling")


# ### Model Building

# In[49]:


from sklearn.tree import DecisionTreeClassifier


# In[50]:


model = DecisionTreeClassifier()


# In[51]:


model.fit(X_train_res, y_train_res)


# In[52]:


y_pred = model.predict(x_test) #Stores in Array


# In[53]:


y_pred[:5]


# In[54]:


y_test[:5]


# In[62]:


print("Training accuracy: ", model.score(X_train_res, y_train_res))
print("Testing accuracy: ", model.score(x_test, y_test))


# In[63]:


from sklearn import tree


# In[64]:


#Hyper Parameter Tuning
model1 = DecisionTreeClassifier(max_leaf_nodes=50)
model1.fit(X_train_res, y_train_res)


# In[85]:


#Hyper Parameter Tuning
model2 = DecisionTreeClassifier(max_leaf_nodes=100)
model2.fit(X_train_res, y_train_res)

# In[87]:


print("Training Accuracy: ", model1.score(X_train_res, y_train_res))
print("Testing Accuracy: ", model1.score(x_test, y_test))


# In[67]:


tree.plot_tree(model1)


# ### Random Forest Classifier with n_estimators = 80

# In[76]:


from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_estimators = 15)


# In[77]:


model3.fit(X_train_res, y_train_res)


# In[78]:


print("Training Accuracy: ", model3.score(X_train_res, y_train_res))
print("Testing Accuracy: ", model3.score(x_test, y_test))



# Saving model to disk
pickle.dump(model1, open('model1.pkl','wb'))

# Loading model to compare the results
model1 = pickle.load(open('model1.pkl','rb'))
print(model1.predict([[2, 9, 6]]))


