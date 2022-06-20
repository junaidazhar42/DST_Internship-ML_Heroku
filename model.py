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
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler

from collections import Counter

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("Insurance_Dataset.csv")


# In[3]:


df.columns = df.columns.to_series().apply(lambda x: x.replace(' ', '_')).to_list()
df.columns = df.columns.to_series().apply(lambda x: x.replace('/', '_')).to_list()
df.columns.values


# In[4]:


df.rename(columns={'Home_or_self_care,': 'Home_or_self_care'}, inplace=True)


# In[5]:


df.drop_duplicates(inplace=True)


# In[6]:


df.dropna(inplace = True)


# In[7]:


# After executing the following code we get '120+' as a unique value
for i in df.Days_spend_hsptl.unique():
    try:
        int(i)
    except ValueError:
        print(i)


# In[8]:


# Replacing '120 +' with '120' the numerical value
df.replace({'Days_spend_hsptl': '120 +'}, 120, inplace=True)


# In[9]:


# Converting the columns which are 'object' datatype into 'int' datatype
df.Days_spend_hsptl = df.Days_spend_hsptl.astype('int')
df.Hospital_Id = df.Hospital_Id.astype('int')
df.Mortality_risk = df.Mortality_risk.astype('int')


# In[10]:


df['Result'].value_counts()


# In[11]:


le = LabelEncoder()


# In[12]:


var = df.select_dtypes(include = 'object').columns


# In[13]:


for i in var:
    df[i] = le.fit_transform(df[i])


# In[14]:


# The variable x will store the dataset without the 'Result' column
# The variable y will store the dropped column 'Result'
x = df[['Area_Service','Hospital_County',	'Age',	'Gender',	'Cultural_group',	'ethnicity',	'Days_spend_hsptl',	'Admission_type',	'Home_or_self_care',	'ccs_diagnosis_code',	'ccs_procedure_code',	'Code_illness',	'Mortality_risk',	'Surg_Description',	'Emergency_dept_yes_No',	'Tot_charg',	'Tot_cost',	'ratio_of_total_costs_to_total_charges',	'Payment_Typology']]
y = df['Result']


# In[15]:


x_train, x_test, y_train, y_test = tts(x, y, test_size=0.20, random_state=0)


# In[16]:


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


# In[17]:


#Balancing the dataset
os =  RandomOverSampler(sampling_strategy=1)


# In[18]:


X_train_res, y_train_res = os.fit_sample(x_train, y_train)


# In[19]:


print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(y_train_res)))


# In[20]:


from sklearn.tree import DecisionTreeClassifier


# In[21]:


model = DecisionTreeClassifier()


# In[22]:


model.fit(x_train, y_train)


# In[24]:


y_pred = model.predict(x_test)


# In[25]:


y_pred[:5]


# In[26]:


y_test[:5]


# In[27]:


print("Training accuracy: ", model.score(x_train, y_train))
print("Testing accuracy: ", model.score(x_test, y_test))


# In[28]:


from sklearn import tree


# In[29]:


#Hyper Parameter Tuning
model1 = DecisionTreeClassifier(max_leaf_nodes=50)
model1.fit(x_train, y_train)


# In[30]:


y_pred = model1.predict(x_test)


# In[31]:


print("Training Accuracy: ", model1.score(x_train, y_train))
print("Testing Accuracy: ", model1.score(x_test, y_test))


# In[32]:


tree.plot_tree(model1)


# In[ ]:

# Saving model to disk
pickle.dump(model1, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))



