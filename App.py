import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



df = pd.read_csv("social_media_usage-2.csv")


# In[134]:


# In[8]:


#2 define clean_sm and test

def clean_sm(x):
    is_one = np.where(x==1)
    if is_one[0] == 0:
        x = 1
    else:
        x = 0
    return x

# In[60]:


#3
ss1 = df[["income", "educ2","par","marital","gender","age","web1h"]]
ss1.columns = ["income","education","parents","marital","gender","age","web1h"]
for i in ss1.index:
    ss1.at[i, 'sm_li'] = clean_sm(ss1.at[i,'web1h'])

ss = ss1.drop(ss1[(ss1['income'] > 9) | (ss1['education'] > 8) | (ss1['age'] > 98)].index)

# Comparison of Income and Linkedin Usage
# 
# As a preliminary analysis, this bar chart shows that as income increases, so does the
# mean of the index indicating linkedin usage. There is a particular jump in the $75,000
# to $100,000 range
# 
# We can infer that there will be a strong positive correlation here

# In[61]:


# Continuted Exploratory Analysis
# 
# Comparison of Education and Linkedin Usage
# 
# As an additional preliminary analysis this bar graph shows a positive relationship between one's level of education and the likelihood that they use linkedin. Interestingly, the mean likelihood of linkedin usage dips at the postgraduate degree level. One possible inference for this is that many people have academia-focused post graduate degrees that do not require school.
# 
# We can infer that there will be a strong positive correlation here

# In[65]:


# 4 target and feature selections
y = ss["sm_li"]
x = ss[["income","education","parents","marital","age","gender"]]


# In[71]:


# 5 split data into test and training datasets, holding out 20 percent for testing
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    stratify = y,
                                                    test_size = .2 ,# holds out 20% for test
                                                    random_state = 987)
# x_train has 80% of the data and contains the features used to predict Linkedin Usage when training the to identify patters of usage
# x_test has 20% of the data and contains the features used to test the model and evaluate performance of the model on data not used to build the model
# y _train has 80% of the data and contains the target of linkedin usage that we are training the model to predict
# y_ test has 20% of the data and contains the target we will be predicting for, this data not used by the model is being used to
# evaluate the model's performance on a seperate section of the data


# In[74]:


#6 Initialize algorithm for regression 
lr = LogisticRegression(class_weight = "balanced")
# Fit algorithm to training data
lr.fit(x_train, y_train)


# In[116]:


# 7 Evaluate Model Performance

# Making a prediction using the model which we can then evaluate
y_pred = lr.predict(x_test)


# In[128]:


#10 Make Predictions on two sample individuals

# New data income, education, parents, marital, age, gender

# the probability that the person uses linkedin dipped from .77 to .51 when their
# age was changed from 42 to 82 and nothing else was changed

st.write("# Linkedin Prediction App")
st.write("## The purpose of this application is to predict whether you are a Linkedin user based on the given information")

income = st.selectbox("Income Range",
            options = 
                    ("",
                    "Less than $10,000",
                    "10 to under $20,000",
                    "20 to under $30,000",
                    "30 to under $40,000",
                    "40 to under $50,000",
                    "50 to under $75,000",
                    "75 to under $100,000",
                    "100 to under $150,000",
                    "$150,000 or more"))

if income  == "Less than $10,000":
    income = 1 
elif income == "10 to under 20,000":
    income = 2 
elif income == "20 to under $30,000":
    income = 3
elif income == "30 to under $40,000":
    income = 4
elif income == "40 to under $50,000":
    income = 5     
elif income == "50 to under $75,000":
    income = 6
elif income == "75 to under $100,000":                    
     income = 7                
elif income == "100 to under $150,000":
     income = 8
else:
     income = 9

education = st.selectbox("Education Level",
            options = 
                    ("",
                    "Less than high school",
                    "High school incomplete",
                    "High school graduate",
                    "Some college, no degree",
                    "Two-year associate degree from a college or university",
                    "Four-year college or university degree",
                    "Some postgraduate or professional schooling, no postgraduate degree",
                    "Postgraduate or professional degree"))

if education  == "Less than high school":
    education = 1 
elif education == "High school incomplete":
    education = 2 
elif education == "High school graduate":
    education = 3
elif education == "Some college, no degree":
    education = 4
elif education == "Two-year associate degree from a college or university":
    education = 5     
elif education == "Four-year college or university degree":
    education = 6
elif education == "Some postgraduate or professional schooling, no postgraduate degree":                    
     education = 7                
else:
     education = 8


parents = st.selectbox("Are you a parent to a child over 18?",
            options = 
                    ("",
                    "Yes",
                    "No"))
if parents == "Yes":
    parents = 1
else:
    parents = 0


gender = st.selectbox("Gender",
            options = 
                    ("",
                    "Male",
                    "Female",
                    "Other"))
if gender == "Male":
    gender = 1
elif gender == "Female":
    gender = 2
else: gender = 3 


marital = st.selectbox("Marital Status",
            options = 
                    ("",
                    "Married",
                    "Living with a partner",
                    "Divorced",
                    "Separated",
                    "Widowed",
                    "Never been married"))
if marital == "Yes":
    marital = 1
elif marital == "Living with a partner":
    marital = 2
elif marital == "Divorced":
    marital = 3
elif marital == "Separated":
    marital = 4
elif marital == "Widowed":
    marital = 5
else:
    marital = 6




age = st.slider(label = "Slide to your age",
                    min_value= 0,
                    max_value= 97,
                    value = 0)

submit = st.button("Submit")

if submit:
    newdata = pd.DataFrame({
    "income": [income],
    "education": [education],
     "parents": [parents],
    "marital": [marital],
    "age": [age],
    "gender": [gender]
    })                    

    predictedclass = lr.predict(newdata)
    probability = lr.predict_proba(newdata)
    classification = "LinkedIn User" if predictedclass[0] ==1 else "Not LinkedIn user"
    st.write(f"I think you are {classification}")
    st.write(f"The probability that you use LinkedIn is {probability[0][1]}")


# %%
