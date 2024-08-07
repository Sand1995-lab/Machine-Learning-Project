#!/usr/bin/env python
# coding: utf-8

# Consider the data present in the Insurance dataset file.<br>
# Following is the attribute related information:<br><br>
# 
# age: age of primary beneficiary<br>
# sex: insurance contractor gender, female, male<br>
# bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
# objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9<br>
# children: Number of children covered by health insurance / Number of dependents<br>
# smoker: Smoking, yes or no<br>
# region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.<br>
# charges: Individual medical costs billed by health insurance<br><br>
# 
# Problem statement: To predict the approximate insurance cost based upon the rest of the features provided for each individual.

# Import the libraries- Pandas, Numpy, Matplotlib and Seaborn

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import warnings
warnings.filterwarnings("ignore")


# Create a dataframe named "data" from the given datafile and print it's head

# In[ ]:


data=pd.read_csv(r"C:\Users\adars\Downloads\datasets_insurance.csv",header=0)
data.head()


# Understand the problem statement and the data, try understanding importance of each variable given.<br>
# Perform Exploratory Data Analysis- print datatypes, dimension and descriptive statistics.

# In[ ]:


print(data.dtypes)
print(data.shape)
print(data.describe())


# Check if there are missing values. If yes, handle them.

# In[ ]:


data.isnull().sum()


# Check for Assumption 1- There should be no outliers in the data.<br>
# Hint: Think logically, nothing needs to be worked upon here.

# In[ ]:


for x in data.columns:
    if data[x].dtypes!='object':
        data.boxplot(column = x)
        plt.show()


# Check for Assumption 2-Assumption of Linearity<br>
# Hint: Use kind="scatter" in the pairplot, because it wont be able to fit a line in categorical variables. Some might show no relationship, but remember they are categorical or important variables, leave it as it is.

# In[ ]:


sns.pairplot(data,x_vars=['age', 'bmi', 'children'],
             y_vars="charges", kind='reg')


# Create X and Y

# In[ ]:


data.columns


# In[ ]:


X=data[['age','sex','bmi','children','smoker','region']]
Y=data['charges']


# Check for Assumption 3-Assumption of Normality <br>
# Hint: You will find the data is highly positively skewed. So log transform the data.

# In[ ]:


print(X.shape)
print(Y.shape)


# In[ ]:


sns.distplot(Y)
plt.show()


# In[ ]:


sns.distplot(Y)
plt.show()


# In[ ]:


X.hist(bins=20)
plt.show()


# Convert Categorical variables to numerical- Sex, Smoker and Region<br>
# Hint: Make use of replace function.<br>
# Sex: Female-0,Male-1<br>
# Smoker: No-0, Yes-1<br>
# Region: northeast-0,northwest-1,southeast-2,southwest-3

# In[ ]:


data['sex'].replace(to_replace = 'female', value = 0, inplace=True)
data['sex'].replace(to_replace = 'male', value = 1, inplace=True)
data['smoker'].replace(to_replace = 'no', value = 0, inplace=True)
data['smoker'].replace(to_replace = 'yes', value = 1, inplace=True)
data['region'].replace(to_replace = 'northeast', value = 0, inplace=True)
data['region'].replace(to_replace = 'northwest', value = 1, inplace=True)
data['region'].replace(to_replace = 'southeast', value = 2, inplace=True)
data['region'].replace(to_replace = 'southwest', value = 3, inplace=True)


# In[ ]:


data.head()


# Check for the normality in the X variables. <br>
# Hint: Some variables make not look normal but realize that they are actually discrete valued.
#     No transformation required.

# In[ ]:





# Check for Assumption 4-No multicollinearity in the data<br>
# Try both the approaches-correlation and VIF.<br>
# Hint: You will find no high correlation. VIF might be high for a few variables but do not eliminate them because they are important as per the domain knowledge.

# In[ ]:


corr_df = X.corr(method = "pearson")
print(corr_df)

sns.heatmap(corr_df,vmax=1.0,vmin=-1.0,annot=True)
plt.show()


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

vif_df = pd.DataFrame()
vif_df["features"] = X.columns
vif_df["VIF Factor"] = [vif(X.values, i) for i in range(X.shape[1])]
vif_df.round(2)


# Split the data into train and test.<br>
# Hint: Make sure you are considering the log transformed Y.

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=10)
print(X_train.shape)


# In[ ]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print("Percent of train data",X_train.shape[0]/X.shape[0]*100)


# Build a base Linear Regression model using sklearn.

# In[1]:


#create a model object
lm = LinearRegression()
#train the model object
lm.fit(X_train,Y_train)

# print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)


# Create a zipped list of the slope coefficients to understand the equation better.<br>
# Write down the equation manually and try making sense out of it.

# In[ ]:


print(list(zip(X.columns, lm.coef_)))


# In[ ]:


X1=100
X2=100
X3=np.log1p(100)

Y_pred=3.3532913858151545+(0.0437425*X1)+(0.19303708*X2)+(-0.04895137*X3)
print(Y_pred)


# In[ ]:


X1=100
X2=200
X3=np.log1p(0)

Y_pred=3.3532913858151545+(0.0437425*X1)+(0.19303708*X2)+(-0.04895137*X3)
print(Y_pred)


# In[ ]:


X1=100
X2=100
X3=np.log1p(0)

Y_pred=3.3532913858151545+(0.0437425*X1)+(0.19303708*X2)+(-0.04895137*X3)
print(Y_pred)


# Predict using the model.

# In[ ]:


Y_pred=lm.predict(X_test)
print(Y_pred)


# In[ ]:


new_df=pd.DataFrame()
new_df=X_test.copy()

new_df["Actual sales"]=Y_test
new_df["Predicted sales"]=Y_pred
new_df=new_df.reset_index().drop("index", axis=1)
new_df


# Evaluate the model.

# In[ ]:


lm.score(X_train, Y_train)


# In[ ]:


r2=r2_score(Y_test,Y_pred)
print("R-squared:",r2)

rmse=np.sqrt(mean_squared_error(Y_test,Y_pred))
print("RMSE:",rmse)

adjusted_r_squared=1-(1-r2)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print("Adj R-square:",adjusted_r_squared)


# In[ ]:


new_df["Deviation"]=new_df["Actual sales"]-new_df["Predicted sales"]
new_df.to_excel("Sales Prediction.xlsx",header=True,index=True)
new_df.head()


# In[ ]:


sns.regplot(x=Y_train,y=lm.predict(X_train),ci=95)


# In[ ]:


sns.regplot(x=Y_test,y=lm.predict(X_test),ci=95)


# Perform Ridge and Lasso regression. Evaluate them as well.<br>
# Hint:Look at the fun in the Lasso Regression, ignore such model.

# In[ ]:


## RIDGE

lm = Ridge()
#train the model object
lm.fit(X_train,Y_train)

# print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)


# In[ ]:


Y_pred=lm.predict(X_test)

print(Y_pred)


# In[ ]:


new_df=pd.DataFrame()
new_df=X_test.copy()

new_df["Actual charges"]=Y_test
new_df["Predicted charges"]=Y_pred
new_df=new_df.reset_index().drop("index", axis=1)

new_df


# In[ ]:


r2=r2_score(Y_test,Y_pred)
print("R-squared:",r2)

rmse=np.sqrt(mean_squared_error(Y_test,Y_pred))
print("RMSE:",rmse)

adjusted_r_squared=1-(1-r2)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print("Adj R-square:",adjusted_r_squared)


# In[ ]:


# LASSO

from sklearn.linear_model import Lasso

lm = Lasso()
#train the model object
lm.fit(X_train,Y_train)

# print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)


# In[ ]:


Y_pred=lm.predict(X_test)
print(Y_pred)


# In[ ]:


r2=r2_score(Y_test,Y_pred)
print("R-squared:",r2)

rmse=np.sqrt(mean_squared_error(Y_test,Y_pred))
print("RMSE:",rmse)

adjusted_r_squared=1-(1-r2)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print("Adj R-square:",adjusted_r_squared)


# Scale the data using Standard Scaler to prepare it for SGD.

# In[ ]:


X=data[['age','sex','bmi','children','smoker','region']]
Y=data['charges']


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X=scaler.fit_transform(X)

#scaler.fit(X)
#X=scaler.transform(X)

print(X)


# In[ ]:


scaler=StandardScaler()
X=scaler.fit_transform(X)

np.set_printoptions(suppress=True) ##converting the scientific notations to float by suppressing the values

print(X)


# Split the data into train and test.<br>
# Hint: Make sure you are considering the log transformed Y.

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y_log, test_size=0.2, random_state=10)


# Build the SGDRegressor.

# In[ ]:


from sklearn.linear_model import SGDRegressor

lm = SGDRegressor(learning_rate="constant",  
                  #want to use a constant learning rate
                  eta0=0.1,  #alpha
                  shuffle=True, 
        #while going ahead with the next epoch shuffle the obs
                  random_state=10, #set seed
                  max_iter=1000, #max no of epochs
                  early_stopping=True,
                  #stop if zero convergence is reached first
                  n_iter_no_change=5) 
    #no of obs to wait for before concluding upon early stopping
lm.fit(X_train,Y_train)


# In[ ]:


print (lm.intercept_)
print (lm.coef_)


# Predict using the model. Evaluate the model. Perform trial and error to reach the optimum model.

# In[ ]:


Y_pred_new=lm.predict(X_test)
print(Y_pred)


# In[ ]:


r2=r2_score(Y_test,Y_pred_new)
print("R-squared:",r2)

rmse=np.sqrt(mean_squared_error(Y_test,Y_pred_new))
print("RMSE:",rmse)

adjusted_r_squared=1-(1-r2)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print("Adj R-square:",adjusted_r_squared)


# Write down your conclusion as to which is the final model that you would go with along with the reasons.

# In[ ]:


# Since the R-squared value of the RIDGE model is the highest,
# therefore I will use that model only to predict the charges of the test sample data.


# Once finalized the model, try predicting the following observations for me.<br>
# Create the dataframe named "X_test_sample" using the given dictionary object.<br>
# 
# sample_data={"age":[56,51,31,64,44,43,56],
#              "sex":[1,0,1,1,0,1,1],
#              "bmi":[19.95,18.05,34.39,25.6,23.98,32.6,33.725],
#              "children":[0,0,3,2,2,2,0],"smoker":[1,0,1,0,0,0,0],"region":[0,1,1,3,2,3,1]}
# 
# 
# 
# 

# In[ ]:


age = [56, 51, 31, 64, 44, 43, 56]
sex = [1, 0, 1, 1, 0, 1, 1]
bmi = [19.95,18.05,34.39,25.6,23.98,32.6,33.725]
children = [19.95,18.05,34.39,25.6,23.98,32.6,33.725]
smoker = [1,0,1,0,0,0,0]
region = [0,1,1,3,2,3,1]

X_test_sample = pd.DataFrame()

X_test_sample['age'] = age
X_test_sample['sex'] =  sex
X_test_sample['bmi'] = bmi
X_test_sample['children'] = children
X_test_sample['smoker'] = smoker
X_test_sample['region'] = region


print(X_test_sample)


# In[ ]:


lm = Ridge()
lm.fit(X_train,Y_train)

print(lm.intercept_)
print(lm.coef_)


# In[ ]:


X_test_sample=X_test_sample.reset_index(drop=True)
X_test_sample


# In[ ]:


charges_pred = lm.predict(X_test_sample)
print(charges_pred)


# The predicted values would be log transformed.Convert them back to original values.<br>
# Hint: Use np.exp()

# In[ ]:


np.exp(charges_pred)


# Save the predicted values along with the observations into an excel file.

# In[ ]:


charges_pred = [2.30011609e+16, 1.06662367e+15, 1.49974497e+15, 1.54480194e+18,
       9.71044554e+14, 2.00827231e+16, 4.69818259e+18]
X_test_sample['charges_pred']= charges_pred

print(X_test_sample)


# In[ ]:


file_name = 'X_test_sample.xlsx'
X_test_sample.to_excel(file_name)
print('dataframe is written to Excel File Successfully')

