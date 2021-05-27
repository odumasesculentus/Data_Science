#!/usr/bin/env python
# coding: utf-8

# In[34]:


#import the required modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import *
style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix, f1_score

#The PolynomialFeatures will be used for feature creation to explore the nonlinear pattern of the numerical data.
from sklearn.preprocessing import PolynomialFeatures
#The Pipeline is used to package the feature creator and the classifier.
from sklearn.pipeline import Pipeline


# For feature creation
# Degree 2 is used here but one can set the degree to be a hyperparameter to further explore the accuracy of the model
poly = PolynomialFeatures(degree = 2)

#importing the classifiers
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

#for combining multiple models into one 
from sklearn.ensemble import VotingClassifier


# In[2]:


#read the dataset 
#the data was saved on my computer, so you can edit the file path to suit you
#this could be a url

loan_train = pd.read_csv("/Users/apple/Documents/OPEN_UNIVERSITY/Data_for_DataScience/Loan-Eligibility-Prediction-Project/train.csv")
loan_test = pd.read_csv("/Users/apple/Documents/OPEN_UNIVERSITY/Data_for_DataScience/Loan-Eligibility-Prediction-Project/test.csv")


# In[3]:


#set what the datasets look like
print(loan_train.shape)
loan_train.head()


# In[4]:


print(loan_test.shape)
loan_test.head()


# In[5]:


# info about the datasets
print(loan_train.info())
print(loan_train.isnull().sum())


# In[6]:


print(loan_test.info())
print(loan_test.isnull().sum())


# #One would notice that the test.csv data does not contain the loan status column. The implication of this is that one would not be able to evaluate the accuracy of the models from the data. Hence, only the train.csv data would be used for this project.

# # Data Analysis 

# In[7]:


#Some analysis of the data is done to check the dependence of the feature variables to the label variable
plt.rcParams['figure.figsize'] = (10.0, 5.0)
fig, ax = plt.subplots(nrows = 1, ncols = 3)
sns.countplot(ax = ax[0], x = loan_train["Loan_Status"])
ax[0].set_title("Loan Status count")
sns.countplot(ax = ax[1], x = loan_train["Gender"])
ax[1].set_title("Gender count")
sns.countplot(ax = ax[2], x = loan_train["Self_Employed"])
ax[2].set_title("Self-employed count")


# In[8]:


#checking dependencies
plt.rcParams['figure.figsize'] = (15.0, 10.0)
fig, ax = plt.subplots(nrows = 2, ncols = 3)
sns.countplot(ax = ax[0,0], x = "Loan_Status", hue = "Married", data = loan_train)
#for minor ticks
ax[0,0].yaxis.set_minor_locator(MultipleLocator(10))
sns.countplot(ax = ax[0,1], x = "Loan_Status", hue = "Gender", data = loan_train)
ax[0,1].yaxis.set_minor_locator(MultipleLocator(10))
sns.countplot(ax = ax[0,2], x = "Loan_Status", hue = "Education", data = loan_train)
ax[0,2].yaxis.set_minor_locator(MultipleLocator(10))
sns.countplot(ax = ax[1,0], x = "Loan_Status", hue = "Property_Area", data = loan_train)
ax[1,0].yaxis.set_minor_locator(MultipleLocator(5))
sns.countplot(ax = ax[1,1], x = "Loan_Status", hue = "Dependents", data = loan_train)
ax[1,1].yaxis.set_minor_locator(MultipleLocator(10))
sns.countplot(ax = ax[1,2], x = "Loan_Status", hue = "Credit_History", data = loan_train)
ax[1,2].yaxis.set_minor_locator(MultipleLocator(10))
plt.show()


# From the charts above, 
# 
#   - about 59% of the unmarried people are not eligible to get a loan while only about 42% of the married people are eligible. Hence, one is more likely to get a loan if the person is married.
#   
#   - about 43% of the female gender was rejected loan while about 44% of the male gender was rejected loan. Hence, one can say that the loan eligibility does not necessarily depend on the person's gender.   
#   
#   - about 63% of the non-graduates are not eligible to get a loan while only about 41% of the graduates are eligible. Hence, a non-graduate is less likely to be eligible for a loan.
#   
#   - one is more likely to get a loan if the property is in a semiurban area giving that about 39%, 34% and 23% of the people whose properties are in the rural, urban and semiurban areas respectively are not eligible for a loan,
#   
#   - the number of people with or without dependent that are not eligible for a loan are as follows:  0: 45%, 1:35%, 2:26%, and 3+: 40%. Hence, one can say that a person is more likely to be eligible for a loan if he/she has 1 or 2 dependents. 
#  
#   - one is more less likely to be eligible for a loan if their credit history is bad

# In[9]:


#The distribution of the applicant's income
plt.rcParams['figure.figsize'] = (15.0, 10.0)
figs, ax = plt.subplots(nrows = 2, ncols = 3)
loan_train['ApplicantIncome'].plot.hist(ax = ax[0,0], title = 'Applicant Income')
ax[0,0].yaxis.set_minor_locator(MultipleLocator(20))
ax[0,0].xaxis.set_minor_locator(MultipleLocator(5000))

loan_train['CoapplicantIncome'].plot.hist(ax = ax[0,1], title = 'Co-applicant Income')
ax[0,1].yaxis.set_minor_locator(MultipleLocator(20))
ax[0,1].xaxis.set_minor_locator(MultipleLocator(5000))


loan_train['LoanAmount'].plot.hist(ax = ax[0,2], title = 'Loan Amount')
ax[0,2].yaxis.set_minor_locator(MultipleLocator(10))
ax[0,2].xaxis.set_minor_locator(MultipleLocator(20))

loan_train['Loan_Amount_Term'].plot.hist(ax = ax[1,0], title = 'Loan Amount Term')
ax[1,0].yaxis.set_minor_locator(MultipleLocator(20))
ax[1,0].xaxis.set_minor_locator(MultipleLocator(20))

ax[1,1].scatter(loan_train['ApplicantIncome'], loan_train['Loan_Status'])
ax[1,1].set_title('Applicant Income')
ax[1,1].xaxis.set_minor_locator(MultipleLocator(2000))

ax[1,2].scatter( x = loan_train['CoapplicantIncome'], y = loan_train['Loan_Status'])
ax[1,2].set_title('Co-applicant Income')
ax[1,2].xaxis.set_minor_locator(MultipleLocator(2000))

plt.show()
print("   ")
print('Mean Applicant Income  = ', np.mean(loan_train['ApplicantIncome']))
print('Mean Co-applicant Income  = ', np.mean(loan_train['CoapplicantIncome']))
print('Mean Loan Amount  = ', np.mean(loan_train['LoanAmount']))
print('Mean Loan Amount Term  = ', np.mean(loan_train['Loan_Amount_Term']))


# # Data Wrangling

# In[10]:


#get rid of all the null values in the data

   #first the heat map for the null values
sns.heatmap(loan_train.isnull(), cmap="viridis")


# In[11]:


#now remove all the null values
loan_train.dropna(inplace = True)

#check the heatmap again
sns.heatmap(loan_train.isnull(), cmap="viridis", cbar = False)


# In[12]:


#or you can print out the isnull count
print(loan_train.isnull().sum())
print(loan_train.shape)


# # Data Preprocessing

# In[13]:


#the Loan_ID column is not needed, so it is dropped
loan_train.drop("Loan_ID", axis = 1, inplace = True)


# In[14]:


#encode the string values in the necessary column
encoder  = LabelEncoder()

#THE GENDER COLUMN: Female = 0, Male  = 1
loan_train["Gender"] = encoder.fit_transform(loan_train["Gender"])
loan_train = loan_train.rename(columns = {"Gender":"Male_Gender"})
loan_train["Male_Gender"].value_counts()


# In[15]:


#THE MARRIED COLUMN: Yes = 1, No = 0
loan_train["Married"] = encoder.fit_transform(loan_train["Married"])
loan_train["Married"].value_counts()


# In[16]:


#THE EDUCATION COLUMN: Graduate = 0, Not Graduate  = 1
loan_train["Education"] = encoder.fit_transform(loan_train["Education"])
loan_train = loan_train.rename(columns = {"Education":"Not_Graduate"})
loan_train["Not_Graduate"].value_counts()


# In[17]:


#THE SELF-ENPMPLOYED COLUMN: No = 1, Yes = 0
loan_train["Self_Employed"] = encoder.fit_transform(loan_train["Self_Employed"])
loan_train["Self_Employed"].value_counts()


# In[18]:


#THE SELF-ENPMPLOYED COLUMN: No = 0, Yes = 1
loan_train["Loan_Status"] = encoder.fit_transform(loan_train["Loan_Status"])
loan_train["Loan_Status"].value_counts()


# In[19]:


#THE PROPERTY AREA COLUMN
#This column has a multivalued. So, one can either use get_dummies or one-hot-encoder

property_area = pd.get_dummies(loan_train["Property_Area"], drop_first = True)
print(property_area.head())


# In[20]:


#THE DEPENDENT COLUMN: This is also multivalued
dependents_ = pd.get_dummies(loan_train["Dependents"])
print(dependents_.head())


# In[21]:


#one can then concatenate this with the main data
loan_train = pd.concat([loan_train, property_area, dependents_], axis = 1)

#drop the previous property area, dependents and the 3+ columns
loan_train.drop(["Property_Area", "Dependents", "3+" ], axis = 1, inplace = True)
loan_train = loan_train.rename(columns = {"0":"0_dependent", "1":"1_dependent", "2":"2_dependents"})
loan_train.head()


# # Data Splitting

# In[22]:


#split the data into features and labels
x = loan_train.drop(["Loan_Status"], axis = 1)
y = loan_train["Loan_Status"]

#split the data into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 20)


# In[23]:


#scale the data due to large range of of the distribution
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # Employing the models

# In[81]:


#Keep all the classifiers in a list so that the testing and training can be done once and for all
#then one can choose the one with the best accuracy
classifiers_ = [
    ("AdaBoost", AdaBoostClassifier()),
    ("Decision Tree", DecisionTreeClassifier(max_depth=3)),
    ("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))),
    ("Linear SVM", SVC(kernel="linear", C=0.025,probability=True)),
    ("Naive Bayes",GaussianNB()),
    ("Nearest Neighbors",KNeighborsClassifier(3)),
    ("Neural Net",MLPClassifier(alpha=1)),
    ("QDA", QuadraticDiscriminantAnalysis()),
    ("Random Forest",RandomForestClassifier(n_jobs=2, random_state=1)),
    ("RBF SVM",SVC(gamma=2, C=1,probability=True)),
    ("SGDClassifier", SGDClassifier(max_iter=1000, tol=10e-3,penalty='elasticnet')),
    ("LogisticRegression", LogisticRegression()), 
    ("Perceptron", Perceptron(tol=1e-3, random_state=0)), 
    ("BaggingClassifier", BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0))
    ] 


# In[25]:


for n,clf in classifiers_:
    print("n = ",n, " and clf = ", clf)


# In[74]:


#use each Classifier to take its training results.
clf_names = []
train_scores = []
test_scores = []
accuracy_scores = []
predict_sums = []
test_f1score = []
i = 0
for n,clf in classifiers_:
    clf_names.append(n)
    # Model declaration with pipeline
    clf = Pipeline([('POLY', poly),('CLF',clf)])
    
    # Model training
    clf.fit(X_train, y_train)
    print(i, ":",  n+" training done! \n ")
    
    # The prediction
    clf.predict(X_test)
    predict_sums.append(clf.predict(X_test).sum()) #this gives the number of mines the classifier predicted
        #you can print the classification report and confusion matrix if you like
    print(classification_report(y_test, clf.predict(X_test)))
    print(confusion_matrix(y_test, clf.predict(X_test)))
    
    #you can also plot the confussion matrix if you like
    disp1 = plot_confusion_matrix(clf, X_train, y_train,
                              display_labels=['YES','NO'],
                              cmap=plt.cm.Blues,
                              normalize=None)
    disp1.ax_.set_title('Confusion matrix')
    plt.show()
    
    disp = plot_confusion_matrix(clf, X_test, y_test,
                              display_labels=['YES','NO'],
                              cmap=plt.cm.Blues,
                              normalize=None)
    disp.ax_.set_title('Confusion matrix')
    plt.show()
    
    # Measure training accuracy and score
    train_scores.append(round(clf.score(X_train, y_train), 3))
    print("The Training Score: ", clf.score(X_train, y_train) )
    print(n+" training score done!")
    
    # Measure test accuracy and score
    test_scores.append(round(clf.score(X_test, y_test), 3))
    accuracy_scores.append(round(accuracy_score(y_test, clf.predict(X_test)), 3))
    test_f1score.append(round(f1_score(y_test, clf.predict(X_test)),3))
    print("The Testing Score: ", clf.score(X_test, y_test) )
    print("The Accuracy Score: ", accuracy_score(y_test, clf.predict(X_test)))
    print("Test F1 Score: ",f1_score(y_test,clf.predict(X_test)))
    print(n+" testing score done!")
    print("-------------------------------------------------------")
    print("  ")
    i = i+1
print("Names: ", clf_names)
print("Predict Sum: ", predict_sums)
print("Train Scores: ", train_scores)
print("Test Scores: ", test_scores)
print("Accuracy Scores: ", accuracy_scores)
print("Test F1 Scores: ", test_f1score)


# In[75]:


#Plot results
#plt.title('Accuracy Training Score')
plt.rcParams['figure.figsize'] = (10.0, 20.0)
figs, ax = plt.subplots(3)
#plt.grid()
ax[0].scatter(x =  train_scores,y = clf_names)
ax[0].set_title("Train Accuracy")
ax[0].grid(True, color = 'g')
#plt.title('Test F1 Score')
#plt.grid()

ax[1].scatter(test_scores,clf_names)
ax[1].set_title("Test Accuracy")
ax[1].grid(True, color = 'g')
#plt.legend(loc = "lower left")

ax[2].scatter(test_f1score,clf_names)
ax[2].set_title("Test F1 Score")
ax[2].grid(True, color = 'g')
#plt.grid(True, color = 'g')

plt.show()

# print("   ")
# print("Actual number of loans given out: ", y_test.sum())
# plt.title('Loan eligibility prediction plot')
# plt.grid()
# plt.scatter(predict_sums,clf_names , color = "g")
#plt.legend(loc = "lower left")
# plt.title('Test F1 Score')
# plt.grid()
# plt.scatter(test_f1score,clf_names, label = "Test F1 Score")
plt.show()


# From the graphs above, one would notice overfitting on the Random Forest classifier despite not showing the best test accuracy. The Bagging Clasifier appears to have the best test accuracy and f1 scores. This is followed by the QDA, Decision Tree, Gaussian Process and Nearest Neighbours.  
# 
# One can combine the  few best multiple models into a single model to obtain a hopefully better accuracy. The models are combined with specific weights. It is therefore necesary to iterate through various combinations of the weights, and obtain the maximum accuracy score. 
# 

# In[76]:


#use the VotingClassifier to combine the models: take Bagging Classifier and QDA
combine_score1 = []
for i in range(20):
    for j in range(20):
                ensemble=VotingClassifier(estimators=[("BaggingClassifier", BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0)), ("QDA", QuadraticDiscriminantAnalysis())], 
                       voting='hard', weights=[i,j]).fit(X_train, y_train)
                combine_score1.append(ensemble.score(X_test, y_test))
print('The accuracy for Bagging Classifier and QDA is:',round(max(combine_score1),3))


# This accuracy score is less than using the Bagging Classifier alone (with score 0.875). Let's give three models a try

# In[70]:


#use the VotingClassifier to combine the models: take Bagging Classifier, QDA and Decision Tree
combine_score = []
for i in range(20):
    for j in range(20):
        for k in range(20):
                ensemble=VotingClassifier(estimators=[("BaggingClassifier", BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0)), ("QDA", QuadraticDiscriminantAnalysis()), ("Decision Tree", DecisionTreeClassifier(max_depth=3))], 
                       voting='hard', weights=[i,j,k]).fit(X_train, y_train)
                combine_score.append(ensemble.score(X_test, y_test))
print('The accuracy for Bagging Classifier, QDA and Decision Tree is: ',round(max(combine_score),3))


# We can also get the most importance features for determing the result of the loan eligibility. This can basically be included when deploying the models but it is done here for clarity purpose. This is not straight-forward since we are using the pipepline. 

# In[169]:


#to obtain the feature importance, 

theclassifiers = classifiers_ = [
    ("AdaBoost", AdaBoostClassifier()),
    ("Decision Tree", DecisionTreeClassifier(max_depth=3)),
    ("Random Forest",RandomForestClassifier(n_jobs=2, random_state=1))
    ] 
#NB: these classifiers are considered because they are only once amongst the classfiers considered that run with the 
# "feature_importances_" code line

for n,clif in theclassifiers:
    clif = Pipeline([('POLY', poly),('CLF',clif)])
    clif.fit(X_train, y_train)
    feature_names = clif.named_steps["POLY"].get_feature_names()
#put the name of the column heads in a list
    column_head = loan_train.columns
#obtain the coefficients of the features
    coefs = clif.named_steps["CLF"].feature_importances_.flatten()

# Zip coefficients and names together and make a DataFrame
    zipped = zip(feature_names, coefs)
    df = pd.DataFrame(zipped, columns=["feature", "value"])

#since we need the names of the column heads as the tick labels
    for i in range(len(column_head)):
        df["feature"][i] = column_head[i]
    
#Make a bar chart of the coefficients
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.barplot(x=df["feature"][0:len(column_head)],
                y=df["value"][0:len(column_head)])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=20)
    ax.set_title(n, fontsize=25)
    ax.set_ylabel("Coefficients", fontsize=22)
    ax.set_xlabel("Feature Name", fontsize=22)
    ax.grid(True, color = "g")
    plt.show()


# The scores suggest that the important features, which are features with non-zero coefficients, are dependent on the model deployed. However, the three plots above show that the Loan_Status appear to be generally "the most" important feature. Basically, all other features with a zero coefficient are essentially removed them from the model.
