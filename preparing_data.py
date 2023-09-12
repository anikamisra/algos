import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

# ------------------------------------------------------------------------------------------------------------
# PREPARE DATA 
# ------------------------------------------------------------------------------------------------------------
dataset = pd.read_csv("Name gender dataset - name_gender_dataset.csv")
dataset.head()
del dataset['Probability']
del dataset['Count']

print(dataset.shape) 
# There are 147,268 rows and 2 columns. Meaning there are 147,268 names. Let's see how many of those names are repeated.

print(len(dataset['Name'].unique()))
# 13359 repeated names 

# ------------------------------------------------------------------------------------------------------------
# VISUALIZATIONS
# ------------------------------------------------------------------------------------------------------------
# visualizing class imbalance 
sns.countplot(x='Gender', data = dataset)
plt.title('No. of male and female names in the dataset')
plt.xticks([0,1],('Female','Male'))

# visualizing letter counts 
alphabets = ['A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
startletter_count = {}
for i in alphabets:
  startletter_count[i] = len(dataset[dataset['Name'].str.startswith(i)])
print(startletter_count)
plt.figure(figsize = (16,8))
plt.bar(startletter_count.keys(), startletter_count.values())
plt.xlabel('Letter that the name starts with')
plt.ylabel('No. of names')


# ------------------------------------------------------------------------------------------------------------
# PREPARING FOR MODEL 
# ------------------------------------------------------------------------------------------------------------
# predicting gender from name! 
X = list(dataset['Name'])
Y = list(dataset['Gender'])

#now we will convert the F and M labels to machine-readable format
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

# count vectorize names with character level n-grams as features 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer='char')
X=cv.fit_transform(X).toarray()

# randomize data and split into testing and training sets 
from sklearn.model_selection import train_test_split 
dataset = dataset.sample(n = 147268, random_state = 0)
dataset = dataset.drop([15041])
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42) 

# ------------------------------------------------------------------------------------------------------------
# MODEL 
# ------------------------------------------------------------------------------------------------------------
# logistic regression 
from sklearn.linear_model import LogisticRegression  
LR_model = LogisticRegression(max_iter=1000)
LR_model.fit(x_train, y_train)
LR_y_pred = LR_model.predict(x_test)

# Naive Bayes 
from sklearn.naive_bayes import MultinomialNB
NB_model = MultinomialNB()
NB_model.fit(x_train, y_train)
NB_y_pred = NB_model.predict(x_test)

# 








