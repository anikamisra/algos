import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

dataset = pd.read_csv("Name gender dataset - name_gender_dataset.csv")
del dataset['Probability']
del dataset['Count']

# visualizing class imbalance 
sns.countplot(x='Gender', data = dataset)
plt.title('No. of male and female names in the dataset')
plt.xticks([0,1],('Female','Male'))

# visualizing first letter of each name 
alphabets = ['A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
startletter_count = {}
for i in alphabets:
  startletter_count[i] = len(dataset[dataset['Name'].str.startswith(i)])
print(startletter_count)
plt.figure(figsize = (16,8))
plt.bar(startletter_count.keys(), startletter_count.values())
plt.xlabel('Letter that the name starts with')
plt.ylabel('No. of names')
