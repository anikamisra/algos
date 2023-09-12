import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

dataset = pd.read_csv("Name gender dataset - name_gender_dataset.csv")
del dataset['Probability']
del dataset['Count']

print(dataset.shape) 

# There are 147,268 rows and 2 columns. Meaning there are 147,268 names. Let's see how many of those names are repeated.

print(len(dataset['Name'].unique()))

# 13359 repeated names 


