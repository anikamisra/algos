sns.countplot(x='Gender', data = dataset)
plt.title('No. of male and female names in the dataset')
plt.xticks([0,1],('Female','Male'))
