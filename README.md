### Context ### 
Recreating a model that can predict gender based on names, taken from an article from Analytics Vidhya. 
However, the dataset I am using will be different. This might change the results of the model from the article.

### Sources ### 
[Article](https://www.analyticsvidhya.com/blog/2023/03/name-based-gender-identification-using-nlp-and-python/)

[Dataset](https://data.world/howarder/gender-by-name)

### Skills used ### 
ML: 
- NLP, character vectorization 
- Logistic Regression
- Naive Bayes
- XGBoost
- LSTM 

Visualizations:
- heat maps
- confusion matrix
- column charts 
- technologies: matplotlib, sklearn.metrics

### Outcome ### 
XGBoost model has the highest accuracy, with 74% accuracy in predicting boys' names, 66% in girls' names. LSTM model has slighty higher accuracies - if possible, train the entire dataset on the LSTM model (~ 3 hours)
