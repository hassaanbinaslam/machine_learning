# Machine Learning A-Z

# import the data
import pandas as pd
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

transactions = []
for i in range(0, dataset.shape[0]):
    row = []
    for j in range(0, dataset.shape[1]):
        row.append(str(dataset.values[i,j])) 
    transactions.append(row)
    
# training aprioir on the dataset
from apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)