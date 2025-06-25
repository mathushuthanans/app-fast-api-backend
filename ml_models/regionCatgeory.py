import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("regionCategory.csv")

features = ['pm25', 'pm10', 'no2', 'o3', 'co']
X = df[features]
y = df['region_class']

xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state = 42)

bestModel = GradientBoostingClassifier(learning_rate= 0.01, max_depth= 3, n_estimators= 50)

bestModel.fit(xtrain, ytrain)
ypred = bestModel.predict(xtest)
print("Model - Ready to be Called")