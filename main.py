import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

a = pd.read_csv('calories.csv')

b = pd.read_csv('exercise.csv')

c = pd.concat([b, a['Calories']], axis=1)
c.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)
print(c)
k = c.describe()

sns.countplot(x='Gender', data=c)
correlation = c.corr()
# constructing a heatmap to understand correlation between the columns
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 18})

x = c.drop(columns={'User_ID', 'Calories'}, axis=1)
y = c['Calories']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LinearRegression()
model.fit(x_train, y_train)
print(model.predict(x_test))

with open('calorie_burnt_predictor', 'wb') as f:
    pickle.dump(model, f)
    