# Basic data loading for machine learning project.

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_absolute_error, mean_squared_error
import numpy as np
# Load the dataset
data=pd.read_csv('Project1/data.csv')
X=data[['Hours']] # double brackets=2d input.- find the column named 'Hours' and assign it to X.
y=data[['Scores']] # Target colmn is 'Scores' and assign it to y.

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())  

# Handle missing values (if any) - for example, by filling with mean
data.fillna(data.mean(), inplace=True)

'''
# Define features and target variable
X = data.drop('Scores', axis=1)  # Replace 'target' with the actual name of your target variable
y = data['Scores']  # Replace 'target' with the actual name of your target variable


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''

# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions on the test set
pred_score = model.predict(X)

# Evaluate the model
mae = mean_absolute_error(y, pred_score)
mse = mean_squared_error(y, pred_score)
rmse = np.sqrt(mse)

# results
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')   

# new prediction
new_prediction=float(input("Enter hours studied: "))
new_pred=model.predict([[new_prediction]])
print(f'Predicted score: {new_pred[0][0]}')

