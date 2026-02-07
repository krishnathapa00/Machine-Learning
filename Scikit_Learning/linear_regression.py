# Linear Regression-Predicting marks obtained by students.

from sklearn.linear_model import LinearRegression
import numpy as np
# Training data
X = [[1],[2],[3],[4],[5]]  # Features (hours studied)
y = [40,50,65,75,90]        # Target (marks)

# Create model and train
model = LinearRegression() # Create a linear regression model
model.fit(X, y) # Train the model on the data

# User input
hours = float(input("Enter the number of hours studied: ")) 

# Predict marks
predicted_marks = model.predict([[hours]]) # Predict marks based on hours studied
print(f"Based on the hours {hours} studied, the predicted marks are: {predicted_marks[0]:.2f}") # Output the predicted marks with 2 decimal places
