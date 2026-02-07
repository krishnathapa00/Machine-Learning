# Mean Absolute Error (MAE) implementation
# The Mean Absolute Error is a common metric used to evaluate the performance of regression models. It measures the average magnitude of the errors between predicted and actual values, without considering their direction.
#  The formula for MAE is:   
# MAE = (1/n) * Σ|y_i - ŷ_i|

 # Iportant Topics:
#1. Take the mistake difference
#2. Remove the minus sign
#3. Add all the mistakes
#4. Divide by the total students

def mean_absolute_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.

    Parameters:
    y_true (list): A list of true values.
    y_pred (list): A list of predicted values.

    Returns:
    float: The Mean Absolute Error.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("The length of true values and predicted values must be the same.")
    
    total_error = 0
    n = len(y_true)
    
    for true, pred in zip(y_true, y_pred):
        error = abs(true - pred)  # Step 1 and Step 2: Take the mistake difference and remove the minus sign
        total_error += error       # Step 3: Add all the mistakes
    
    mae = total_error / n          # Step 4: Divide by the total number of samples
    return mae
# Example usage: Students real scores and predicted scores
real_scores = [90, 60, 80, 100]
pred_scores = [85, 70, 70, 95]
# Model Prediction.
#Mean Absolute Error Calculation
mae = mean_absolute_error(real_scores, pred_scores)
print(f"Mean Absolute Error: {mae}")

'''
# Mean squared error calculation
mse = mean_squared_error(real_scores, pred_scores)
print(f"Mean Squared Error: {mse}")


# Root mean squared error calculation
rms = root_mean_squared_error(real_scores, pred_scores)
print(f"Root Mean Squared Error: {rms}")    

'''
# Comparison of MAE, MSE, and RMS
print(f"Comparison of MAE, MSE, and RMS:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMS: {rms}")

# Uses of MAE, MSE, and RMS:
# 1. MAE is more robust to outliers than MSE and RMS, as it does not square the errors. It is often used when the cost of large errors is not significantly higher than the cost of small errors.
# 2. MSE is more sensitive to outliers than MAE, as it squares the errors. It is often used when the cost of large errors is significantly higher than the cost of small errors.
# 3. RMS is similar to MSE but provides a more interpretable metric, as it is in the same units as the original data. It is often used when the cost of large errors is significantly higher than the cost of small errors,