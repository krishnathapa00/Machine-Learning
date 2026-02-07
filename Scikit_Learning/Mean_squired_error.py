# Mean Squired Error (MSE) implementation
# The Mean Squared Error is a common metric used to evaluate the performance of regression models. It measures the average of the squares of the errors between predicted and actual values. The formula for MSE is:   
# MSE = (1/n) * Σ(y_i - ŷ_i)^2  
# Important Topics:
#1. Take the mistake difference
#2. Square the mistake
#3. Add all the squared mistakes
#4. Divide by the total students    

def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    y_true (list): A list of true values.
    y_pred (list): A list of predicted values.

    Returns:
    float: The Mean Squared Error.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("The length of true values and predicted values must be the same.")
    
    total_error = 0
    n = len(y_true)
    
    for true, pred in zip(y_true, y_pred):
        error = (true - pred) ** 2  # Step 1 and Step 2: Take the mistake difference and square the mistake
        total_error += error          # Step 3: Add all the squared mistakes
    
    mse = total_error / n             # Step 4: Divide by the total number of samples
    return mse
# Example usage:
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse}") 