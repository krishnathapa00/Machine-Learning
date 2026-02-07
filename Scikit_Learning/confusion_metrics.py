# Confusion Metrics: 
# Used to evaluate classification models by comparing predicted and actual class labels.
# A confusion matrix is a table that shows the counts of correct and incorrect predictions for each class.

# Example:
from sklearn.metrics import confusion_matrix
#   True Answers(What actually happened)
y_true = [0, 1, 1, 0, 1]
#   Predicted Answers(What the model predicted)
y_pred = [0, 1, 0, 0, 1]    
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
# Output:
# Confusion Matrix:
# [[2 0]
#  [1 2]]
# Interpretation of the confusion matrix:
# - True Negatives (TN): 2 (correctly predicted class 0)
# - False Positives (FP): 0 (incorrectly predicted class 1 as class
# - False Negatives (FN): 1 (incorrectly predicted class 0 as class 1)
# - True Positives (TP): 2 (correctly predicted class 1)    

# Confusion Metrics: gives more accurate results. it shows where the model is making errors.
