# Sklearn Metrics:
# Used for evaluating the performance of machine learning models.For example, accuracy_score, precision_score, recall_score, f1_score, etc.


# Example:

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#   True Answers(What actually happened)
y_true = [0, 1, 1, 0, 1]
#   Predicted Answers(What the model predicted)
y_pred = [0, 1, 0, 0, 1]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
# Output:
# Accuracy: 0.8
# Precision: 1.0
# Recall: 0.6666666666666666
# F1 Score: 0.8   
