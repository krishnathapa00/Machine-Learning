# Classification algorithms-predicting the category of a data point.
from sklearn.linear_model import LogisticRegression

# Input data
X = [[1],[2],[3],[4],[5]]  # Hours studied
y = [0,0,1,1,1]             # 0 = Fail, 1 = Pass

# Create model and train
model = LogisticRegression()
model.fit(X, y)

# User input
hours = float(input("Enter how many hours you study: "))

# Predict
result = model.predict([[hours]])[0]

# Output
if result == 1:
    print(f"Based on hours {hours}, you are likely to PASS")
else:
    print(f"Based on hours {hours}, you are likely to FAIL")

