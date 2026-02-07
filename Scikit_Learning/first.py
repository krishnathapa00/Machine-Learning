import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Create simple dataset
data = {
    "hours_studied": [1, 2, 3, 4, 5],
    "score": [40, 50, 60, 70, 80]
}

df = pd.DataFrame(data)

# 2. Split features & target
X = df[["hours_studied"]]   # feature
y = df["score"]             # target

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create & train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict
predictions = model.predict(X_test)

# 6. Evaluate
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
