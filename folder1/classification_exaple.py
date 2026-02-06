import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset: hours studied vs pass/fail
data = {
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8],
    "pass_exam":    [0, 0, 0, 1, 1, 1, 1, 1]  # 0=fail, 1=pass
}

df = pd.DataFrame(data)

# Features & target
X = df[["hours_studied"]]
y = df["pass_exam"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Scaling (good habit for ML)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
