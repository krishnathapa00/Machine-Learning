import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Sample dataset (with numeric and categorical data)
data = {
    "hours_studied": [1, 2, 3, 4, 5, 6, None, 7],
    "assignment_completed": ["yes", "no", "yes", "no", "yes", "yes", "no", "yes"],
    "pass_exam": [0, 0, 0, 1, 1, 1, 0, 1]  # target
}

df = pd.DataFrame(data)

# 2. Features & target
X = df[["hours_studied", "assignment_completed"]]
y = df["pass_exam"]

# 3. Preprocessing
numeric_features = ["hours_studied"]
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),   # fill missing values
    ("scaler", StandardScaler())                   # scale numeric values
])

categorical_features = ["assignment_completed"]
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # fill missing categories
    ("onehot", OneHotEncoder(handle_unknown="ignore"))     # encode categories
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 4. Create ML pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 6. Train model
model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
