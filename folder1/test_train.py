import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Create data
data = {
    'StudyHours': [1, 2, 3, 4, 5],
    'TestScore': [40, 50, 60, 70, 80]
}

# Step 2: Create DataFrame
df = pd.DataFrame(data)

# Step 3: Features and target
X = df[['StudyHours']]   # input
y = df['TestScore']      # output

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:")
print(X_train)

print("\nX_test:")
print(X_test)
