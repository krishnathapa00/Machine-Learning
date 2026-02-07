import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Sample data
data = {
    'StudyHours': [1, 2, 3, 4, 5],
    'TestScore': [40, 50, 60, 70, 80]
}

df = pd.DataFrame(data)

# StandardScaler
standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(df)

print('Standard Scaler Output:')
print(pd.DataFrame(standard_scaled, columns=['StudyHours', 'TestScore']))

# MinMaxScaler
minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(df)

print('\nMinMax Scaler Output:')
print(pd.DataFrame(minmax_scaled, columns=['StudyHours', 'TestScore']))




