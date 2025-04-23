import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Collect Data from API
api_url = "https://api.example.com/data"  # Replace with actual API URL
response = requests.get(api_url)

if response.status_code == 200:
    data = response.json()
    print("Data fetched successfully!")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
    exit()

# Step 2: Preprocess Data
df = pd.DataFrame(data)

# Handling missing values
df.fillna(df.mean(), inplace=True)

# Encoding categorical columns (if necessary)
df = pd.get_dummies(df, drop_first=True)

# Splitting features and target
X = df.drop(columns=['target'])  # Replace 'target' with your actual target column
y = df['target']

# Splitting into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Save Preprocessed Data
df.to_csv('preprocessed_data.csv', index=False)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
X_train_scaled_df.to_csv('X_train_scaled.csv', index=False)
X_test_scaled_df.to_csv('X_test_scaled.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Preprocessed data saved!")
