import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings for cleaner output

# Step 1: Load your dataset
data = pd.read_csv('final_crime_dataset.csv')

# Step 2: Create the target variable
# If 'Incident ID' is not null, a crime occurred (1); otherwise, no crime (0)
# Since all rows in your dataset have an Incident ID, this will initially set all to 1
data['CrimeOccurred'] = np.where(data['Incident ID'].notnull(), 1, 0)

# Verify the initial distribution (should show all 1s)
print("Initial Target Variable Distribution:")
print(data['CrimeOccurred'].value_counts())

# Step 3: Simulate negative examples (no-crime data)
# Since your dataset only has crime incidents, we need to create rows where no crime occurred
negative_data = data.copy()
negative_data['Incident ID'] = np.nan
negative_data['Crime Category'] = np.nan
negative_data['Incident Description'] = np.nan
negative_data['CrimeOccurred'] = 0

# Adjust features slightly to simulate realistic variations for no-crime scenarios
# Shift the timestamp by 1 hour to avoid overlap
negative_data['Timestamp'] = pd.to_datetime(negative_data['Timestamp']) + pd.Timedelta(hours=1)
# Slightly adjust numerical features to reflect possible differences
negative_data['Passenger Count'] = negative_data['Passenger Count'] * np.random.uniform(0.8, 1.2, size=len(negative_data))
negative_data['Trip Duration (mins)'] = negative_data['Trip Duration (mins)'] * np.random.uniform(0.9, 1.1, size=len(negative_data))
negative_data['Vehicle Count'] = negative_data['Vehicle Count'] * np.random.uniform(0.8, 1.2, size=len(negative_data))

# Combine the original (crime) and simulated (no-crime) data
balanced_data = pd.concat([data, negative_data], ignore_index=True)

# Step 4: Preprocess the data
# Convert Timestamp to datetime and extract useful features
balanced_data['Timestamp'] = pd.to_datetime(balanced_data['Timestamp'])
balanced_data['Hour'] = balanced_data['Timestamp'].dt.hour
balanced_data['DayOfWeek'] = balanced_data['Timestamp'].dt.dayofweek

# Select features for the model
features = ['Passenger Count', 'Trip Duration (mins)', 'District/Zone', 'Latitude', 
            'Longitude', 'Vehicle Count', 'Weather Condition', 'Hour', 'DayOfWeek']
X = balanced_data[features]
y = balanced_data['CrimeOccurred']

# Handle categorical variables (District/Zone) using one-hot encoding
X = pd.get_dummies(X, columns=['District/Zone'], drop_first=True)

# Normalize numerical features to ensure they're on the same scale
scaler = StandardScaler()
numerical_cols = ['Passenger Count', 'Trip Duration (mins)', 'Latitude', 'Longitude', 
                  'Vehicle Count', 'Weather Condition', 'Hour', 'DayOfWeek']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Step 5: Split the data into training and test sets
# Use stratify to ensure the train/test sets have the same proportion of 1s and 0s
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Verify the split
print("\nTraining Set Target Distribution:")
print(y_train.value_counts())
print("\nTest Set Target Distribution:")
print(y_test.value_counts())

# Step 6: Train the Random Forest model
model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
# Predict on the test set
y_pred = model.predict(X_test)

# Print evaluation metrics
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Feature Importance
# Check which features are most important for predicting crimes
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Step 9: Make a prediction on new data
# Example: Predict crime likelihood for a new scenario
new_data = pd.DataFrame({
    'Passenger Count': [50],
    'Trip Duration (mins)': [30],
    'District/Zone': [2],
    'Latitude': [22.1362303565],
    'Longitude': [73.73615288848],
    'Vehicle Count': [25],
    'Weather Condition': [0],
    'Hour': [22],  # 10 PM
    'DayOfWeek': [5]  # Friday
})

# Preprocess the new data to match the training data
new_data = pd.get_dummies(new_data, columns=['District/Zone'], drop_first=True)
new_data = new_data.reindex(columns=X_train.columns, fill_value=0)  # Ensure same columns
new_data[numerical_cols] = scaler.transform(new_data[numerical_cols])

# Predict
prediction = model.predict(new_data)
print("\nPrediction for New Data:")
print("Crime Likelihood:", "Yes" if prediction[0] == 1 else "No")
