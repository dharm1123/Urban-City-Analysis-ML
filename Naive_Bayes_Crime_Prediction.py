import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  # Changed to Gaussian Naive Bayes
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load your dataset
data = pd.read_csv('final_crime_dataset.csv')

# Step 2: Create the target variable
data['CrimeOccurred'] = np.where(data['Incident ID'].notnull(), 1, 0)

# Verify the initial distribution
print("Initial Target Variable Distribution:")
print(data['CrimeOccurred'].value_counts())

# Step 3: Simulate negative examples
negative_data = data.copy()
negative_data['Incident ID'] = np.nan
negative_data['Crime Category'] = np.nan
negative_data['Incident Description'] = np.nan
negative_data['CrimeOccurred'] = 0

# Adjust features
negative_data['Timestamp'] = pd.to_datetime(negative_data['Timestamp']) + pd.Timedelta(hours=1)
negative_data['Passenger Count'] = negative_data['Passenger Count'] * np.random.uniform(0.8, 1.2, size=len(negative_data))
negative_data['Trip Duration (mins)'] = negative_data['Trip Duration (mins)'] * np.random.uniform(0.9, 1.1, size=len(negative_data))
negative_data['Vehicle Count'] = negative_data['Vehicle Count'] * np.random.uniform(0.8, 1.2, size=len(negative_data))

# Combine data
balanced_data = pd.concat([data, negative_data], ignore_index=True)

# Step 4: Preprocess the data
balanced_data['Timestamp'] = pd.to_datetime(balanced_data['Timestamp'])
balanced_data['Hour'] = balanced_data['Timestamp'].dt.hour
balanced_data['DayOfWeek'] = balanced_data['Timestamp'].dt.dayofweek

# Select features
features = ['Passenger Count', 'Trip Duration (mins)', 'District/Zone', 'Latitude', 
            'Longitude', 'Vehicle Count', 'Weather Condition', 'Hour', 'DayOfWeek']
X = balanced_data[features]
y = balanced_data['CrimeOccurred']

# Handle categorical variables
X = pd.get_dummies(X, columns=['District/Zone'], drop_first=True)

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = ['Passenger Count', 'Trip Duration (mins)', 'Latitude', 'Longitude', 
                  'Vehicle Count', 'Weather Condition', 'Hour', 'DayOfWeek']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Step 5: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Verify the split
print("\nTraining Set Target Distribution:")
print(y_train.value_counts())
print("\nTest Set Target Distribution:")
print(y_test.value_counts())

# Step 6: Train the Naive Bayes model
model = GaussianNB()  # Changed to Gaussian Naive Bayes
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)

# Print evaluation metrics
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Feature Importance
# Note: Naive Bayes doesn't provide feature importance like Random Forest
# Instead, we can look at the means of features for each class
print("\nFeature Means by Class:")
means_df = pd.DataFrame({
    'Feature': X.columns,
    'Mean (No Crime)': model.theta_[0],  # Means for class 0
    'Mean (Crime)': model.theta_[1]     # Means for class 1
})
print(means_df)

# Step 9: Make a prediction on new data
new_data = pd.DataFrame({
    'Passenger Count': [50],
    'Trip Duration (mins)': [30],
    'District/Zone': [2],
    'Latitude': [22.1362303565],
    'Longitude': [73.73615288848],
    'Vehicle Count': [25],
    'Weather Condition': [0],
    'Hour': [22],
    'DayOfWeek': [5]
})

# Preprocess new data
new_data = pd.get_dummies(new_data, columns=['District/Zone'], drop_first=True)
new_data = new_data.reindex(columns=X_train.columns, fill_value=0)
new_data[numerical_cols] = scaler.transform(new_data[numerical_cols])

# Predict
prediction = model.predict(new_data)
print("\nPrediction for New Data:")
print("Crime Likelihood:", "Yes" if prediction[0] == 1 else "No")