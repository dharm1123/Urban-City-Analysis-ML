import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Changed to Logistic Regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load your dataset
data = pd.read_csv('final_crime_dataset.csv')

# Step 2: Create the target variable
data['CrimeOccurred'] = np.where(data['Incident ID'].notnull(), 1, 0)

# Step 3: Simulate negative examples
negative_data = data.copy()
negative_data['Incident ID'] = np.nan
negative_data['Crime Category'] = np.nan
negative_data['Incident Description'] = np.nan
negative_data['CrimeOccurred'] = 0
negative_data['Timestamp'] = pd.to_datetime(negative_data['Timestamp']) + pd.Timedelta(hours=1)
negative_data['Passenger Count'] *= np.random.uniform(0.8, 1.2, size=len(negative_data))
negative_data['Trip Duration (mins)'] *= np.random.uniform(0.9, 1.1, size=len(negative_data))
negative_data['Vehicle Count'] *= np.random.uniform(0.8, 1.2, size=len(negative_data))

# Combine data
balanced_data = pd.concat([data, negative_data], ignore_index=True)

# Step 4: Preprocess the data
balanced_data['Timestamp'] = pd.to_datetime(balanced_data['Timestamp'])
balanced_data['Hour'] = balanced_data['Timestamp'].dt.hour
balanced_data['DayOfWeek'] = balanced_data['Timestamp'].dt.dayofweek

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

# Step 6: Train the Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)  # Increased max_iter for convergence
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)

print("\nLogistic Regression Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Feature Coefficients
# Check which features contribute most to predictions
feature_coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)  # Sort by absolute value
print("\nFeature Coefficients:")
print(feature_coef)

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

new_data = pd.get_dummies(new_data, columns=['District/Zone'], drop_first=True)
new_data = new_data.reindex(columns=X_train.columns, fill_value=0)
new_data[numerical_cols] = scaler.transform(new_data[numerical_cols])

prediction = model.predict(new_data)
print("\nPrediction for New Data:")
print("Crime Likelihood:", "Yes" if prediction[0] == 1 else "No")