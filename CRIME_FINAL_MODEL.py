import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
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
negative_data['Passenger Count'] = negative_data['Passenger Count'] * np.random.uniform(0.8, 1.2, size=len(negative_data))
negative_data['Trip Duration (mins)'] = negative_data['Trip Duration (mins)'] * np.random.uniform(0.9, 1.1, size=len(negative_data))
negative_data['Vehicle Count'] = negative_data['Vehicle Count'] * np.random.uniform(0.8, 1.2, size=len(negative_data))

balanced_data = pd.concat([data, negative_data], ignore_index=True)

# Step 4: Preprocess the data
balanced_data['Timestamp'] = pd.to_datetime(balanced_data['Timestamp'])
balanced_data['Hour'] = balanced_data['Timestamp'].dt.hour
balanced_data['DayOfWeek'] = balanced_data['Timestamp'].dt.dayofweek
balanced_data['IsWeekend'] = balanced_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

features = ['Passenger Count', 'Trip Duration (mins)', 'District/Zone', 'Latitude', 
            'Longitude', 'Vehicle Count', 'Weather Condition', 'Hour', 'DayOfWeek', 'IsWeekend']
X = balanced_data[features]
y = balanced_data['CrimeOccurred']

# One-hot encoding for categorical variables
X = pd.get_dummies(X, columns=['District/Zone'], drop_first=True)

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = ['Passenger Count', 'Trip Duration (mins)', 'Latitude', 'Longitude', 
                  'Vehicle Count', 'Weather Condition', 'Hour', 'DayOfWeek', 'IsWeekend']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Step 5: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Define base models for stacking
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42))
]

# Define the stacking classifier
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5
)

# Step 7: Train the stacking model
stacking_model.fit(X_train, y_train)

# Step 8: Evaluate the model
y_pred = stacking_model.predict(X_test)
print("\nStacking Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
'''
# Step 9: Hyperparameter tuning for GradientBoostingClassifier (optional)
# Set n_jobs=1 to avoid parallel processing issues
param_grid = {
    'n_estimators': [100],
    'max_depth': [3],
    'learning_rate': [0.01]
}
gb_model = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(gb_model, param_grid, cv=5, scoring='accuracy', n_jobs=1)
grid_search.fit(X_train, y_train)
print("\nBest Parameters for Gradient Boosting:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)
y_pred_tuned = grid_search.predict(X_test)
print("Tuned Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_tuned))

# Step 10: Feature Importance (for Random Forest)
rf_model = stacking_model.named_estimators_['rf']
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance (Random Forest):")
print(feature_importance)'''

