import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv('final_crime_dataset.csv')

# Step 1: Drop irrelevant or duplicate columns
columns_to_drop = ['Incident ID', 'Incident Description', 'Timestamp.1', 'Date/Time']
data = data.drop(columns=columns_to_drop)

# Step 2: Feature Engineering (Streamlined)
# Convert Timestamp to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d-%m-%Y %H:%M')

# Time-based features
data['Hour'] = data['Timestamp'].dt.hour
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
data['Month'] = data['Timestamp'].dt.month
data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
data['TimeOfDay'] = pd.cut(data['Hour'], 
                          bins=[0, 6, 12, 18, 24], 
                          labels=['Night', 'Morning', 'Afternoon', 'Evening'], 
                          include_lowest=True)

# Geospatial clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Reduced clusters for speed
data['GeoCluster'] = kmeans.fit_predict(data[['Latitude', 'Longitude']])

# Interaction feature
data['TripDuration_VehicleCount'] = data['Trip Duration (mins)'] * data['Vehicle Count']

# Handle outliers in Passenger Count
data = data[data['Passenger Count'].between(data['Passenger Count'].quantile(0.01), 
                                           data['Passenger Count'].quantile(0.99))]

# Drop original Timestamp
data = data.drop(columns=['Timestamp'])

# Step 3: Define features and target
target = 'Passenger Count'
features = [col for col in data.columns if col != target]

# Separate features and target
X = data[features]
y = data[target]

# Step 4: Preprocessing
categorical_cols = ['Stop Locations', 'TimeOfDay', 'GeoCluster', 'Crime Category', 
                   'Weather Condition']
numeric_cols = [col for col in features if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'), 
         categorical_cols)
    ])

# Step 5: Create pipeline with RandomForestRegressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Step 6: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Simplified hyperparameter tuning
param_dist = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [5, 10],
    'regressor__min_samples_split': [2, 5]
}

random_search = RandomizedSearchCV(
    model, param_distributions=param_dist, n_iter=5, cv=3,  # Reduced iterations and folds
    scoring='r2', n_jobs=1, random_state=42
)

# Step 8: Train with tuning
random_search.fit(X_train, y_train)

# Step 9: Best model
best_model = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)
print("Best CV R^2:", random_search.best_score_)

# Step 10: Predict and evaluate
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nTest Set Performance:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Step 11: Feature importance
feature_names = (numeric_cols + 
                 list(best_model.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .get_feature_names_out(categorical_cols)))
importances = best_model.named_steps['regressor'].feature_importances_
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='Importance', ascending=False).head(10))