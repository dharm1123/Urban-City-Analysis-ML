import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from termcolor import colored
import warnings
import sys

warnings.filterwarnings('ignore')

file_path = "final_crime_dataset.csv"
df = pd.read_csv(file_path)

# Check for null values

missing_values = df.isnull().sum()
print("Null values per column before filling:")
print(missing_values)

def fill_with_random(col):
    if col.isnull().sum() > 0:
        fill_value = col.dropna().sample(n=1, random_state=42).values[0]
        return col.fillna(fill_value)
    return col

# Fill null values with any value from the same column
df_filled = df.apply(fill_with_random)

# Check null values after filling
filled_values = missing_values - df_filled.isnull().sum()
print("Filled null values per column:")
print(filled_values)

# Save the processed file
df_filled.to_csv('final_crime_dataset.csv', index=False)

# Print unique values, min, and max for each column excluding 'Incident'
print("\nUnique Values, Min, and Max for Each Attribute:")
for col in df_filled.columns:
    if col != "Incident":
        unique_vals = df_filled[col].nunique()
        min_val = df_filled[col].min()
        max_val = df_filled[col].max()
        print(f"{col}: Unique = {unique_vals}, Min = {min_val}, Max = {max_val}")

# Convert categorical columns to numerical using label encoding
df_encoded = df_filled.copy()
for col in df_filled.select_dtypes(include=['object']).columns:
    df_encoded[col] = df_encoded[col].astype("category").cat.codes

# Normalize Data for Box Plot
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)

# Box Plot - Passenger Count vs Other Features
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot Passenger Count separately
sns.boxplot(data=df_scaled[['Passenger Count']], ax=axes[0], palette="Set2")
axes[0].set_title("Box Plot of Passenger Count (Scaled)")

# Plot remaining numerical variables
sns.boxplot(data=df_scaled.drop(columns=['Passenger Count']), ax=axes[1], palette="Set3")
axes[1].set_title("Box Plot of Other Features (Scaled)")

plt.tight_layout()
plt.show()

# Compute the correlation matrix for all numerical features
correlation_matrix = df_filled.corr()

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Overall Feature Correlation Heatmap")
plt.show()

# Scatter Plots for Key Features
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Passenger Count vs. Trip Duration
if "Trip Duration (mins)" in df_filled.columns:
    sns.scatterplot(x=df_filled["Passenger Count"], y=df_filled["Trip Duration (mins)"], alpha=0.6, color="blue", ax=axes[0])
    axes[0].set_title("Passenger Count vs. Trip Duration")
    axes[0].set_xlabel("Passenger Count")
    axes[0].set_ylabel("Trip Duration (mins)")
    axes[0].grid(True)
else:
    print("Error: 'Trip Duration (mins)' column not found in dataset.")

# Weather Condition vs. Trip Duration
if "Trip Duration (mins)" in df_filled.columns:
    sns.scatterplot(x=df_filled["Weather Condition"], y=df_filled["Trip Duration (mins)"], alpha=0.6, color="green", ax=axes[1])
    axes[1].set_title("Weather Condition vs. Trip Duration")
    axes[1].set_xlabel("Weather Condition")
    axes[1].set_ylabel("Trip Duration (mins)")
    axes[1].grid(True)
else:
    print("Error: 'Trip Duration (mins)' column not found in dataset.")

# Crime Category vs. Accident Severity
sns.scatterplot(x=df_filled["Crime Category"], y=df_filled["Accident Severity"], alpha=0.6, color="red", ax=axes[2])
axes[2].set_title("Crime Category vs. Accident Severity")
axes[2].set_xlabel("Crime Category")
axes[2].set_ylabel("Accident Severity")
axes[2].grid(True)

plt.tight_layout()
plt.show()

# Histogram for model-specific features
model_features = ["Passenger Count", "Trip Duration (mins)", "Accident Severity"]
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
axes = axes.flatten()

for i, col in enumerate(model_features):
    if col in df_filled.columns:
        sns.histplot(df_filled[col], kde=True, bins=30, ax=axes[i], color='blue')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()






# Load dataset
df = pd.read_csv("final_crime_dataset.csv")
print("Dataset shape:", df.shape)

# Ask user for choice
choice = input("Enter 1 for Crime Prediction, 2 for Accident Severity Prediction, or 3 for Passenger Count Prediction: ")

if choice == '1':
    # ============================
    # Task 1: Crime Prediction
    # ============================
    df_crime = df.copy()
    df_crime['CrimeOccurred'] = np.where(df_crime['Incident ID'].notnull(), 1, 0)

    # Simulate negative samples
    negative_data = df_crime.copy()
    negative_data['Incident ID'] = np.nan
    negative_data['Crime Category'] = np.nan
    negative_data['Incident Description'] = np.nan
    negative_data['CrimeOccurred'] = 0
    negative_data['Timestamp'] = pd.to_datetime(negative_data['Timestamp']) + pd.Timedelta(hours=1)
    negative_data['Passenger Count'] *= np.random.uniform(0.8, 1.2, size=len(negative_data))
    negative_data['Trip Duration (mins)'] *= np.random.uniform(0.9, 1.1, size=len(negative_data))
    negative_data['Vehicle Count'] *= np.random.uniform(0.8, 1.2, size=len(negative_data))

    balanced_data = pd.concat([df_crime, negative_data], ignore_index=True)

    # Time-based features
    balanced_data['Timestamp'] = pd.to_datetime(balanced_data['Timestamp'])
    balanced_data['Hour'] = balanced_data['Timestamp'].dt.hour
    balanced_data['DayOfWeek'] = balanced_data['Timestamp'].dt.dayofweek
    balanced_data['IsWeekend'] = balanced_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    features_crime = ['Passenger Count', 'Trip Duration (mins)', 'District/Zone', 'Latitude', 'Longitude',
                      'Vehicle Count', 'Weather Condition', 'Hour', 'DayOfWeek', 'IsWeekend']
    X_crime = balanced_data[features_crime]
    y_crime = balanced_data['CrimeOccurred']

    # One-hot encode categorical columns
    X_crime = pd.get_dummies(X_crime, columns=['District/Zone'], drop_first=True)

    # Normalize
    scaler_crime = StandardScaler()
    numerical_cols = ['Passenger Count', 'Trip Duration (mins)', 'Latitude', 'Longitude',
                      'Vehicle Count', 'Weather Condition', 'Hour', 'DayOfWeek', 'IsWeekend']
    X_crime[numerical_cols] = scaler_crime.fit_transform(X_crime[numerical_cols])

    # Train-test split
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_crime, y_crime, test_size=0.2, stratify=y_crime, random_state=42)

    # Individual models
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

    rf_model.fit(X_train_c, y_train_c)
    gb_model.fit(X_train_c, y_train_c)

    y_pred_rf = rf_model.predict(X_test_c)
    y_pred_gb = gb_model.predict(X_test_c)

    acc_rf = accuracy_score(y_test_c, y_pred_rf)
    acc_gb = accuracy_score(y_test_c, y_pred_gb)

    print("\n=== Individual Base Model Accuracies ===")
    print(colored(f"Random Forest Accuracy: {acc_rf:.4f}", "cyan"))
    print(colored(f"Gradient Boosting Accuracy: {acc_gb:.4f}", "cyan"))

    # Stacking model
    base_models = [
        ('rf', rf_model),
        ('gb', gb_model)
    ]
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=5)
    stacking_model.fit(X_train_c, y_train_c)
    y_pred_c = stacking_model.predict(X_test_c)

    print("\n=== Crime Prediction with Stacking ===")
    acc = accuracy_score(y_test_c, y_pred_c)
    print(colored(f"Stacking Model Accuracy: {acc:.4f}", "green"))
    print(confusion_matrix(y_test_c, y_pred_c))
    print(classification_report(y_test_c, y_pred_c))

elif choice == '2':
    # ============================
    # Task 2: Accident Severity
    # ============================
    df_severity = df.copy()
    df_severity = df_severity.drop(columns=["Incident ID", "Timestamp", "Date/Time", "Timestamp.1", "Incident Description"])

    # Encode categoricals
    label_encoders = {}
    for col in df_severity.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_severity[col] = le.fit_transform(df_severity[col])
        label_encoders[col] = le

    def simplify_severity(sev):
        if sev <= 3:
            return 0  # Low
        elif sev <= 6:
            return 1  # Medium
        else:
            return 2  # High

    df_severity["Severity_Level"] = df_severity["Accident Severity"].apply(simplify_severity)
    X_sev = df_severity.drop(columns=["Accident Severity", "Severity_Level"])
    y_sev = df_severity["Severity_Level"]

    # Train-test split
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sev, y_sev, test_size=0.3, random_state=42)

    # Scale data
    scaler_sev = StandardScaler()
    X_train_s = scaler_sev.fit_transform(X_train_s)
    X_test_s = scaler_sev.transform(X_test_s)

    print("\n=== Accident Severity Prediction ===")
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1, random_state=42),
        "SVM": SVC(kernel='linear', C=0.1, max_iter=10000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train_s, y_train_s)
        y_pred_s = model.predict(X_test_s)
        acc = accuracy_score(y_test_s, y_pred_s)
        print(colored(f"{name}: {acc:.4f}", "green"))

elif choice == '3':
    # ============================
    # Task 3: Passenger Count Prediction
    # ============================
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

    # Load passenger count dataset
    data = pd.read_csv('passenger_count_dataset_modified.csv')

    # Data inspection
    print("Dataset Shape:", data.shape)
    print("\nPassenger Count Summary:")
    print(data['passenger_count'].describe())
    print("\nMissing Values:\n", data.isnull().sum())

    # Feature engineering
    data['weather_time_interaction'] = data['weather_condition'] + '_' + data['time']

    # Cleaning
    data = data.dropna(subset=['passenger_count'])
    data = data[data['passenger_count'].apply(lambda x: isinstance(x, (int, float)) and x >= 0)]
    q99 = data['passenger_count'].quantile(0.99)
    data = data[data['passenger_count'] <= q99]

    # Feature & target setup
    X = data[['weather_condition', 'time', 'stop_location', 'accident_severity', 'crime_category', 'weather_time_interaction']]
    y = data['passenger_count']

    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False), categorical_cols)
        ]
    )

    base_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    param_grid = {
        'regressor__n_estimators': [100],
        'regressor__max_depth': [10, 15],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1]
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='r2', n_jobs=1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    train_r2 = best_model.score(X_train, y_train)
    test_r2 = best_model.score(X_test, y_test)
    cv_r2 = grid_search.best_score_

    print(f"\nBest Hyperparameters: {grid_search.best_params_}")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Testing R² Score: {test_r2:.4f}")
    print(f"Cross-Validated R² Score: {cv_r2:.4f}")

    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    importances = best_model.named_steps['regressor'].feature_importances_
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    print("\nTop 5 Feature Importances:")
    print(feature_importance.sort_values(by='Importance', ascending=False).head())

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"\nMean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")
    print(f"Approximate Accuracy: {100 - mape * 100:.2f}%")

else:
    print("Invalid choice. Please enter 1, 2, or 3.")
