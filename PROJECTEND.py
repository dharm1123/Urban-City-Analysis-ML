"""
Urban City Analysis ML - Main Project File

This script provides a comprehensive machine learning analysis suite for urban city data.
It includes three main analysis types:
1. Crime Prediction - Predicts likelihood of crime occurrence
2. Accident Severity Prediction - Classifies accident severity levels  
3. Passenger Count Prediction - Estimates passenger counts using regression

Author: DHARM DUDHAGARA
Repository: https://github.com/dharm1123/Urban-City-Analysis-ML
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from termcolor import colored
import warnings
import sys

warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """
    Load and preprocess the urban city dataset.
    
    This function:
    1. Loads the crime dataset from CSV
    2. Handles missing values by filling them with random values from the same column
    3. Displays dataset statistics and information
    
    Returns:
        pandas.DataFrame: Preprocessed dataset ready for analysis
    """
    print(colored("="*80, "cyan"))
    print(colored("URBAN CITY ANALYSIS - DATA LOADING & PREPROCESSING", "cyan", attrs=["bold"]))
    print(colored("="*80, "cyan"))
    
    file_path = "final_crime_dataset.csv"
    print(f"Loading dataset from: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded successfully! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(colored(f"❌ Error: Could not find {file_path}", "red"))
        print(colored("Please ensure the dataset file is in the project directory.", "red"))
        sys.exit(1)

def preprocess_dataset(df):
    """
    Preprocess the dataset by handling missing values and displaying basic statistics.
    
    Args:
        df (pandas.DataFrame): Raw dataset
        
    Returns:
        pandas.DataFrame: Cleaned dataset with filled missing values
    """
    print(colored("\n--- DATA PREPROCESSING ---", "yellow", attrs=["bold"]))
    
    # Check for null values
    missing_values = df.isnull().sum()
    print("Null values per column before filling:")
    print(missing_values)

    def fill_with_random(col):
        """Fill missing values with random values from the same column"""
        if col.isnull().sum() > 0:
            fill_value = col.dropna().sample(n=1, random_state=42).values[0]
            return col.fillna(fill_value)
        return col

    # Fill null values with any value from the same column
    df_filled = df.apply(fill_with_random)

    # Check null values after filling
    filled_values = missing_values - df_filled.isnull().sum()
    print("✓ Filled null values per column:")
    print(filled_values)

    # Save the processed file
    df_filled.to_csv('final_crime_dataset.csv', index=False)
    print("✓ Preprocessed data saved successfully!")

    # Print unique values, min, and max for each column excluding text columns
    print(colored("\n--- DATASET STATISTICS ---", "yellow", attrs=["bold"]))
    print("Unique Values, Min, and Max for Each Attribute:")
    for col in df_filled.columns:
        if col not in ["Incident", "Incident Description", "Incident ID", "Timestamp", "Date/Time", "Timestamp.1"]:
            try:
                unique_vals = df_filled[col].nunique()
                min_val = df_filled[col].min()
                max_val = df_filled[col].max()
                print(f"  {col}: Unique = {unique_vals}, Min = {min_val}, Max = {max_val}")
            except:
                unique_vals = df_filled[col].nunique()
                print(f"  {col}: Unique = {unique_vals} (Non-numeric data)")
                
    return df_filled

def create_visualizations(df_filled):
    """
    Create comprehensive visualizations for exploratory data analysis.
    
    This function generates:
    1. Box plots for data distribution analysis
    2. Correlation heatmap for numerical features
    3. Scatter plots for key feature relationships
    4. Histograms for target variables
    
    Args:
        df_filled (pandas.DataFrame): Preprocessed dataset
    """
    print(colored("\n--- EXPLORATORY DATA ANALYSIS & VISUALIZATIONS ---", "green", attrs=["bold"]))
    
    # Convert categorical columns to numerical using label encoding for visualization
    print("📊 Encoding categorical variables for visualization...")
    df_encoded = df_filled.copy()
    label_encoders = {}
    
    for col in df_filled.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    # Normalize Data for Box Plot
    print("📈 Creating normalized box plots...")
    scaler = MinMaxScaler()
    numerical_cols = df_encoded.select_dtypes(include=[np.number]).columns
    df_scaled = pd.DataFrame(scaler.fit_transform(df_encoded[numerical_cols]), 
                            columns=numerical_cols)

    # Box plots for data distribution analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Passenger Count Box Plot
    if "Passenger Count" in df_scaled.columns:
        sns.boxplot(y=df_scaled["Passenger Count"], ax=axes[0], palette="Set2")
        axes[0].set_title("Box Plot of Passenger Count (Normalized)", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("Normalized Passenger Count")
    else:
        axes[0].text(0.5, 0.5, "Passenger Count\nNot Available", 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=axes[0].transAxes, fontsize=12)
        axes[0].set_title("Passenger Count - Not Available")

    # Plot remaining numerical variables (excluding Passenger Count)
    other_cols = [col for col in df_scaled.columns if col != 'Passenger Count'][:10]  # Limit to 10 for readability
    if other_cols:
        sns.boxplot(data=df_scaled[other_cols], ax=axes[1], palette="Set3")
        axes[1].set_title("Box Plot of Other Features (Normalized)", fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
    else:
        axes[1].text(0.5, 0.5, "Other Features\nNot Available", 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=axes[1].transAxes, fontsize=12)

    plt.tight_layout()
    plt.show()

    # Compute the correlation matrix for numerical features only
    print("🔥 Computing correlation matrix for numerical features...")
    numerical_data = df_encoded.select_dtypes(include=[np.number])
    correlation_matrix = numerical_data.corr()

    # Create the correlation heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
                linewidths=0.5, square=True, cbar_kws={"shrink": .8})
    plt.title("Feature Correlation Heatmap (Numerical Features Only)", 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

    # Scatter Plots for Key Features
    print("🎯 Creating scatter plots for key feature relationships...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Passenger Count vs. Trip Duration
    if "Trip Duration (mins)" in df_filled.columns and "Passenger Count" in df_filled.columns:
        sns.scatterplot(x=df_filled["Passenger Count"], y=df_filled["Trip Duration (mins)"], 
                       alpha=0.6, color="blue", ax=axes[0])
        axes[0].set_title("Passenger Count vs. Trip Duration", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Passenger Count")
        axes[0].set_ylabel("Trip Duration (mins)")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "Data Not Available\nfor this relationship", 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=axes[0].transAxes, fontsize=12)
        axes[0].set_title("Passenger Count vs. Trip Duration - N/A")

    # Weather Condition vs. Trip Duration
    if "Trip Duration (mins)" in df_filled.columns and "Weather Condition" in df_filled.columns:
        sns.scatterplot(x=df_filled["Weather Condition"], y=df_filled["Trip Duration (mins)"], 
                       alpha=0.6, color="green", ax=axes[1])
        axes[1].set_title("Weather Condition vs. Trip Duration", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Weather Condition")
        axes[1].set_ylabel("Trip Duration (mins)")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "Data Not Available\nfor this relationship", 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title("Weather vs. Trip Duration - N/A")

    # Crime Category vs. Accident Severity
    if "Crime Category" in df_filled.columns and "Accident Severity" in df_filled.columns:
        sns.scatterplot(x=df_filled["Crime Category"], y=df_filled["Accident Severity"], 
                       alpha=0.6, color="red", ax=axes[2])
        axes[2].set_title("Crime Category vs. Accident Severity", fontsize=12, fontweight='bold')
        axes[2].set_xlabel("Crime Category")
        axes[2].set_ylabel("Accident Severity")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "Data Not Available\nfor this relationship", 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=axes[2].transAxes, fontsize=12)
        axes[2].set_title("Crime vs. Accident - N/A")

    plt.tight_layout()
    plt.show()

    # Histogram for model-specific features
    print("📊 Creating histograms for key target variables...")
    model_features = ["Passenger Count", "Trip Duration (mins)", "Accident Severity"]
    available_features = [feat for feat in model_features if feat in df_filled.columns]
    
    if available_features:
        fig, axes = plt.subplots(1, len(available_features), figsize=(6*len(available_features), 5))
        if len(available_features) == 1:
            axes = [axes]
            
        for i, feature in enumerate(available_features):
            df_filled[feature].hist(bins=30, alpha=0.7, color=['blue', 'green', 'red'][i], ax=axes[i])
            axes[i].set_title(f"Distribution of {feature}", fontsize=12, fontweight='bold')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️  No key features available for histogram display.")
        
    print(colored("✓ Exploratory Data Analysis completed successfully!", "green", attrs=["bold"]))

def display_menu():
    """
    Display the main menu options for Urban City Analysis.
    
    Returns:
        str: User's choice (1, 2, or 3)
    """
    print(colored("\n" + "="*80, "magenta"))
    print(colored("          URBAN CITY ANALYSIS - MACHINE LEARNING MENU", "magenta", attrs=["bold"]))
    print(colored("="*80, "magenta"))
    print(colored("Please select the analysis you want to perform:", "white", attrs=["bold"]))
    print()
    print(colored("1. Crime Prediction Analysis", "cyan", attrs=["bold"]))
    print(colored("   → Predict whether a crime will occur based on urban factors", "cyan"))
    print(colored("   → Uses ensemble stacking models (Random Forest + Gradient Boosting)", "cyan"))
    print()
    print(colored("2. Accident Severity Prediction", "yellow", attrs=["bold"]))
    print(colored("   → Classify accident severity levels (low/medium/high)", "yellow"))
    print(colored("   → Uses multiple classification algorithms", "yellow"))
    print()
    print(colored("3. Passenger Count Prediction", "green", attrs=["bold"]))
    print(colored("   → Estimate passenger count using regression analysis", "green"))
    print(colored("   → Uses advanced feature engineering and Random Forest", "green"))
    print()
    print(colored("0. Exit", "red", attrs=["bold"]))
    print(colored("="*80, "magenta"))
    
    try:
        choice = input(colored("Enter your choice (0-3): ", "white", attrs=["bold"]))
        return choice
    except (EOFError, KeyboardInterrupt):
        return '0'  # Default to exit if input fails

def crime_prediction_analysis(df):
    """
    Perform crime prediction analysis using ensemble machine learning models.
    
    This function:
    1. Creates balanced dataset with positive and negative crime samples
    2. Engineers time-based features (hour, day of week, weekend indicator)
    3. Uses ensemble stacking with Random Forest and Gradient Boosting
    4. Evaluates model performance with detailed metrics
    
    Args:
        df (pandas.DataFrame): Preprocessed dataset
    """
    print(colored("\n🚨 CRIME PREDICTION ANALYSIS", "red", attrs=["bold"]))
    print(colored("="*60, "red"))
    print("This analysis predicts the likelihood of crime occurrence based on:")
    print("• Passenger count and trip duration")
    print("• Geographic location (district, latitude, longitude)")
    print("• Weather conditions and vehicle count")
    print("• Time-based features (hour, day of week, weekend)")
    print()
    
    # Create a copy for crime analysis
    df_crime = df.copy()
    
    # Create target variable: CrimeOccurred (1 if crime incident exists, 0 otherwise)
    df_crime['CrimeOccurred'] = np.where(df_crime['Incident ID'].notnull(), 1, 0)
    print(f"📊 Original positive samples (crimes): {df_crime['CrimeOccurred'].sum()}")

    # Create balanced dataset by simulating negative samples
    print("🔄 Creating balanced dataset with negative samples...")
    negative_data = df_crime.copy()
    negative_data['Incident ID'] = np.nan
    negative_data['Crime Category'] = np.nan
    negative_data['Incident Description'] = np.nan
    negative_data['CrimeOccurred'] = 0
    
    # Add slight variations to make negative samples realistic
    negative_data['Timestamp'] = pd.to_datetime(negative_data['Timestamp']) + pd.Timedelta(hours=1)
    negative_data['Passenger Count'] *= np.random.uniform(0.8, 1.2, size=len(negative_data))
    negative_data['Trip Duration (mins)'] *= np.random.uniform(0.9, 1.1, size=len(negative_data))
    negative_data['Vehicle Count'] *= np.random.uniform(0.8, 1.2, size=len(negative_data))

    # Combine positive and negative samples
    balanced_data = pd.concat([df_crime, negative_data], ignore_index=True)
    print(f"✓ Total balanced dataset: {len(balanced_data)} samples")
    print(f"  - Positive samples: {balanced_data['CrimeOccurred'].sum()}")
    print(f"  - Negative samples: {len(balanced_data) - balanced_data['CrimeOccurred'].sum()}")

    # Feature Engineering: Time-based features
    print("🛠️  Engineering time-based features...")
    balanced_data['Timestamp'] = pd.to_datetime(balanced_data['Timestamp'])
    balanced_data['Hour'] = balanced_data['Timestamp'].dt.hour
    balanced_data['DayOfWeek'] = balanced_data['Timestamp'].dt.dayofweek
    balanced_data['IsWeekend'] = balanced_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    # Select features for the model
    features_crime = ['Passenger Count', 'Trip Duration (mins)', 'District/Zone', 'Latitude', 'Longitude',
                      'Vehicle Count', 'Weather Condition', 'Hour', 'DayOfWeek', 'IsWeekend']
    
    # Check which features are available
    available_features = [f for f in features_crime if f in balanced_data.columns]
    print(f"📋 Available features for modeling: {len(available_features)}")
    for feat in available_features:
        print(f"  ✓ {feat}")
    
    if len(available_features) < 3:
        print(colored("❌ Error: Not enough features available for modeling", "red"))
        return
    
    X_crime = balanced_data[available_features]
    y_crime = balanced_data['CrimeOccurred']

    # One-hot encode categorical columns if present
    categorical_features = X_crime.select_dtypes(include=['object']).columns.tolist()
    if 'District/Zone' in X_crime.columns:
        X_crime = pd.get_dummies(X_crime, columns=['District/Zone'], drop_first=True)
        print("✓ One-hot encoded categorical features")

    # Normalize numerical features
    print("📊 Normalizing features...")
    scaler_crime = StandardScaler()
    numerical_cols = X_crime.select_dtypes(include=[np.number]).columns.tolist()
    X_crime[numerical_cols] = scaler_crime.fit_transform(X_crime[numerical_cols])

    # Train-test split with stratification
    print("🔀 Splitting data into train/test sets...")
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_crime, y_crime, test_size=0.2, stratify=y_crime, random_state=42
    )
    print(f"  - Training set: {len(X_train_c)} samples")
    print(f"  - Test set: {len(X_test_c)} samples")

    # Train individual models
    print("🤖 Training ensemble models...")
    
    # Random Forest Model
    print("  → Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_c, y_train_c)
    
    # Gradient Boosting Model  
    print("  → Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train_c, y_train_c)

    # Make predictions with individual models
    y_pred_rf = rf_model.predict(X_test_c)
    y_pred_gb = gb_model.predict(X_test_c)

    # Calculate individual model accuracies
    acc_rf = accuracy_score(y_test_c, y_pred_rf)
    acc_gb = accuracy_score(y_test_c, y_pred_gb)

    print(f"  ✓ Random Forest Accuracy: {acc_rf:.4f}")
    print(f"  ✓ Gradient Boosting Accuracy: {acc_gb:.4f}")

    # Create and train stacking ensemble
    print("  → Training Stacking Ensemble...")
    stacking_model = StackingClassifier(
        estimators=[('rf', rf_model), ('gb', gb_model)],
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    stacking_model.fit(X_train_c, y_train_c)

    # Final ensemble predictions
    y_pred_stack = stacking_model.predict(X_test_c)
    acc_stack = accuracy_score(y_test_c, y_pred_stack)

    # Display results
    print(colored("\n📊 CRIME PREDICTION RESULTS", "red", attrs=["bold"]))
    print(colored("="*50, "red"))
    print(f"🎯 Random Forest Accuracy:      {acc_rf:.4f} ({acc_rf*100:.2f}%)")
    print(f"🎯 Gradient Boosting Accuracy:  {acc_gb:.4f} ({acc_gb*100:.2f}%)")
    print(f"🏆 Stacking Ensemble Accuracy:  {acc_stack:.4f} ({acc_stack*100:.2f}%)")
    
    # Feature importance from best individual model
    best_model = rf_model if acc_rf > acc_gb else gb_model
    best_name = "Random Forest" if acc_rf > acc_gb else "Gradient Boosting"
    
    print(f"\n🔍 Feature Importance ({best_name}):")
    feature_importance = pd.DataFrame({
        'Feature': X_crime.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for _, row in feature_importance.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

    # Confusion Matrix for ensemble model
    print(f"\n🎭 Confusion Matrix (Stacking Ensemble):")
    cm = confusion_matrix(y_test_c, y_pred_stack)
    print(cm)
    
    # Classification Report
    print(f"\n📋 Classification Report (Stacking Ensemble):")
    print(classification_report(y_test_c, y_pred_stack))
    
    print(colored("✅ Crime Prediction Analysis completed successfully!", "green", attrs=["bold"]))

def accident_severity_analysis(df):
    """
    Perform accident severity prediction analysis.
    
    This function classifies accidents into severity levels using the Accident Severity column.
    It uses multiple classification algorithms to predict severity levels.
    
    Args:
        df (pandas.DataFrame): Preprocessed dataset
    """
    print(colored("\n🚗 ACCIDENT SEVERITY PREDICTION ANALYSIS", "yellow", attrs=["bold"]))
    print(colored("="*60, "yellow"))
    print("This analysis predicts accident severity levels based on:")
    print("• Weather conditions and vehicle count")
    print("• Geographic location and time factors")
    print("• Traffic and urban environment data")
    print()
    
    if 'Accident Severity' not in df.columns:
        print(colored("❌ Error: 'Accident Severity' column not found in dataset", "red"))
        return
    
    # Prepare data for accident severity prediction
    print("🔄 Preparing data for accident severity analysis...")
    df_accident = df.copy()
    
    # Remove rows with missing accident severity
    df_accident = df_accident.dropna(subset=['Accident Severity'])
    print(f"📊 Dataset size after cleaning: {len(df_accident)} samples")
    
    # Display accident severity distribution
    severity_counts = df_accident['Accident Severity'].value_counts().sort_index()
    print(f"📈 Accident Severity Distribution:")
    for severity, count in severity_counts.items():
        print(f"  Severity {severity}: {count} cases")
    
    # Feature selection for accident prediction
    accident_features = ['Vehicle Count', 'Weather Condition', 'Latitude', 'Longitude', 
                        'Passenger Count', 'Trip Duration (mins)']
    
    # Check available features
    available_features = [f for f in accident_features if f in df_accident.columns]
    print(f"📋 Available features for modeling: {len(available_features)}")
    
    if len(available_features) < 2:
        print(colored("❌ Error: Not enough features available for modeling", "red"))
        return
    
    X_accident = df_accident[available_features]
    y_accident = df_accident['Accident Severity']
    
    # Handle missing values in features
    X_accident = X_accident.fillna(X_accident.mean())
    
    # Normalize features
    print("📊 Normalizing features...")
    scaler = StandardScaler()
    X_accident_scaled = scaler.fit_transform(X_accident)
    X_accident_scaled = pd.DataFrame(X_accident_scaled, columns=available_features)
    
    # Train-test split
    print("🔀 Splitting data into train/test sets...")
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_accident_scaled, y_accident, test_size=0.2, random_state=42, stratify=y_accident
    )
    print(f"  - Training set: {len(X_train_a)} samples")
    print(f"  - Test set: {len(X_test_a)} samples")
    
    # Train multiple models
    print("🤖 Training multiple classification models...")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    for name, model in models.items():
        print(f"  → Training {name}...")
        model.fit(X_train_a, y_train_a)
        y_pred = model.predict(X_test_a)
        accuracy = accuracy_score(y_test_a, y_pred)
        results[name] = {'model': model, 'accuracy': accuracy, 'predictions': y_pred}
        print(f"    ✓ {name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    best_predictions = results[best_model_name]['predictions']
    
    # Display results
    print(colored(f"\n📊 ACCIDENT SEVERITY PREDICTION RESULTS", "yellow", attrs=["bold"]))
    print(colored("="*50, "yellow"))
    
    for name, result in results.items():
        status = "🏆" if name == best_model_name else "🎯"
        print(f"{status} {name}: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    
    print(f"\n🏆 Best Model: {best_model_name}")
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n🔍 Feature Importance ({best_model_name}):")
        feature_importance = pd.DataFrame({
            'Feature': available_features,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for _, row in feature_importance.iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # Confusion Matrix
    print(f"\n🎭 Confusion Matrix ({best_model_name}):")
    cm = confusion_matrix(y_test_a, best_predictions)
    print(cm)
    
    # Classification Report
    print(f"\n📋 Classification Report ({best_model_name}):")
    print(classification_report(y_test_a, best_predictions))
    
    print(colored("✅ Accident Severity Analysis completed successfully!", "green", attrs=["bold"]))

def passenger_count_prediction(df):
    """
    Perform passenger count prediction using regression analysis.
    
    This function predicts passenger counts using advanced feature engineering
    and Random Forest regression with hyperparameter tuning.
    
    Args:
        df (pandas.DataFrame): Preprocessed dataset
    """
    print(colored("\n🚌 PASSENGER COUNT PREDICTION ANALYSIS", "green", attrs=["bold"]))
    print(colored("="*60, "green"))
    print("This analysis predicts passenger counts based on:")
    print("• Weather conditions and time factors")
    print("• Stop locations and route information")
    print("• Accident severity and crime category")
    print("• Advanced feature engineering interactions")
    print()
    
    # Prepare passenger count dataset
    print("🔄 Preparing passenger count prediction dataset...")
    
    # Load the specific passenger count dataset if available
    passenger_file = "passenger_count_dataset_modified.csv"
    try:
        data = pd.read_csv(passenger_file)
        print(f"✓ Loaded passenger count dataset: {data.shape}")
    except FileNotFoundError:
        print(f"⚠️  {passenger_file} not found, using main dataset...")
        data = df.copy()
    
    # Rename columns to match expected names (if needed)
    column_mapping = {
        'Weather Condition': 'weather_condition',
        'Passenger Count': 'passenger_count',
        'Stop Locations': 'stop_location',
        'Accident Severity': 'accident_severity',
        'Crime Category': 'crime_category'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in data.columns:
            data = data.rename(columns={old_name: new_name})
    
    # Check required columns
    required_cols = ['passenger_count']
    if not all(col in data.columns for col in required_cols):
        print(colored("❌ Error: Required columns not found for passenger count prediction", "red"))
        print(f"Available columns: {list(data.columns)}")
        return
    
    # Feature engineering
    print("🛠️  Engineering features for passenger count prediction...")
    
    # Create time feature if timestamp available
    time_cols = [col for col in data.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
    if time_cols:
        data['time'] = pd.to_datetime(data[time_cols[0]], errors='coerce')
        data['time'] = data['time'].dt.hour if data['time'].dt.hour.notna().any() else 12
    else:
        data['time'] = 12  # Default noon hour
    
    # Create interaction features
    if 'weather_condition' in data.columns:
        data['weather_time_interaction'] = data.get('weather_condition', 0) * data['time']
    else:
        data['weather_condition'] = 0
        data['weather_time_interaction'] = 0
    
    # Fill missing values with defaults
    feature_defaults = {
        'stop_location': 'City Center',
        'accident_severity': 0,
        'crime_category': 0
    }
    
    for feature, default_val in feature_defaults.items():
        if feature not in data.columns:
            data[feature] = default_val
        else:
            data[feature] = data[feature].fillna(default_val)
    
    print(f"📊 Dataset prepared: {len(data)} samples")
    
    # Data cleaning
    print("🧹 Cleaning dataset...")
    original_size = len(data)
    
    # Remove invalid passenger counts
    data = data.dropna(subset=['passenger_count'])
    data = data[data['passenger_count'].apply(lambda x: isinstance(x, (int, float)) and x >= 0)]
    
    # Remove outliers (99th percentile threshold)
    q99 = data['passenger_count'].quantile(0.99)
    data = data[data['passenger_count'] <= q99]
    
    print(f"✓ Cleaned dataset: {len(data)} samples (removed {original_size - len(data)} outliers)")
    print(f"📈 Passenger count range: {data['passenger_count'].min():.0f} - {data['passenger_count'].max():.0f}")
    
    # Feature & target setup
    feature_columns = ['weather_condition', 'time', 'stop_location', 'accident_severity', 
                      'crime_category', 'weather_time_interaction']
    
    # Use only available features
    available_features = [col for col in feature_columns if col in data.columns]
    print(f"📋 Using {len(available_features)} features for prediction:")
    for feat in available_features:
        print(f"  ✓ {feat}")
    
    X = data[available_features]
    y = data['passenger_count']
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"🏷️  Encoding {len(categorical_cols)} categorical features...")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Create model pipeline
    base_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    # Hyperparameter tuning
    print("🔧 Performing hyperparameter tuning...")
    param_grid = {
        'regressor__n_estimators': [100, 150],
        'regressor__max_depth': [10, 15],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2]
    }
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"🔀 Split data: {len(X_train)} training, {len(X_test)} test samples")
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Calculate performance metrics
    train_r2 = best_model.score(X_train, y_train)
    test_r2 = best_model.score(X_test, y_test)
    cv_r2 = grid_search.best_score_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    # Display results
    print(colored("\n📊 PASSENGER COUNT PREDICTION RESULTS", "green", attrs=["bold"]))
    print(colored("="*50, "green"))
    print(f"🎯 Best Hyperparameters: {grid_search.best_params_}")
    print(f"📈 Training R² Score:     {train_r2:.4f}")
    print(f"📉 Testing R² Score:      {test_r2:.4f}")
    print(f"🔄 Cross-Validated R²:    {cv_r2:.4f}")
    print()
    print(f"📊 Mean Squared Error (MSE):     {mse:.2f}")
    print(f"📏 Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"📐 Mean Absolute Percentage Error: {mape:.2%}")
    print(f"🎯 Model Accuracy (approx):        {100 - mape * 100:.2f}%")
    
    # Feature importance
    if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
        print(f"\n🔍 Top 5 Feature Importances:")
        feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
        importances = best_model.named_steps['regressor'].feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names, 
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        for _, row in feature_importance.head(5).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    print(colored("✅ Passenger Count Prediction completed successfully!", "green", attrs=["bold"]))

def main():
    """
    Main function that orchestrates the Urban City Analysis workflow.
    
    This function:
    1. Loads and preprocesses the dataset
    2. Creates exploratory visualizations
    3. Displays menu and handles user choices
    4. Executes the selected analysis
    """
    print(colored("🏙️  WELCOME TO URBAN CITY ANALYSIS ML", "blue", attrs=["bold"]))
    print(colored("Author: DHARM DUDHAGARA", "blue"))
    print(colored("Repository: https://github.com/dharm1123/Urban-City-Analysis-ML", "blue"))
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        df_filled = preprocess_dataset(df)
        
        # Show visualizations
        try:
            show_viz = input(colored("\n📊 Do you want to see data visualizations? (y/n): ", "cyan")).lower()
            if show_viz in ['y', 'yes']:
                create_visualizations(df_filled)
        except (EOFError, KeyboardInterrupt):
            print(colored("\nSkipping visualizations...", "yellow"))
        
        # Main menu loop
        while True:
            try:
                choice = display_menu()
                
                if choice == '1':
                    crime_prediction_analysis(df_filled)
                elif choice == '2':
                    accident_severity_analysis(df_filled)
                elif choice == '3':
                    passenger_count_prediction(df_filled)
                elif choice == '0':
                    print(colored("\n👋 Thank you for using Urban City Analysis ML!", "blue", attrs=["bold"]))
                    print(colored("Goodbye!", "blue"))
                    break
                else:
                    print(colored("❌ Invalid choice. Please enter 0, 1, 2, or 3.", "red"))
                
                # Ask if user wants to continue
                if choice in ['1', '2', '3']:
                    try:
                        continue_choice = input(colored("\n🔄 Do you want to perform another analysis? (y/n): ", "cyan")).lower()
                        if continue_choice not in ['y', 'yes']:
                            print(colored("\n👋 Thank you for using Urban City Analysis ML!", "blue", attrs=["bold"]))
                            break
                    except (EOFError, KeyboardInterrupt):
                        print(colored("\n👋 Thank you for using Urban City Analysis ML!", "blue", attrs=["bold"]))
                        break
            except (EOFError, KeyboardInterrupt):
                print(colored("\n👋 Thank you for using Urban City Analysis ML!", "blue", attrs=["bold"]))
                break
    
    except KeyboardInterrupt:
        print(colored("\n\n⚠️  Analysis interrupted by user. Goodbye!", "yellow"))
    except Exception as e:
        print(colored(f"\n❌ An error occurred: {str(e)}", "red"))
        print(colored("Please check your data files and try again.", "red"))

if __name__ == "__main__":
    main()