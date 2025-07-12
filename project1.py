import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
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

