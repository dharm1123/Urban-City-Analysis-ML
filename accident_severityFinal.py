import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("final_crime_dataset.csv")

# Drop unneeded columns
df = df.drop(columns=[
    "Incident ID", "Timestamp", "Date/Time", "Timestamp.1", "Incident Description"
])

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Convert Accident Severity to 3 levels
def simplify_severity(sev):
    if sev <= 3:
        return 0  # Low
    elif sev <= 6:
        return 1  # Medium
    else:
        return 2  # High

df["Severity_Level"] = df["Accident Severity"].apply(simplify_severity)

# Features and target
X = df.drop(columns=["Accident Severity", "Severity_Level"])
y = df["Severity_Level"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

