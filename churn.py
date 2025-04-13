import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('churn.csv')

# Drop customerID if it exists (assuming it's not useful for model training)
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Convert 'TotalCharges' to numeric and drop rows with NaN values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Encode categorical features using LabelEncoder
le = LabelEncoder()
# Apply encoding only to categorical columns (i.e., object columns)
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop('Churn', axis=1)  # Features (dropping target 'Churn')
y = df['Churn']  # Target variable

# Initialize and apply StandardScaler for feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Apply SMOTE to handle class imbalance (resample the minority class)
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ✅ Train RandomForestClassifier with balanced class weights to address class imbalance
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model, scaler, and feature names
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')  # Save the column names (features)

