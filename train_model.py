import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv('data/raw/sdss_galaxy_dr18.csv', skiprows=1, low_memory=False)

# Inspect the dataset
print("First 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns)

# Calculate the required features
df['u-g'] = df['u'] - df['g']
df['g-r'] = df['g'] - df['r']
df['r-i'] = df['r'] - df['i']
df['i-z'] = df['i'] - df['z']

# Select 10 specific features
features = [
    'u-g', 'g-r', 'r-i', 'i-z', 'petroR50_r', 
    'petroFlux_r', 'psfMag_r', 'expAB_r', 'modelFlux_r', 'redshift'
]

# Drop rows with missing values
df = df.dropna(subset=features + ['subclass'])

# Encode the 'subclass' column
df['subclass'] = df['subclass'].map({'STARFORMING': 0, 'STARBURST': 1})

# Split data into features and target
X = df[features]  # Features (10 selected columns)
y = df['subclass']  # Target variable ('subclass')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'models/best_model_10_features.pkl')