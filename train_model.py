import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import sys

# Load dataset
df = pd.read_csv('data/avocado.csv')

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Encode categorical columns
le_region = LabelEncoder()
df['region_encoded'] = le_region.fit_transform(df['region'])

le_type = LabelEncoder()
df['type_encoded'] = le_type.fit_transform(df['type'])

# Features & Target
feature_cols = [
    'Total Volume', '4046', '4225', '4770', 'Total Bags',
    'Small Bags', 'Large Bags', 'XLarge Bags', 'Year',
    'region_encoded', 'type_encoded', 'Month', 'Day', 'DayOfWeek'
]
X = df[feature_cols]
y = df['AveragePrice']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Save Model & Encoders
joblib.dump(model, 'models/avocado_price_model.pkl')
joblib.dump(le_region, 'models/region_encoder.pkl')
joblib.dump(le_type, 'models/type_encoder.pkl')

# Safe print with emoji fallback
try:
    print("✅ Model and encoders saved successfully!")
except UnicodeEncodeError:
    print("Model and encoders saved successfully!")
