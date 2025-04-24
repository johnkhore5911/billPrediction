# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("electricity_bill_dataset.csv")  # Replace with your actual path

# Sample 10% of the data
sample_df = df.sample(frac=0.1, random_state=42)

# Define features and target
X = sample_df.drop("ElectricityBill", axis=1)
y = sample_df["ElectricityBill"]

# Identify columns
categorical_cols = ["City", "Company"]
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.difference(categorical_cols).tolist()

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Store metrics
results = {}

# Train and evaluate each model
for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    results[name] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2 Score": r2_score(y_test, y_pred)
    }

    if name == "Random Forest":
        rf_pipeline = pipe  # Save pipeline for user input prediction

# Convert results to DataFrame
results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})

# Display table
print("Performance Metrics for Both Models:")
print(results_df)

# Plot performance
metrics = ["MAE", "MSE", "RMSE", "R2 Score"]
plt.figure(figsize=(12, 6))
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    sns.barplot(x="Model", y=metric, data=results_df)
    plt.title(f"{metric} Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()

plt.suptitle("Model Performance Comparison (10% Sample)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# 🔍 Take user input for prediction using Random Forest
print("\nPredict Electricity Bill (Random Forest Model)")
input_data = {}
for col in X.columns:
    if col in categorical_cols:
        input_data[col] = input(f"{col} (Options: {df[col].unique().tolist()}): ")
    else:
        input_data[col] = float(input(f"{col}: "))

# Make prediction
input_df = pd.DataFrame([input_data])
predicted_bill = rf_pipeline.predict(input_df)[0]
print(f"\n💡 Predicted Electricity Bill: ₹{predicted_bill:.2f}")
