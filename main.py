import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import os
import subprocess

def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name=None)
    return df

def preprocess_data(df):
    orignal_data = df.copy()
    df_mills = orignal_data['Mill_Yield_Quality']
    df_cane = orignal_data['Sugarcane_Quality']
    df_machinery = orignal_data['Machinery']
    df_weather = orignal_data['Weather']
    df = df_mills.merge(df_cane, on=['Date', 'Mill_ID'], how='left')
    df = df.merge(df_weather, on=['Date', 'Mill_ID'], how='left')
    mach_summary = df_machinery.groupby(["Date", "Mill_ID"]).agg({"Usage_Hours": "mean","Fault_Count": "sum"}).reset_index()
    df = df.merge(mach_summary, on=["Date", "Mill_ID"], how="left")

    # Data features
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    
    # Keep Date and Mill_ID for test set before dropping Date
    X_full = df.copy()
    y_full = df["Sugar_Produced_kg"]
    dates = X_full["Date"]
    mill_ids = X_full["Mill_ID"]
    
    df.drop("Date", axis=1, inplace=True)
    
    # Encode categorical variables
    df = pd.get_dummies(df, columns=["Mill_ID", "Variety"], drop_first=True)

    # Feature Engineering
    df["Cane_to_Sugar_Ratio"] = df["Sugar_Produced_kg"] / df["Total_Cane_Processed"]
    df["Fault_Rate"] = df["Fault_Count"] / df["Usage_Hours"]
    # Outlier removal on target
    Q3 = df["Sugar_Produced_kg"].quantile(0.75)
    Q1 = df["Sugar_Produced_kg"].quantile(0.25)
    IQR = Q3 - Q1
    df = df[(df["Sugar_Produced_kg"] >= Q1 - 1.5 * IQR) & (df["Sugar_Produced_kg"] <= Q3 + 1.5 * IQR)]
    # Scaling
    features_to_scale = [
        "Total_Cane_Processed", "Brix", "Fiber", "Pol", 
        "Temperature_C", "Rainfall_mm", "Humidity_%", 
        "Usage_Hours", "Fault_Count", "Cane_to_Sugar_Ratio", "Fault_Rate"
    ]
    
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # Define features and target
    X = df.drop("Sugar_Produced_kg", axis=1)
    y = df["Sugar_Produced_kg"]

    # Train-test split with indices to keep track of Date and Mill_ID
    X_train, X_test, y_train, y_test, dates_train, dates_test, mill_ids_train, mill_ids_test = train_test_split(
        X, y, dates, mill_ids, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler, dates_test, mill_ids_test

def apply_model(X_train, y_train, X_test, y_test, dates_test, mill_ids_test):
    # Linear Regression Model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    
    # Decision Tree Regressor
    dt_regressor = DecisionTreeRegressor(random_state=42)
    dt_regressor.fit(X_train, y_train)
    y_pred_dt = dt_regressor.predict(X_test)
    
    # Random Forest Regressor
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_regressor.fit(X_train, y_train)
    y_pred_rf = rf_regressor.predict(X_test)

    # Evaluate models using R^2 score
    print("Linear Regression R^2 Score:", r2_score(y_test, y_pred_linear))
    print("Decision Tree Regressor R^2 Score:", r2_score(y_test, y_pred_dt))
    print("Random Forest Regressor R^2 Score:", r2_score(y_test, y_pred_rf))

    # Save models
    joblib.dump(linear_model, r'E:\My_Projects\Sugar_mill_yeild_and_quality\linear_model.pkl')
    joblib.dump(dt_regressor, r'E:\My_Projects\Sugar_mill_yeild_and_quality\dt_regressor.pkl')
    joblib.dump(rf_regressor, r'E:\My_Projects\Sugar_mill_yeild_and_quality\rf_regressor.pkl')

    # result to xlsx with Date and Mill_ID
    results = pd.DataFrame({
        'Date': dates_test,
        'Mill_ID': mill_ids_test,
        'Actual_Sugar_Produced_kg': y_test,
        'Linear_Predicted': y_pred_linear,
        'Decision_Tree_Predicted': y_pred_dt,
        'Random_Forest_Predicted': y_pred_rf
    })
    results.to_excel(r'E:\My_Projects\Sugar_mill_yeild_and_quality\model_results.xlsx', index=False)

if __name__ == "__main__":
    file_path = r"E:\My_Projects\Sugar_mill_yeild_and_quality\sugar_mill_erp_dataset.xlsx"
    load_data(file_path)
    df = load_data(file_path)
    X_train, X_test, y_train, y_test, scaler, dates_test, mill_ids_test = preprocess_data(df)
    apply_model(X_train, y_train, X_test, y_test, dates_test, mill_ids_test)
    print("Models trained and saved successfully.")

    # Path to Power BI Desktop executable (usually in this location)
    pbi_path = r"C:\Program Files\Microsoft Power BI Desktop\bin\PBIDesktop.exe"

    # Path to your PBIX file
    dashboard_path = r"C:\Users\Dell M4600\Documents\Sugar_Mill_Production.pbix"

    # Run Power BI and open your dashboard
    subprocess.Popen([pbi_path, dashboard_path])

    print("📊 Power BI dashboard opened.")
