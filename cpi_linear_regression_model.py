import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from scipy import stats

# Set pandas options to display all columns
pd.set_option('display.max_columns', None)

def load_data():
    """
    Load and prepare data from CSV file.
    """
    print("Loading data...")
    df = pd.read_csv("economic_data_filled_knn (1).csv")
    
    # Create datetime index
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
    df = df.set_index('Date')
    
    # Sort by date
    df = df.sort_index()
    
    # Convert all columns to float
    for col in df.columns:
        if col not in ['Year', 'Month']:  # Keep Year and Month as integers
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for NaN values
    if df.isnull().sum().sum() > 0:
        print(f"Warning: Dataset contains {df.isnull().sum().sum()} NaN values")
        print(df.isnull().sum())
        
        # Fill NaN values with column means
        df = df.fillna(df.mean())
        
    print(f"Data loaded with shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    return df

def check_stationarity(series, name):
    """
    Check stationarity of a time series using ADF test.
    """
    print(f"\nChecking stationarity for {name}...")
    
    # Drop NaN values for the test
    series_cleaned = series.dropna()
    
    # Perform ADF test
    result = adfuller(series_cleaned)
    
    print(f"ADF Test for {name}:")
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    
    # If p-value > 0.05, series is non-stationary
    if result[1] > 0.05:
        print(f"{name} is non-stationary. Applying first-order differencing.")
        differenced_series = series.diff().dropna()
        
        # Check stationarity again
        result = adfuller(differenced_series.dropna())
        print(f"After differencing - ADF Statistic: {result[0]}, p-value: {result[1]}")
        
        if result[1] <= 0.05:
            print(f"{name} is now stationary after differencing.")
            return differenced_series, True
        else:
            print(f"Warning: {name} remains non-stationary after first-order differencing.")
            return differenced_series, True
    else:
        print(f"{name} is already stationary.")
        return series, False

def select_features(df, target_col):
    """
    Select features based on correlation with target and VIF.
    """
    print("\n--- Feature Selection ---")
    
    potential_features = [
        'Brent', 'China_CPI', 'VN_money_supply', 
        'VN_Interest_Rate', 'USD_VND', 'Food_Inflation'
    ]
    
    # Calculate correlation with target
    correlations = {}
    for feature in potential_features:
        if feature in df.columns:
            corr = df[feature].corr(df[target_col])
            correlations[feature] = corr
            print(f"Correlation between {feature} and {target_col}: {corr:.4f}")
    
    # Select features with absolute correlation > 0.2
    selected_features = [feature for feature, corr in correlations.items() if abs(corr) > 0.2]
    print(f"\nFeatures with correlation > 0.2: {selected_features}")
    
    if len(selected_features) > 1:
        # Check for multicollinearity using VIF
        X = df[selected_features].dropna()
        
        # Add constant for statsmodels
        X = sm.add_constant(X)
        
        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        print("\nVariance Inflation Factors:")
        print(vif_data)
        
        # Remove features with VIF > 5 (excluding the constant)
        high_vif_features = vif_data[(vif_data["Feature"] != "const") & (vif_data["VIF"] > 5)]["Feature"].tolist()
        
        if high_vif_features:
            print(f"Removing features with VIF > 5: {high_vif_features}")
            selected_features = [f for f in selected_features if f not in high_vif_features]
    
    print(f"\nFinal selected features: {selected_features}")
    return selected_features

def prepare_model_data(df, selected_features, target_col, lag=1):
    """
    Prepare data for modeling with lagged features and seasonal dummies.
    """
    print(f"\n--- Preparing Model Data with lag {lag} ---")
    
    # Create a copy of the dataframe
    model_df = df.copy()
    
    # Create lagged features
    for feature in selected_features + [target_col]:
        model_df[f'{feature}_lag{lag}'] = model_df[feature].shift(lag)
    
    # Add month dummy variables
    model_df['month'] = model_df.index.month
    month_dummies = pd.get_dummies(model_df['month'], prefix='Month', drop_first=True)
    model_df = pd.concat([model_df, month_dummies], axis=1)
    model_df.drop('month', axis=1, inplace=True)
    
    # Convert boolean columns to int
    bool_columns = model_df.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        model_df[col] = model_df[col].astype(int)
    
    # Drop rows with NaN values
    model_df = model_df.dropna()
    
    # Verify data types
    for col in model_df.columns:
        if model_df[col].dtype == 'object':
            print(f"Converting column {col} to numeric")
            model_df[col] = pd.to_numeric(model_df[col], errors='coerce')
    
    # Drop any remaining rows with NaN
    model_df = model_df.dropna()
    
    print(f"Prepared data shape: {model_df.shape}")
    print(f"Data types after preparation:\n{model_df.dtypes}")
    
    return model_df

def split_data(df, target_col, test_size=0.2):
    """
    Split data into training and testing sets, preserving time order.
    """
    print("\n--- Splitting Data ---")
    
    # Determine split point
    split_idx = int(len(df) * (1 - test_size))
    
    # Split data
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Create feature matrix and target vector
    feature_cols = [col for col in df.columns if col != target_col]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def build_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Build and evaluate the linear regression model.
    """
    print("\n--- Building Linear Regression Model ---")
    
    # Print data info
    print(f"X_train shape: {X_train.shape}, dtypes: {X_train.dtypes.value_counts()}")
    print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
    
    # Add constant for intercept
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)
    
    try:
        # Fit the model
        model = sm.OLS(y_train, X_train_const).fit()
        
        # Print summary
        print(model.summary())
        
        # Predictions
        y_train_pred = model.predict(X_train_const)
        y_test_pred = model.predict(X_test_const)
        
        # Evaluation metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print("\n--- Model Evaluation ---")
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Testing MSE: {test_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Testing R²: {test_r2:.4f}")
        
        # Model diagnostics
        residuals = model.resid
        
        # Check for autocorrelation
        lb_test = acorr_ljungbox(residuals, lags=[10])
        lb_pvalue = lb_test.iloc[0, 1]
        print(f"Ljung-Box test p-value: {lb_pvalue:.4f}")
        
        # Check for heteroscedasticity
        bp_test = het_breuschpagan(residuals, X_train_const)
        bp_pvalue = bp_test[1]
        print(f"Breusch-Pagan test p-value: {bp_pvalue:.4f}")
        
        # Durbin-Watson test for autocorrelation
        dw_stat = durbin_watson(residuals)
        print(f"Durbin-Watson statistic: {dw_stat:.4f}")
        
        # Apply Newey-West if needed
        if lb_pvalue < 0.05 or bp_pvalue < 0.05:
            print("\nDetected autocorrelation or heteroscedasticity. Applying Newey-West standard errors.")
            model_robust = sm.OLS(y_train, X_train_const).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
            print("\nModel with Newey-West standard errors:")
            print(model_robust.summary())
            model = model_robust
        
        return model, {
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'residuals': residuals
        }
    except Exception as e:
        print(f"Error building model: {e}")
        print("\nAnalyzing data for issues:")
        print(f"X_train contains NaN: {X_train.isnull().sum().sum() > 0}")
        print(f"y_train contains NaN: {y_train.isnull().sum() > 0}")
        
        if X_train.isnull().sum().sum() > 0:
            print("NaN counts by column:")
            print(X_train.isnull().sum())
        
        return None, None

def test_granger_causality(df, target_col):
    """
    Test Granger causality between interest rate and CPI.
    """
    if 'VN_Interest_Rate' not in df.columns:
        print("VN_Interest_Rate not in dataframe, skipping Granger causality test.")
        return
    
    print("\n--- Granger Causality Test ---")
    
    # Create dataframe with the two variables
    causality_df = pd.DataFrame({
        'MonthlyCPI': df[target_col],
        'VN_Interest_Rate': df['VN_Interest_Rate']
    }).dropna()
    
    # Run Granger causality test
    max_lag = 4
    test_result = grangercausalitytests(causality_df, max_lag, verbose=False)
    
    print("Granger Causality Test Results:")
    for lag in range(1, max_lag + 1):
        p_value = test_result[lag][0]['ssr_chi2test'][1]
        print(f"Lag {lag}: p-value = {p_value:.4f}")

def visualize_results(y_train, y_test, results):
    """
    Visualize model results with plots.
    """
    if results is None:
        print("No results to visualize.")
        return
    
    print("\n--- Visualizing Results ---")
    
    # Set plot style
    sns.set(style="whitegrid")
    
    # Get predictions
    y_train_pred = results['y_train_pred']
    y_test_pred = results['y_test_pred']
    residuals = results['residuals']
    
    # 1. Actual vs Predicted (combined plot)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(y_train.index, y_train, label='Actual (Train)', alpha=0.7)
    plt.plot(y_train.index, y_train_pred, label='Predicted (Train)', alpha=0.7)
    plt.plot(y_test.index, y_test, label='Actual (Test)', alpha=0.7)
    plt.plot(y_test.index, y_test_pred, label='Predicted (Test)', alpha=0.7)
    
    plt.title('Actual vs Predicted MonthlyCPI')
    plt.xlabel('Date')
    plt.ylabel('MonthlyCPI')
    plt.legend()
    plt.tight_layout()
    
    # 2. Residuals over time
    plt.subplot(2, 2, 2)
    plt.scatter(y_train.index, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    
    plt.title('Residuals over Time')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.tight_layout()
    
    # 3. Q-Q Plot
    plt.subplot(2, 2, 3)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.tight_layout()
    
    # 4. Residuals vs Predicted
    plt.subplot(2, 2, 4)
    plt.scatter(y_train_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    
    plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
    print("Plots saved as 'model_evaluation_plots.png'")
    
    # Save individual required plots
    
    # Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_train.index, y_train, label='Actual (Train)', alpha=0.7)
    plt.plot(y_train.index, y_train_pred, label='Predicted (Train)', alpha=0.7)
    plt.plot(y_test.index, y_test, label='Actual (Test)', alpha=0.7)
    plt.plot(y_test.index, y_test_pred, label='Predicted (Test)', alpha=0.7)
    
    plt.title('Actual vs Predicted MonthlyCPI')
    plt.xlabel('Date')
    plt.ylabel('MonthlyCPI')
    plt.legend()
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    
    # Residuals over time
    plt.figure(figsize=(12, 6))
    plt.scatter(y_train.index, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    
    plt.title('Residuals over Time')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.tight_layout()
    plt.savefig('residuals_over_time.png', dpi=300, bbox_inches='tight')
    
    # Q-Q Plot
    plt.figure(figsize=(10, 10))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.tight_layout()
    plt.savefig('qq_plot.png', dpi=300, bbox_inches='tight')
    
    print("Additional plots saved as required.")

def main():
    print("=== CPI Linear Regression Model Analysis ===\n")
    
    # 1. Load data
    df = load_data()
    
    # 2. Check stationarity
    target_col = 'MonthlyCPI'
    stationary_series, is_differenced = check_stationarity(df[target_col], target_col)
    
    # Update dataframe if differencing was applied
    if is_differenced:
        print("Updating dataframe with differenced series")
        df['MonthlyCPI_original'] = df[target_col].copy()
        df[target_col] = stationary_series
    
    # 3. Feature selection
    selected_features = select_features(df, target_col)
    
    # 4. Prepare data with lag=1 (based on typical economic models)
    model_df = prepare_model_data(df, selected_features, target_col, lag=1)
    
    # 5. Split data
    X_train, X_test, y_train, y_test = split_data(model_df, target_col)
    
    # 6. Build and evaluate model
    model, results = build_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    # Only continue if model was successfully built
    if model is not None:
        # 7. Granger causality test
        test_granger_causality(df, target_col)
        
        # 8. Visualize results
        visualize_results(y_train, y_test, results)
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main() 