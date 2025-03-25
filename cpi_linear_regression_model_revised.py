import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from scipy import stats
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as path_effects
from matplotlib import cm

# Set pandas options
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
    
    # Convert all numeric columns to float
    for col in df.columns:
        if col not in ['Year', 'Month']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for NaN values
    if df.isnull().sum().sum() > 0:
        print(f"Warning: Dataset contains {df.isnull().sum().sum()} NaN values")
        print(df.isnull().sum())
        df = df.fillna(df.mean())
    
    print(f"Data loaded with shape: {df.shape}")
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

def prepare_model_data(df, target_col, lag=1):
    """
    Prepare data for modeling with a simplified approach.
    """
    print(f"\n--- Preparing Model Data ---")
    
    # Create a copy of the dataframe with only essential variables
    variables = [
        'Year', 'Month', 'Brent', 'VN_Interest_Rate', 
        'VN_rice_price', 'VN_Trade_Balance', 'MonthlyCPI'
    ]
    
    model_df = df[variables].copy()
    
    # Ensure all columns are numeric
    for col in model_df.columns:
        model_df[col] = pd.to_numeric(model_df[col], errors='coerce')
    
    # Create lagged features
    for feature in variables:
        if feature != 'Year' and feature != 'Month' and feature != target_col:
            model_df[f'{feature}_lag{lag}'] = model_df[feature].shift(lag)
    
    # Add autoregressive term
    model_df[f'{target_col}_lag{lag}'] = model_df[target_col].shift(lag)
    
    # Add seasonal dummies (quarterly instead of monthly to reduce parameters)
    model_df['Quarter'] = (model_df.index.month - 1) // 3 + 1
    quarter_dummies = pd.get_dummies(model_df['Quarter'], prefix='Q', drop_first=True)
    
    # Convert dummy variables to int
    for col in quarter_dummies.columns:
        quarter_dummies[col] = quarter_dummies[col].astype(int)
    
    model_df = pd.concat([model_df, quarter_dummies], axis=1)
    model_df.drop('Quarter', axis=1, inplace=True)
    
    # Drop rows with NaN values
    model_df = model_df.dropna()
    
    # Final check for any non-numeric columns
    for col in model_df.columns:
        if not np.issubdtype(model_df[col].dtype, np.number):
            print(f"Converting column {col} to numeric")
            model_df[col] = pd.to_numeric(model_df[col], errors='coerce')
    
    # Drop any remaining rows with NaN after conversion
    model_df = model_df.dropna()
    
    print(f"Prepared data shape: {model_df.shape}")
    print(f"Columns in prepared data: {model_df.columns.tolist()}")
    print(f"Data types in prepared data: {model_df.dtypes}")
    
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
    
    # Check for and convert any non-numeric data
    for col in X_train.columns:
        if not np.issubdtype(X_train[col].dtype, np.number):
            print(f"Converting column {col} to numeric in X_train")
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    
    for col in X_test.columns:
        if not np.issubdtype(X_test[col].dtype, np.number):
            print(f"Converting column {col} to numeric in X_test")
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # Convert target variables to numeric
    if not np.issubdtype(y_train.dtype, np.number):
        print("Converting y_train to numeric")
        y_train = pd.to_numeric(y_train, errors='coerce')
    
    if not np.issubdtype(y_test.dtype, np.number):
        print("Converting y_test to numeric")
        y_test = pd.to_numeric(y_test, errors='coerce')
    
    # Drop any NaN values that might have been introduced
    mask_train = ~np.isnan(y_train)
    X_train = X_train.loc[mask_train]
    y_train = y_train[mask_train]
    
    mask_test = ~np.isnan(y_test)
    X_test = X_test.loc[mask_test]
    y_test = y_test[mask_test]
    
    # Print data shapes and types for debugging
    print(f"X_train shape: {X_train.shape}, dtypes: {X_train.dtypes}")
    print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
    
    # Add constant for intercept
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)
    
    # Fit the model
    try:
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
        print(f"X_train shape: {X_train.shape}")
        print(f"X_train contains NaN: {X_train.isnull().sum().sum() > 0}")
        print(f"y_train contains NaN: {y_train.isnull().sum() > 0}")
        
        if X_train.isnull().sum().sum() > 0:
            print("NaN counts by column:")
            print(X_train.isnull().sum())
        
        # Try converting to numpy arrays as a last resort
        print("\nAttempting to convert to numpy arrays directly:")
        X_train_np = np.asarray(X_train.values, dtype=float)
        y_train_np = np.asarray(y_train.values, dtype=float)
        
        print(f"X_train_np shape: {X_train_np.shape}, dtype: {X_train_np.dtype}")
        print(f"y_train_np shape: {y_train_np.shape}, dtype: {y_train_np.dtype}")
        
        try:
            X_train_np = sm.add_constant(X_train_np)
            model_np = sm.OLS(y_train_np, X_train_np).fit()
            print("Model successfully built using numpy arrays")
            print(model_np.summary())
            return model_np, None
        except Exception as e2:
            print(f"Error building model with numpy arrays: {e2}")
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

def visualize_results(X_train, X_test, y_train, y_test, results, model=None):
    """
    Visualize model results with enhanced, visually impressive plots.
    """
    print("\n--- Creating Advanced Visualizations ---")
    
    # Check if there are results to visualize
    if results is None:
        if model is None:
            print("No results or model to visualize.")
            return
        
        # Create results from model if we have it
        print("No results dictionary, creating predictions from model...")
        X_train_const = sm.add_constant(X_train) if 'const' not in X_train.columns else X_train
        X_test_const = sm.add_constant(X_test) if 'const' not in X_test.columns else X_test
        
        y_train_pred = model.predict(X_train_const)
        y_test_pred = model.predict(X_test_const)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Testing MSE: {test_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Testing R²: {test_r2:.4f}")
        
        # Create results dictionary
        results = {
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'residuals': model.resid
        }
    
    # Set modern plot style with dark background for dramatic effect
    plt.style.use('dark_background')
    
    # Custom color palette with vibrant, modern colors
    colors = {
        'actual_train': '#2CA02C',  # Green
        'pred_train': '#FF7F0E',    # Orange
        'actual_test': '#1F77B4',   # Blue
        'pred_test': '#D62728',     # Red
        'residual': '#9467BD',      # Purple
        'zero_line': '#FCFF33',     # Yellow
        'grid': '#555555',          # Dark gray
        'text': '#FFFFFF'           # White
    }
    
    # Get predictions
    y_train_pred = results['y_train_pred']
    y_test_pred = results['y_test_pred']
    residuals = results['residuals']
    
    # Create a figure with custom layout
    fig = plt.figure(figsize=(20, 15), facecolor='black')
    gs = GridSpec(6, 6, figure=fig)
    
    # Add a bold title to the entire figure
    fig.suptitle('VIETNAM MONTHLY CPI: ADVANCED MODEL ANALYSIS', 
                 fontsize=28, fontweight='bold', color='white', y=0.98)
    fig_text = fig.text(0.5, 0.94, 
             'Predictive Analytics & Diagnostic Visualization', 
             ha='center', color='#AAAAAA', fontsize=18)
    
    # Add subtle drop shadow to title for depth
    fig_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='#333333')])
    
    # 1. Enhanced Actual vs Predicted Plot
    ax1 = fig.add_subplot(gs[0:3, 0:3])
    
    # Add semi-transparent gradient background to highlight the test region
    train_end = y_train.index[-1]
    test_start = y_test.index[0]
    
    # Plot with increased line thickness and markers
    ax1.plot(y_train.index, y_train, color=colors['actual_train'], linewidth=3, 
             marker='o', markersize=5, label='Actual (Train)', alpha=0.9)
    ax1.plot(y_train.index, y_train_pred, color=colors['pred_train'], linewidth=3, 
             marker='s', markersize=5, label='Predicted (Train)', alpha=0.9)
    ax1.plot(y_test.index, y_test, color=colors['actual_test'], linewidth=3, 
             marker='o', markersize=5, label='Actual (Test)', alpha=0.9)
    ax1.plot(y_test.index, y_test_pred, color=colors['pred_test'], linewidth=3, 
             marker='s', markersize=5, label='Predicted (Test)', alpha=0.9)
    
    # Add shaded background for test period
    ax1.axvspan(test_start, y_test.index[-1], color='white', alpha=0.05)
    
    # Add annotation for test period
    ax1.annotate('Test Period', xy=(test_start, ax1.get_ylim()[0]), 
                xytext=(test_start, ax1.get_ylim()[0] - 0.1), 
                color='white', fontsize=12, ha='center',
                arrowprops=dict(facecolor='white', shrink=0.05, width=2, headwidth=8))
    
    # Customize grid and ticks
    ax1.grid(True, linestyle='--', alpha=0.3, color=colors['grid'])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.tick_params(axis='both', which='major', labelsize=12, colors=colors['text'])
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add model performance metrics as annotations
    performance_text = (
        f"Training R² = {results['train_r2']:.3f}\n"
        f"Testing R² = {results['test_r2']:.3f}\n"
        f"Train MSE = {results['train_mse']:.3f}\n"
        f"Test MSE = {results['test_mse']:.3f}"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7, edgecolor='white')
    ax1.text(0.02, 0.98, performance_text, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, color=colors['text'])
    
    # Enhanced title and labels with shadow effect for depth
    title = ax1.set_title('Actual vs Predicted Monthly CPI', fontsize=20, fontweight='bold', color='white', pad=20)
    title.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    xlabel = ax1.set_xlabel('Date', fontsize=14, fontweight='bold', color='white', labelpad=10)
    xlabel.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    ylabel = ax1.set_ylabel('Monthly CPI', fontsize=14, fontweight='bold', color='white', labelpad=10)
    ylabel.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    # Enhanced legend with custom styling
    legend = ax1.legend(loc='upper left', fontsize=12, frameon=True, framealpha=0.8)
    legend.get_frame().set_facecolor('black')
    legend.get_frame().set_edgecolor('white')
    for text in legend.get_texts():
        text.set_color('white')
    
    # 2. Enhanced Residuals over time - with color gradient by magnitude
    ax2 = fig.add_subplot(gs[0:3, 3:6])
    
    # Create colorful scatter with color mapped to residual magnitude
    norm = plt.Normalize(-abs(residuals).max(), abs(residuals).max())
    scatter = ax2.scatter(y_train.index, residuals, 
                         c=residuals, cmap='coolwarm', 
                         s=80, alpha=0.7, edgecolor='white', linewidth=0.5,
                         norm=norm)
    
    # Add zero line with enhanced styling
    ax2.axhline(y=0, color=colors['zero_line'], linestyle='-', linewidth=2, alpha=0.8)
    
    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax2, pad=0.01)
    cbar.set_label('Residual Magnitude', fontsize=12, fontweight='bold', color='white', labelpad=10)
    cbar.ax.tick_params(colors='white')
    
    # Calculate and display Durbin-Watson statistic
    if model is not None:
        dw_stat = durbin_watson(residuals)
        dw_text = f"Durbin-Watson: {dw_stat:.3f}"
        props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7, edgecolor='white')
        ax2.text(0.02, 0.02, dw_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='bottom', bbox=props, color='white')
        
        # Add interpretation
        if dw_stat < 1.5:
            interpretation = "Positive Autocorrelation"
        elif dw_stat > 2.5:
            interpretation = "Negative Autocorrelation"
        else:
            interpretation = "No Significant Autocorrelation"
        
        ax2.text(0.98, 0.02, interpretation, transform=ax2.transAxes, fontsize=12,
                horizontalalignment='right', verticalalignment='bottom', 
                bbox=props, color='white')
    
    # Customize grid and ticks
    ax2.grid(True, linestyle='--', alpha=0.3, color=colors['grid'])
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.tick_params(axis='both', which='major', labelsize=12, colors=colors['text'])
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Enhanced title and labels
    title = ax2.set_title('Residuals Analysis Over Time', fontsize=20, fontweight='bold', color='white', pad=20)
    title.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    xlabel = ax2.set_xlabel('Date', fontsize=14, fontweight='bold', color='white', labelpad=10)
    xlabel.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    ylabel = ax2.set_ylabel('Residual', fontsize=14, fontweight='bold', color='white', labelpad=10)
    ylabel.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    # 3. Enhanced Q-Q Plot
    ax3 = fig.add_subplot(gs[3:6, 0:3])
    
    # Create custom Q-Q plot with colored points
    qq = stats.probplot(residuals, dist="norm", fit=True, plot=ax3)
    
    # Extract the data points and line
    ax3.get_lines()[0].set_markerfacecolor('#1F77B4')
    ax3.get_lines()[0].set_markeredgecolor('white')
    ax3.get_lines()[0].set_markersize(10)
    ax3.get_lines()[0].set_alpha(0.7)
    
    # Make the line thicker
    ax3.get_lines()[1].set_color('#FF7F0E')
    ax3.get_lines()[1].set_linewidth(3)
    
    # Add shaded confidence intervals
    sorted_residuals = np.sort(residuals)
    n = len(sorted_residuals)
    confidence = 0.95
    j = stats.norm.ppf((1 + confidence) / 2)
    stderr = np.std(sorted_residuals) * (1 / np.sqrt(n))
    
    x = qq[0][0]
    y = qq[0][1]
    y_fit = qq[1][0] * x + qq[1][1]
    
    ax3.fill_between(x, y_fit - j * stderr, y_fit + j * stderr, 
                     color='cyan', alpha=0.2, label=f'{int(confidence*100)}% Confidence Interval')
    
    # Add correlation coefficient
    r, _ = stats.pearsonr(x, y)
    corr_text = f"Correlation: {r:.3f}"
    
    # Perform Shapiro-Wilk test for normality
    _, p_value = stats.shapiro(residuals)
    sw_text = f"Shapiro-Wilk p-value: {p_value:.5f}"
    
    # Determine if residuals are normal
    if p_value < 0.05:
        norm_text = "Non-Normal Residuals"
    else:
        norm_text = "Normal Residuals"
    
    # Add annotation box
    stats_text = f"{corr_text}\n{sw_text}\n{norm_text}"
    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7, edgecolor='white')
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, color='white')
    
    # Customize grid and ticks
    ax3.grid(True, linestyle='--', alpha=0.3, color=colors['grid'])
    ax3.tick_params(axis='both', which='major', labelsize=12, colors=colors['text'])
    
    # Enhanced title and labels
    title = ax3.set_title('Q-Q Plot for Normality Assessment', fontsize=20, fontweight='bold', color='white', pad=20)
    title.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    xlabel = ax3.set_xlabel('Theoretical Quantiles', fontsize=14, fontweight='bold', color='white', labelpad=10)
    xlabel.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    ylabel = ax3.set_ylabel('Sample Quantiles', fontsize=14, fontweight='bold', color='white', labelpad=10)
    ylabel.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    # Make legend for Q-Q plot
    ax3.legend(loc='lower right', fontsize=12, framealpha=0.8)
    
    # 4. Residuals vs Predicted with LOWESS trend
    ax4 = fig.add_subplot(gs[3:6, 3:6])
    
    # Create scatter with predefined colormap for residual magnitude
    scatter = ax4.scatter(y_train_pred, residuals, 
                         c=np.abs(residuals), cmap='viridis', 
                         s=80, alpha=0.7, edgecolor='white', linewidth=0.5)
    
    # Add zero line
    ax4.axhline(y=0, color=colors['zero_line'], linestyle='-', linewidth=2, alpha=0.8)
    
    # Add LOWESS smoothed trend line to check for patterns
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        z = lowess(residuals, y_train_pred, frac=0.6, it=3)
        ax4.plot(z[:, 0], z[:, 1], color='red', linewidth=3, label='LOWESS Trend')
        
        # Check if trend line deviates significantly from horizontal
        trend_variation = np.std(z[:, 1])
        if trend_variation > 0.1:
            pattern_text = "⚠️ Pattern Detected"
        else:
            pattern_text = "✓ No Pattern Detected"
            
        props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7, edgecolor='white')
        ax4.text(0.02, 0.02, pattern_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='bottom', bbox=props, color='white')
    except:
        pass  # Skip LOWESS if not available
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4, pad=0.01)
    cbar.set_label('|Residual|', fontsize=12, fontweight='bold', color='white', labelpad=10)
    cbar.ax.tick_params(colors='white')
    
    # Calculate and display Breusch-Pagan test for heteroscedasticity
    if model is not None and 'const' in X_train.columns:
        bp_test = het_breuschpagan(residuals, X_train)
        bp_pvalue = bp_test[1]
        bp_text = f"Breusch-Pagan p-value: {bp_pvalue:.5f}"
        
        if bp_pvalue < 0.05:
            interp = "⚠️ Heteroscedasticity Present"
        else:
            interp = "✓ Homoscedasticity (Constant Variance)"
            
        bp_full = f"{bp_text}\n{interp}"
        props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7, edgecolor='white')
        ax4.text(0.98, 0.02, bp_full, transform=ax4.transAxes, fontsize=12,
                horizontalalignment='right', verticalalignment='bottom', bbox=props, color='white')
        
    # Customize grid and ticks
    ax4.grid(True, linestyle='--', alpha=0.3, color=colors['grid'])
    ax4.tick_params(axis='both', which='major', labelsize=12, colors=colors['text'])
    
    # Enhanced title and labels
    title = ax4.set_title('Residuals vs Predicted Values', fontsize=20, fontweight='bold', color='white', pad=20)
    title.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    xlabel = ax4.set_xlabel('Predicted Values', fontsize=14, fontweight='bold', color='white', labelpad=10)
    xlabel.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    ylabel = ax4.set_ylabel('Residuals', fontsize=14, fontweight='bold', color='white', labelpad=10)
    ylabel.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    # Add watermark with version info
    fig.text(0.99, 0.01, 'Advanced Visualization v2.0', 
             fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)
    
    # Adjust layout and save high-resolution figure
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('revised_model_evaluation_plots.png', dpi=300, bbox_inches='tight', facecolor='black')
    print("Enhanced plots saved as 'revised_model_evaluation_plots.png'")
    
    # Save individual enhanced plots
    
    # Actual vs Predicted
    plt.figure(figsize=(14, 8), facecolor='black')
    
    plt.plot(y_train.index, y_train, color=colors['actual_train'], linewidth=3, 
             marker='o', markersize=6, label='Actual (Train)', alpha=0.9)
    plt.plot(y_train.index, y_train_pred, color=colors['pred_train'], linewidth=3, 
             marker='s', markersize=6, label='Predicted (Train)', alpha=0.9)
    plt.plot(y_test.index, y_test, color=colors['actual_test'], linewidth=3, 
             marker='o', markersize=6, label='Actual (Test)', alpha=0.9)
    plt.plot(y_test.index, y_test_pred, color=colors['pred_test'], linewidth=3, 
             marker='s', markersize=6, label='Predicted (Test)', alpha=0.9)
    
    # Add model performance metrics
    performance_text = (
        f"Training R² = {results['train_r2']:.3f}\n"
        f"Testing R² = {results['test_r2']:.3f}\n"
        f"Train MSE = {results['train_mse']:.3f}\n"
        f"Test MSE = {results['test_mse']:.3f}"
    )
    
    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7, edgecolor='white')
    plt.text(0.02, 0.98, performance_text, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props, color='white')
    
    # Shade test period
    plt.axvspan(test_start, y_test.index[-1], color='white', alpha=0.05)
    
    # Add annotation for test period
    plt.annotate('Test Period', xy=(test_start, plt.gca().get_ylim()[0]), 
                xytext=(test_start, plt.gca().get_ylim()[0] - 0.1), 
                color='white', fontsize=12, ha='center',
                arrowprops=dict(facecolor='white', shrink=0.05, width=2, headwidth=8))
    
    # Customize grid and ticks
    plt.grid(True, linestyle='--', alpha=0.3, color=colors['grid'])
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.tick_params(axis='both', which='major', labelsize=12, colors=colors['text'])
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Enhanced title and labels
    title = plt.title('Actual vs Predicted Monthly CPI', fontsize=22, fontweight='bold', color='white', pad=20)
    title.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    xlabel = plt.xlabel('Date', fontsize=16, fontweight='bold', color='white', labelpad=10)
    xlabel.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    ylabel = plt.ylabel('Monthly CPI', fontsize=16, fontweight='bold', color='white', labelpad=10)
    ylabel.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    # Enhanced legend
    legend = plt.legend(loc='upper left', fontsize=12, frameon=True, framealpha=0.8)
    legend.get_frame().set_facecolor('black')
    legend.get_frame().set_edgecolor('white')
    for text in legend.get_texts():
        text.set_color('white')
    
    plt.tight_layout()
    plt.savefig('revised_actual_vs_predicted.png', dpi=300, bbox_inches='tight', facecolor='black')
    
    # Residuals over time with color gradient
    plt.figure(figsize=(14, 8), facecolor='black')
    
    # Create a colormap based on residual values
    norm = plt.Normalize(-abs(residuals).max(), abs(residuals).max())
    scatter = plt.scatter(y_train.index, residuals, 
                         c=residuals, cmap='coolwarm', 
                         s=100, alpha=0.7, edgecolor='white', linewidth=0.5,
                         norm=norm)
    
    # Add zero line
    plt.axhline(y=0, color=colors['zero_line'], linestyle='-', linewidth=2, alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, pad=0.01)
    cbar.set_label('Residual Magnitude', fontsize=14, fontweight='bold', color='white', labelpad=10)
    cbar.ax.tick_params(colors='white')
    
    # Calculate and display Durbin-Watson statistic
    if model is not None:
        dw_stat = durbin_watson(residuals)
        dw_text = f"Durbin-Watson: {dw_stat:.3f}"
        props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7, edgecolor='white')
        plt.text(0.02, 0.02, dw_text, transform=plt.gca().transAxes, fontsize=14,
                verticalalignment='bottom', bbox=props, color='white')
        
        # Add interpretation
        if dw_stat < 1.5:
            interpretation = "Positive Autocorrelation"
        elif dw_stat > 2.5:
            interpretation = "Negative Autocorrelation"
        else:
            interpretation = "No Significant Autocorrelation"
        
        plt.text(0.98, 0.02, interpretation, transform=plt.gca().transAxes, fontsize=14,
                horizontalalignment='right', verticalalignment='bottom', 
                bbox=props, color='white')
    
    # Customize grid and ticks
    plt.grid(True, linestyle='--', alpha=0.3, color=colors['grid'])
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.tick_params(axis='both', which='major', labelsize=12, colors=colors['text'])
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Enhanced title and labels
    title = plt.title('Residuals Analysis Over Time', fontsize=22, fontweight='bold', color='white', pad=20)
    title.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    xlabel = plt.xlabel('Date', fontsize=16, fontweight='bold', color='white', labelpad=10)
    xlabel.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    ylabel = plt.ylabel('Residual', fontsize=16, fontweight='bold', color='white', labelpad=10)
    ylabel.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    plt.tight_layout()
    plt.savefig('revised_residuals_over_time.png', dpi=300, bbox_inches='tight', facecolor='black')
    
    # Enhanced Q-Q Plot
    plt.figure(figsize=(12, 12), facecolor='black')
    
    # Create Q-Q plot
    qq = stats.probplot(residuals, dist="norm", fit=True, plot=plt)
    
    # Extract the data points and line to style them
    plt.gca().get_lines()[0].set_markerfacecolor('#1F77B4')
    plt.gca().get_lines()[0].set_markeredgecolor('white')
    plt.gca().get_lines()[0].set_markersize(12)
    plt.gca().get_lines()[0].set_alpha(0.7)
    
    plt.gca().get_lines()[1].set_color('#FF7F0E')
    plt.gca().get_lines()[1].set_linewidth(3)
    
    # Add confidence intervals
    sorted_residuals = np.sort(residuals)
    n = len(sorted_residuals)
    confidence = 0.95
    j = stats.norm.ppf((1 + confidence) / 2)
    stderr = np.std(sorted_residuals) * (1 / np.sqrt(n))
    
    x = qq[0][0]
    y = qq[0][1]
    y_fit = qq[1][0] * x + qq[1][1]
    
    plt.fill_between(x, y_fit - j * stderr, y_fit + j * stderr, 
                     color='cyan', alpha=0.2, label=f'{int(confidence*100)}% Confidence Interval')
    
    # Add correlation coefficient
    r, _ = stats.pearsonr(x, y)
    corr_text = f"Correlation: {r:.3f}"
    
    # Perform Shapiro-Wilk test for normality
    _, p_value = stats.shapiro(residuals)
    sw_text = f"Shapiro-Wilk p-value: {p_value:.5f}"
    
    # Determine if residuals are normal
    if p_value < 0.05:
        norm_text = "Non-Normal Residuals"
    else:
        norm_text = "Normal Residuals"
    
    # Add annotation box
    stats_text = f"{corr_text}\n{sw_text}\n{norm_text}"
    props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7, edgecolor='white')
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props, color='white')
    
    # Customize grid and ticks
    plt.grid(True, linestyle='--', alpha=0.3, color=colors['grid'])
    plt.tick_params(axis='both', which='major', labelsize=12, colors=colors['text'])
    
    # Enhanced title and labels
    title = plt.title('Q-Q Plot for Normality Assessment', fontsize=22, fontweight='bold', color='white', pad=20)
    title.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    xlabel = plt.xlabel('Theoretical Quantiles', fontsize=16, fontweight='bold', color='white', labelpad=10)
    xlabel.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    ylabel = plt.ylabel('Sample Quantiles', fontsize=16, fontweight='bold', color='white', labelpad=10)
    ylabel.set_path_effects([path_effects.withStroke(linewidth=1, foreground='#333333')])
    
    # Add legend
    plt.legend(loc='lower right', fontsize=12, framealpha=0.8)
    
    plt.tight_layout()
    plt.savefig('revised_qq_plot.png', dpi=300, bbox_inches='tight', facecolor='black')
    
    print("Enhanced individual plots saved successfully.")

def main():
    print("=== Revised CPI Linear Regression Model Analysis ===\n")
    
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
    
    # 3. Prepare data with a simplified approach
    model_df = prepare_model_data(df, target_col, lag=1)
    
    # 4. Split data
    X_train, X_test, y_train, y_test = split_data(model_df, target_col)
    
    # 5. Build and evaluate model
    model, results = build_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    # 6. Granger causality test (only if we can run it)
    if model is not None:
        test_granger_causality(df, target_col)
    
    # 7. Visualize results (pass both model and results)
    if model is not None:
        visualize_results(X_train, X_test, y_train, y_test, results, model)
    else:
        print("No model was successfully built. Skipping visualization.")
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main() 