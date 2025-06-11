import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

class GameSequencePredictor:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.models = {}
        self.scalers = {}
        self.best_model_name = None
        self.feature_names = []
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI without TA-Lib"""
        if len(prices) < period + 1:
            return 50.0  # neutral RSI
        
        prices = np.array(prices)
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands without TA-Lib"""
        if len(prices) < period:
            mean_price = np.mean(prices)
            return mean_price * 1.02, mean_price, mean_price * 0.98, 0.5
        
        prices = np.array(prices)
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        # Calculate position within bands
        current_price = prices[-1]
        if upper == lower:
            position = 0.5
        else:
            position = (current_price - lower) / (upper - lower)
            position = max(0, min(1, position))  # Clamp between 0 and 1
        
        return upper, sma, lower, position
    
    def calculate_ema(self, prices, period=12):
        """Calculate Exponential Moving Average"""
        if len(prices) < 2:
            return np.mean(prices) if prices else 0
        
        prices = np.array(prices)
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def create_enhanced_targets(self, sequence, i):
        """Create multiple enhanced target types"""
        if i+1 >= len(sequence):
            return None, None, None, None
        
        current = sequence[i]
        next_val = sequence[i+1]
        
        # Calculate percentage change
        pct_change = (next_val - current) / (current + 1e-8)
        
        # 1. Magnitude-based direction (5 classes) - More granular than original
        if pct_change > 0.15:  # 15% increase - big jump
            magnitude_class = 4  # much_higher
        elif pct_change > 0.03:  # 3% increase - moderate jump
            magnitude_class = 3  # higher
        elif pct_change < -0.15:  # 15% decrease - big drop
            magnitude_class = 0  # much_lower
        elif pct_change < -0.03:  # 3% decrease - moderate drop
            magnitude_class = 1  # lower
        else:
            magnitude_class = 2  # same/stable
        
        # 2. Volatility class (3 classes) - predict if next change will be volatile
        volatility_class = 2 if abs(pct_change) > 0.2 else (1 if abs(pct_change) > 0.05 else 0)  # high/medium/low volatility
        
        # 3. Threshold class (3 classes) - specific to your domain
        if next_val > 10.0:
            threshold_class = 2  # very_high
        elif next_val > 3.0:
            threshold_class = 1  # high
        else:
            threshold_class = 0  # normal/low
        
        # 4. Trend class (3 classes) - predict trend direction
        if i >= 4:  # Need at least 5 points for trend
            recent_trend = np.polyfit(range(5), sequence[i-4:i+1], 1)[0]
            if recent_trend > 0.1:
                trend_class = 2  # uptrend
            elif recent_trend < -0.1:
                trend_class = 0  # downtrend
            else:
                trend_class = 1  # sideways
        else:
            trend_class = 1  # neutral
        
        return magnitude_class, volatility_class, threshold_class, trend_class
        
    def create_features_from_sequence(self, sequence):
        """
        Create features from a sequence of game results
        This generates mean, var, next_approximate and many other features
        """
        print(f"Starting feature creation for a sequence of length {len(sequence)}") # DEBUG
        sequence = np.array(sequence)
        features = []
        
        # Need at least 3 points to create meaningful features
        for i in range(2, len(sequence)):
            if i % 1000 == 0: # DEBUG: Print progress every 1000 iterations
                print(f"Processing feature creation, iteration {i}/{len(sequence)}")
            current_window = sequence[:i+1]  # All values up to current point
            
            # Corrected: A window of at most self.sequence_length items ending at index i
            start_index_recent = max(0, i - self.sequence_length + 1)
            recent_window = sequence[start_index_recent : i+1]
            
            # Basic statistical features from recent window
            mean_recent = np.mean(recent_window)
            var_recent = np.var(recent_window)
            std_recent = np.std(recent_window)
            median_recent = np.median(recent_window)
            # Statistical features from all history
            mean_all = np.mean(current_window)
            var_all = np.var(current_window)
            std_all = np.std(current_window)
            
            # Advanced statistical features
            rolling_skew = skew(recent_window) if len(recent_window) >= 3 else 0.0
            rolling_kurtosis = kurtosis(recent_window) if len(recent_window) >= 3 else 0.0
            
            # Technical indicators
            rsi = self.calculate_rsi(recent_window, period=min(14, len(recent_window)))
            bb_upper, bb_middle, bb_lower, bb_position = self.calculate_bollinger_bands(recent_window, 
                                                                                       period=min(20, len(recent_window)))
            ema_short = self.calculate_ema(recent_window, period=min(12, len(recent_window)))
            ema_long = self.calculate_ema(recent_window, period=min(26, len(recent_window)))
            
              # Trend analysis
            if len(recent_window) >= 3:
                # Linear trend (slope)
                try:
                    x_trend = np.arange(len(recent_window))
                    trend_slope = np.polyfit(x_trend, recent_window, 1)[0]
                    if not np.isfinite(trend_slope):
                        trend_slope = 0
                except (np.linalg.LinAlgError, ValueError):
                    trend_slope = 0
            else:
                trend_slope = 0
                
            # Momentum features
            if i >= 1:
                momentum_1 = sequence[i] - sequence[i-1]
                momentum_ratio_1 = sequence[i] / (sequence[i-1] + 1e-8)
            else:
                momentum_1 = 0
                momentum_ratio_1 = 1
                
            if i >= 2:
                momentum_2 = sequence[i] - sequence[i-2]
                momentum_avg_2 = (sequence[i] + sequence[i-1]) / 2 - sequence[i-2]
            else:
                momentum_2 = 0
                momentum_avg_2 = 0
            
            # Pattern features
            min_recent = np.min(recent_window)
            max_recent = np.max(recent_window)
            range_recent = max_recent - min_recent
            
            # Position features
            current_vs_mean = sequence[i] / (mean_recent + 1e-8)
            current_vs_median = sequence[i] / (median_recent + 1e-8)
            current_vs_max = sequence[i] / (max_recent + 1e-8)
            current_vs_min = sequence[i] / (min_recent + 1e-8)
            
            # Volatility features
            volatility_recent = std_recent / (mean_recent + 1e-8)  # Coefficient of variation
            
            # Percentile features
            percentile_25 = np.percentile(recent_window, 25)
            percentile_75 = np.percentile(recent_window, 75)
            iqr = percentile_75 - percentile_25
              # Frequency features (how often we see similar values)
            current_val = sequence[i]
            similar_count = np.sum(np.abs(recent_window - current_val) < 0.1)
            
            # Consecutive patterns
            consecutive_up = 0
            consecutive_down = 0
            for j in range(min(5, len(recent_window)-1)):
                if recent_window[-(j+1)] > recent_window[-(j+2)]:
                    consecutive_up += 1
                else:
                    break
            for j in range(min(5, len(recent_window)-1)):
                if recent_window[-(j+1)] < recent_window[-(j+2)]:
                    consecutive_down += 1
                else:
                    break
            
            # Local extrema detection
            is_local_max = (i >= 2 and sequence[i] > sequence[i-1] and sequence[i-1] > sequence[i-2])
            is_local_min = (i >= 2 and sequence[i] < sequence[i-1] and sequence[i-1] < sequence[i-2])
            
            # Time since last extreme
            time_since_max = 1
            time_since_min = 1
            max_val = sequence[i]
            min_val = sequence[i]
            for j in range(1, min(20, i+1)):
                if sequence[i-j] > max_val:
                    max_val = sequence[i-j]
                    time_since_max = j
                if sequence[i-j] < min_val:
                    min_val = sequence[i-j]
                    time_since_min = j
            
            # Rolling statistics with different windows
            rolling_features = {}
            for window in [3, 5, 10]:
                if len(recent_window) >= window:
                    roll_window = recent_window[-window:]
                    rolling_features[f'mean_{window}'] = np.mean(roll_window)
                    rolling_features[f'std_{window}'] = np.std(roll_window)
                    rolling_features[f'min_{window}'] = np.min(roll_window)
                    rolling_features[f'max_{window}'] = np.max(roll_window)
                else:
                    rolling_features[f'mean_{window}'] = mean_recent
                    rolling_features[f'std_{window}'] = std_recent
                    rolling_features[f'min_{window}'] = min_recent
                    rolling_features[f'max_{window}'] = max_recent
            
            # Lag features (previous values)
            lag_features = {}
            for lag in range(1, min(6, len(current_window))):
                if i >= lag:
                    lag_features[f'lag_{lag}'] = sequence[i-lag]
                    lag_features[f'lag_diff_{lag}'] = sequence[i] - sequence[i-lag]
                    lag_features[f'lag_ratio_{lag}'] = sequence[i] / (sequence[i-lag] + 1e-8)
                else:
                    lag_features[f'lag_{lag}'] = sequence[i]
                    lag_features[f'lag_diff_{lag}'] = 0
                    lag_features[f'lag_ratio_{lag}'] = 1
            
            # Next approximate (simple prediction based on recent trend)
            if len(recent_window) >= 3:
                # Linear extrapolation
                x_vals = np.arange(len(recent_window))
                coeffs = np.polyfit(x_vals, recent_window, 1)
                next_approximate = coeffs[0] * len(recent_window) + coeffs[1]
                
                # Exponential smoothing
                alpha = 0.3
                exp_smooth = recent_window[-1]
                for j in range(len(recent_window)-2, -1, -1):
                    exp_smooth = alpha * recent_window[j] + (1-alpha) * exp_smooth
                next_exp_smooth = alpha * recent_window[-1] + (1-alpha) * exp_smooth
            else:
                next_approximate = mean_recent
                next_exp_smooth = mean_recent
            
            # Get enhanced targets
            magnitude_target, volatility_target, threshold_target, trend_target = self.create_enhanced_targets(sequence, i)
            
            # Original simple target
            if i+1 < len(sequence):
                if sequence[i+1] > sequence[i] + 1e-4:
                    direction = 1
                elif sequence[i+1] < sequence[i] - 1e-4:
                    direction = -1
                else:
                    direction = 0
            else:
                direction = None
            
            # Combine all features
            feature_row = {
                # Basic statistics
                'mean_recent': mean_recent,
                'var_recent': var_recent,
                'std_recent': std_recent,
                'median_recent': median_recent,
                'mean_all': mean_all,
                'var_all': var_all,
                'std_all': std_all,
                
                # Advanced statistics
                'rolling_skew': rolling_skew,
                'rolling_kurtosis': rolling_kurtosis,
                
                # Technical indicators
                'rsi': rsi,
                'bb_position': bb_position,
                'ema_short': ema_short,
                'ema_long': ema_long,
                'ema_diff': ema_short - ema_long,
                
                # Trend and momentum
                'trend_slope': trend_slope,
                'momentum_1': momentum_1,
                'momentum_2': momentum_2,
                'momentum_ratio_1': momentum_ratio_1,
                'momentum_avg_2': momentum_avg_2,
                
                # Pattern features
                'min_recent': min_recent,
                'max_recent': max_recent,
                'range_recent': range_recent,
                'percentile_25': percentile_25,
                'percentile_75': percentile_75,
                'iqr': iqr,
                'consecutive_up': consecutive_up,
                'consecutive_down': consecutive_down,
                'is_local_max': int(is_local_max),
                'is_local_min': int(is_local_min),
                'time_since_max': time_since_max,
                'time_since_min': time_since_min,
                
                # Position features
                'current_vs_mean': current_vs_mean,
                'current_vs_median': current_vs_median,
                'current_vs_max': current_vs_max,
                'current_vs_min': current_vs_min,
                
                # Volatility
                'volatility_recent': volatility_recent,
                'similar_count': similar_count,
                
                # Next approximate
                'next_approximate': next_approximate,
                'next_exp_smooth': next_exp_smooth,
                
                # Enhanced targets
                'target_magnitude': magnitude_target,
                'target_volatility': volatility_target,
                'target_threshold': threshold_target,
                'target_trend': trend_target,
                
                # Original target
                'target': direction
            }

            feature_row['target'] = direction

            # Add rolling features
            feature_row.update(rolling_features)
            
            # Add lag features
            feature_row.update(lag_features)              # Add interaction features
            feature_row['mean_var_ratio'] = mean_recent / (var_recent + 1e-8)
            feature_row['mean_times_var'] = mean_recent * var_recent
            feature_row['trend_times_volatility'] = trend_slope * volatility_recent
            feature_row['momentum_times_volatility'] = momentum_1 * volatility_recent
            feature_row['rsi_bb_interaction'] = rsi * bb_position
            feature_row['ema_momentum_interaction'] = (ema_short - ema_long) * momentum_1
            
            # Validate all features are finite
            for key, value in feature_row.items():
                if value is not None:
                    problematic_value = False
                    try:
                        if not np.isfinite(value):  # Check for np.inf, np.nan
                            problematic_value = True
                    except TypeError:  # Handle non-numeric types that np.isfinite can't process
                        problematic_value = True
                    
                    if problematic_value:
                        if key.startswith('target'):
                            feature_row[key] = None
                        else:
                            feature_row[key] = 0.0
            
            features.append(feature_row)
        
        print(f"Finished feature creation. Generated {len(features)} feature rows.") # DEBUG
        return pd.DataFrame(features)
    
    def select_top_features(self, X, y, top_k=30, random_state=42):
        """
        Select the top_k most important features using a Random Forest.
        Returns a reduced X DataFrame with only the selected features.
        """
        print(f"Selecting top {top_k} features using Random Forest importance...")
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        rf.fit(X, y)
        importances = rf.feature_importances_
        feature_importance = pd.Series(importances, index=X.columns)
        top_features = feature_importance.sort_values(ascending=False).head(top_k).index.tolist()
        print("Top features selected:", top_features)
        return X[top_features], top_features
    
    def log_transform(self, sequence):
        """Apply log1p transform to a sequence (handles zeros safely)"""
        return [np.log1p(x) for x in sequence]

    def prepare_training_data(self, full_historical_sequence): # Renamed arg
        """
        Prepare training data from a single long historical sequence
        full_historical_sequence: list of numbers (the entire history)
        """
        print("Starting to prepare training data...") # DEBUG
        transformed_sequence = self.log_transform(full_historical_sequence)
        if len(full_historical_sequence) < 4:
            print("Warning: Full historical sequence is too short to generate any training samples.")
            # This will likely lead to an empty DataFrame later, which is handled.
            # Or, raise an error immediately:
            # raise ValueError("Full historical sequence is too short (must be >= 4).")
            # For now, let downstream checks handle empty df.
            df = pd.DataFrame([]) # Start with an empty df if too short
        else:
            # Create features from the entire historical sequence
            print("Calling create_features_from_sequence...") # DEBUG
            df = self.create_features_from_sequence(transformed_sequence)
            print(f"Finished create_features_from_sequence. DataFrame shape: {df.shape}") # DEBUG
        
        # Remove rows where target is None
        print("Dropping rows with None target...") # DEBUG
        df = df.dropna(subset=['target'])
        print(f"DataFrame shape after dropping None target: {df.shape}") # DEBUG
        
        # Handle NaN and infinite values
        print("Replacing infinite values with NaN...") # DEBUG
        df = df.replace([np.inf, -np.inf], np.nan)
        print(f"DataFrame shape after replacing infinite values: {df.shape}") # DEBUG
        
        # Check for NaN values before dropping (optional, can be verbose)
        # nan_counts = df.isnull().sum()
        # if nan_counts.sum() > 0:
        #     print(f"Found NaN values in features/target before final dropna:")
        #     for col, count in nan_counts.items():
        #         if count > 0:
        #             print(f"  {col}: {count} NaN values")

        initial_rows = len(df)
        print("Dropping rows with any NaN values...") # DEBUG
        df = df.dropna() # Drop rows with any NaN values (in features or target)
        final_rows = len(df)
        print(f"DataFrame shape after final dropna: {df.shape}") # DEBUG
        
        if initial_rows != final_rows:
            print(f"Dropped {initial_rows - final_rows} rows due to NaN values after feature creation ({final_rows} remaining)")
        
        if len(df) == 0:
            raise ValueError("No valid data remaining after cleaning NaN values. Input sequence might be too short or all generated features/targets resulted in NaNs.")
        
        feature_cols = [col for col in df.columns if col != 'target']
        if not feature_cols:
             raise ValueError("No feature columns found. 'target' column might be missing or DataFrame is structured unexpectedly.")

        X = df[feature_cols]
        y = df['target']

        # Remap target values for classification: -1 -> 0, 0 -> 1, 1 -> 2
        y = y.map({-1: 0, 0: 1, 1: 2})
        print("Remapped target values for classification: -1 -> 0, 0 -> 1, 1 -> 2") # DEBUG

        # Impute any remaining NaN values in features (safety net)
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        print("Imputing remaining NaN values in features...") # DEBUG
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=feature_cols, index=X.index) # Preserve index        print("Finished imputing NaN values.") # DEBUG
        
        self.feature_names = feature_cols
        print(f"Created {len(feature_cols)} features from {len(df)} samples (derived from the long sequence)")
        print("Finished preparing training data.") # DEBUG
        
        return X, y
    
    def prepare_multi_target_training_data(self, full_historical_sequence):
        """
        Prepare training data with multiple target types
        """
        print("Starting to prepare multi-target training data...") # DEBUG
        transformed_sequence = self.log_transform(full_historical_sequence)
        
        if len(full_historical_sequence) < 4:
            print("Warning: Full historical sequence is too short to generate any training samples.")
            return None, None, None, None, None, None
        
        # Create features from the entire historical sequence
        df = self.create_features_from_sequence(transformed_sequence)
        
        # Remove rows where any target is None
        target_cols = ['target', 'target_magnitude', 'target_volatility', 'target_threshold', 'target_trend']
        initial_rows = len(df)
        df = df.dropna(subset=target_cols)
        print(f"Dropped {initial_rows - len(df)} rows with None targets")
        
        # Handle NaN and infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        if len(df) == 0:
            raise ValueError("No valid data remaining after cleaning.")
        
        # Separate features and targets
        feature_cols = [col for col in df.columns if not col.startswith('target')]
        X = df[feature_cols]
        
        # Original target (remapped)
        y_original = df['target'].map({-1: 0, 0: 1, 1: 2})
        
        # Enhanced targets
        y_magnitude = df['target_magnitude']
        y_volatility = df['target_volatility'] 
        y_threshold = df['target_threshold']
        y_trend = df['target_trend']
        
        # Impute any remaining NaN values in features
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=feature_cols, index=X.index)
        
        self.feature_names = feature_cols
        print(f"Created {len(feature_cols)} features from {len(df)} samples with enhanced targets")
        
        return X, y_original, y_magnitude, y_volatility, y_threshold, y_trend
    
    def train_models(self, X, y, test_size=0.2, tune_hyperparameters=False):
        """Train multiple models, with an option for hyperparameter tuning"""
        print(f"Training models on {len(X)} samples...")
        if tune_hyperparameters:
            print("Hyperparameter tuning enabled.")
        else:
            print("Using base models without hyperparameter tuning.")

        # Use TimeSeriesSplit for better validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False  # Don't shuffle time series
        )
        
        # Initialize scalers
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        
        # Fit scalers
        X_train_scaled = self.scalers['robust'].fit_transform(X_train)
        X_test_scaled = self.scalers['robust'].transform(X_test)
        
        # Define base models
        base_models_for_tuning = {
            'XGBoost': xgb.XGBClassifier(random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Neural Network': MLPClassifier(max_iter=500, random_state=42),
            'Logistic Regression': LogisticRegression()
        }

        trained_models = {}

        if tune_hyperparameters:
            # Define hyperparameter grids for tuning
            param_grids = {
                'XGBoost': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                },
                'LightGBM': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                },
                'Random Forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [8, 10, 12, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Gradient Boosting': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'subsample': [0.7, 0.8, 0.9]
                },
                'Neural Network': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 75)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate_init': [0.001, 0.01, 0.1]
                },
                'Ridge': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            }
            
            print("Starting hyperparameter tuning...")
            for name, base_model in base_models_for_tuning.items():
                print(f"Tuning {name}...")
                try:
                    search = RandomizedSearchCV(
                        base_model,
                        param_grids[name],
                        n_iter=20,  # Number of parameter settings sampled
                        cv=tscv,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        random_state=42,
                        verbose=0 
                    )
                    
                    if name == 'Neural Network':
                        search.fit(X_train_scaled, y_train)
                    else:
                        search.fit(X_train, y_train)
                    
                    trained_models[name] = search.best_estimator_
                    print(f"{name} best params: {search.best_params_}")
                
                except Exception as e:
                    print(f"Error tuning {name}: {str(e)}")
                    trained_models[name] = base_model
        else:
            trained_models = base_models_for_tuning
        
        models_config = {}
        for name, model_instance in trained_models.items():
            if name == 'Neural Network':
                models_config[name] = {
                    'model': model_instance,
                    'data': (X_train_scaled, X_test_scaled)
                }
            else:
                models_config[name] = {
                    'model': model_instance,
                    'data': (X_train, X_test)
                }
        
        results = {}
        
        for name, config in models_config.items():
            print(f"Training {name}...")
            try:
                model = config['model']
                X_tr, X_te = config['data']
                
                model.fit(X_tr, y_train)
                
                y_pred = model.predict(X_te)
                
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                
                results[name] = {
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1,
                    'ConfusionMatrix': cm,
                    'model': model
                }
                
                print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                results[name] = {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0, 'ConfusionMatrix': None, 'model': None}
        
        self.models = results # Ensure self.models is updated with all training results

        MIN_ACCEPTABLE_F1 = 0.0
        valid_models_from_instance = { # Check against the updated self.models
            k: v for k, v in self.models.items()
            if v['model'] is not None and v['F1'] > MIN_ACCEPTABLE_F1
        }

        if valid_models_from_instance:
            self.best_model_name = max(
                valid_models_from_instance.keys(),
                key=lambda x: (self.models[x]['F1'], self.models[x]['Accuracy']) # Access self.models
            )
            print(f"\nBest model (F1 > {MIN_ACCEPTABLE_F1}): {self.best_model_name} (F1: {self.models[self.best_model_name]['F1']:.4f}, Accuracy: {self.models[self.best_model_name]['Accuracy']:.4f})")
        else:
            print(f"\nNo model found with F1 > {MIN_ACCEPTABLE_F1}.")
            self.best_model_name = None
            # self.models = results # This line is now handled above
        return results
    
    def predict_next_value(self, sequence):
        """
        Predict the next value in a sequence
        sequence: list of numbers (the game results)
        """
        if self.best_model_name is None:
            raise ValueError("No best model was selected during training. Cannot make predictions.")

        if len(sequence) < 3:
            raise ValueError("Need at least 3 values to make a prediction")
        
        # Apply the same log transformation used during training
        transformed_sequence = self.log_transform(sequence)
        features_df = self.create_features_from_sequence(transformed_sequence)
        
        X = features_df[self.feature_names].iloc[[-1]]
        
        model_name = self.best_model_name
        model = self.models[model_name]['model']
        
        if model_name == 'Neural Network':
            X = self.scalers['robust'].transform(X)
        
        prediction = model.predict(X)
        return prediction[0]
    
    def get_feature_importance(self, model_name=None, top_k=20):
        """Get feature importance"""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name is None:
            print("No best model available to get feature importance.")
            return None
            
        if model_name not in self.models or self.models[model_name]['model'] is None:
            return None
            
        model = self.models[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_k)
            return feature_imp
        
        return None
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'best_model_name': self.best_model_name,
            'sequence_length': self.sequence_length
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_names = model_data['feature_names']
        self.best_model_name = model_data['best_model_name']
        self.sequence_length = model_data['sequence_length']
        print(f"Model loaded from {filepath}")


def make_dataset(file_path):
    data = []
    with open(file_path, "r") as file:
        lines = file.readlines()[1:]
        for line_number, line in enumerate(lines, 1):
            try:
                data.append(float(line.strip()))
            except ValueError:
                print(f"Warning: Could not convert line {line_number+1} to float: '{line.strip()}'")
    return data


def load_dataset(file_path):
    """Load dataset from a file"""
    try:
        data = make_dataset(file_path)
        print(f"Loaded {len(data)} sequences from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return []


def example_usage():
    sample_sequences = load_dataset('../data/multipliers.csv')
    sample_sequences = sample_sequences[:5000]
    
    predictor = GameSequencePredictor(sequence_length=20)
    
    X, y = predictor.prepare_training_data(sample_sequences)
    print(X[0:5])  # Display first 5 rows of features for debugging
    print(y[0:5])  # Display first 5 target values for debugging

    # X_selected, selected_features = predictor.select_top_features(X, y, top_k=30)
    # predictor.feature_names = selected_features
    
    results = predictor.train_models(X, y, tune_hyperparameters=False) # Set to True to run tuning
    
    print("\nModel Performance:")
    for name, metrics in results.items():
        if metrics['model'] is not None:
            print(f"{name}: Accuracy={metrics['Accuracy']:.4f}, Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}, F1={metrics['F1']:.4f}")

    importance = predictor.get_feature_importance(top_k=10)
    if importance is not None:
        print(f"\nTop 10 Features ({predictor.best_model_name}):")
        print(importance)
    
    test_sequence = [3.0, 3.89, 1.65, 4.45, 1.13, 3.8, 4.42, 5.2, 2.88, 1.96, 1.55, 13.14, 3.08, 1.13, 15.91, 10.46, 1.68, 1.58, 1.44, 2.11]
    next_prediction = predictor.predict_next_value(test_sequence)
    direction_map = {0: 'lower', 1: 'same', 2: 'higher'}
    predicted_direction = direction_map.get(next_prediction, 'unknown')
    print(f"\nPredicted next value: {predicted_direction}")
    
    predictor.save_model('game_sequence_model.pkl')
    
    return predictor

# Production inference
class FastGamePredictor:
    """Fast predictor for production use"""
    
    def __init__(self, model_path):
        self.predictor = GameSequencePredictor()
        self.predictor.load_model(model_path)
    
    def predict(self, sequence):
        """Predict next value from sequence"""
        return self.predictor.predict_next_value(sequence)
    
    def predict_with_confidence(self, sequence, n_models=3):
        """Get prediction with confidence using multiple models"""
        predictions = []
        
        # Get predictions from top N models        
        sorted_models = sorted(
            [(name, info) for name, info in self.predictor.models.items() if info['model'] is not None],
            key=lambda x: x[1]['RMSE']
        )
        
        for name, info in sorted_models[:n_models]:
            try:
                # Apply the same log transformation used during training
                transformed_sequence = self.predictor.log_transform(sequence)
                # Create features
                features_df = self.predictor.create_features_from_sequence(transformed_sequence)
                X = features_df[self.predictor.feature_names].iloc[[-1]]
                
                # Apply scaling if needed
                if name == 'Neural Network':
                    X = self.predictor.scalers['robust'].transform(X)
                
                pred = info['model'].predict(X)[0]
                predictions.append(pred)
            except:
                continue
        
        if predictions:
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            return mean_pred, std_pred
        else:
            return self.predict(sequence), 0.0


class MultiTargetGamePredictor:
    """Enhanced predictor with multiple target types"""
    
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.models = {
            'original': {},
            'magnitude': {},
            'volatility': {},
            'threshold': {},
            'trend': {}
        }
        self.best_models = {}
        self.scalers = {}
        self.feature_names = []
        self.base_predictor = GameSequencePredictor(sequence_length)
    
    def prepare_data(self, full_historical_sequence):
        """Prepare data for multi-target training"""
        return self.base_predictor.prepare_multi_target_training_data(full_historical_sequence)
    
    def train_multi_target_models(self, X, y_original, y_magnitude, y_volatility, y_threshold, y_trend, tune_hyperparameters=False):
        """Train separate models for each target type"""
        targets = {
            'original': y_original,
            'magnitude': y_magnitude,
            'volatility': y_volatility,
            'threshold': y_threshold,
            'trend': y_trend
        }
        
        self.feature_names = X.columns.tolist()
        
        for target_name, y in targets.items():
            print(f"Training models for {target_name} prediction...")
            
            # Create a new predictor for this target
            predictor = GameSequencePredictor(self.sequence_length)
            predictor.feature_names = self.feature_names
            
            # Train models
            results = predictor.train_models(X, y, tune_hyperparameters=tune_hyperparameters)
            
            self.models[target_name] = results
            self.best_models[target_name] = predictor.best_model_name
            if target_name == 'original':  # Copy scalers from the first trained model
                self.scalers = predictor.scalers
    
    def predict_multi_target(self, sequence):
        """Get predictions for all target types"""
        predictions = {}
        
        # Apply the same log transformation used during training
        transformed_sequence = self.base_predictor.log_transform(sequence)
        # Create features from sequence
        features_df = self.base_predictor.create_features_from_sequence(transformed_sequence)
        X = features_df[self.feature_names].iloc[[-1]]
        
        for target_name in self.models.keys():
            if self.best_models.get(target_name) and self.models[target_name].get(self.best_models[target_name]):
                best_model_name = self.best_models[target_name]
                model = self.models[target_name][best_model_name]['model']
                
                if model is not None:
                    try:
                        # Apply scaling if needed
                        if best_model_name == 'Neural Network':
                            X_scaled = self.scalers['robust'].transform(X)
                            pred = model.predict(X_scaled)[0]
                        else:
                            pred = model.predict(X)[0]
                        
                        predictions[target_name] = pred
                    except Exception as e:
                        print(f"Error predicting {target_name}: {str(e)}")
                        predictions[target_name] = None
                else:
                    predictions[target_name] = None
            else:
                predictions[target_name] = None
        
        return predictions
    
    def get_prediction_summary(self, sequence):
        """Get a comprehensive prediction summary"""
        predictions = self.predict_multi_target(sequence)
        
        # Map predictions to readable labels
        direction_map = {0: 'lower', 1: 'same', 2: 'higher'}
        magnitude_map = {0: 'much_lower', 1: 'lower', 2: 'same', 3: 'higher', 4: 'much_higher'}
        volatility_map = {0: 'low', 1: 'medium', 2: 'high'}
        threshold_map = {0: 'normal', 1: 'high', 2: 'very_high'}
        trend_map = {0: 'downtrend', 1: 'sideways', 2: 'uptrend'}
        
        summary = {
            'direction': direction_map.get(predictions.get('original'), 'unknown'),
            'magnitude': magnitude_map.get(predictions.get('magnitude'), 'unknown'),
            'volatility': volatility_map.get(predictions.get('volatility'), 'unknown'),
            'threshold': threshold_map.get(predictions.get('threshold'), 'unknown'),
            'trend': trend_map.get(predictions.get('trend'), 'unknown'),
            'raw_predictions': predictions
        }
        
        return summary
    
    def save_model(self, filepath):
        """Save the multi-target model"""
        model_data = {
            'models': self.models,
            'best_models': self.best_models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length
        }
        joblib.dump(model_data, filepath)
        print(f"Multi-target model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained multi-target model"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.best_models = model_data['best_models']
        self.scalers = model_data['scalers']
        self.feature_names = model_data['feature_names']
        self.sequence_length = model_data['sequence_length']
        print(f"Multi-target model loaded from {filepath}")


def example_usage_multi_target():
    """Example usage of multi-target predictor"""
    sample_sequences = load_dataset('../data/multipliers.csv')
    sample_sequences = sample_sequences[:5000]  # Use smaller dataset for testing
    
    predictor = MultiTargetGamePredictor(sequence_length=20)
    
    # Prepare multi-target data
    data = predictor.prepare_data(sample_sequences)
    
    if data[0] is not None:
        X, y_original, y_magnitude, y_volatility, y_threshold, y_trend = data
        print(f"Training data shape: {X.shape}")
        print(f"Target distributions:")
        print(f"  Original: {np.bincount(y_original)}")
        print(f"  Magnitude: {np.bincount(y_magnitude)}")
        print(f"  Volatility: {np.bincount(y_volatility)}")
        print(f"  Threshold: {np.bincount(y_threshold)}")
        print(f"  Trend: {np.bincount(y_trend)}")
        
        # Train multi-target models
        predictor.train_multi_target_models(X, y_original, y_magnitude, y_volatility, y_threshold, y_trend, tune_hyperparameters=False)
        
        # Test prediction
        test_sequence = [3.0, 3.89, 1.65, 4.45, 1.13, 3.8, 4.42, 5.2, 2.88, 1.96, 1.55, 13.14, 3.08, 1.13, 15.91, 10.46, 1.68, 1.58, 1.44, 2.11]
        summary = predictor.get_prediction_summary(test_sequence)
        
        print(f"\nMulti-target prediction summary:")
        print(f"  Direction: {summary['direction']}")
        print(f"  Magnitude: {summary['magnitude']}")
        print(f"  Volatility: {summary['volatility']}")
        print(f"  Threshold: {summary['threshold']}")
        print(f"  Trend: {summary['trend']}")
        
        # Save the model
        predictor.save_model('multi_target_game_model.pkl')
        
        return predictor
    else:
        print("No valid training data generated.")
        return None

def main():
    fastpredictor = FastGamePredictor('multi_target_game_model.pkl')
    test_sequence = [3.0, 3.89, 1.65, 4.45, 1.13, 3.8, 4.42, 5.2, 2.88, 1.96, 1.55, 13.14, 3.08, 1.13, 15.91, 10.46, 1.68, 1.58, 1.44, 2.11]
    prediction = fastpredictor.predict(test_sequence)
    print(f"Fast prediction for sequence {test_sequence} is: {prediction}")

if __name__ == "__main__":
    main()
    # # Run original example
    # print("=== Running Original Example ===")
    # predictor = example_usage()
    # 
    # print("\n=== Running Multi-Target Example ===")
    # # Run multi-target example
    # multi_predictor = example_usage_multi_target()