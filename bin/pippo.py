import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
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
        
    def create_features_from_sequence(self, sequence):
        """
        Create features from a sequence of game results
        This generates mean, var, next_approximate and many other features
        """
        sequence = np.array(sequence)
        features = []
        
        # Need at least 3 points to create meaningful features
        for i in range(2, len(sequence)):
            current_window = sequence[:i+1]  # All values up to current point
            recent_window = sequence[max(0, i-self.sequence_length):i+1]  # Last 20 values
            
            # Basic statistical features from recent window
            mean_recent = np.mean(recent_window)
            var_recent = np.var(recent_window)
            std_recent = np.std(recent_window)
            median_recent = np.median(recent_window)
            
            # Statistical features from all history
            mean_all = np.mean(current_window)
            var_all = np.var(current_window)
            std_all = np.std(current_window)
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
                
                # Target (next value)
                'target': sequence[i+1] if i+1 < len(sequence) else None
            }
            
            # Add rolling features
            feature_row.update(rolling_features)
            
            # Add lag features
            feature_row.update(lag_features)
              # Add interaction features
            feature_row['mean_var_ratio'] = mean_recent / (var_recent + 1e-8)
            feature_row['mean_times_var'] = mean_recent * var_recent
            feature_row['trend_times_volatility'] = trend_slope * volatility_recent
            feature_row['momentum_times_volatility'] = momentum_1 * volatility_recent
            
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
                        if key == 'target':
                            feature_row[key] = None
                        else:
                            feature_row[key] = 0.0
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def prepare_training_data(self, sequences):
        """
        Prepare training data from multiple sequences
        sequences: list of sequences (each sequence is a list of numbers)
        """
        all_features = []
        
        for seq in sequences:
            if len(seq) >= 4:  # Need at least 4 points to create features and target
                seq_features = self.create_features_from_sequence(seq)
                all_features.append(seq_features)
          # Combine all sequences
        df = pd.concat(all_features, ignore_index=True)
        
        # Remove rows where target is None
        df = df.dropna(subset=['target'])
        
        # Handle NaN and infinite values
        df = df.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
        
        # Check for NaN values before dropping
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"Found NaN values in features:")
            for col, count in nan_counts.items():
                if count > 0:
                    print(f"  {col}: {count} NaN values")
        
        # Drop rows with any NaN values
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        if initial_rows != final_rows:
            print(f"Dropped {initial_rows - final_rows} rows with NaN values ({final_rows} remaining)")
        
        # Ensure we still have data after cleaning
        if len(df) == 0:
            raise ValueError("No valid data remaining after cleaning NaN values")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != 'target']
        X = df[feature_cols]
        y = df['target']

        # Impute any remaining NaN values in features (safety net)
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from scipy.stats import randint, uniform
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
        
        self.feature_names = feature_cols
        print(f"Created {len(feature_cols)} features from {len(df)} samples")
        
        return X, y
    
    def train_models(self, X, y, test_size=0.2):
        """Train multiple models"""
        print(f"Training models on {len(X)} samples...")
        
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
        
        # Define models
        # Hyperparameter tuning configurations
        
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
        
        # Base models for tuning
        base_models = {
            'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Neural Network': MLPRegressor(max_iter=500, random_state=42),
            'Ridge': Ridge()
        }
        
        # Perform hyperparameter tuning
        tuned_models = {}
        print("Starting hyperparameter tuning...")
        
        for name, base_model in base_models.items():
            print(f"Tuning {name}...")
            try:
                # Use RandomizedSearchCV for faster tuning
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
                
                # Fit on appropriate data (scaled for Neural Network)
                if name == 'Neural Network':
                    search.fit(X_train_scaled, y_train)
                else:
                    search.fit(X_train, y_train)
                
                tuned_models[name] = search.best_estimator_
                print(f"{name} best params: {search.best_params_}")
            
            except Exception as e:
                print(f"Error tuning {name}: {str(e)}")
                # Fall back to default model
                tuned_models[name] = base_model
        
        # Create models config with tuned models
        models_config = {}
        for name, tuned_model in tuned_models.items():
            if name == 'Neural Network':
                models_config[name] = {
                    'model': tuned_model,
                    'data': (X_train_scaled, X_test_scaled)
                }
            else:
                models_config[name] = {
                    'model': tuned_model,
                    'data': (X_train, X_test)
                }
        
        results = {}
        
        for name, config in models_config.items():
            print(f"Training {name}...")
            try:
                model = config['model']
                X_tr, X_te = config['data']
                
                # Train model
                model.fit(X_tr, y_train)
                
                # Predict
                y_pred = model.predict(X_te)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'RMSE': np.sqrt(mse),
                    'MAE': mae,
                    'R²': r2,
                    'model': model
                }
                
                print(f"{name} - RMSE: {np.sqrt(mse):.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                results[name] = {'RMSE': float('inf'), 'MAE': float('inf'), 'R²': -float('inf'), 'model': None}
        
        # Find best model
        valid_models = {
            k: v for k, v in results.items() 
            if v['model'] is not None and not np.isclose(v['R²'], 0.0)
        }
        if valid_models:
            self.best_model_name = min(valid_models.keys(), key=lambda x: valid_models[x]['RMSE'])
            print(f"\nBest model: {self.best_model_name}")
        else:
            print("\nNo suitable model found after filtering by R².")
            # Fallback or handle the case where no model meets the criteria
            # For example, pick the best RMSE model from the original set if no R² > 0 model exists
            all_models_with_rmse = {k: v for k, v in results.items() if v['model'] is not None and 'RMSE' in v and np.isfinite(v['RMSE'])}
            if all_models_with_rmse:
                self.best_model_name = min(all_models_with_rmse.keys(), key=lambda x: all_models_with_rmse[x]['RMSE'])
                print(f"Warning: No model with R² > 0. Falling back to best RMSE model: {self.best_model_name} (R²: {results[self.best_model_name]['R²']:.4f})")
            else:
                self.best_model_name = None
                print("Error: No valid models found at all.")
        
        self.models = results
        return results
    
    def predict_next_value(self, sequence):
        """
        Predict the next value in a sequence
        sequence: list of numbers (the game results)
        """
        if len(sequence) < 3:
            raise ValueError("Need at least 3 values to make a prediction")
        
        # Create features for the sequence
        features_df = self.create_features_from_sequence(sequence)
        
        # Get the last row (most recent features)
        X = features_df[self.feature_names].iloc[[-1]]
        
        # Make prediction with best model
        model_name = self.best_model_name
        model = self.models[model_name]['model']
        
        # Apply scaling if needed
        if model_name == 'Neural Network':
            X = self.scalers['robust'].transform(X)
        
        prediction = model.predict(X)
        return prediction[0]
    
    def get_feature_importance(self, model_name=None, top_k=20):
        """Get feature importance"""
        if model_name is None:
            model_name = self.best_model_name
            
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
    temp = []
    with open(file_path, "r") as file:
        lines = file.readlines()[1:]
        for line in lines:
            temp.append(float(line.strip()))
            if len(temp) == 20:  # Assuming each sequence has 20 values
                data.append(temp)
                temp = []
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


# Usage example
def example_usage():
    # Example: Your game sequences
    sample_sequences = load_dataset('../data/multipliers.csv')
    
    # Initialize predictor
    predictor = GameSequencePredictor(sequence_length=20)
    
    # Prepare training data from sequences
    X, y = predictor.prepare_training_data(sample_sequences)
    
    # Train models
    results = predictor.train_models(X, y)
    
    # Display results
    print("\nModel Performance:")
    for name, metrics in results.items():
        if metrics['model'] is not None:
            print(f"{name}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R²={metrics['R²']:.4f}")
    
    # Show feature importance
    importance = predictor.get_feature_importance(top_k=10)
    if importance is not None:
        print(f"\nTop 10 Features ({predictor.best_model_name}):")
        print(importance)
    
    # Make prediction
    test_sequence = [2.8, 6.55, 1.1, 1.06, 1.88, 1.89, 2.36, 8.23, 1.76, 1.68, 1.44, 1.35, 2.56, 1.49, 3.03, 1.82, 1.69, 7.81, 3.8]
    next_prediction = predictor.predict_next_value(test_sequence)
    print(f"\nPredicted next value: {next_prediction:.4f}")
    
    # Save model
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
                # Create features
                features_df = self.predictor.create_features_from_sequence(sequence)
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

if __name__ == "__main__":
    # Run example
    predictor = example_usage()