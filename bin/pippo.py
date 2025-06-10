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
                # 'target': sequence[i+1] if i+1 < len(sequence) else None
            }
            
            if i+1 < len(sequence):
                if sequence[i+1] > sequence[i] + 1e-4:
                    direction = 1
                elif sequence[i+1] < sequence[i] - 1e-4:
                    direction = -1
                else:
                    direction = 0
            else:
                direction = None

            feature_row['target'] = direction

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
        X = pd.DataFrame(X_imputed, columns=feature_cols, index=X.index) # Preserve index
        print("Finished imputing NaN values.") # DEBUG
        
        self.feature_names = feature_cols
        print(f"Created {len(feature_cols)} features from {len(df)} samples (derived from the long sequence)")
        print("Finished preparing training data.") # DEBUG
        
        return X, y
    
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
        MIN_ACCEPTABLE_F1 = 0.0
        valid_models = {
            k: v for k, v in results.items()
            if v['model'] is not None and v['F1'] > MIN_ACCEPTABLE_F1
        }

        if valid_models:
            self.best_model_name = max(
                valid_models.keys(),
                key=lambda x: (valid_models[x]['F1'], valid_models[x]['Accuracy'])
            )
            print(f"\nBest model (F1 > {MIN_ACCEPTABLE_F1}): {self.best_model_name} (F1: {results[self.best_model_name]['F1']:.4f}, Accuracy: {results[self.best_model_name]['Accuracy']:.4f})")
        else:
            print(f"\nNo model found with F1 > {MIN_ACCEPTABLE_F1}.")
            self.best_model_name = None

        self.models = results
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
        
        features_df = self.create_features_from_sequence(sequence)
        
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
    sample_sequences = sample_sequences[:10000]
    
    predictor = GameSequencePredictor(sequence_length=20)
    
    X, y = predictor.prepare_training_data(sample_sequences)
    print(X[0:5])  # Display first 5 rows of features for debugging
    print(y[0:5])  # Display first 5 target values for debugging

    X_selected, selected_features = predictor.select_top_features(X, y, top_k=30)
    predictor.feature_names = selected_features
    
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