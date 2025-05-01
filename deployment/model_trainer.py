import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import optuna
import joblib
import os
from datetime import datetime, timedelta
import warnings
import json

# Suppress warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Advanced model training with hyperparameter tuning and model management.
    Combines the best approaches from multiple solutions.
    """

    def __init__(self, models_dir='models', random_seed=42):
        """
        Initialize the ModelTrainer.

        Args:
            models_dir (str): Directory to save trained models
            random_seed (int): Random seed for reproducibility
        """
        self.models_dir = models_dir
        self.random_seed = random_seed

        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

    def prepare_training_data(self, df, target_col='price', test_size=0.2, validation=False):
        """
        Prepare data for training by splitting into train and test sets.
        Uses time-based split to avoid data leakage.

        Args:
            df (pandas.DataFrame): Input dataframe with features
            target_col (str): Target column for prediction
            test_size (float): Proportion of data to use for testing
            validation (bool): Whether to create a validation set

        Returns:
            tuple: X_train, X_test, y_train, y_test, (X_val, y_val) if validation=True
        """
        if df.empty:
            if validation:
                return None, None, None, None, None, None
            return None, None, None, None

        # Drop rows with NaN values
        df_clean = df.dropna()

        if df_clean.empty:
            if validation:
                return None, None, None, None, None, None
            return None, None, None, None

        # Sort by date
        df_clean = df_clean.sort_values('date')

        # Get split points
        if validation:
            train_end = int(len(df_clean) * (1 - 2 * test_size))
            val_end = int(len(df_clean) * (1 - test_size))

            # Split data
            train_df = df_clean.iloc[:train_end]
            val_df = df_clean.iloc[train_end:val_end]
            test_df = df_clean.iloc[val_end:]
        else:
            split_idx = int(len(df_clean) * (1 - test_size))

            # Split data
            train_df = df_clean.iloc[:split_idx]
            test_df = df_clean.iloc[split_idx:]

        # Drop non-feature columns
        feature_cols = [col for col in df_clean.columns
                       if col not in [target_col, 'date', 'region', 'commodity', 'type']]

        # Create train and test sets
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        if validation:
            X_val = val_df[feature_cols]
            y_val = val_df[target_col]
            return X_train, X_test, y_train, y_test, X_val, y_val

        return X_train, X_test, y_train, y_test

    def xgboost_objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Objective function for XGBoost hyperparameter tuning with Optuna.

        Args:
            trial: Optuna trial object
            X_train, y_train, X_val, y_val: Training and validation data

        Returns:
            float: RMSE on the validation set
        """
        param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': self.random_seed
        }

        # Add dart-specific parameters if dart booster is selected
        if param['booster'] == 'dart':
            param['rate_drop'] = trial.suggest_float('rate_drop', 0.0, 0.5)
            param['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 0.5)

        # Create dataset for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train model
        model = xgb.train(
            param,
            dtrain,
            num_boost_round=10000,
            evals=[(dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=False
        )

        # Make predictions
        preds = model.predict(dval)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_val, preds))

        return rmse

    def lightgbm_objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Objective function for LightGBM hyperparameter tuning with Optuna.

        Args:
            trial: Optuna trial object
            X_train, y_train, X_val, y_val: Training and validation data

        Returns:
            float: RMSE on the validation set
        """
        param = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'random_state': self.random_seed
        }

        # Add boosting type specific parameters
        if param['boosting_type'] == 'dart':
            param['drop_rate'] = trial.suggest_float('drop_rate', 0.0, 0.5)
            param['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 0.5)
        elif param['boosting_type'] == 'goss':
            # GOSS doesn't use bagging
            param.pop('bagging_fraction')
            param.pop('bagging_freq')
            param['top_rate'] = trial.suggest_float('top_rate', 0.0, 0.5)
            param['other_rate'] = trial.suggest_float('other_rate', 0.0, 0.5)

        # Create dataset for LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)

        # Train model
        model = lgb.train(
            param,
            train_data,
            num_boost_round=10000,
            valid_sets=[lgb.Dataset(X_val, label=y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )

        # Make predictions
        preds = model.predict(X_val)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_val, preds))

        return rmse

    def train_model_with_optuna(self, X_train, y_train, X_test, y_test, model_type='xgboost', n_trials=50):
        """
        Train a model with hyperparameter tuning using Optuna.

        Args:
            X_train, y_train, X_test, y_test: Training and testing data
            model_type (str): Type of model to train ('xgboost' or 'lightgbm')
            n_trials (int): Number of Optuna trials

        Returns:
            tuple: Trained model, best parameters, and evaluation metrics
        """
        if X_train is None or y_train is None or X_test is None or y_test is None:
            return None, None, None

        # Split training data into train and validation sets for hyperparameter tuning
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_seed
        )

        # Create Optuna study
        study = optuna.create_study(direction='minimize')

        # Run optimization
        if model_type == 'xgboost':
            study.optimize(
                lambda trial: self.xgboost_objective(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt),
                n_trials=n_trials
            )

            # Get best parameters
            best_params = study.best_params
            best_params['objective'] = 'reg:squarederror'
            best_params['eval_metric'] = 'rmse'
            best_params['random_state'] = self.random_seed

            # Train final model with best parameters on full training data
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            final_model = xgb.train(
                best_params,
                dtrain,
                num_boost_round=10000,
                evals=[(dtest, 'test')],
                early_stopping_rounds=100,
                verbose_eval=False
            )

        elif model_type == 'lightgbm':
            study.optimize(
                lambda trial: self.lightgbm_objective(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt),
                n_trials=n_trials
            )

            # Get best parameters
            best_params = study.best_params
            best_params['objective'] = 'regression'
            best_params['metric'] = 'rmse'
            best_params['random_state'] = self.random_seed

            # Train final model with best parameters on full training data
            train_data = lgb.Dataset(X_train, label=y_train)

            final_model = lgb.train(
                best_params,
                train_data,
                num_boost_round=10000,
                valid_sets=[lgb.Dataset(X_test, label=y_test)],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
            )

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Evaluate final model
        if model_type == 'xgboost':
            preds = final_model.predict(dtest)
        else:  # lightgbm
            preds = final_model.predict(X_test)

        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Calculate normalized RMSE
        y_range = y_test.max() - y_test.min()
        nrmse = rmse / y_range if y_range > 0 else 0

        # Create metrics dictionary
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'nrmse': nrmse
        }

        # Print best parameters and score
        print(f"Best {model_type} parameters: {best_params}")
        print(f"Best {model_type} RMSE: {study.best_value:.4f}")
        print(f"Final {model_type} RMSE: {rmse:.4f}, NRMSE: {nrmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        return final_model, best_params, metrics

    def train_model_for_commodity(self, df, commodity, region=None, model_type='xgboost', n_trials=50):
        """
        Train a model for a specific commodity (and optionally region).

        Args:
            df (pandas.DataFrame): Input dataframe with features
            commodity (str): Commodity to train model for
            region (str, optional): Region to train model for
            model_type (str): Type of model to train ('xgboost' or 'lightgbm')
            n_trials (int): Number of Optuna trials

        Returns:
            tuple: Trained model, best parameters, feature columns, and metrics
        """
        # Filter data for the commodity
        commodity_df = df[df['commodity'] == commodity].copy()

        if region:
            commodity_df = commodity_df[commodity_df['region'] == region].copy()

        if commodity_df.empty:
            print(f"No data available for {commodity}")
            return None, None, None, None

        print(f"Training model for {commodity} with {len(commodity_df)} records")

        # Prepare training data
        X_train, X_test, y_train, y_test = self.prepare_training_data(commodity_df)

        if X_train is None or X_train.empty:
            print(f"Insufficient data for {commodity} after preprocessing")
            return None, None, None, None

        # Train model with hyperparameter tuning
        model, best_params, metrics = self.train_model_with_optuna(
            X_train, y_train, X_test, y_test, model_type, n_trials
        )

        if model is None:
            print(f"Failed to train model for {commodity}")
            return None, None, None, None

        # Get feature columns
        feature_cols = X_train.columns.tolist()

        return model, best_params, feature_cols, metrics

    def save_model(self, model, commodity, region=None, model_type='xgboost', params=None, feature_cols=None, metrics=None):
        """
        Save a trained model to disk.

        Args:
            model: Trained model
            commodity (str): Commodity the model was trained for
            region (str, optional): Region the model was trained for
            model_type (str): Type of model ('xgboost' or 'lightgbm')
            params (dict): Model parameters
            feature_cols (list): Feature columns used for training
            metrics (dict): Evaluation metrics

        Returns:
            str: Path to the saved model
        """
        if model is None:
            return None

        # Create filename
        if region:
            filename = f"{commodity.lower().replace(' ', '_')}_{region.lower().replace(' ', '_')}"
        else:
            filename = f"{commodity.lower().replace(' ', '_')}"

        # Save model
        model_path = os.path.join(self.models_dir, f"{filename}.model")

        if model_type == 'xgboost':
            model.save_model(model_path)
        else:  # lightgbm
            model.save_model(model_path)

        # Save metadata
        metadata = {
            'commodity': commodity,
            'region': region,
            'model_type': model_type,
            'params': params,
            'feature_cols': feature_cols,
            'metrics': metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        metadata_path = os.path.join(self.models_dir, f"{filename}.meta.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model for {commodity} saved to {model_path}")

        return model_path

    def load_model(self, commodity, region=None):
        """
        Load a trained model from disk.

        Args:
            commodity (str): Commodity to load model for
            region (str, optional): Region to load model for

        Returns:
            tuple: Loaded model, model type, feature columns, and metrics
        """
        # Create filename
        if region:
            filename = f"{commodity.lower().replace(' ', '_')}_{region.lower().replace(' ', '_')}"
        else:
            filename = f"{commodity.lower().replace(' ', '_')}"

        # Load metadata
        metadata_path = os.path.join(self.models_dir, f"{filename}.meta.json")

        if not os.path.exists(metadata_path):
            print(f"No model found for {commodity}")
            return None, None, None, None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load model
        model_path = os.path.join(self.models_dir, f"{filename}.model")

        if metadata['model_type'] == 'xgboost':
            model = xgb.Booster()
            model.load_model(model_path)
        else:  # lightgbm
            model = lgb.Booster(model_file=model_path)

        return model, metadata['model_type'], metadata['feature_cols'], metadata.get('metrics')

    def train_all_commodity_models(self, df, model_type='xgboost', n_trials=50, by_region=False):
        """
        Train models for all commodities in the dataset.

        Args:
            df (pandas.DataFrame): Input dataframe with features
            model_type (str): Type of model to train ('xgboost' or 'lightgbm')
            n_trials (int): Number of Optuna trials
            by_region (bool): Whether to train separate models for each region

        Returns:
            dict: Dictionary of trained models
        """
        # Get unique commodities
        commodities = df['commodity'].unique()

        # Train models for each commodity
        models = {}
        overall_metrics = []

        for commodity in commodities:
            if by_region:
                # Get unique regions for this commodity
                regions = df[df['commodity'] == commodity]['region'].unique()

                for region in regions:
                    print(f"\n=== Training model for {commodity} in {region} ===")

                    model, params, feature_cols, metrics = self.train_model_for_commodity(
                        df, commodity, region, model_type, n_trials
                    )

                    if model is not None:
                        model_path = self.save_model(model, commodity, region, model_type, params, feature_cols, metrics)
                        models[(commodity, region)] = (model, model_type, feature_cols, metrics)

                        if metrics:
                            metrics['commodity'] = commodity
                            metrics['region'] = region
                            overall_metrics.append(metrics)
            else:
                print(f"\n=== Training model for {commodity} ===")

                model, params, feature_cols, metrics = self.train_model_for_commodity(
                    df, commodity, None, model_type, n_trials
                )

                if model is not None:
                    model_path = self.save_model(model, commodity, None, model_type, params, feature_cols, metrics)
                    models[commodity] = (model, model_type, feature_cols, metrics)

                    if metrics:
                        metrics['commodity'] = commodity
                        overall_metrics.append(metrics)

        # Calculate and print overall metrics
        if overall_metrics:
            avg_rmse = np.mean([m['rmse'] for m in overall_metrics])
            avg_nrmse = np.mean([m['nrmse'] for m in overall_metrics])
            avg_mae = np.mean([m['mae'] for m in overall_metrics])
            avg_r2 = np.mean([m['r2'] for m in overall_metrics])

            print("\n=== Overall Model Performance ===")
            print(f"Average RMSE: {avg_rmse:.4f}")
            print(f"Average NRMSE: {avg_nrmse:.4f}")
            print(f"Average MAE: {avg_mae:.4f}")
            print(f"Average R²: {avg_r2:.4f}")

        return models

    def update_model(self, model, model_type, X_new, y_new, params=None):
        """
        Update an existing model with new data.

        Args:
            model: Existing model
            model_type (str): Type of model ('xgboost' or 'lightgbm')
            X_new: New features
            y_new: New targets
            params (dict, optional): Model parameters

        Returns:
            Updated model
        """
        if model is None or X_new is None or y_new is None:
            return model

        if model_type == 'xgboost':
            # Create DMatrix for new data
            dnew = xgb.DMatrix(X_new, label=y_new)

            # Update model
            updated_model = xgb.train(
                params or {'objective': 'reg:squarederror', 'eval_metric': 'rmse'},
                dnew,
                num_boost_round=100,
                xgb_model=model
            )

            return updated_model

        elif model_type == 'lightgbm':
            # Create Dataset for new data
            new_data = lgb.Dataset(X_new, label=y_new)

            # Update model
            updated_model = lgb.train(
                params or {'objective': 'regression', 'metric': 'rmse'},
                new_data,
                num_boost_round=100,
                init_model=model
            )

            return updated_model

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def predict(self, model, model_type, X, feature_cols=None):
        """
        Make predictions with a trained model.

        Args:
            model: Trained model
            model_type (str): Type of model ('xgboost' or 'lightgbm')
            X: Features to predict on
            feature_cols (list, optional): Feature columns to use

        Returns:
            numpy.ndarray: Predictions
        """
        if model is None or X is None:
            return None

        # Select feature columns if provided
        if feature_cols is not None:
            # Ensure all feature columns are present
            missing_cols = set(feature_cols) - set(X.columns)
            if missing_cols:
                print(f"Warning: Missing feature columns: {missing_cols}")
                # Create missing columns with zeros
                for col in missing_cols:
                    X[col] = 0

            X = X[feature_cols]

        # Make predictions
        if model_type == 'xgboost':
            dtest = xgb.DMatrix(X)
            preds = model.predict(dtest)
        else:  # lightgbm
            preds = model.predict(X)

        return preds

    def evaluate_model(self, model, model_type, X_test, y_test, feature_cols=None):
        """
        Evaluate a trained model on test data.

        Args:
            model: Trained model
            model_type (str): Type of model ('xgboost' or 'lightgbm')
            X_test: Test features
            y_test: Test targets
            feature_cols (list, optional): Feature columns to use

        Returns:
            dict: Evaluation metrics
        """
        if model is None or X_test is None or y_test is None:
            return None

        # Make predictions
        preds = self.predict(model, model_type, X_test, feature_cols)

        if preds is None:
            return None

        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Calculate normalized RMSE
        y_range = y_test.max() - y_test.min()
        nrmse = rmse / y_range if y_range > 0 else 0

        # Create metrics dictionary
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'nrmse': nrmse
        }

        return metrics
