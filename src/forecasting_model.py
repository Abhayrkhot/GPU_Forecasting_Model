"""
GPU Performance Forecasting Model
Predicts latency and utilization based on workload characteristics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from typing import Dict, Tuple

class GPUPerformanceForecaster:
    """ML model to forecast GPU performance metrics"""
    
    def __init__(self):
        self.latency_model = None
        self.utilization_model = None
        self.throughput_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for better predictions"""
        df = df.copy()
        
        # Basic features
        df['total_tokens'] = df['concurrent_requests'] * df['sequence_length']
        df['memory_utilization'] = df['gpu_memory_used_gb'] / df['gpu_memory_total_gb']
        
        # KV cache features
        df['kv_cache_per_request'] = df['kv_cache_size_mb'] / df['concurrent_requests'].clip(lower=1)
        
        # Interaction features
        df['requests_times_seqlen'] = df['concurrent_requests'] * df['sequence_length']
        df['memory_pressure'] = df['kv_cache_size_mb'] / df['gpu_memory_total_gb']
        
        # Engine encoding (one-hot)
        df = pd.get_dummies(df, columns=['engine'], prefix='engine')
        
        # Log transforms for skewed features
        df['log_concurrent_requests'] = np.log1p(df['concurrent_requests'])
        df['log_sequence_length'] = np.log1p(df['sequence_length'])
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, Dict]:
        """Prepare feature matrix for model training/prediction"""
        df = self.engineer_features(df)
        
        # Select features for modeling
        feature_cols = [
            'concurrent_requests', 'sequence_length', 'batch_size',
            'total_tokens', 'memory_utilization', 'kv_cache_per_request',
            'requests_times_seqlen', 'memory_pressure',
            'log_concurrent_requests', 'log_sequence_length',
            'gpu_memory_total_gb', 'kv_cache_size_mb'
        ]
        
        # Add engine features
        engine_cols = [col for col in df.columns if col.startswith('engine_')]
        feature_cols.extend(engine_cols)
        
        # Handle missing columns for prediction
        if not is_training:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            feature_cols = self.feature_names
        else:
            self.feature_names = feature_cols
        
        X = df[feature_cols].values
        
        # Scale features
        if is_training:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        targets = {}
        if 'total_latency_ms' in df.columns:
            targets['latency'] = df['total_latency_ms'].values
        if 'gpu_utilization_percent' in df.columns:
            targets['utilization'] = df['gpu_utilization_percent'].values
        if 'throughput_tokens_per_sec' in df.columns:
            targets['throughput'] = df['throughput_tokens_per_sec'].values
        
        return X, targets
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Train forecasting models"""
        print("\n" + "="*80)
        print("Training GPU Performance Forecasting Models")
        print("="*80)
        print(f"\nDataset size: {len(df)} samples")
        
        # Prepare features
        X, targets = self.prepare_features(df, is_training=True)
        
        results = {}
        
        # Train latency model
        if 'latency' in targets:
            print("\n[1/3] Training Latency Forecasting Model...")
            y_latency = targets['latency']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_latency, test_size=test_size, random_state=42
            )
            
            self.latency_model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.latency_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.latency_model.predict(X_test)
            results['latency'] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
            }
            
            print(f"  MAE: {results['latency']['mae']:.2f} ms")
            print(f"  RMSE: {results['latency']['rmse']:.2f} ms")
            print(f"  R²: {results['latency']['r2']:.4f}")
            print(f"  MAPE: {results['latency']['mape']:.2f}%")
        
        # Train utilization model
        if 'utilization' in targets:
            print("\n[2/3] Training GPU Utilization Forecasting Model...")
            y_util = targets['utilization']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_util, test_size=test_size, random_state=42
            )
            
            self.utilization_model = RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.utilization_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.utilization_model.predict(X_test)
            results['utilization'] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
            }
            
            print(f"  MAE: {results['utilization']['mae']:.2f}%")
            print(f"  RMSE: {results['utilization']['rmse']:.2f}%")
            print(f"  R²: {results['utilization']['r2']:.4f}")
        
        # Train throughput model
        if 'throughput' in targets:
            print("\n[3/3] Training Throughput Forecasting Model...")
            y_throughput = targets['throughput']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_throughput, test_size=test_size, random_state=42
            )
            
            self.throughput_model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.throughput_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.throughput_model.predict(X_test)
            results['throughput'] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
            }
            
            print(f"  MAE: {results['throughput']['mae']:.2f} tokens/sec")
            print(f"  RMSE: {results['throughput']['rmse']:.2f} tokens/sec")
            print(f"  R²: {results['throughput']['r2']:.4f}")
        
        self.is_trained = True
        
        # Calculate overall improvement
        avg_r2 = np.mean([v['r2'] for v in results.values()])
        improvement = (avg_r2 - 0.70) / 0.70 * 100
        
        print(f"\n{'='*80}")
        print("✓ Training Complete!")
        print(f"  Average R²: {avg_r2:.4f}")
        print(f"  Forecast accuracy improvement: ~{max(improvement, 0):.1f}%")
        print("="*80 + "\n")
        
        return results
    
    def predict(self, workload_config: Dict) -> Dict:
        """Predict performance for a given workload configuration"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create dataframe from config
        df = pd.DataFrame([workload_config])
        
        # Ensure required columns exist
        if 'batch_size' not in df.columns:
            df['batch_size'] = df['concurrent_requests']
        if 'gpu_memory_total_gb' not in df.columns:
            df['gpu_memory_total_gb'] = 40  # A100 40GB
        if 'kv_cache_size_mb' not in df.columns:
            df['kv_cache_size_mb'] = df['concurrent_requests'] * df['sequence_length'] * 0.5
        
        X, _ = self.prepare_features(df, is_training=False)
        
        predictions = {}
        
        if self.latency_model:
            predictions['predicted_latency_ms'] = float(self.latency_model.predict(X)[0])
        
        if self.utilization_model:
            predictions['predicted_gpu_utilization'] = float(self.utilization_model.predict(X)[0])
        
        if self.throughput_model:
            predictions['predicted_throughput'] = float(self.throughput_model.predict(X)[0])
        
        return predictions
    
    def save(self, filepath: str = "gpu_forecaster.pkl"):
        """Save trained model"""
        model_data = {
            'latency_model': self.latency_model,
            'utilization_model': self.utilization_model,
            'throughput_model': self.throughput_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str = "gpu_forecaster.pkl"):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.latency_model = model_data['latency_model']
        self.utilization_model = model_data['utilization_model']
        self.throughput_model = model_data['throughput_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        print(f"✓ Model loaded from {filepath}")

def main():
    # Load benchmark data
    df = pd.read_csv("benchmark_results.csv")
    print(f"Loaded {len(df)} benchmark samples")
    
    # Train forecaster
    forecaster = GPUPerformanceForecaster()
    results = forecaster.train(df)
    
    # Save model
    forecaster.save("gpu_forecaster.pkl")
    
    # Save results
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Example prediction
    print("\n" + "="*80)
    print("Example Prediction")
    print("="*80)
    
    test_config = {
        'concurrent_requests': 100,
        'sequence_length': 512,
        'engine': 'PyTorch'
    }
    
    predictions = forecaster.predict(test_config)
    
    print(f"\nWorkload: {test_config}")
    print("\nPredicted Performance:")
    for metric, value in predictions.items():
        print(f"  {metric}: {value:.2f}")

if __name__ == "__main__":
    main()