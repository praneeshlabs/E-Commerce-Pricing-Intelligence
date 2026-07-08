
class RegressionPipeline:
    """
    Price prediction using multiple regression models
    NO CROSS-VALIDATION - Simple 80/20 split for large datasets
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_engineer = EnhancedFeatureEngineering()

    def train_all_models(self, X, y, test_size=0.2):
        """Train all regression models"""
        print("\n" + "="*80)
        print("TRAINING REGRESSION MODELS (80/20 SPLIT)")
        print("="*80)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        print(f"\nData Split:")
        print(f"  Training: {len(X_train):,} samples ({(1-test_size)*100:.0f}%)")
        print(f"  Testing:  {len(X_test):,} samples ({test_size*100:.0f}%)")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        results = {}

        # Train models
        print("\n1. XGBoost...")
        xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train_scaled, y_train)
        results['XGBoost'] = self._evaluate(xgb_model, X_train_scaled, X_test_scaled, y_train, y_test)

        print("\n2. LightGBM...")
        lgb_model = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                      subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
        lgb_model.fit(X_train_scaled, y_train)
        results['LightGBM'] = self._evaluate(lgb_model, X_train_scaled, X_test_scaled, y_train, y_test)

        print("\n3. CatBoost...")
        cat_model = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, random_state=42, verbose=0)
        cat_model.fit(X_train_scaled, y_train)
        results['CatBoost'] = self._evaluate(cat_model, X_train_scaled, X_test_scaled, y_train, y_test)

        print("\n4. Ridge...")
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train_scaled, y_train)
        results['Ridge'] = self._evaluate(ridge_model, X_train_scaled, X_test_scaled, y_train, y_test)

        # Print comparison
        self._print_comparison(results)

        return results, self.scaler, (X_train, X_test, y_train, y_test)

    def _evaluate(self, model, X_train, X_test, y_train, y_test):
        """Evaluate a single model"""
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        residuals = y_test - y_test_pred
        std_residual = np.std(residuals)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        metrics = {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'test_mape': mean_absolute_percentage_error(y_test, y_test_pred) * 100,
            'predictions': y_test_pred,
            'actuals': y_test,
            'prediction_std': std_residual
        }

        overfit_ratio = train_rmse / test_rmse

        print(f" Train RMSE: ${train_rmse:.2f} | Test RMSE: ${test_rmse:.2f} | R²: {metrics['test_r2']:.4f} | MAPE: {metrics['test_mape']:.2f}%")
        if overfit_ratio < 0.95:
            print(f" ⚠️ Tr ain/Test ratio: {overfit_ratio:.3f} (possible overfitting)")

        return metrics

    def _print_comparison(self, results):
        """Print model comparison"""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(f"\n{'Model':<20} {'Test RMSE':<12} {'Test MAE':<12} {'Test R²':<10} {'MAPE':<10}")
        print("─"*80)

        for name, result in results.items():
            print(f"{name:<20} ${result['test_rmse']:<11.2f} ${result['test_mae']:<11.2f} "
                  f"{result['test_r2']:<9.4f} {result['test_mape']:<9.2f}%")

        best_model_name = min(results.keys(), key=lambda k: results[k]['test_rmse'])
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f" Test RMSE: ${results[best_model_name]['test_rmse']:.2f}")
        print(f" Test R²: {results[best_model_name]['test_r2']:.4f}")

        # Feature importance
        if best_model_name in ['XGBoost', 'LightGBM', 'Random Forest', 'CatBoost']:
            model = results[best_model_name]['model']
            if hasattr(model, 'feature_importances_'):
                print(f"\n📊 Top 10 Most Important Features:")
                importances = model.feature_importances_
                feature_names = self.feature_engineer.feature_names
                indices = np.argsort(importances)[::-1][:10]

                for i, idx in enumerate(indices, 1):
                    feat_name = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
                    print(f"   {i:2d}. {feat_name:<30} {importances[idx]:.4f}")