

def run_complete_pipeline(df, text_column='catalog_content', image_column=None,
                          price_column='price', n_clusters=5):
    """
    COMPLETE END-TO-END PIPELINE

    Args:
        df: Input DataFrame with columns: catalog_content, images (optional), price, Value, Unit
        text_column: Name of text description column
        image_column: Name of image path column (or None)
        price_column: Name of price column
        n_clusters: Number of clusters for K-means

    Returns:
        Dictionary with all results
    """

    print("="*80)
    print("COMPLETE UNIFIED PIPELINE")
    print("="*80)
    print(f"\nDataset: {len(df):,} samples")
    print(f"Columns: {list(df.columns)}")

    # PHASE 1: LLM EXTRACTION
    print("\n" + "="*80)
    print("PHASE 1: LLM/VISION EXTRACTION & EMBEDDING CREATION")
    print("="*80)

    extraction_pipeline = ProductAnalysisPipeline()

    df_processed, embeddings = extraction_pipeline.process_dataframe(
        df,
        text_column=text_column,
        image_column=image_column,
        price_column=price_column,
        save_raw_extracts=False
    )

    # Save intermediate results
    df_processed.to_csv('step1_extracted_features.csv', index=False)
    np.save('step1_embeddings.npy', embeddings)
    print("\n✓ Saved extraction results to 'step1_extracted_features.csv'")

    # PHASE 2: CLUSTERING 
    print("\n" + "="*80)
    print("PHASE 2: K-MEANS CLUSTERING")
    print("="*80)

    cluster_results = extraction_pipeline.perform_clustering(embeddings, n_clusters=n_clusters)
    df_processed['cluster'] = cluster_results['labels']

    # PHASE 3: FEATURE ENGINEERING
    print("\n" + "="*80)
    print("PHASE 3: COMPREHENSIVE FEATURE ENGINEERING")
    print("="*80)

    regression_pipeline = RegressionPipeline()
    X, feature_names = regression_pipeline.feature_engineer.extract_structured_features(df_processed)
    y = df_processed[price_column].values

    print(f"\nTarget Variable Statistics:")
    print(f"  Mean: ${y.mean():.2f}")
    print(f"  Std: ${y.std():.2f}")
    print(f"  Min: ${y.min():.2f}")
    print(f"  Max: ${y.max():.2f}")

    # PHASE 4: REGRESSION 
    print("\n" + "="*80)
    print("PHASE 4: PRICE PREDICTION MODELS")
    print("="*80)

    results, scaler, split_data = regression_pipeline.train_all_models(X, y, test_size=0.2)

    # PHASE 5: SAVE RESULTS
    print("\n" + "="*80)
    print("PHASE 5: SAVING RESULTS")
    print("="*80)

    # Save best model
    best_model_name = min(results.keys(), key=lambda k: results[k]['test_rmse'])
    best_model = results[best_model_name]['model']

    model_package = {
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'model_name': best_model_name,
        'cluster_model': cluster_results['model'],
        'cluster_scaler': cluster_results['scaler']
    }

    with open('final_model_package.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    print(f"✓ Saved model package: final_model_package.pkl")

    # Save processed data
    df_processed.to_csv('final_processed_data.csv', index=False)
    print(f"✓ Saved processed data: final_processed_data.csv")

    # Save embeddings
    np.save('final_embeddings.npy', embeddings)
    print(f"✓ Saved embeddings: final_embeddings.npy")

    # Save summary
    X_train, X_test, y_train, y_test = split_data
    summary = {
        'dataset_info': {
            'total_samples': len(df),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X.shape[1],
            'n_clusters': n_clusters
        },
        'price_stats': {
            'mean': float(y.mean()),
            'std': float(y.std()),
            'min': float(y.min()),
            'max': float(y.max())
        },
        'model_performance': {
            name: {
                'test_rmse': float(result['test_rmse']),
                'test_mae': float(result['test_mae']),
                'test_r2': float(result['test_r2']),
                'test_mape': float(result['test_mape'])
            }
            for name, result in results.items()
        },
        'best_model': best_model_name
    }

    with open('pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: pipeline_summary.json")

    # FINAL SUMMARY 
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nDataset: {len(df):,} samples")
    print(f"Features: {X.shape[1]}")
    print(f"Clusters: {n_clusters}")
    print(f"\nBest Model: {best_model_name}")
    print(f"  Test RMSE: ${results[best_model_name]['test_rmse']:.2f}")
    print(f"  Test MAE: ${results[best_model_name]['test_mae']:.2f}")
    print(f"  Test R²: {results[best_model_name]['test_r2']:.4f}")
    print(f"  Test MAPE: {results[best_model_name]['test_mape']:.2f}%")

    print("\nOutput Files:")
    print("  - final_model_package.pkl (trained model)")
    print("  - final_processed_data.csv (all features)")
    print("  - final_embeddings.npy (384-dim vectors)")
    print("  - pipeline_summary.json (metrics)")

    return {
        'df_processed': df_processed,
        'embeddings': embeddings,
        'features': (X, y, feature_names),
        'results': results,
        'best_model': best_model,
        'model_package': model_package,
        'summary': summary
    }
