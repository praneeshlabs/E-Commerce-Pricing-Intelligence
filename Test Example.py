
if __name__ == "__main__":
    """
    COMPLETE USAGE EXAMPLE

    Required DataFrame columns:
    - catalog_content: Product text description
    - images: Path to product image (optional, can be None)
    - price: Product price
    - Value: Packaging quantity (e.g., 16, 500)
    - Unit: Unit type (e.g., 'oz', 'ml', 'g')
    """

    df_path = '/content/drive/MyDrive/Amazon_ML_Challenge/Amazon_ML_dataset/Copy_of_test.csv'
    df = pd.read_csv(df_path)

    # Run complete pipeline
    print("="*80)
    print("STARTING COMPLETE PIPELINE")
    print("="*80)

    results = run_complete_pipeline(
        df[:20],
        text_column='catalog_content',
        image_column='image_link',
        price_column='price',
        n_clusters=5
    )

    #Access results
    best_model = results['best_model']
    df_processed = results['df_processed']
    embeddings = results['embeddings']
    summary = results['summary']

    print("\nPipeline ready to run!")
    print("\nUncomment the run_complete_pipeline() call above to execute.")
    print("Make sure your DataFrame has the required columns:")
    print("  - catalog_content (text)")
    print("  - images (paths or None)")
    print("  - price (numeric)")
    print("  - Value (numeric quantity)")
    print("  - Unit (text unit type)")