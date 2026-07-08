
class EnhancedFeatureEngineering:
    """
    Comprehensive feature engineering using ALL available data
    """

    def __init__(self):
        # Placeholder for feature names, as required by the print_comparison method
        self.feature_names = [f'embedding_{i}' for i in range(384)] + ['Item Name', 'Value', 'Unit']

    def extract_structured_features(self, df):
        """Extract and engineer features from ALL columns"""
        print("\n" + "="*80)
        print("ENHANCED FEATURE ENGINEERING")
        print("="*80)

        features_dict = {}

        # 1. Embeddings (384d)
        print("\n1. Text Embeddings (384 dimensions)")
        if 'embedding' in df.columns:
            embeddings = np.vstack(df['embedding'].values)
            features_dict['embeddings'] = embeddings
            print(f"   ✓ Added {embeddings.shape[1]} embedding features")

        # 2. Value (quantity)
        print("\n2. Value (Packaging Quantity)")
        if 'Value' in df.columns:
            value_feature = pd.to_numeric(df['Value'], errors='coerce')
            value_feature = value_feature.fillna(0).values.reshape(-1, 1)
            features_dict['value'] = value_feature
            print(f"   ✓ Added Value feature (range: {value_feature.min():.2f} - {value_feature.max():.2f})")

        # 3. Unit (categorical)
        print("\n3. Unit (Packaging Unit Type)")
        if 'Unit' in df.columns:
            units = df['Unit'].fillna('unknown').astype(str)
            unique_units = units.unique()
            print(f"   Found {len(unique_units)} unique units: {list(unique_units)[:10]}")

            unit_dummies = pd.get_dummies(units)
            features_dict['unit'] = unit_dummies.values
            print(f"   ✓ Added {unit_dummies.shape[1]} unit features")

        # 4. Clusters
        print("\n4. Cluster Assignments")
        if 'cluster' in df.columns:
            cluster_dummies = pd.get_dummies(df['cluster'], prefix='cluster')
            features_dict['cluster'] = cluster_dummies.values
            print(f"   ✓ Added {cluster_dummies.shape[1]} cluster features")


        # 6. Text statistics
        print("\n6. Text Statistics")
        if 'catalog_content' in df.columns:
            text_lengths = df['catalog_content'].str.len().fillna(0).values.reshape(-1, 1)
            word_counts = df['catalog_content'].str.split().str.len().fillna(0).values.reshape(-1, 1)

            features_dict['text_stats'] = np.hstack([text_lengths, word_counts])
            print(f"   ✓ Added 2 text statistics")

        # 7. Image availability
        print("\n7. Image Features")
        if 'image_link' in df.columns or 'images' in df.columns:
            img_col = 'image_path' if 'image_path' in df.columns else 'images'
            has_image = df[img_col].notna().astype(int).values.reshape(-1, 1)
            features_dict['has_image'] = has_image
            print(f"   ✓ Added image availability feature")

        # 8. Entity features
        print("\n8. Extracted Entities")
        if 'synthesis' in df.columns:
            entity_features = self._extract_entities(df['synthesis'])
            features_dict['entities'] = entity_features
            print(f"   ✓ Added {entity_features.shape[1]} entity features")

        # Combine all features
        print("\n" + "="*80)
        print("COMBINING FEATURES")
        print("="*80)

        feature_matrices = []
        self.feature_names = []

        feature_order = ['value', 'unit', 'embeddings', 'cluster', 'confidence',
                        'text_stats', 'has_image', 'entities']

        for feat_name in feature_order:
            if feat_name in features_dict:
                matrix = features_dict[feat_name]
                feature_matrices.append(matrix)

                if feat_name == 'embeddings':
                    self.feature_names.extend([f'emb_{i}' for i in range(matrix.shape[1])])
                elif feat_name == 'value':
                    self.feature_names.append('value')
                elif feat_name == 'unit':
                    self.feature_names.extend([f'unit_{i}' for i in range(matrix.shape[1])])
                elif feat_name == 'cluster':
                    self.feature_names.extend([f'cluster_{i}' for i in range(matrix.shape[1])])
                elif feat_name == 'confidence':
                    self.feature_names.extend(['conf_high', 'conf_medium', 'conf_low'])
                elif feat_name == 'text_stats':
                    self.feature_names.extend(['text_length', 'word_count'])
                elif feat_name == 'has_image':
                    self.feature_names.append('has_image')
                elif feat_name == 'entities':
                    self.feature_names.extend([f'entity_{i}' for i in range(matrix.shape[1])])

        X = np.hstack(feature_matrices)

        print(f"\nFinal Feature Matrix: {X.shape}")
        print(f"  Samples: {X.shape[0]:,}")
        print(f"  Features: {X.shape[1]}")

        return X, self.feature_names

    def _extract_entities(self, synthesis_series):
        """Extract entity features from synthesis text"""
        entity_features = []

        for text in synthesis_series:
            if pd.isna(text) or text == '':
                entity_features.append([0, 0, 0, 0, 0])
                continue

            text_lower = str(text).lower()
            has_brand = 1 if 'brand:' in text_lower else 0
            has_category = 1 if 'category:' in text_lower else 0
            has_size = 1 if 'size:' in text_lower else 0
            has_claims = 1 if 'claims:' in text_lower or 'organic' in text_lower else 0
            has_cert = 1 if 'certification' in text_lower else 0

            entity_features.append([has_brand, has_category, has_size, has_claims, has_cert])

        return np.array(entity_features)