## Problem Statement

Develop a machine learning system that predicts product prices by understanding both product descriptions and images. The pipeline extracts structured product information from unstructured data, generates semantic representations, engineers meaningful features, and learns complex relationships between product characteristics and pricing.

## Multimodal Product Price Prediction

An end-to-end multimodal AI pipeline that predicts product prices by combining textual product descriptions and product images. The system leverages multiple Large Language Models (LLMs), Vision Language Models (VLMs), semantic embeddings, feature engineering, and ensemble regression models to accurately estimate product prices.




## Features

- Multi-LLM information extraction (Phi-2, TinyLlama, Flan-T5)
- Vision-based feature extraction (SmolVLM, TrOCR, BLIP)
- LLM-based information fusion using Qwen2.5
- Semantic embedding generation using Sentence Transformers
- K-Means clustering for product segmentation
- Automated feature engineering
- Price prediction using:
  - XGBoost
  - LightGBM
  - CatBoost
  - Ridge Regression
- Complete end-to-end inference pipeline


## Project Workflow

```
Product Text + Product Image
            │
            ▼
 Multi-LLM & Vision Extraction
            │
            ▼
    Information Fusion (Qwen2.5)
            │
            ▼
     Semantic Embedding Creation
            │
            ▼
      Feature Engineering
            │
            ▼
     K-Means Clustering
            │
            ▼
     Regression Model Training
            │
            ▼
      Product Price Prediction
```



## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Pandas
- NumPy


## Repository Structure

```
├── Extraction pipeline.py
├── Feature Engineering.py
├── Regression Pipeline.py
├── Workflow.py
├── Test Example.py
├── requirements.py
└── README.md
```


## Results

The pipeline automatically:

- Extracts structured product attributes
- Generates semantic embeddings
- Creates engineered features
- Trains multiple regression models
- Selects the best-performing model
- Saves production-ready artifacts for inference


## Future Improvements

- FastAPI deployment
- Docker support
- MLflow experiment tracking
- Hyperparameter optimization
- Distributed batch inference
- Cloud deployment on AWS/GCP



## Author: **Praneesh B**


