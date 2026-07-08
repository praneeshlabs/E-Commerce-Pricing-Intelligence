""" Run
1. 'python -m pip install --prefer-binary catboost num2words accelerate bitsandbytes -U'
2. 'python -m pip install torch torchvision torchaudio --prefer-binary'
in terminal 
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoProcessor, AutoModelForVision2Seq,
    TrOCRProcessor, VisionEncoderDecoderModel,
    BlipProcessor, BlipForConditionalGeneration,
    AutoModelForImageTextToText, AutoModel
)
from PIL import Image
import time
import requests
from io import BytesIO
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import json
import gc
from tqdm import tqdm
import warnings
import zipfile
import os
from pathlib import Path
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import json
import pickle
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForImageTextToText

print("Completed all imports successfully.")

warnings.filterwarnings('ignore')