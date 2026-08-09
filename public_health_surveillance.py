# ============================================
# PUBLIC HEALTH SURVEILLANCE DATA ANALYSIS
# ============================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Display settings
pd.set_option("display.max_columns", None)

# Seaborn style
sns.set_theme(style="whitegrid")

df = pd.read_csv(
    r"C:\Users\User\Documents\Khingfm_projects\data_science\assets\personal_datasets\public_health_surveillance_dataset (1).csv"
)

print(df.head())
print(df.tail())

