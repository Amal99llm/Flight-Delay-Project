# ✈️ Flight Delay Prediction Model

This repository contains a machine learning project for predicting flight delays using **GPU-accelerated libraries**: [RAPIDS](https://rapids.ai/), `cuML`, and `XGBoost`.  
The final model achieves **82.90% accuracy** in predicting whether a flight will be delayed by more than 15 minutes.

---

## 📦 Dataset

We use a dataset of **3 million flight records** (`flights_sample_3m.csv`) with rich features including:

- Flight dates, airlines, and flight numbers  
- Origin and destination airports  
- Scheduled and actual departure/arrival times  
- Flight distance and elapsed time  
- Delay information categorized by cause  

### 🧾 Data Overview

- **3,000,000** flight records  
- **32 columns**:  
  - 9 categorical  
  - 4 integer  
  - 19 float features  
- **Target variable**: `IS_DELAYED`  
  (1 if arrival delay > 15 minutes)  
- **Class distribution**: ~17.18% delayed flights

---

## 📊 Exploratory Data Analysis

Key insights from EDA:

1. **Delay Distribution**: Most flights are on time or early; delay distribution is right-skewed  
2. **Airline Performance**: Major differences in average delay by airline  
3. **Distance vs Delay**: No strong linear relationship observed

---

## 🔧 Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values (mean/mode imputation)  
- Feature engineering:
  - Time features (month, day of week, hour, year)
  - Airport frequency (major origin/destination)
  - Distance binning
  - Airline encoding  
- Class imbalance handling using weighted classes  
- Conversion to **GPU format** using RAPIDS `cuDF` for acceleration

---

## 🤖 Models Compared

Three models were trained and evaluated using GPU and CPU where applicable:

### 1. 💠 Logistic Regression (cuML)
- **Accuracy**: 57.67%  
- **Training time**: ~0.97 sec  
- **Prediction time**: ~0.65 sec  
- Used for baseline comparison

---

### 2. 🌲 Random Forest Classifier (cuML)
- **Accuracy**: 82.86%  
- **Training time**: 13.73 sec  
- **Prediction time**: 19.48 sec

---

### 3. ⚡ XGBoost Classifier (GPU)
- **Accuracy**: 82.90% ✅  
- **Training time**: 18.53 sec  
- **Prediction time**: 2.15 sec  
- **Selected as final model** due to best performance and fastest prediction

---
