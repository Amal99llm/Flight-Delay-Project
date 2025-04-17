✈️ Flight Delay Prediction Model
This repository contains a machine learning project for predicting flight delays using GPU-accelerated libraries (RAPIDS, cuML, and XGBoost).
The final model achieves 82.90% accuracy in predicting whether a flight will be delayed.

📊 Dataset
We use a dataset of 3 million flight records (flights_sample_3m.csv) with rich features including:

Flight dates, airlines, and flight numbers

Origin and destination airports

Scheduled vs actual departure/arrival times

Flight distance and elapsed time

Delay information categorized by cause

🔍 Data Overview
3,000,000 flight records

32 columns:

9 categorical

4 integer

19 float features

Target variable: IS_DELAYED
(1 if arrival delay > 15 minutes)

Class distribution: ~17.18% delayed flights

🧪 Exploratory Data Analysis
Key insights:

Delay Distribution: Most flights are on time or early; delay distribution is right-skewed.

Airline Performance: Big variance in average delay across airlines; some consistently outperform others.

Distance vs Delay: No clear linear relationship between flight distance and delay time.

⚙️ Data Preprocessing
Steps included:

Missing value imputation (mean/mode)

Feature engineering:

Time features (day of week, month, year, departure hour)

Major origin/destination indicators

Distance categorization

Airline encoding

Data conversion to GPU format using RAPIDS cuDF for accelerated processing

🧠 Models Compared
Two models were trained and evaluated:

1. 🌲 Random Forest Classifier (cuML)
Accuracy: 82.86%

Training time: 13.73 sec

Prediction time: 19.48 sec

2. ⚡ XGBoost Classifier (GPU)
Accuracy: 82.90%

Training time: 18.53 sec

Prediction time: 2.15 sec ✅

🏆 XGBoost was selected as the final model due to its higher accuracy and significantly faster inference time.

🚀 Usage
🔧 Prerequisites
Python 3.x

Required packages:
numpy, pandas, joblib, xgboost, cudf, cuml

📌 Make Predictions
Use the provided script:

bash
Copy
Edit
python prediction.py --input your_data.csv --output predictions.csv --model XGBoost_model.joblib
Arguments:
--input: Path to your input CSV file (required)

--output: Where to save the predictions (required)

--model: Path to the model file (default: XGBoost_model.joblib)

📁 Repository Contents
flight_delay_prediction.ipynb: Full notebook with code and analysis

XGBoost_model.joblib: Trained XGBoost model

scalar.joblib: Scaler used during preprocessing

prediction.py: Prediction script

sampled_data.csv: Sample of 200 flight records

README.md: This file

📈 Performance Summary
Accuracy: 82.90%

Prediction time: 2.15 seconds for 600k test samples

Class imbalance handled effectively (~17% delayed flights)

🔮 Future Improvements
Add features (weather, airport congestion)

Hyperparameter tuning

Ensemble methods

Try neural networks or boosting variants

Deploy as a real-time web API
