Comparative Analysis of Temporal Feature Engineering Techniques for Short-Term Household Power Consumption Forecasting





A complete end-to-end machine learning pipeline that compares multiple temporal feature engineering strategies and evaluates a wide range of regression models for short-term household electricity demand forecasting.



📌 Project Overview

This repository implements a systematic study of temporal feature engineering techniques applied to the classic Individual Household Electric Power Consumption dataset.

The work answers two practical questions:





Which temporal features (lags, rolling statistics, calendar encodings, Fourier terms, etc.) most improve short-term forecasting accuracy?



Which machine-learning models perform best once those features are engineered?

The pipeline covers the full lifecycle:





Raw data ingestion & cleaning



Missing-value handling



Resampling to hourly resolution



Multiple temporal feature-engineering strategies



Large-scale model benchmarking



Best-model selection & residual analysis



SHAP-based model interpretability



📁 Repository Structure

├── data/
│   ├── household_power_consumption.txt          # Original UCI dataset
│   ├── household_power_hourly.csv                # Cleaned hourly series
│   ├── household_power_hourly_clean.csv          # After missing-value treatment
│   └── household_power_featured.csv              # Final feature matrix
│
├── scripts/
│   ├── step1.py                                  # Data loading
│   ├── step2.py                                  # Initial exploration
│   ├── step3.py                                  # Basic preprocessing
│   ├── step4_timedate.py                         # Datetime parsing & indexing
│   ├── step5_hourly.py                           # Resampling to hourly
│   ├── step6_missinghandaling.py                 # Missing-value strategies
│   ├── step7_feautre.py                          # Temporal feature engineering
│   ├── step8_experimental.py                     # Experiment design
│   ├── step9_ml_model.py                         # Model training & evaluation
│   ├── step10_best_model.py                      # Best-model selection
│   ├── step11_visualization.py                   # Performance plots
│   ├── step12_shape.py                           # SHAP analysis
│   └── step13_final_result.py                    # Final report generation
│
├── results/
│   ├── ml_experiment_results.csv
│   ├── ml_experiment_results_sorted.csv
│   ├── results_sorted_by_rmse.csv
│   ├── top10_models.csv
│   ├── model_average_performance.csv
│   ├── best_model_per_experiment.csv
│   ├── best_model_predictions.csv
│   ├── best_model_feature_importance.csv
│   ├── shap_feature_importance.csv
│   └── ...
│
├── figures/
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   ├── feature_importance.png
│   ├── top10_rmse_comparison.png
│   ├── top10_r2_comparison.png
│   ├── shap_summary_plot.png
│   ├── shap_feature_importance.png
│   └── shap_individual_prediction.png
│
├── docs/
│   └── Comparative Analysis of Temporal Feature Engineering Techniques for Short.docx
│
└── README.md



🗃️ Dataset

Source: UCI Machine Learning Repository – Individual Household Electric Power Consumption
Original file: household_power_consumption.txt







Attribute



Description





Date / Time



Timestamp (minute resolution)





Global_active_power



Household global active power (kW)





Global_reactive_power



Household global reactive power (kW)





Voltage



Minute-averaged voltage (V)





Global_intensity



Minute-averaged current intensity (A)





Sub_metering_1/2/3



Energy sub-metering (Wh)

After cleaning and resampling the series is converted to hourly resolution for short-term forecasting experiments.



⚙️ Pipeline Stages







Step



Script



Purpose





1



step1.py



Load raw text file





2



step2.py



Exploratory data analysis





3



step3.py



Basic cleaning & type conversion





4



step4_timedate.py



Parse datetime & set index





5



step5_hourly.py



Resample to hourly frequency





6



step6_missinghandaling.py



Imputation / missing-value strategies





7



step7_feautre.py



Temporal feature engineering (core contribution)





8



step8_experimental.py



Define experiment grid





9



step9_ml_model.py



Train & evaluate many regressors





10



step10_best_model.py



Select best model per experiment





11



step11_visualization.py



Generate comparison plots





12



step12_shape.py



SHAP global & local explanations





13



step13_final_result.py



Aggregate final metrics & export



🧪 Temporal Feature Engineering Techniques Evaluated

The project systematically compares several families of temporal features:





Lag features – previous 1–24 h (and selected longer lags)



Rolling / expanding statistics – mean, std, min, max over multiple windows



Calendar encodings – hour-of-day, day-of-week, month, is-weekend, is-holiday



Cyclical encodings – sine/cosine transforms of hour and day-of-week



Fourier terms – seasonal harmonics



Difference & rate-of-change features



Interaction features between calendar and lag variables

Each combination is tested under controlled experimental conditions so that the contribution of feature sets can be isolated.



🤖 Models Benchmarked

A broad set of regressors is evaluated (exact list available in ml_experiment_results.csv):





Linear models (Ridge, Lasso, ElasticNet)



Tree-based ensembles (Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost)



Support Vector Regression



k-Nearest Neighbors



Neural-network baselines (MLP)



Naive seasonal baselines (for reference)

Primary metrics: RMSE, MAE, R²
Secondary: training time, prediction latency, feature-importance stability



📊 Key Results & Visualisations







File



Description





top10_rmse_comparison.png



Top-10 models ranked by RMSE





top10_r2_comparison.png



Top-10 models ranked by R²





actual_vs_predicted.png



Actual vs predicted time-series plot





residual_plot.png



Residual diagnostics





feature_importance.png



Classical feature importance





shap_summary_plot.png



SHAP summary (global feature impact)





shap_feature_importance.png



Mean |SHAP| ranking





shap_individual_prediction.png



Local explanation for a single forecast

Detailed numeric results are stored in:





ml_experiment_results_sorted.csv



results_sorted_by_rmse.csv



top10_models.csv



best_model_per_experiment.csv



best_model_feature_importance.csv



shap_feature_importance.csv



🚀 How to Run

# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Place the original dataset
#    Download from UCI and put it in data/household_power_consumption.txt

# 4. Execute the pipeline sequentially
python scripts/step1.py
python scripts/step2.py
# ... continue through step13_final_result.py



Note: Some steps are computationally intensive (especially model grid search and SHAP). A machine with ≥ 8 GB RAM is recommended.



📦 Requirements

Typical dependencies (create requirements.txt from your environment):

pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
shap
matplotlib
seaborn
statsmodels
joblib
tqdm



📄 Documentation

A full comparative analysis write-up is available in:

docs/Comparative Analysis of Temporal Feature Engineering Techniques for Short.docx



🔍 Key Findings (Summary)





Carefully engineered temporal features significantly outperform naïve lag-only baselines.



Tree-based ensembles (especially gradient boosting variants) consistently rank in the top tier for this dataset.



SHAP analysis reveals that recent lags, hour-of-day cyclical encodings, and short-term rolling means are among the most influential predictors.



Residual plots indicate that remaining errors are largely heteroscedastic around peak-consumption hours — a useful direction for future work (e.g., quantile regression or peak-specific models).

(Exact numbers and rankings are in the CSV/PNG artefacts.)



🤝 Contributing

Pull requests that add new feature-engineering strategies, additional models, or improved evaluation protocols are welcome. Please open an issue first to discuss major changes.



📜 License

This project is released under the MIT License. See LICENSE for details.



🙏 Acknowledgements





UCI Machine Learning Repository for the Household Power Consumption dataset



The open-source community behind scikit-learn, XGBoost, LightGBM, CatBoost and SHAP



📧 Contact

For questions or collaboration:

[Your Name]
[your.email@example.com]
[LinkedIn / Personal website]



Last updated: September 2026
