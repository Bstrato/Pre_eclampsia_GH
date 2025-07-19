**Preeclampsia Prediction Using Machine Learning: Addressing Class Imbalance**

This project develops a machine learning pipeline to predict the presence of preeclampsia in pregnant women using clinical data. Preeclampsia is a serious pregnancy complication characterized by high blood pressure and protein in urine, making early detection crucial for maternal and fetal health. The project addresses the inherent class imbalance problem commonly found in medical datasets through comprehensive sampling techniques and multi-model evaluation.

**Problem Statement**

Preeclampsia affects 2-8% of pregnancies worldwide and is a leading cause of maternal and perinatal morbidity and mortality. Early prediction can significantly improve clinical outcomes by enabling timely intervention. However, the rarity of the condition creates a class imbalance challenge in machine learning models, where the minority class (preeclampsia cases) is often underrepresented.

**Key Features**
- Data Preprocessing: Handles missing values, feature encoding, and correlation analysis
- Class Imbalance Mitigation: Implements multiple resampling techniques (SMOTE, Random Over/Under Sampling)
- Multi-Model Evaluation: Compares performance across 5 different machine learning algorithms
- Extensive Visualization: Generates confusion matrices, correlation heatmaps, and class distribution plots
- Robust Metrics: Evaluates models using accuracy, precision, recall, F1-score, and ROC-AUC

**Dataset**
The project uses Dataset_V1.csv containing clinical features related to pregnancy complications. The target variable is "Principal Diagnosis" with binary classification (preeclampsia vs. normal).
Data Processing Steps:

-Missing Value Imputation: Mode for categorical variables, mean for numerical variables
-Feature Engineering: One-hot encoding for categorical variables
-Feature Selection: Correlation-based selection (threshold > 0.05)
-Data Shuffling: Ensures randomization for better model training

**Resampling Techniques**
-SMOTE (Synthetic Minority Over-sampling Technique): Generates synthetic examples for minority class
-Random Over Sampler: Duplicates minority class samples randomly
-Random Under Sampler: Reduces majority class samples randomly

**Classification Models**
- Logistic Regression: Linear baseline model
- Random Forest: Ensemble tree-based method
- Support Vector Machine (SVM): Kernel-based classification
- K-Nearest Neighbors (KNN): Instance-based learning
- Gradient Boosting: Sequential ensemble method

**Evaluation Metrics**
- Accuracy: Overall correctness of predictions
- Precision: Positive predictive value
- Recall (Sensitivity): True positive rate
- F1-Score: Harmonic mean of precision and recall
- ROC-AUC: Area under the receiver operating characteristic curve

**Core Libraries**
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- scikit-learn: Machine learning algorithms and utilities
- imbalanced-learn: Specialized library for handling imbalanced datasets

**Visualization**
- matplotlib: Static plotting
- seaborn: Statistical data visualization
