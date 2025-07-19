import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore")


def plot_confusion(y_true, y_pred, labels, sampler_name, model_name):
    """Save confusion matrix as an image"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix\n({sampler_name} - {model_name})")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    filename = f"confusion_{sampler_name}_{model_name}.png".replace(" ", "_")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_post_balance(y_resampled, labels, sampler_name):
    """Plot class distribution after applying imbalance technique"""
    class_counts = pd.Series(y_resampled).value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.xticks([0, 1], labels=labels)
    plt.title(f"Class Distribution After {sampler_name}")
    plt.ylabel("Count")
    plt.xlabel("Principal Diagnosis")
    plt.tight_layout()
    filename = f"post_balance_{sampler_name}.png".replace(" ", "_")
    plt.savefig(filename)
    plt.close()

# Function for Data loadingm shuffling and handling missing data
def main():
    file_path = "Dataset_V1.csv"
    if not os.path.exists(file_path):
        print(f"ERROR: File {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('S/N', errors='ignore')

    imputer_mode = SimpleImputer(strategy='most_frequent')
    imputer_mean = SimpleImputer(strategy='mean')

    df[cat_cols] = imputer_mode.fit_transform(df[cat_cols])
    df[num_cols] = imputer_mean.fit_transform(df[num_cols])

    target = 'Principal Diagnosis'
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    class_labels = le.classes_

    #Feature Engineering
    df_encoded = pd.get_dummies(df.drop(columns=['S/N']), drop_first=True)
    corr = df_encoded.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("heatmap.png")
    plt.close()

    # Feature selection
    target_corr = corr[target].abs().sort_values(ascending=False)
    selected_features = target_corr[target_corr > 0.05].index.drop(target)
    X = df_encoded[selected_features]
    y = df_encoded[target]

    #Class distribution of balance
    class_counts = y.value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.xticks([0, 1], labels=le.inverse_transform([0, 1]))
    plt.title("Class Distribution (Before Resampling)")
    plt.ylabel("Count")
    plt.xlabel("Principal Diagnosis")
    plt.tight_layout()
    plt.savefig("class_distribution_before.png")
    plt.close()

    #Function for evaluation
    def evaluate_models(X, y, sampler, sampler_name):
        X_res, y_res = sampler.fit_resample(X, y)

        # Visualize post-balance distribution
        plot_post_balance(y_res, le.inverse_transform([0, 1]), sampler_name)

        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        results = []
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

            # Save confusion matrix
            plot_confusion(y_test, y_pred, le.inverse_transform([0, 1]), sampler_name, model_name)

            results.append({
                "Sampler": sampler_name,
                "Model": model_name,
                "Accuracy": round(acc, 3),
                "Precision": round(prec, 3),
                "Recall": round(rec, 3),
                "F1-score": round(f1, 3),
                "ROC-AUC": round(auc, 3)
            })
        return results

  
    samplers = {
        "SMOTE": SMOTE(random_state=42),
        "RandomOverSampler": RandomOverSampler(random_state=42),
        "RandomUnderSampler": RandomUnderSampler(random_state=42)
    }

    all_results = []
    for name, sampler in samplers.items():
        all_results.extend(evaluate_models(X, y, sampler, name))

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("classification_results.csv", index=False)

    print("\n=== Evaluation Complete ===")
    print(results_df)


if __name__ == "__main__":
    main()
