# CardioGuard ML

> **Intelligent Cardiac Risk Assessment Powered by Machine Learning**

An end-to-end Machine Learning project that predicts the likelihood of cardiovascular disease based on clinical patient attributes using **Random Forest** and **Logistic Regression**.

---

## 📌 Project Overview

CardioGuard ML analyzes clinical health indicators (such as blood pressure, cholesterol levels, resting heart rate, and chest pain type) to classify whether a patient is at risk of heart disease (`target = 1`) or healthy (`target = 0`).

It includes:
1. **Interactive Desktop GUI (`app_gui.py`)**: A modern desktop application featuring single-patient clinical assessment, instant batch CSV imports, risk probability scores, and report export.
2. **Terminal Analysis Script (`Heart Disease Prediction System.py`)**: End-to-end exploratory data analysis (EDA), data cleaning, model training, and sample prediction.

---

## ✨ Key Enhancements & Algorithm Fixes

- **Duplicate Data Elimination:** Automatically detected and removed **723 duplicate rows** from the raw dataset, preventing data leakage and training honestly on **302 unique patients**.
- **Ensemble Learning (Random Forest):** Replaced basic linear fitting with Random Forest (120 estimators) to capture complex non-linear clinical interactions.
- **High Sensitivity / Recall Focus:** Optimized to detect true positive cardiac risk cases effectively.
- **Risk Probability Scores:** Instead of returning a plain binary (0 or 1), the system outputs precise risk probability percentages (e.g. *84.2% Cardiac Risk*).

---

## 📊 Dataset & Features

The model is trained on diagnostic features from clinical records:

| Feature | Description |
| :--- | :--- |
| `age` | Age of patient (years) |
| `sex` | Gender (1 = Male, 0 = Female) |
| `cp` | Chest pain type (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic) |
| `trestbps` | Resting blood pressure (in mm Hg) |
| `chol` | Serum cholesterol (in mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) |
| `restecg` | Resting electrocardiographic results (0: Normal, 1: ST-T Abnormality, 2: LV Hypertrophy) |
| `thalach` | Maximum heart rate achieved during exercise |
| `exang` | Exercise induced angina (1 = Yes, 0 = No) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping) |
| `ca` | Number of major vessels colored by fluoroscopy (0-3) |
| `thal` | Thalassemia status (1: Normal, 2: Fixed Defect, 3: Reversible Defect) |

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Desktop GUI Application (Recommended)
```bash
python app_gui.py
```
**GUI Features:**
- **Tab 1 (Patient Assessment):** Fill in clinical lab inputs or click *"Sample: High Risk Patient"* / *"Sample: Healthy Patient"* to get an instant visual diagnosis with color-coded risk percentage.
- **Tab 2 (Batch CSV Import):** Import `sample_test_patients.csv` (or any patient list), run batch predictions, view results in an interactive table, and export the report to CSV.
- **Tab 3 (Model Insights):** Switch between Random Forest and Logistic Regression, retrain dynamically, and inspect performance metrics.

### 3. Run the Terminal Script
```bash
python "Heart Disease Prediction System.py"
```

---

## 📁 Repository Structure

```text
├── .gitignore                          # Ignored temporary files & caches
├── README.md                           # Project documentation
├── requirements.txt                    # Project dependencies
├── heart.csv                           # Clinical dataset
├── sample_test_patients.csv            # Sample CSV for testing batch predictions
├── app_gui.py                          # Modern Desktop GUI (Tkinter + ML Engine)
├── Heart Disease Prediction System.py  # Terminal pipeline script
├── targetDistribution.png              # Generated target distribution plot
└── heartDiseaseAndAges.png             # Generated age frequency plot
```
