# CredAI – AI-Based Credit Risk Assessment and Loan Default Prediction System

**AI Credit Intelligence | Artificial Intelligence + Data Science + FinTech**

CredAI is an AI-powered credit risk assessment and loan default prediction system designed to help evaluate borrowers' creditworthiness and identify potential loan default risks using machine learning.

The system uses a Random Forest classification model to analyze borrower information and classify loan applications into low-risk and high-risk categories.

## 🎯 Project Objectives

- Predict the probability of loan default using machine learning.
- Assess borrower creditworthiness based on financial and personal attributes.
- Identify important factors influencing credit risk.
- Support batch assessment of multiple borrowers through CSV uploads.
- Provide clear risk insights through an interactive dashboard.

## ✨ Key Features

- **AI-Based Risk Prediction:** Uses a Random Forest classifier to predict loan default risk.
- **CSV Batch Assessment:** Upload CSV files to assess multiple borrowers.
- **Risk Classification:** Categorizes borrowers into low-risk and high-risk groups.
- **Prediction Probability:** Displays the estimated probability of loan default.
- **Prediction History:** Stores and tracks previous assessments using SQLite.
- **Data Visualization:** Displays interactive charts and risk analytics.
- **Feature Importance:** Identifies the major factors influencing predictions.
- **PDF Reports:** Generates downloadable prediction reports.
- **CSV Export:** Supports exporting assessment results.
- **User Authentication:** Provides login and registration functionality.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| Machine Learning | Scikit-learn, Random Forest |
| Data Processing | Pandas, NumPy |
| Database | SQLite |
| Model Storage | Joblib |
| Visualization | Chart.js |
| PDF Generation | ReportLab |
| Development Tools | VS Code, Git, GitHub |

## 🧠 Machine Learning Model

**Algorithm:** Random Forest Classifier

The model is trained using borrower-related attributes, including:

- Age
- Income
- Loan Amount
- Credit Score
- Months Employed
- Number of Credit Lines
- Interest Rate
- Loan Term
- Debt-to-Income Ratio
- Education
- Employment Type
- Marital Status
- Mortgage Status
- Dependents
- Loan Purpose
- Co-signer Status

### Model Performance

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 61.78% |
| Precision | 19.90% |
| Recall | 75.74% |

*These are the reported evaluation results of the current model.*

## ⚙️ Project Methodology

1. **Data Collection:** Collect borrower information from the loan default dataset.
2. **Data Preprocessing:** Clean the dataset and prepare numerical and categorical features.
3. **Feature Selection:** Identify relevant attributes influencing loan default.
4. **Model Training:** Train the Random Forest classification model.
5. **Risk Prediction:** Predict the probability and category of loan default.
6. **Visualization:** Display prediction results and analytics through the dashboard.

## 📁 Project Structure

```text
CredAI/
│
├── backend/
│   ├── app.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── test_api.py
│   ├── requirements.txt
│   ├── model_threshold.json
│   └── threshold_comparison.csv
│
├── data/
│   └── Loan_default.csv
│
├── main/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── style.css
│   └── script.js
│
├── .gitignore
└── README.md
```

## 🚀 Installation and Setup

### Prerequisites

- Python 3.10 or later (use a version compatible with the project's dependencies).
- Git
- Visual Studio Code
- A modern web browser

### Step 1: Clone the Repository

```bash
git clone https://github.com/lssatwika29-sys/CredAI.git
```

### Step 2: Navigate to the Project

```bash
cd CredAI
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### Step 5: Prepare the Machine Learning Model

The trained `.joblib` model is not included in this repository because of its large file size.

Train or generate the model locally using the provided training script:

```bash
cd backend
python train_model.py
```

Ensure the generated model is saved at the location expected by `app.py`.

### Step 6: Run the Backend

From the `backend` directory:

```bash
python app.py
```

The Flask backend runs at:

```text
http://127.0.0.1:5000
```

### Step 7: Run the Frontend

Open the `main` folder in VS Code and launch `index.html` using the Live Server extension.

Make sure the Flask backend is running before using the assessment features.

## 🔗 Backend API Endpoints

| API Endpoint | Purpose |
|---|---|
| `/api/health` | Check backend and model status |
| `/api/predict` | Predict individual loan default risk |
| `/api/predict-csv` | Assess multiple borrowers using CSV |
| `/api/history` | Retrieve prediction history |
| `/api/feature-importance` | Retrieve feature importance |

## 📸 Project Screenshots

Add your actual application screenshots here.

- Dashboard Overview
- Borrower Risk Assessment
- CSV Batch Assessment Results
- Prediction History
- Model Performance
- Feature Importance and Explainability

## 🔮 Future Enhancements

- Integrate additional machine learning algorithms.
- Improve prediction accuracy and model calibration.
- Add advanced borrower risk profiling.
- Introduce real-time financial data integration.
- Enhance explainable AI capabilities.
- Deploy the application on a cloud platform.

## 🎓 Project Information

**Project Name:** CredAI – AI-Based Credit Risk Assessment and Loan Default Prediction System

**Domain:** Artificial Intelligence, Data Science and FinTech

**Project Type:** Academic Mini Project

**Repository:** https://github.com/lssatwika29-sys/CredAI

---

*CredAI – Making Credit Risk Assessment Smarter with Artificial Intelligence.*
