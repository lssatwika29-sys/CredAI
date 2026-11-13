from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from flask import session
import re
import pandas as pd
import joblib
import os
import sqlite3
import json

from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


# --------------------------------------------------
# FLASK APPLICATION SETUP
# --------------------------------------------------


app = Flask(__name__)
# Allow CSV uploads up to 100 MB
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

app.config["SECRET_KEY"] = os.environ.get(
    "CREDAI_SECRET_KEY",
    "change-this-to-a-random-secret-key"
)

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_PATH="/",
    SESSION_REFRESH_EACH_REQUEST=True
)

CORS(
    app,
    supports_credentials=True,
    origins=["http://127.0.0.1:5500"]
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "loan_default_model_optimized.joblib"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "model_threshold.json"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "credai_history.db"
)

# Load optimized machine learning model
model = joblib.load(MODEL_PATH)

# Load optimized classification threshold
with open(THRESHOLD_PATH, "r") as file:
    MODEL_THRESHOLD = float(
        json.load(file)["threshold"]
    )


# --------------------------------------------------
# PDF COLORS
# --------------------------------------------------

GREEN = colors.HexColor("#166534")
DARK_GREEN = colors.HexColor("#14532d")
LIGHT_GREEN = colors.HexColor("#dcfce7")
LIGHT_BG = colors.HexColor("#f8fafc")
DARK = colors.HexColor("#1e293b")
MUTED = colors.HexColor("#64748b")
BORDER = colors.HexColor("#e2e8f0")
RED = colors.HexColor("#b91c1c")


# --------------------------------------------------
# DATABASE SETUP
# --------------------------------------------------



def init_database():

    with sqlite3.connect(DATABASE_PATH) as connection:

        # Create predictions table
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                credit_score INTEGER,
                loan_amount REAL,
                prediction INTEGER NOT NULL,
                probability REAL NOT NULL,
                borrower_data TEXT
            )
        """)

        # Get existing prediction columns
        columns = [
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(predictions)"
            ).fetchall()
        ]

        # Add borrower_data if missing
        if "borrower_data" not in columns:
            connection.execute("""
                ALTER TABLE predictions
                ADD COLUMN borrower_data TEXT
            """)

        # REFRESH column list after adding borrower_data
        columns = [
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(predictions)"
            ).fetchall()
        ]

        # Add classification_threshold if missing
        if "classification_threshold" not in columns:
            connection.execute("""
                ALTER TABLE predictions
                ADD COLUMN classification_threshold REAL
            """)

        # REFRESH column list again
        columns = [
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(predictions)"
            ).fetchall()
        ]

        # Add user_id if missing
        if "user_id" not in columns:
            connection.execute("""
                ALTER TABLE predictions
                ADD COLUMN user_id INTEGER
            """)

        # Create users table
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        connection.commit()



def get_history():

    with sqlite3.connect(DATABASE_PATH) as connection:

        connection.row_factory = sqlite3.Row

        rows = connection.execute("""
            SELECT id, timestamp, credit_score,
                   loan_amount, prediction, probability
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 100
        """, (session["user_id"],)).fetchall()

        return [dict(row) for row in rows]

init_database()

# --------------------------------------------------
# USER REGISTRATION
# --------------------------------------------------

@app.route("/api/register", methods=["POST"])
def register():

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Please provide registration details."
            }), 400

        full_name = str(data.get("full_name", "")).strip()
        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", ""))

        # Validate input
        if not full_name or not email or not password:
            return jsonify({
                "status": "error",
                "message": "All fields are required."
            }), 400

        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            return jsonify({
                "status": "error",
                "message": "Please enter a valid email address."
            }), 400

        if len(password) < 8:
            return jsonify({
                "status": "error",
                "message": "Password must contain at least 8 characters."
            }), 400

        # Securely hash the password
        password_hash = generate_password_hash(password)

        created_at = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        with sqlite3.connect(DATABASE_PATH) as connection:

            connection.execute("""
                INSERT INTO users (
                    full_name,
                    email,
                    password_hash,
                    created_at
                )
                VALUES (?, ?, ?, ?)
            """, (
                full_name,
                email,
                password_hash,
                created_at
            ))

        return jsonify({
            "status": "success",
            "message": "Account created successfully."
        }), 201

    except sqlite3.IntegrityError:
        return jsonify({
            "status": "error",
            "message": "An account with this email already exists."
        }), 409

    except Exception:
        app.logger.exception("Registration failed")
        return jsonify({
            "status": "error",
            "message": "Registration failed. Please try again."
        }), 500



# --------------------------------------------------
# GENERAL ROUTES
# --------------------------------------------------

# --------------------------------------------------
# USER LOGIN
# --------------------------------------------------

@app.route("/api/login", methods=["POST"])
def login():

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Please provide login details."
            }), 400

        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", ""))

        if not email or not password:
            return jsonify({
                "status": "error",
                "message": "Email and password are required."
            }), 400

        with sqlite3.connect(DATABASE_PATH) as connection:

            connection.row_factory = sqlite3.Row

            user = connection.execute("""
                SELECT id, full_name, email, password_hash
                FROM users
                WHERE email = ?
            """, (email,)).fetchone()

        if user is None or not check_password_hash(
            user["password_hash"], password
        ):
            return jsonify({
                "status": "error",
                "message": "Invalid email or password."
            }), 401
        

        # Create login session
        session.clear()
        session["user_id"] = user["id"]
        session["user_name"] = user["full_name"]

        return jsonify({
            "status": "success",
            "message": "Login successful.",
            "user": {
                "id": user["id"],
                "full_name": user["full_name"],
                "email": user["email"]
            }
        }), 200

    except Exception:
        app.logger.exception("Login failed")
        return jsonify({
            "status": "error",
            "message": "Login failed. Please try again."
        }), 500
# --------------------------------------------------
# USER LOGOUT
# --------------------------------------------------

@app.route("/api/logout", methods=["POST"])
def logout():

    session.clear()

    return jsonify({
        "status": "success",
        "message": "Logged out successfully."
    }), 200
# --------------------------------------------------
# CHECK LOGIN STATUS
# --------------------------------------------------

@app.route("/api/session", methods=["GET"])
def check_session():

    if "user_id" not in session:
        return jsonify({
            "authenticated": False
        }), 401

    return jsonify({
        "authenticated": True,
        "user": {
            "id": session["user_id"],
            "full_name": session["user_name"]
        }
    }), 200
# --------------------------------------------------
# GENERAL ROUTES
# --------------------------------------------------
# --------------------------------------------------
# LOGIN REQUIRED DECORATOR
# --------------------------------------------------

from functools import wraps

def login_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:
            return jsonify({
                "status": "error",
                "message": "Please log in first."
            }), 401

        return function(*args, **kwargs)

    return wrapper
@app.route("/")
def home():

    return jsonify({
        "message": "Credit Risk Assessment API is running!",
        "project": (
            "AI-Based Credit Risk Assessment and "
            "Loan Default Prediction System"
        ),
        "model": "Optimized Random Forest",
        "threshold": MODEL_THRESHOLD
    })


@app.route("/api/health")
def health():

    return jsonify({
        "status": "success",
        "message": "Backend is working correctly",
        "model_loaded": model is not None,
        "model": "Optimized Random Forest",
        "classification_threshold": MODEL_THRESHOLD,
        "database_connected": os.path.exists(DATABASE_PATH)
    })


# --------------------------------------------------
# PREDICTION ROUTE
# --------------------------------------------------
@app.route("/api/predict", methods=["POST"])
@login_required
def predict():

    try:

        data = request.get_json(silent=True)

        if not isinstance(data, dict):

            return jsonify({
                "error": "Please send borrower details as JSON."
            }), 400

        required_columns = (
            list(model.feature_names_in_)
            if hasattr(model, "feature_names_in_")
            else None
        )

        if not required_columns:

            return jsonify({
                "error": "Could not determine model input columns."
            }), 500

        missing = [
            col for col in required_columns
            if col not in data
        ]

        if missing:

            return jsonify({
                "error": "Missing required fields.",
                "missing_fields": missing
            }), 400

        borrower = pd.DataFrame([
            {col: data[col] for col in required_columns}
        ])

        # Get prediction probabilities
        probabilities = model.predict_proba(borrower)[0]

        classes = list(model.classes_)

        default_probability = (
            float(probabilities[classes.index(1)])
            if 1 in classes else 0.0
        )

        # Apply optimized threshold
        prediction = int(
            default_probability >= MODEL_THRESHOLD
        )
        used_threshold = MODEL_THRESHOLD
        # Generate individual borrower explanation
        risk_factors = []
        positive_factors = []

        credit_score = float(data.get("CreditScore", 0))
        dti_ratio = float(data.get("DTIRatio", 0))
        interest_rate = float(data.get("InterestRate", 0))
        months_employed = float(data.get("MonthsEmployed", 0))

        if credit_score < 600:
            risk_factors.append(
                "Credit score is below 600."
            )
        elif credit_score >= 700:
            positive_factors.append(
                "Credit score is 700 or above."
            )

        if dti_ratio > 0.4:
            risk_factors.append(
                "Debt-to-income ratio is relatively high."
            )
        elif dti_ratio <= 0.3:
            positive_factors.append(
                "Debt-to-income ratio is relatively low."
            )

        if interest_rate > 15:
            risk_factors.append(
                "Interest rate is relatively high."
            )

        if months_employed < 12:
            risk_factors.append(
                "Employment duration is less than one year."
            )
        elif months_employed >= 24:
            positive_factors.append(
                "Employment duration is at least two years."
            )

        if prediction == 1:
            explanation_summary = (
                "The model classified this borrower as high default risk. "
                "Review the risk indicators below as part of human assessment."
            )
        else:
            explanation_summary = (
                "The model classified this borrower as low default risk. "
                "Continue normal verification before making any decision."
            )

        individual_explanation = {
            "summary": explanation_summary,
            "risk_factors": risk_factors,
            "positive_factors": positive_factors
        }
        
        timestamp = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        borrower_json = json.dumps(
            data,
            ensure_ascii=False,
            default=str
        )

        # Save prediction in SQLite
        with sqlite3.connect(DATABASE_PATH) as connection:

            cursor = connection.execute("""
    INSERT INTO predictions (
        timestamp,
        credit_score,
        loan_amount,
        prediction,
        probability,
        borrower_data,
        classification_threshold,
        user_id
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", (
    timestamp,
    data.get("CreditScore"),
    data.get("LoanAmount"),
    prediction,
    default_probability,
    borrower_json,
    used_threshold,
    session["user_id"]
))

            prediction_id = cursor.lastrowid

        return jsonify({
            "status": "success",
            "id": prediction_id,
            "prediction": prediction,
            "result": (
                "High default risk"
                if prediction == 1
                else "Low default risk"
            ),
            "default_probability": round(
                default_probability, 4
            ),
            "classification_threshold": round(
                MODEL_THRESHOLD, 4
            ),
            "individual_explanation": individual_explanation,

"note": (
    "This is a machine-learning estimate, not "
    "a loan approval or rejection decision."
)
        })

    except (TypeError, ValueError) as error:

        return jsonify({
            "error": "Invalid borrower details.",
            "details": str(error)
        }), 400

    except Exception:

        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed. Check backend logs."
        }), 500





# --------------------------------------------------
# LARGE CSV BATCH PREDICTION - OPTIMIZED
# --------------------------------------------------

@app.route("/api/predict-csv", methods=["POST"])
@login_required
def predict_csv():

    try:
        uploaded_file = request.files.get("file")

        if uploaded_file is None or not uploaded_file.filename:
            return jsonify({
                "status": "error",
                "message": "Please select a CSV file."
            }), 400

        if not uploaded_file.filename.lower().endswith(".csv"):
            return jsonify({
                "status": "error",
                "message": "Only CSV files are supported."
            }), 400

        # ------------------------------------------
        # READ CSV
        # ------------------------------------------

        file_bytes = uploaded_file.read()

        if not file_bytes:
            return jsonify({
                "status": "error",
                "message": "The uploaded CSV file is empty."
            }), 400

        df = pd.read_csv(BytesIO(file_bytes))

        df.columns = [
            str(col).strip().lstrip("\ufeff")
            for col in df.columns
        ]

        if df.empty:
            return jsonify({
                "status": "error",
                "message": "No borrower records found."
            }), 400

        total_records = len(df)

        # Maximum borrower limit
        MAX_RECORDS = 10000

        if total_records > MAX_RECORDS:
            return jsonify({
                "status": "error",
                "message": f"Maximum {MAX_RECORDS} borrowers allowed per upload.",
                "total_records": total_records
            }), 400

        print("\n========== CREDAI CSV UPLOAD ==========")
        print("Filename:", uploaded_file.filename)
        print("Total records:", total_records)
        print("File size:", len(file_bytes) / (1024 * 1024), "MB")
        print("=======================================\n")

        # ------------------------------------------
        # BORROWER IDS
        # ------------------------------------------

        if "LoanID" in df.columns:
            borrower_ids = df["LoanID"].fillna("").astype(str).tolist()
        else:
            borrower_ids = [
                f"Borrower {i + 1}"
                for i in range(total_records)
            ]

        # ------------------------------------------
        # CHECK MODEL COLUMNS
        # ------------------------------------------

        required_columns = list(model.feature_names_in_)

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            return jsonify({
                "status": "error",
                "message": "Required CSV columns are missing.",
                "missing_columns": missing_columns
            }), 400

        borrower_df = df[required_columns].copy()

        # ------------------------------------------
        # VALIDATE EMPTY VALUES
        # ------------------------------------------

        borrower_df = borrower_df.replace(
            r"^\s*$",
            pd.NA,
            regex=True
        )

        if borrower_df.isna().any().any():

            missing_rows = (
                borrower_df.index[
                    borrower_df.isna().any(axis=1)
                ] + 2
            ).tolist()

            return jsonify({
                "status": "error",
                "message": "CSV contains missing borrower information.",
                "csv_rows": missing_rows[:30],
                "total_invalid_rows": len(missing_rows)
            }), 400

        # ------------------------------------------
        # NUMERIC VALIDATION
        # ------------------------------------------

        numeric_columns = [
            "Age",
            "Income",
            "LoanAmount",
            "CreditScore",
            "MonthsEmployed",
            "NumCreditLines",
            "InterestRate",
            "LoanTerm",
            "DTIRatio"
        ]

        for col in numeric_columns:

            if col in borrower_df.columns:

                borrower_df[col] = pd.to_numeric(
                    borrower_df[col],
                    errors="coerce"
                )

        if borrower_df.isna().any().any():

            return jsonify({
                "status": "error",
                "message": "Invalid or non-numeric values found in numeric columns."
            }), 400

        import numpy as np

        for col in numeric_columns:

            if col in borrower_df.columns:

                if not np.isfinite(
                    borrower_df[col].to_numpy(dtype=float)
                ).all():

                    return jsonify({
                        "status": "error",
                        "message": f"Invalid numeric values in {col}."
                    }), 400

        if "DTIRatio" in borrower_df.columns:

            if not borrower_df["DTIRatio"].between(0, 1).all():

                return jsonify({
                    "status": "error",
                    "message": "DTIRatio must be between 0 and 1."
                }), 400

        # ------------------------------------------
        # MODEL CLASS VALIDATION
        # ------------------------------------------

        classes = list(model.classes_)

        if 1 not in classes:
            return jsonify({
                "status": "error",
                "message": "Default class not found in model."
            }), 500

        default_index = classes.index(1)

        # ------------------------------------------
        # PROCESS IN SMALLER BATCHES
        # ------------------------------------------

        BATCH_SIZE = 1000

        results = []
        database_records = []

        timestamp = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        user_id = session["user_id"]

        for start in range(0, total_records, BATCH_SIZE):

            end = min(start + BATCH_SIZE, total_records)

            batch_df = borrower_df.iloc[start:end]

            # Predict only 1000 records at a time
            batch_probabilities = model.predict_proba(batch_df)

            default_probabilities = batch_probabilities[:, default_index]

            batch_predictions = (
                default_probabilities >= MODEL_THRESHOLD
            ).astype(int)

            # --------------------------------------
            # BUILD INDIVIDUAL RESULTS
            # --------------------------------------

            for position, (index, row) in enumerate(batch_df.iterrows()):

                probability = float(
                    default_probabilities[position]
                )

                prediction = int(
                    batch_predictions[position]
                )

                credit_score = float(row["CreditScore"])
                dti_ratio = float(row["DTIRatio"])
                interest_rate = float(row["InterestRate"])
                months_employed = float(row["MonthsEmployed"])
                loan_amount = float(row["LoanAmount"])

                risk_factors = []
                positive_factors = []
                recommendations = []

                # Credit score
                if credit_score < 600:
                    risk_factors.append("Low credit score")

                elif credit_score >= 700:
                    positive_factors.append("Good credit score")

                # DTI
                if dti_ratio > 0.4:
                    risk_factors.append("High DTI ratio")

                elif dti_ratio <= 0.3:
                    positive_factors.append("Manageable DTI ratio")

                # Interest rate
                if interest_rate > 15:
                    risk_factors.append("High interest rate")

                # Employment
                if months_employed < 12:
                    risk_factors.append("Short employment history")

                elif months_employed >= 24:
                    positive_factors.append("Stable employment history")

                # Recommendations
                if prediction == 1:

                    if not risk_factors:
                        risk_factors.append(
                            "Elevated risk identified from overall borrower profile."
                        )

                    if credit_score < 600:
                        recommendations.append(
                            "Improve repayment consistency."
                        )

                    if dti_ratio > 0.4:
                        recommendations.append(
                            "Reduce outstanding debt."
                        )

                    if interest_rate > 15:
                        recommendations.append(
                            "Review loan affordability."
                        )

                    if months_employed < 12:
                        recommendations.append(
                            "Build a stable employment history."
                        )

                    recommendations.append(
                        "Verify financial details and repayment records."
                    )

                    summary = (
                        "The model classified this borrower as high "
                        "default risk. Review the identified indicators."
                    )

                else:

                    summary = (
                        "The model classified this borrower as low "
                        "default risk. Normal verification is required."
                    )

                    recommendations = [
                        "Continue making repayments on time.",
                        "Maintain manageable outstanding debt.",
                        "Monitor loan affordability."
                    ]

                individual_explanation = {
                    "summary": summary,
                    "risk_factors": risk_factors,
                    "positive_factors": positive_factors
                }

                borrower_id = borrower_ids[index]

                # Store original model input
                borrower_data = json.dumps(
                    row.to_dict(),
                    ensure_ascii=False,
                    default=str
                )

                # ----------------------------------
                # PREPARE DATABASE RECORD
                # ----------------------------------

                database_records.append((
                    timestamp,
                    credit_score,
                    loan_amount,
                    prediction,
                    probability,
                    borrower_data,
                    MODEL_THRESHOLD,
                    user_id
                ))

                # ----------------------------------
                # PREPARE FRONTEND RESULT
                # ----------------------------------

                results.append({
                    "borrower": borrower_id,
                    "borrower_id": borrower_id,
                    "credit_score": credit_score,
                    "loan_amount": loan_amount,
                    "prediction": prediction,
                    "result": (
                        "High default risk"
                        if prediction == 1
                        else "Low default risk"
                    ),
                    "default_probability": round(probability, 4),
                    "individual_explanation": individual_explanation,
                    "recommendations": recommendations
                })

            print(
                f"Processed {end}/{total_records} borrowers"
            )

        # ------------------------------------------
        # BULK DATABASE INSERT WITH PDF REPORT IDs
        # ------------------------------------------

        with sqlite3.connect(DATABASE_PATH) as connection:

            cursor = connection.cursor()

            last_id = cursor.execute(
                "SELECT COALESCE(MAX(id), 0) FROM predictions"
            ).fetchone()[0]

            cursor.executemany("""
                INSERT INTO predictions (
                    timestamp,
                    credit_score,
                    loan_amount,
                    prediction,
                    probability,
                    borrower_data,
                    classification_threshold,
                    user_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, database_records)

            inserted_ids = cursor.execute("""
                SELECT id
                FROM predictions
                WHERE id > ? AND user_id = ?
                ORDER BY id ASC
            """, (last_id, user_id)).fetchall()

            connection.commit()

        for result, inserted_id in zip(results, inserted_ids):
            result["id"] = inserted_id[0]

       

            connection.commit()

        # ------------------------------------------
        # RETURN ALL RESULTS
        # ------------------------------------------

        print(
            f"SUCCESS: {len(results)} borrowers assessed."
        )

        return jsonify({
            "status": "success",
            "message": "Large CSV assessment completed successfully.",
            "total": total_records,
            "processed": len(results),
            "batch_size": BATCH_SIZE,
            "results": results
        }), 200

    except pd.errors.EmptyDataError:

        return jsonify({
            "status": "error",
            "message": "The CSV file does not contain valid data."
        }), 400

    except Exception:

        app.logger.exception("Large CSV prediction failed")

        return jsonify({
            "status": "error",
            "message": "CSV assessment failed. Check backend terminal."
        }), 500


    try:
        uploaded_file = request.files.get("file")

        if uploaded_file is None or uploaded_file.filename == "":
            return jsonify({
                "status": "error",
                "message": "Please select a CSV file."
            }), 400

        if not uploaded_file.filename.lower().endswith(".csv"):
            return jsonify({
                "status": "error",
                "message": "Only CSV files are supported."
            }), 400

        # Read CSV file with a 10 MB limit
        file_bytes = uploaded_file.read(10 * 1024 * 1024 + 1)

        if len(file_bytes) > 10 * 1024 * 1024:
            return jsonify({
                "status": "error",
                "message": "CSV file must be smaller than 10 MB."
            }), 400

        if not file_bytes:
            return jsonify({
                "status": "error",
                "message": "The uploaded CSV file is empty."
            }), 400

        # Read CSV
        df = pd.read_csv(BytesIO(file_bytes))
        # DEBUG: Verify which CSV file Flask actually receives
        print("\n========== CSV UPLOAD DEBUG ==========")
        print("Uploaded filename:", uploaded_file.filename)
        print("Total borrower rows:", len(df))
        print("CSV columns:", df.columns.tolist())

        if "LoanID" in df.columns:
            print("First 5 borrower IDs:", df["LoanID"].head().tolist())

        print("======================================\n")

        # Clean column names, including UTF-8 BOM
        df.columns = [
            str(column).strip().lstrip("\ufeff")
            for column in df.columns
        ]

        if df.empty:
            return jsonify({
                "status": "error",
                "message": "The CSV file contains no borrower records."
            }), 400

        if len(df) > 500:
            return jsonify({
                "status": "error",
                "message": "Upload a maximum of 500 borrowers at a time."
            }), 400

        # Preserve borrower IDs if LoanID exists
        if "LoanID" in df.columns:
            borrower_ids = df["LoanID"].tolist()
        else:
            borrower_ids = [
                f"Borrower {i + 1}"
                for i in range(len(df))
            ]

        # Get model input columns
        required_columns = list(model.feature_names_in_)

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            return jsonify({
                "status": "error",
                "message": "Some required CSV columns are missing.",
                "missing_columns": missing_columns
            }), 400

        # Keep only the required model columns
        borrower_df = df[required_columns].copy()

        # Treat blank cells as missing
        borrower_df = borrower_df.replace(
            r"^\s*$",
            pd.NA,
            regex=True
        )

        if borrower_df.isna().any().any():

            missing_rows = (
                borrower_df.index[
                    borrower_df.isna().any(axis=1)
                ] + 2
            ).tolist()

            return jsonify({
                "status": "error",
                "message": "Some borrower records contain missing values.",
                "csv_rows": missing_rows[:20]
            }), 400

        # Convert numeric columns
        numeric_columns = [
            "Age",
            "Income",
            "LoanAmount",
            "CreditScore",
            "MonthsEmployed",
            "NumCreditLines",
            "InterestRate",
            "LoanTerm",
            "DTIRatio"
        ]

        for column in numeric_columns:

            if column in borrower_df.columns:

                borrower_df[column] = pd.to_numeric(
                    borrower_df[column],
                    errors="coerce"
                )

        if borrower_df.isna().any().any():
            return jsonify({
                "status": "error",
                "message": "One or more numeric fields contain invalid values."
            }), 400

        # Validate finite numeric values
        import math

        for column in numeric_columns:

            if column in borrower_df.columns:

                valid = borrower_df[column].map(
                    lambda value: math.isfinite(float(value))
                )

                if not valid.all():
                    return jsonify({
                        "status": "error",
                        "message": f"Invalid numeric value in {column}."
                    }), 400

        # Validate DTI ratio
        if "DTIRatio" in borrower_df.columns:

            if not borrower_df["DTIRatio"].between(0, 1).all():
                return jsonify({
                    "status": "error",
                    "message": "DTIRatio must be between 0 and 1."
                }), 400

        # Generate predictions for ALL borrowers
        probabilities = model.predict_proba(borrower_df)

        classes = list(model.classes_)

        if 1 not in classes:
            return jsonify({
                "status": "error",
                "message": "The model does not contain the default class."
            }), 500

        default_index = classes.index(1)

        results = []

        # Process every borrower individually
        with sqlite3.connect(DATABASE_PATH) as connection:

            for index, (_, row) in enumerate(borrower_df.iterrows()):

                # Get this borrower's probability
                default_probability = float(
                    probabilities[index, default_index]
                )

                # Apply classification threshold
                prediction = int(
                    default_probability >= MODEL_THRESHOLD
                )

                # -----------------------------------------
                # INDIVIDUAL BORROWER RISK EXPLANATION
                # -----------------------------------------

                credit_score = float(row["CreditScore"])
                dti_ratio = float(row["DTIRatio"])
                interest_rate = float(row["InterestRate"])
                months_employed = float(row["MonthsEmployed"])

                risk_factors = []
                positive_factors = []
                recommendations = []

                # Credit score analysis
                if credit_score < 600:
                    risk_factors.append("Low credit score")

                elif credit_score >= 700:
                    positive_factors.append("Good credit score")

                # Debt-to-income analysis
                if dti_ratio > 0.4:
                    risk_factors.append("High DTI ratio")

                elif dti_ratio <= 0.3:
                    positive_factors.append("Manageable DTI ratio")

                # Interest rate analysis
                if interest_rate > 15:
                    risk_factors.append("High interest rate")

                # Employment stability
                if months_employed < 12:
                    risk_factors.append("Short employment history")

                elif months_employed >= 24:
                    positive_factors.append("Stable employment history")

                # Recommendations for both risk categories
                if prediction == 1:

                    if not risk_factors:
                        risk_factors.append(
                            "The model identified elevated risk "
                            "from the overall borrower profile."
                        )

                    if credit_score < 600:
                        recommendations.append(
                            "Improve repayment consistency."
                        )

                    if dti_ratio > 0.4:
                        recommendations.append(
                            "Reduce outstanding debt."
                        )

                    if interest_rate > 15 or dti_ratio > 0.4:
                        recommendations.append(
                            "Review loan affordability."
                        )

                    if months_employed < 12:
                        recommendations.append(
                            "Build and document a stable employment history."
                        )

                    recommendations.append(
                        "Verify financial details and repayment records."
                    )

                    explanation_summary = (
                        "The model classified this borrower as high "
                        "default risk. Review the identified risk "
                        "indicators before making a decision."
                    )

                else:

                    explanation_summary = (
                        "The model classified this borrower as low "
                        "default risk. Normal verification is still required."
                    )

                    recommendations = [
                        "Continue making repayments on time.",
                        "Maintain manageable outstanding debt.",
                        "Continue monitoring loan affordability."
                    ]

                # -----------------------------------------
                # SAVE EVERY BORROWER TO SQLITE
                # -----------------------------------------

                timestamp = datetime.now().astimezone().isoformat(
                    timespec="seconds"
                )

                borrower_data = json.dumps(
                    row.to_dict(),
                    ensure_ascii=False,
                    default=str
                )

                cursor = connection.execute("""
                    INSERT INTO predictions (
                        timestamp,
                        credit_score,
                        loan_amount,
                        prediction,
                        probability,
                        borrower_data,
                        classification_threshold,
                        user_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    credit_score,
                    float(row["LoanAmount"]),
                    prediction,
                    default_probability,
                    borrower_data,
                    MODEL_THRESHOLD,
                    session["user_id"]
                ))

                prediction_id = cursor.lastrowid

                # -----------------------------------------
                # ADD EVERY BORROWER TO RESULTS
                # -----------------------------------------

                borrower_id = borrower_ids[index]

                individual_explanation = {
                    "summary": explanation_summary,
                    "risk_factors": risk_factors,
                    "positive_factors": positive_factors
                }

                results.append({
                    "id": prediction_id,
                    "borrower": str(borrower_id),
                    "borrower_id": str(borrower_id),
                    "credit_score": credit_score,
                    "loan_amount": float(row["LoanAmount"]),
                    "prediction": prediction,
                    "result": (
                        "High default risk"
                        if prediction == 1
                        else "Low default risk"
                    ),
                    "default_probability": round(
                        default_probability, 4
                    ),
                    "individual_explanation": individual_explanation,
                    "recommendations": recommendations
                })

            # Save all borrower records
            connection.commit()

        # Return all results to frontend
        return jsonify({
            "status": "success",
            "message": "CSV assessment completed successfully.",
            "total": len(results),
            "results": results
        }), 200

    except pd.errors.EmptyDataError:

        return jsonify({
            "status": "error",
            "message": "The CSV file does not contain valid data."
        }), 400

    except Exception:

        app.logger.exception("CSV prediction failed")

        return jsonify({
            "status": "error",
            "message": "CSV assessment failed. Check the backend terminal."
        }), 500

# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------


@app.route("/api/history", methods=["GET"])
@login_required

def prediction_history():

    try:
        return jsonify({
            "status": "success",
            "history": get_history()
        })

    except Exception:
        app.logger.exception("History retrieval failed")

        return jsonify({
            "error": "Could not retrieve prediction history."
        }), 500


@app.route("/api/history", methods=["DELETE"])
@login_required
def clear_prediction_history():

    try:
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute(
                "DELETE FROM predictions WHERE user_id = ?",
                (session["user_id"],)
            )

        return jsonify({
            "status": "success",
            "message": "Your prediction history has been cleared."
        })

    except Exception:
        app.logger.exception("History deletion failed")

        return jsonify({
            "error": "Could not clear prediction history."
        }), 500





# --------------------------------------------------
# PDF DESIGN HELPERS
# --------------------------------------------------

def safe_text(value):

    if value is None or value == "":
        return "N/A"

    return escape(str(value))


def format_money(value):

    try:

        if value is None:
            return "N/A"

        return f"INR {float(value):,.2f}"

    except (TypeError, ValueError):

        return "N/A"


def make_pdf_table(data, col_widths):

    table = Table(
        data,
        colWidths=col_widths,
        repeatRows=1,
        hAlign="LEFT"
    )

    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), GREEN),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, LIGHT_BG]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]

    table.setStyle(TableStyle(commands))

    return table


def get_recommendations(prediction):

    if prediction == 1:

        return [
            "Review the borrower's credit history and repayment records.",
            "Verify income, existing debts and financial obligations.",
            "Carefully assess the affordability of the repayment schedule.",
            "Complete independent verification and human review."
        ]

    return [
        "Continue verifying the borrower's financial information.",
        "Review existing debts and repayment obligations.",
        "Check that the repayment schedule is affordable.",
        "Complete normal documentation and independent checks."
    ]


def add_page_number(canvas, document):

    canvas.saveState()

    width, height = A4

    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)

    canvas.line(
        14 * mm,
        9 * mm,
        width - 14 * mm,
        9 * mm
    )

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)

    canvas.drawString(
        14 * mm,
        5 * mm,
        "CredAI | AI Credit Risk Assessment"
    )

    canvas.drawRightString(
        width - 14 * mm,
        5 * mm,
        f"Page {document.page}"
    )

    canvas.restoreState()


# --------------------------------------------------
# PDF REPORT GENERATION
# --------------------------------------------------

@app.route("/api/report/<int:prediction_id>", methods=["GET"])
@login_required
def generate_report(prediction_id):

    try:

        with sqlite3.connect(DATABASE_PATH) as connection:

            connection.row_factory = sqlite3.Row

            record = connection.execute("""
                SELECT *
                FROM predictions
                WHERE id = ? AND user_id = ?
            """, (prediction_id, session["user_id"])).fetchone()

        if record is None:

            return jsonify({
                "error": "Prediction record not found."
            }), 404

        record = dict(record)

        prediction = int(record["prediction"])
        probability = float(record["probability"]) * 100
        report_threshold = float(
        record.get("classification_threshold")
        if record.get("classification_threshold") is not None
        else MODEL_THRESHOLD
)

        risk_label = (
            "HIGH DEFAULT RISK"
            if prediction == 1
            else "LOW DEFAULT RISK"
        )

        risk_color = RED if prediction == 1 else GREEN

        borrower_data = {}

        if record.get("borrower_data"):

            try:

                borrower_data = json.loads(
                    record["borrower_data"]
                )

            except (TypeError, ValueError):

                borrower_data = {}

        pdf_buffer = BytesIO()

        document = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            rightMargin=14 * mm,
            leftMargin=14 * mm,
            topMargin=7 * mm,
            bottomMargin=10 * mm
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "CredAITitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=21,
            textColor=DARK_GREEN,
            alignment=TA_CENTER,
            spaceAfter=2
        )

        subtitle_style = ParagraphStyle(
            "CredAISubtitle",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=6
        )

        section_style = ParagraphStyle(
            "CredAISection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=11,
            textColor=DARK_GREEN,
            spaceBefore=6,
            spaceAfter=4
        )

        normal_style = ParagraphStyle(
            "CredAINormal",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=DARK
        )

        small_style = ParagraphStyle(
            "CredAISmall",
            parent=styles["Normal"],
            fontSize=7,
            leading=9,
            textColor=MUTED
        )

        risk_style = ParagraphStyle(
            "CredAIRisk",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            textColor=risk_color,
            alignment=TA_CENTER
        )

        story = []

        story.append(Paragraph("CredAI", title_style))

        story.append(
            Paragraph(
                "AI-BASED CREDIT RISK ASSESSMENT REPORT",
                subtitle_style
            )
        )

        story.append(
            Paragraph(
                f"Report ID: {prediction_id}",
                normal_style
            )
        )

        story.append(
            Paragraph(
                "Generated: " + safe_text(record.get("timestamp")),
                normal_style
            )
        )

        story.append(Spacer(1, 5))

        # 1. ASSESSMENT SUMMARY

        story.append(
            Paragraph(
                "1. AI Risk Assessment Summary",
                section_style
            )
        )

        summary_data = [
            ["Assessment", "Result"],
            ["Risk Classification", risk_label],
            ["Default Probability", f"{probability:.2f}%"],
            ["Credit Score", safe_text(record.get("credit_score"))],
            ["Loan Amount", format_money(record.get("loan_amount"))]
        ]

        story.append(
            make_pdf_table(
                summary_data,
                [65 * mm, 105 * mm]
            )
        )

        risk_panel = Table(
            [[Paragraph(risk_label, risk_style)]],
            colWidths=[170 * mm]
        )

        risk_panel.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1),
             LIGHT_GREEN if prediction == 0 else colors.HexColor("#fee2e2")),
            ("BOX", (0, 0), (-1, -1), 0.8, risk_color),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))

        story.append(Spacer(1, 5))
        story.append(risk_panel)

        # 2. BORROWER INFORMATION

        story.append(
            Paragraph(
                "2. Borrower Information",
                section_style
            )
        )

        borrower_fields = [
            ("Age", "Age"),
            ("Income", "Income"),
            ("Education", "Education"),
            ("Employment Type", "EmploymentType"),
            ("Marital Status", "MaritalStatus"),
            ("Dependents", "HasDependents"),
            ("Mortgage", "HasMortgage"),
            ("Co-signer", "HasCoSigner")
        ]

        borrower_rows = [
            ["Field", "Submitted Information"]
        ]

        for label, key in borrower_fields:

            borrower_rows.append([
                label,
                safe_text(borrower_data.get(key))
            ])

        story.append(
            make_pdf_table(
                borrower_rows,
                [65 * mm, 105 * mm]
            )
        )

        # 3. FINANCIAL DETAILS

        story.append(
            Paragraph(
                "3. Financial and Loan Details",
                section_style
            )
        )

        financial_fields = [
            ("Credit Score", "CreditScore"),
            ("Annual Income", "Income"),
            ("Loan Amount", "LoanAmount"),
            ("Interest Rate", "InterestRate"),
            ("Loan Term", "LoanTerm"),
            ("Debt-to-Income Ratio", "DTIRatio"),
            ("Months Employed", "MonthsEmployed"),
            ("Number of Credit Lines", "NumCreditLines"),
            ("Loan Purpose", "LoanPurpose")
        ]

        financial_rows = [
            ["Field", "Submitted Information"]
        ]

        for label, key in financial_fields:

            value = borrower_data.get(key)

            if value is None:
                value = "N/A"

            financial_rows.append([
                label,
                safe_text(value)
            ])

        story.append(
            make_pdf_table(
                financial_rows,
                [65 * mm, 105 * mm]
            )
        )

        # 4. MODEL ASSESSMENT

        story.append(
            Paragraph(
                "4. Model Assessment",
                section_style
            )
        )

        story.append(
            Paragraph(
                f"Predicted Result: <b>{risk_label}</b>",
                normal_style
            )
        )

        story.append(Spacer(1, 3))

        story.append(
            Paragraph(
                f"Estimated Default Probability: "
                f"<b>{probability:.2f}%</b>",
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"Classification Threshold: "
                f"<b>{report_threshold:.4f}</b>",
                normal_style
            )
        )

        # 5. RECOMMENDATIONS

        story.append(
            Paragraph(
                "5. Recommendations",
                section_style
            )
        )

        for item in get_recommendations(prediction):

            story.append(
                Paragraph(
                    "• " + safe_text(item),
                    normal_style
                )
            )

        # DISCLAIMER

        story.append(
            Paragraph(
                "Important Disclaimer",
                section_style
            )
        )

        story.append(
            Paragraph(
                "This report contains an estimate generated by a "
                "machine-learning model using the information provided. "
                "It is intended to support credit-risk assessment only. "
                "It is not a loan approval or rejection decision. "
                "Final decisions require independent verification "
                "and appropriate human review.",
                small_style
            )
        )

        document.build(
            story,
            onFirstPage=add_page_number,
            onLaterPages=add_page_number
        )

        pdf_buffer.seek(0)

        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=(
                f"CredAI_Professional_Report_{prediction_id}.pdf"
            )
        )

    except Exception:

        app.logger.exception(
            "PDF report generation failed"
        )

        return jsonify({
            "error": (
                "Could not generate the PDF report. "
                "Check backend logs."
            )
        }), 500


# --------------------------------------------------
# MODEL PERFORMANCE EVALUATION
# --------------------------------------------------

@app.route("/api/evaluation", methods=["GET"])
def get_model_evaluation():

    try:

        from sklearn.model_selection import train_test_split

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            fbeta_score,
            confusion_matrix,
            classification_report
        )

        dataset_path = os.path.join(
            BASE_DIR,
            "data",
            "Loan_default.csv"
        )

        if not os.path.exists(dataset_path):

            return jsonify({
                "status": "error",
                "message": "Loan_default.csv dataset not found."
            }), 404

        df = pd.read_csv(dataset_path)
        df = df.drop_duplicates().copy()

        X = df.drop(columns=["LoanID", "Default"])
        y = df["Default"]

        # Use exactly the same split as train_model.py
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=0.25,
            random_state=42,
            stratify=y_temp
        )

        # Get probabilities and apply selected threshold
        probabilities = model.predict_proba(X_test)[:, 1]

        classes = list(model.classes_)

        if 1 not in classes:

            return jsonify({
                "status": "error",
                "message": "Default class not found in model."
            }), 500

        default_index = classes.index(1)

        probabilities = model.predict_proba(X_test)[:, default_index]

        predictions = (
            probabilities >= MODEL_THRESHOLD
        ).astype(int)

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        f2 = fbeta_score(
            y_test,
            predictions,
            beta=2,
            zero_division=0
        )

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Non-default", "Default"],
            output_dict=True,
            zero_division=0
        )

        return jsonify({
            "status": "success",
            "model": "Optimized Random Forest",
            "threshold": round(float(MODEL_THRESHOLD), 4),
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "f2_score": round(float(f2), 4),
            "confusion_matrix": cm.tolist(),
            "test_samples": int(len(y_test)),
            "classification_report": report
        })

    except Exception:

        app.logger.exception("Model evaluation failed")

        return jsonify({
            "status": "error",
            "message": (
                "Model evaluation failed. Check backend logs."
            )
        }), 500

# --------------------------------------------------
# MODEL EXPLAINABILITY - FEATURE IMPORTANCE
# --------------------------------------------------


# --------------------------------------------------
# MODEL EXPLAINABILITY - FEATURE IMPORTANCE
# --------------------------------------------------

@app.route("/api/feature-importance", methods=["GET"])
def get_feature_importance():

    try:
        # Get the final Random Forest estimator
        classifier = model

        if hasattr(model, "steps"):
            classifier = model.steps[-1][1]

        # Check whether feature importance is supported
        if not hasattr(classifier, "feature_importances_"):
            return jsonify({
                "status": "error",
                "message": "Feature importance is not available."
            }), 500

        importances = classifier.feature_importances_

        # Get feature names after preprocessing
        try:
            feature_names = (
                model[:-1].get_feature_names_out()
                if hasattr(model, "steps")
                else model.feature_names_in_
            )

        except Exception:
            feature_names = [
                f"Feature {i + 1}"
                for i in range(len(importances))
            ]

        # Ensure feature names and importance values match
        if len(feature_names) != len(importances):
            feature_names = [
                f"Feature {i + 1}"
                for i in range(len(importances))
            ]

        results = []

        # Prepare feature importance results
        for name, importance in zip(
            feature_names,
            importances
        ):
            # Remove preprocessing prefixes
            clean_name = str(name).split("__")[-1]

            results.append({
                "feature": clean_name,
                "importance": round(
                    float(importance) * 100, 4
                )
            })

        # Sort from highest to lowest importance
        results.sort(
            key=lambda item: item["importance"],
            reverse=True
        )

        return jsonify({
            "status": "success",
            "model": "Optimized Random Forest",
            "description": (
                "Global model-level feature importance. "
                "These values show relative contribution "
                "within the trained model, not causation."
            ),
            "features": results[:15]
        })

    except Exception:
        app.logger.exception(
            "Feature importance retrieval failed"
        )

        return jsonify({
            "status": "error",
            "message": "Could not retrieve feature importance."
        }), 500

# --------------------------------------------------
# START CRED AI BACKEND
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
