
"""
CredAI - Flask Backend

Supports:
- Local optimized Random Forest model
- Registration and login
- Single borrower prediction
- CSV batch prediction (up to 10,000 rows)
- Per-user prediction history
- PDF reports
- Model evaluation
- Feature importance

Required files:
- loan_default_model_optimized.joblib
- model_threshold.json (optional)
- data/Loan_default.csv

Environment variables:
- CREDAI_SECRET_KEY
- FRONTEND_ORIGIN
"""

import os
import re
import json
import sqlite3
import logging
from io import BytesIO
from datetime import datetime
from functools import wraps
from xml.sax.saxutils import escape

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
app.config["SECRET_KEY"] = os.environ.get("CREDAI_SECRET_KEY", "dev-only-change-this-secret")

# Secure cookies are enabled on deployed HTTPS services.
is_production = os.environ.get("RENDER", "").lower() == "true"
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="None" if is_production else "Lax",
    SESSION_COOKIE_SECURE=is_production,
    SESSION_COOKIE_PATH="/",
    SESSION_PERMANENT=False,
)

frontend_origin = os.environ.get("FRONTEND_ORIGIN", "http://127.0.0.1:5500").rstrip("/")
CORS(
    app,
    supports_credentials=True,
    origins=[
        frontend_origin,
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
    ],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "credai_history.db")
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "loan_default_model_optimized.joblib")
LOCAL_THRESHOLD_PATH = os.path.join(BASE_DIR, "model_threshold.json")
DATASET_PATH = os.path.join(BASE_DIR, "data", "Loan_default.csv")

# --------------------------------------------------
# LOAD LOCAL MODEL
# --------------------------------------------------

def load_model_file():
   

    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {LOCAL_MODEL_PATH}"
        )

    app.logger.info("Loading local CredAI model...")
    return LOCAL_MODEL_PATH


def load_threshold():
    """Load classification threshold from local JSON."""

    if os.path.exists(LOCAL_THRESHOLD_PATH):
        with open(LOCAL_THRESHOLD_PATH, "r", encoding="utf-8") as file:
            value = json.load(file).get("threshold", 0.5)
            value = float(value)

            if not 0 < value < 1:
                raise ValueError(
                    "Model threshold must be between 0 and 1."
                )

            return value

    app.logger.warning(
        "Threshold file not found. Using default threshold 0.5"
    )
    return 0.5


# --------------------------------------------------
# INITIALIZE LOCAL MODEL
# --------------------------------------------------

MODEL_PATH = load_model_file()

model = joblib.load(MODEL_PATH)

MODEL_THRESHOLD = load_threshold()

if not hasattr(model, "predict_proba"):
    raise RuntimeError(
        "The loaded model does not support predict_proba()."
    )

MODEL_COLUMNS = (
    list(model.feature_names_in_)
    if hasattr(model, "feature_names_in_")
    else None
)

app.logger.info("CredAI model loaded successfully.")
app.logger.info("Model: Optimized Random Forest")



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
# DATABASE
# --------------------------------------------------

def init_database():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                credit_score INTEGER,
                loan_amount REAL,
                prediction INTEGER NOT NULL,
                probability REAL NOT NULL,
                borrower_data TEXT,
                classification_threshold REAL,
                user_id INTEGER
            )
        """)

        columns = {
            row[1] for row in connection.execute(
                "PRAGMA table_info(predictions)"
            ).fetchall()
        }

        migrations = {
            "borrower_data": "TEXT",
            "classification_threshold": "REAL",
            "user_id": "INTEGER",
        }
        for column, column_type in migrations.items():
            if column not in columns:
                connection.execute(
                    f"ALTER TABLE predictions ADD COLUMN {column} {column_type}"
                )
        connection.commit()


init_database()


def current_timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def get_default_probability(probability_row):
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Default class (1) is not present in the model.")
    return float(probability_row[classes.index(1)])


def validate_borrower_frame(frame):
    if not MODEL_COLUMNS:
        raise RuntimeError("Model input columns could not be determined.")

    missing = [column for column in MODEL_COLUMNS if column not in frame.columns]
    if missing:
        return None, {
            "status": "error",
            "message": "Required model columns are missing.",
            "missing_columns": missing,
        }, 400

    borrower_frame = frame[MODEL_COLUMNS].copy()
    borrower_frame = borrower_frame.replace(r"^\s*$", pd.NA, regex=True)

    if borrower_frame.isna().any().any():
        bad_rows = (
            borrower_frame.index[borrower_frame.isna().any(axis=1)] + 2
        ).tolist()
        return None, {
            "status": "error",
            "message": "Borrower information contains missing values.",
            "csv_rows": bad_rows[:30],
            "total_invalid_rows": len(bad_rows),
        }, 400

    numeric_columns = [
        "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
        "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"
    ]
    for column in numeric_columns:
        if column in borrower_frame.columns:
            borrower_frame[column] = pd.to_numeric(
                borrower_frame[column], errors="coerce"
            )

    if borrower_frame.isna().any().any():
        return None, {
            "status": "error",
            "message": "Invalid or non-numeric values found in numeric columns."
        }, 400

    for column in numeric_columns:
        if column in borrower_frame.columns:
            values = borrower_frame[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                return None, {
                    "status": "error",
                    "message": f"Invalid numeric values in {column}."
                }, 400

    if "DTIRatio" in borrower_frame.columns:
        if not borrower_frame["DTIRatio"].between(0, 1).all():
            return None, {
                "status": "error",
                "message": "DTIRatio must be between 0 and 1."
            }, 400

    return borrower_frame, None, None


def explain_borrower(row, prediction):
    credit_score = float(row.get("CreditScore", 0))
    dti_ratio = float(row.get("DTIRatio", 0))
    interest_rate = float(row.get("InterestRate", 0))
    months_employed = float(row.get("MonthsEmployed", 0))

    risk_factors = []
    positive_factors = []

    if credit_score < 600:
        risk_factors.append("Low credit score")
    elif credit_score >= 700:
        positive_factors.append("Good credit score")

    if dti_ratio > 0.4:
        risk_factors.append("High debt-to-income ratio")
    elif dti_ratio <= 0.3:
        positive_factors.append("Manageable debt-to-income ratio")

    if interest_rate > 15:
        risk_factors.append("High interest rate")

    if months_employed < 12:
        risk_factors.append("Short employment history")
    elif months_employed >= 24:
        positive_factors.append("Stable employment history")

    if prediction == 1:
        if not risk_factors:
            risk_factors.append(
                "Elevated risk identified from the overall borrower profile."
            )
        summary = (
            "The model classified this borrower as high default risk. "
            "Review the identified indicators as part of human assessment."
        )
        recommendations = [
            "Verify financial details and repayment records.",
            "Review existing debt and loan affordability.",
            "Complete independent verification and human review.",
        ]
        if credit_score < 600:
            recommendations.append("Review credit history and repayment consistency.")
        if dti_ratio > 0.4:
            recommendations.append("Assess whether outstanding debt can be reduced.")
    else:
        summary = (
            "The model classified this borrower as low default risk. "
            "Normal verification is still required."
        )
        recommendations = [
            "Continue making repayments on time.",
            "Maintain manageable outstanding debt.",
            "Monitor loan affordability.",
        ]

    return {
        "summary": summary,
        "risk_factors": risk_factors,
        "positive_factors": positive_factors,
    }, recommendations


def save_prediction(timestamp, row, prediction, probability, user_id):
    borrower_json = json.dumps(row.to_dict(), ensure_ascii=False, default=str)
    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.execute("""
            INSERT INTO predictions (
                timestamp, credit_score, loan_amount, prediction,
                probability, borrower_data, classification_threshold, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            row.get("CreditScore"),
            row.get("LoanAmount"),
            int(prediction),
            float(probability),
            borrower_json,
            MODEL_THRESHOLD,
            user_id,
        ))
        return cursor.lastrowid


# --------------------------------------------------
# GENERAL / AUTH ROUTES
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Credit Risk Assessment API is running!",
        "project": "AI-Based Credit Risk Assessment and Loan Default Prediction System",
        "model": "Optimized Random Forest",
        "threshold": MODEL_THRESHOLD,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "success",
        "message": "Backend is working correctly",
        "model_loaded": model is not None,
        "model": "Optimized Random Forest",
        "classification_threshold": MODEL_THRESHOLD,
        "database_connected": os.path.exists(DATABASE_PATH),
    })


@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Please provide registration details."}), 400

    full_name = str(data.get("full_name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not full_name or not email or not password:
        return jsonify({"status": "error", "message": "All fields are required."}), 400
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return jsonify({"status": "error", "message": "Please enter a valid email address."}), 400
    if len(password) < 8:
        return jsonify({"status": "error", "message": "Password must contain at least 8 characters."}), 400

    try:
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute("""
                INSERT INTO users (full_name, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
            """, (full_name, email, generate_password_hash(password), current_timestamp()))
        return jsonify({"status": "success", "message": "Account created successfully."}), 201
    except sqlite3.IntegrityError:
        return jsonify({"status": "error", "message": "An account with this email already exists."}), 409
    except Exception:
        app.logger.exception("Registration failed")
        return jsonify({"status": "error", "message": "Registration failed. Please try again."}), 500


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Please provide login details."}), 400

    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required."}), 400

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        user = connection.execute(
            "SELECT id, full_name, email, password_hash FROM users WHERE email = ?",
            (email,)
        ).fetchone()

    if user is None or not check_password_hash(user["password_hash"], password):
        return jsonify({"status": "error", "message": "Invalid email or password."}), 401

    session.clear()
    session["user_id"] = user["id"]
    session["user_name"] = user["full_name"]
    return jsonify({
        "status": "success",
        "message": "Login successful.",
        "user": {
            "id": user["id"],
            "full_name": user["full_name"],
            "email": user["email"],
        },
    })


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "success", "message": "Logged out successfully."})


@app.route("/api/session", methods=["GET"])
def check_session():
    if "user_id" not in session:
        return jsonify({"authenticated": False}), 401
    return jsonify({
        "authenticated": True,
        "user": {
            "id": session["user_id"],
            "full_name": session.get("user_name", ""),
        },
    })


# --------------------------------------------------
# SINGLE PREDICTION
# --------------------------------------------------

@app.route("/api/predict", methods=["POST"])
@login_required
def predict():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Please send borrower details as JSON."}), 400

        frame, error, status = validate_borrower_frame(pd.DataFrame([data]))
        if error:
            return jsonify(error), status

        probabilities = model.predict_proba(frame)[0]
        probability = get_default_probability(probabilities)
        prediction = int(probability >= MODEL_THRESHOLD)
        explanation, recommendations = explain_borrower(frame.iloc[0], prediction)
        prediction_id = save_prediction(
            current_timestamp(), frame.iloc[0], prediction, probability, session["user_id"]
        )

        return jsonify({
            "status": "success",
            "id": prediction_id,
            "prediction": prediction,
            "result": "High default risk" if prediction else "Low default risk",
            "default_probability": round(probability, 4),
            "classification_threshold": round(MODEL_THRESHOLD, 4),
            "individual_explanation": explanation,
            "recommendations": recommendations,
            "note": "This is a machine-learning estimate, not a loan approval or rejection decision.",
        })
    except (TypeError, ValueError) as error:
        return jsonify({"error": "Invalid borrower details.", "details": str(error)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check backend logs."}), 500


# --------------------------------------------------
# CSV BATCH PREDICTION
# --------------------------------------------------

@app.route("/api/predict-csv", methods=["POST"])
@login_required
def predict_csv():
    try:
        uploaded_file = request.files.get("file")
        if uploaded_file is None or not uploaded_file.filename:
            return jsonify({"status": "error", "message": "Please select a CSV file."}), 400
        if not uploaded_file.filename.lower().endswith(".csv"):
            return jsonify({"status": "error", "message": "Only CSV files are supported."}), 400

        file_bytes = uploaded_file.read()
        if not file_bytes:
            return jsonify({"status": "error", "message": "The uploaded CSV file is empty."}), 400

        frame = pd.read_csv(BytesIO(file_bytes))
        frame.columns = [str(col).strip().lstrip("\ufeff") for col in frame.columns]

        if frame.empty:
            return jsonify({"status": "error", "message": "No borrower records found."}), 400

        max_records = 10000
        if len(frame) > max_records:
            return jsonify({
                "status": "error",
                "message": f"Maximum {max_records} borrowers allowed per upload.",
                "total_records": len(frame),
            }), 400

        borrower_ids = (
            frame["LoanID"].fillna("").astype(str).tolist()
            if "LoanID" in frame.columns
            else [f"Borrower {i + 1}" for i in range(len(frame))]
        )

        borrower_frame, error, status = validate_borrower_frame(frame)
        if error:
            return jsonify(error), status

        if 1 not in list(model.classes_):
            return jsonify({"status": "error", "message": "Default class not found in model."}), 500

        batch_size = 1000
        results = []
        timestamp = current_timestamp()
        user_id = session["user_id"]

        for start in range(0, len(borrower_frame), batch_size):
            batch = borrower_frame.iloc[start:start + batch_size]
            batch_probabilities = model.predict_proba(batch)

            for offset, (_, row) in enumerate(batch.iterrows()):
                probability = get_default_probability(batch_probabilities[offset])
                prediction = int(probability >= MODEL_THRESHOLD)
                explanation, recommendations = explain_borrower(row, prediction)
                prediction_id = save_prediction(
                    timestamp, row, prediction, probability, user_id
                )
                original_index = start + offset
                borrower_id = str(borrower_ids[original_index])

                results.append({
                    "id": prediction_id,
                    "borrower": borrower_id,
                    "borrower_id": borrower_id,
                    "credit_score": float(row["CreditScore"]),
                    "loan_amount": float(row["LoanAmount"]),
                    "prediction": prediction,
                    "result": "High default risk" if prediction else "Low default risk",
                    "default_probability": round(probability, 4),
                    "individual_explanation": explanation,
                    "recommendations": recommendations,
                })

            app.logger.info("Processed %s/%s borrowers", min(start + batch_size, len(borrower_frame)), len(borrower_frame))

        return jsonify({
            "status": "success",
            "message": "CSV assessment completed successfully.",
            "total": len(results),
            "processed": len(results),
            "batch_size": batch_size,
            "results": results,
        })
    except pd.errors.EmptyDataError:
        return jsonify({"status": "error", "message": "The CSV file does not contain valid data."}), 400
    except Exception:
        app.logger.exception("CSV prediction failed")
        return jsonify({"status": "error", "message": "CSV assessment failed. Check backend logs."}), 500


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

@app.route("/api/history", methods=["GET"])
@login_required
def prediction_history():
    try:
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("""
                SELECT id, timestamp, credit_score, loan_amount, prediction, probability
                FROM predictions
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT 100
            """, (session["user_id"],)).fetchall()
        return jsonify({"status": "success", "history": [dict(row) for row in rows]})
    except Exception:
        app.logger.exception("History retrieval failed")
        return jsonify({"error": "Could not retrieve prediction history."}), 500


@app.route("/api/history", methods=["DELETE"])
@login_required
def clear_prediction_history():
    try:
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute(
                "DELETE FROM predictions WHERE user_id = ?",
                (session["user_id"],)
            )
        return jsonify({"status": "success", "message": "Your prediction history has been cleared."})
    except Exception:
        app.logger.exception("History deletion failed")
        return jsonify({"error": "Could not clear prediction history."}), 500


# --------------------------------------------------
# PDF REPORT HELPERS
# --------------------------------------------------

def safe_text(value):
    if value is None or value == "":
        return "N/A"
    return escape(str(value))


def format_money(value):
    try:
        return "N/A" if value is None else f"INR {float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def make_pdf_table(data, col_widths):
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), GREEN),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def add_page_number(canvas, document):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(BORDER)
    canvas.line(14 * mm, 9 * mm, width - 14 * mm, 9 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(14 * mm, 5 * mm, "CredAI | AI Credit Risk Assessment")
    canvas.drawRightString(width - 14 * mm, 5 * mm, f"Page {document.page}")
    canvas.restoreState()


@app.route("/api/report/<int:prediction_id>", methods=["GET"])
@login_required
def generate_report(prediction_id):
    try:
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.row_factory = sqlite3.Row
            record = connection.execute(
                "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
                (prediction_id, session["user_id"])
            ).fetchone()

        if record is None:
            return jsonify({"error": "Prediction record not found."}), 404

        record = dict(record)
        prediction = int(record["prediction"])
        probability = float(record["probability"]) * 100
        threshold = record.get("classification_threshold")
        threshold = float(threshold) if threshold is not None else MODEL_THRESHOLD
        risk_label = "HIGH DEFAULT RISK" if prediction else "LOW DEFAULT RISK"
        risk_color = RED if prediction else GREEN

        try:
            borrower_data = json.loads(record.get("borrower_data") or "{}")
        except (TypeError, ValueError):
            borrower_data = {}

        buffer = BytesIO()
        document = SimpleDocTemplate(
            buffer, pagesize=A4, rightMargin=14 * mm, leftMargin=14 * mm,
            topMargin=10 * mm, bottomMargin=13 * mm
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CredAITitle", parent=styles["Title"], fontName="Helvetica-Bold",
            fontSize=19, leading=21, textColor=DARK_GREEN,
            alignment=TA_CENTER, spaceAfter=3
        )
        subtitle_style = ParagraphStyle(
            "CredAISubtitle", parent=styles["Normal"], fontSize=8,
            leading=10, textColor=MUTED, alignment=TA_CENTER, spaceAfter=7
        )
        section_style = ParagraphStyle(
            "CredAISection", parent=styles["Heading2"], fontName="Helvetica-Bold",
            fontSize=10, leading=12, textColor=DARK_GREEN, spaceBefore=7, spaceAfter=4
        )
        normal_style = ParagraphStyle(
            "CredAINormal", parent=styles["Normal"], fontSize=8, leading=10, textColor=DARK
        )
        small_style = ParagraphStyle(
            "CredAISmall", parent=styles["Normal"], fontSize=7, leading=9, textColor=MUTED
        )
        risk_style = ParagraphStyle(
            "CredAIRisk", parent=styles["Heading2"], fontName="Helvetica-Bold",
            fontSize=12, leading=14, textColor=risk_color, alignment=TA_CENTER
        )

        story = [
            Paragraph("CredAI", title_style),
            Paragraph("AI-BASED CREDIT RISK ASSESSMENT REPORT", subtitle_style),
            Paragraph(f"Report ID: {prediction_id}", normal_style),
            Paragraph(f"Generated: {safe_text(record.get('timestamp'))}", normal_style),
            Spacer(1, 6),
            Paragraph("1. AI Risk Assessment Summary", section_style),
            make_pdf_table([
                ["Assessment", "Result"],
                ["Risk Classification", risk_label],
                ["Default Probability", f"{probability:.2f}%"],
                ["Credit Score", safe_text(record.get("credit_score"))],
                ["Loan Amount", format_money(record.get("loan_amount"))],
            ], [65 * mm, 105 * mm]),
            Spacer(1, 6),
        ]

        risk_panel = Table([[Paragraph(risk_label, risk_style)]], colWidths=[170 * mm])
        risk_panel.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREEN if prediction == 0 else colors.HexColor("#fee2e2")),
            ("BOX", (0, 0), (-1, -1), 0.8, risk_color),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(risk_panel)

        borrower_fields = [
            ("Age", "Age"), ("Income", "Income"), ("Education", "Education"),
            ("Employment Type", "EmploymentType"), ("Marital Status", "MaritalStatus"),
            ("Dependents", "HasDependents"), ("Mortgage", "HasMortgage"),
            ("Co-signer", "HasCoSigner"),
        ]
        story.extend([
            Paragraph("2. Borrower Information", section_style),
            make_pdf_table(
                [["Field", "Submitted Information"]] +
                [[label, safe_text(borrower_data.get(key))] for label, key in borrower_fields],
                [65 * mm, 105 * mm]
            ),
        ])

        financial_fields = [
            ("Credit Score", "CreditScore"), ("Annual Income", "Income"),
            ("Loan Amount", "LoanAmount"), ("Interest Rate", "InterestRate"),
            ("Loan Term", "LoanTerm"), ("Debt-to-Income Ratio", "DTIRatio"),
            ("Months Employed", "MonthsEmployed"),
            ("Number of Credit Lines", "NumCreditLines"), ("Loan Purpose", "LoanPurpose"),
        ]
        story.extend([
            Paragraph("3. Financial and Loan Details", section_style),
            make_pdf_table(
                [["Field", "Submitted Information"]] +
                [[label, safe_text(borrower_data.get(key))] for label, key in financial_fields],
                [65 * mm, 105 * mm]
            ),
            Paragraph("4. Model Assessment", section_style),
            Paragraph(f"Predicted Result: <b>{risk_label}</b>", normal_style),
            Paragraph(f"Estimated Default Probability: <b>{probability:.2f}%</b>", normal_style),
            Paragraph(f"Classification Threshold: <b>{threshold:.4f}</b>", normal_style),
            Paragraph("5. Recommendations", section_style),
        ])

        _, recommendations = explain_borrower(
            pd.Series(borrower_data), prediction
        )
        for item in recommendations:
            story.append(Paragraph("• " + safe_text(item), normal_style))

        story.extend([
            Paragraph("Important Disclaimer", section_style),
            Paragraph(
                "This report contains an estimate generated by a machine-learning "
                "model using the information provided. It supports credit-risk "
                "assessment only and is not a loan approval or rejection decision. "
                "Final decisions require independent verification and appropriate "
                "human review.",
                small_style
            ),
        ])

        document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"CredAI_Professional_Report_{prediction_id}.pdf"
        )
    except Exception:
        app.logger.exception("PDF report generation failed")
        return jsonify({"error": "Could not generate the PDF report. Check backend logs."}), 500


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

@app.route("/api/evaluation", methods=["GET"])
def get_model_evaluation():
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, fbeta_score, confusion_matrix, classification_report
        )

        if not os.path.exists(DATASET_PATH):
            return jsonify({
                "status": "error",
                "message": "Loan_default.csv dataset not found on the backend."
            }), 404

        frame = pd.read_csv(DATASET_PATH).drop_duplicates().copy()
        X = frame.drop(columns=["LoanID", "Default"])
        y = frame["Default"]

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        classes = list(model.classes_)
        if 1 not in classes:
            return jsonify({"status": "error", "message": "Default class not found in model."}), 500

        probabilities = model.predict_proba(X_test)[:, classes.index(1)]
        predictions = (probabilities >= MODEL_THRESHOLD).astype(int)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        f2 = fbeta_score(y_test, predictions, beta=2, zero_division=0)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        report = classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["Non-default", "Default"],
            output_dict=True, zero_division=0
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
            "confusion_matrix": matrix.tolist(),
            "test_samples": int(len(y_test)),
            "classification_report": report,
        })
    except Exception:
        app.logger.exception("Model evaluation failed")
        return jsonify({"status": "error", "message": "Model evaluation failed. Check backend logs."}), 500


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

@app.route("/api/feature-importance", methods=["GET"])
def get_feature_importance():
    try:
        classifier = model
        if hasattr(model, "steps"):
            classifier = model.steps[-1][1]

        if not hasattr(classifier, "feature_importances_"):
            return jsonify({
                "status": "error",
                "message": "Feature importance is not available."
            }), 500

        importances = classifier.feature_importances_

        try:
            if hasattr(model, "steps"):
                feature_names = model[:-1].get_feature_names_out()
            elif hasattr(model, "feature_names_in_"):
                feature_names = model.feature_names_in_
            else:
                feature_names = [f"Feature {i + 1}" for i in range(len(importances))]
        except Exception:
            feature_names = [f"Feature {i + 1}" for i in range(len(importances))]

        if len(feature_names) != len(importances):
            feature_names = [f"Feature {i + 1}" for i in range(len(importances))]

        features = [
            {
                "feature": str(name).split("__")[-1],
                "importance": round(float(value) * 100, 4),
            }
            for name, value in zip(feature_names, importances)
        ]
        features.sort(key=lambda item: item["importance"], reverse=True)

        return jsonify({
            "status": "success",
            "model": "Optimized Random Forest",
            "description": (
                "Global model-level feature importance. These values show "
                "relative contribution within the trained model, not causation."
            ),
            "features": features[:15],
        })
    except Exception:
        app.logger.exception("Feature importance retrieval failed")
        return jsonify({
            "status": "error",
            "message": "Could not retrieve feature importance."
        }), 500


# --------------------------------------------------
# START SERVER
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
