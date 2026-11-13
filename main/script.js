/* =========================================
   CREDAI — CREDIT RISK DASHBOARD
   Complete updated script.js
========================================= */

const API_BASE = "http://127.0.0.1:5000";

const HEALTH_URL = `${API_BASE}/api/health`;
const PREDICT_URL = `${API_BASE}/api/predict`;
const HISTORY_URL = `${API_BASE}/api/history`;
const FEATURE_IMPORTANCE_URL =
    `${API_BASE}/api/feature-importance`;

let predictionHistory = [];

/* =========================================
   HELPER FUNCTIONS
========================================= */

function getElement(id) {
    return document.getElementById(id);
}

function setText(id, value) {
    const element = getElement(id);

    if (element) {
        element.textContent = value;
    }
}

function formatCurrency(value) {
    const amount = Number(value);

    if (!Number.isFinite(amount)) return "—";

    return new Intl.NumberFormat("en-IN", {
        style: "currency",
        currency: "INR",
        maximumFractionDigits: 0
    }).format(amount);
}

function formatDate(value) {
    if (!value) return "—";

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
        return String(value);
    }

    return date.toLocaleString("en-IN", {
        day: "2-digit",
        month: "short",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit"
    });
}

function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, character => {
        const characters = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        };

        return characters[character];
    });
}

function showMessage(message, type = "") {
    const element = getElement("formMessage");

    if (!element) return;

    element.textContent = message;
    element.className = "form-message";

    if (type) element.classList.add(type);
}

/* =========================================
   RISK CLASSIFICATION
========================================= */

function getRiskType(record) {
    const result = String(
        record.result ??
        record.risk_result ??
        record.risk_level ??
        record.risk ??
        record.prediction_label ??
        ""
    ).toLowerCase();

    const prediction = record.prediction;

    if (result.includes("low") || result.includes("safe")) {
        return "low";
    }

    if (result.includes("high") || result.includes("default")) {
        return "high";
    }

    if (Number(prediction) === 1) return "high";
    if (Number(prediction) === 0) return "low";

    return "unknown";
}

function getRiskLabel(record) {
    const risk = getRiskType(record);

    if (risk === "low") return "Low Risk";
    if (risk === "high") return "High Risk";

    return "Unknown";
}

function getProbability(record) {
    let probability = Number(
        record.default_probability ??
        record.probability ??
        record.defaultProbability
    );

    if (!Number.isFinite(probability)) return null;

    if (probability >= 0 && probability <= 1) {
        probability *= 100;
    }

    return Math.max(0, Math.min(100, probability));
}

/* =========================================
   HISTORY RESPONSE
========================================= */

function getHistoryArray(data) {
    if (Array.isArray(data)) return data;

    if (data && Array.isArray(data.history)) return data.history;
    if (data && Array.isArray(data.predictions)) return data.predictions;
    if (data && Array.isArray(data.records)) return data.records;
    if (data && Array.isArray(data.data)) return data.data;

    return [];
}

/* =========================================
   API HEALTH CHECK
========================================= */

async function checkAPI() {
    const status = getElement("apiStatus");
    const modelStatus = getElement("modelStatus");
    const apiMessage = getElement("apiMessage");
    const liveDot = document.querySelector(".live-dot");

    try {
        const response = await fetch(HEALTH_URL);

        if (!response.ok) {
            throw new Error("Backend health check failed.");
        }

        const data = await response.json();

        if (status) status.textContent = "API Connected";

        if (modelStatus) {
            modelStatus.textContent = data.model_loaded
                ? "Model loaded"
                : "API running";
        }

        if (apiMessage) {
            apiMessage.textContent = data.model_loaded
                ? "Machine learning model is ready"
                : "Check model loading status";
        }

        if (liveDot) {
            liveDot.classList.remove("disconnected");
            liveDot.classList.add("connected");
        }

    } catch (error) {
        if (status) status.textContent = "API Disconnected";
        if (modelStatus) modelStatus.textContent = "Offline";
        if (apiMessage) apiMessage.textContent = "Start your Flask backend";

        if (liveDot) {
            liveDot.classList.remove("connected");
            liveDot.classList.add("disconnected");
        }

        console.error("API connection error:", error);
    }
}

/* =========================================
   FORM DATA
========================================= */

function getBorrowerData(form) {
    const formData = new FormData(form);
    const borrower = {};

    for (const [key, value] of formData.entries()) {
        borrower[key] = value.trim();
    }

    const numericFields = [
        "Age",
        "Income",
        "LoanAmount",
        "CreditScore",
        "MonthsEmployed",
        "NumCreditLines",
        "InterestRate",
        "LoanTerm",
        "DTIRatio"
    ];

    for (const field of numericFields) {
        if (borrower[field] !== undefined) {
            borrower[field] = Number(borrower[field]);
        }
    }

    return borrower;
}

/* =========================================
   RISK FACTORS & RECOMMENDATIONS
========================================= */

function generateRiskInsights(borrower) {
    const factors = [];
    const recommendations = [];

    const creditScore = Number(borrower.CreditScore);
    const dti = Number(borrower.DTIRatio);
    const monthsEmployed = Number(borrower.MonthsEmployed);
    const loanAmount = Number(borrower.LoanAmount);
    const income = Number(borrower.Income);
    const interestRate = Number(borrower.InterestRate);
    const creditLines = Number(borrower.NumCreditLines);

    if (Number.isFinite(creditScore) && creditScore < 600) {
        factors.push("Credit score is below 600.");

        recommendations.push(
            "Review the credit report and repayment history."
        );
    }

    if (Number.isFinite(dti) && dti >= 0.40) {
        factors.push("Debt-to-income ratio is 40% or higher.");

        recommendations.push(
            "Review existing debt obligations and repayment affordability."
        );
    }

    if (
        Number.isFinite(monthsEmployed) &&
        monthsEmployed < 12
    ) {
        factors.push("Employment duration is less than 12 months.");

        recommendations.push(
            "Verify employment and income stability."
        );
    }

    if (
        Number.isFinite(loanAmount) &&
        Number.isFinite(income) &&
        income > 0 &&
        loanAmount > income
    ) {
        factors.push(
            "Loan amount is higher than the reported annual income."
        );

        recommendations.push(
            "Review the requested loan amount and repayment schedule."
        );
    }

    if (
        Number.isFinite(interestRate) &&
        interestRate >= 15
    ) {
        factors.push("Interest rate is 15% or higher.");

        recommendations.push(
            "Review the interest burden and total repayment cost."
        );
    }

    if (
        String(borrower.EmploymentType || "").toLowerCase() === "unemployed"
    ) {
        factors.push("Employment type is listed as unemployed.");

        recommendations.push(
            "Verify current income sources and repayment capacity."
        );
    }

    if (
        Number.isFinite(creditLines) &&
        creditLines >= 8
    ) {
        factors.push("Borrower has 8 or more existing credit lines.");

        recommendations.push(
            "Review outstanding balances across existing credit lines."
        );
    }

    if (factors.length === 0) {
        factors.push(
            "No configured screening indicators were triggered."
        );

        recommendations.push(
            "Continue with normal financial verification and assessment."
        );
    }

    return { factors, recommendations };
}

function displayRiskInsights(borrower) {
    const section = getElement("riskFactorsSection");
    const factorsList = getElement("riskFactorsList");
    const recommendationsList = getElement("recommendationsList");

    if (!section || !factorsList || !recommendationsList) {
        console.warn("Risk insights HTML elements were not found.");
        return;
    }

    const insights = generateRiskInsights(borrower);

    factorsList.innerHTML = insights.factors
        .map(factor => `<li>${escapeHTML(factor)}</li>`)
        .join("");

    recommendationsList.innerHTML = insights.recommendations
        .map(item => `<li>${escapeHTML(item)}</li>`)
        .join("");

    section.hidden = false;
}

/* =========================================
   PREDICTION SUBMISSION
========================================= */

async function handlePrediction(event) {
    event.preventDefault();

    const form = event.currentTarget;
    const button = getElement("submitButton");
    const resultCard = getElement("result");

    if (!form.checkValidity()) {
        form.reportValidity();
        return;
    }

    const borrower = getBorrowerData(form);

    if (
        !Number.isFinite(borrower.DTIRatio) ||
        borrower.DTIRatio < 0 ||
        borrower.DTIRatio > 1
    ) {
        showMessage(
            "Debt-to-income ratio must be between 0 and 1.",
            "error"
        );
        return;
    }

    if (button) {
        button.disabled = true;
        button.innerHTML = "<span>Assessing...</span><span>⌛</span>";
    }

    showMessage("Analyzing borrower information...");

    try {
        const response = await fetch(PREDICT_URL, {
            method: "POST",
            credentials: "include",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(borrower)
        });

        const data = await response.json();

        if (!response.ok || data.status === "error") {
            throw new Error(
                data.message ||
                data.error ||
                "Prediction failed. Check the backend."
            );
        }

        if (
            data.prediction === undefined &&
            data.result === undefined
        ) {
            throw new Error(
                "The API response does not contain a prediction result."
            );
        }

        displayPrediction(data, borrower);

        showMessage("Prediction completed successfully.", "success");

        if (resultCard) {
            resultCard.hidden = false;

            resultCard.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        }

        await loadHistory();

    } catch (error) {
        console.error("Prediction error:", error);
        showMessage(error.message, "error");

    } finally {
        if (button) {
            button.disabled = false;
            button.innerHTML =
                "<span>Assess credit risk</span><span>↗</span>";
        }
    }
}

/* =========================================
   DISPLAY PREDICTION RESULT
========================================= */

function displayPrediction(data, borrower = {}) {
    const resultCard = getElement("result");
    const riskResult = getElement("riskResult");
    const riskBadge = getElement("riskBadge");
    const probabilityResult = getElement("probabilityResult");
    const riskExplanation = getElement("riskExplanation");
    const riskMeterFill = getElement("riskMeterFill");
    const resultNote = getElement("resultNote");

    const risk = getRiskType(data);
    const label = data.result || getRiskLabel(data);
    const probability = getProbability(data);

    if (riskResult) riskResult.textContent = label;

    if (riskBadge) {
        riskBadge.textContent = getRiskLabel(data);
        riskBadge.className = "risk-badge";

        if (risk === "low") riskBadge.classList.add("low-risk");
        if (risk === "high") riskBadge.classList.add("high-risk");
    }

    if (probabilityResult) {
        probabilityResult.textContent =
            probability === null
                ? "Unavailable"
                : `${probability.toFixed(2)}%`;
    }

    if (riskMeterFill) {
        riskMeterFill.style.width =
            probability === null ? "0%" : `${probability}%`;

        riskMeterFill.classList.toggle("high-risk", risk === "high");
    }

    if (riskExplanation) {
        if (risk === "low") {
            riskExplanation.textContent =
                "The model estimates a lower likelihood of loan default.";
        } else if (risk === "high") {
            riskExplanation.textContent =
                "The model estimates an elevated likelihood of loan default.";
        } else {
            riskExplanation.textContent =
                "Review the returned prediction and probability.";
        }
    }

    if (resultNote) {
        resultNote.textContent =
            data.note ||
            "This is a machine-learning estimate, not a loan approval or rejection decision.";
    }

    // Display the new risk factors and recommendations.
    displayRiskInsights(borrower);

    if (resultCard) resultCard.hidden = false;
}

/* =========================================
   LOAD PREDICTION HISTORY
========================================= */

async function loadHistory() {
    const historyMessage = getElement("historyMessage");

    if (historyMessage) {
        historyMessage.textContent = "Loading prediction history...";
        historyMessage.className = "history-message";
    }

   try {
    const response = await fetch(HISTORY_URL, {
        credentials: "include"
    });

    if (!response.ok) {
        throw new Error(
            `History API returned HTTP ${response.status}.`
        );
    }

        const data = await response.json();

        predictionHistory = getHistoryArray(data);

        renderHistory(predictionHistory);

        if (historyMessage) {
            historyMessage.textContent =
                predictionHistory.length > 0
                    ? `${predictionHistory.length} prediction record(s) loaded.`
                    : "No prediction history available yet.";

            historyMessage.className = "history-message";
        }

    } catch (error) {
        console.error("History loading error:", error);

        if (historyMessage) {
            historyMessage.textContent =
                "Could not load history. Check your backend history endpoint.";

            historyMessage.className = "history-message error";
        }

        predictionHistory = [];
        renderHistory([]);
    }
}

/* =========================================
   RENDER HISTORY AND ANALYTICS
========================================= */

function getFilteredHistory(records) {
    const search = (
        getElement("historySearch")?.value || ""
    ).trim().toLowerCase();

    const riskFilter = getElement("riskFilter")?.value || "all";
    const sortOrder = getElement("sortHistory")?.value || "newest";

    const filteredRecords = records.filter(record => {
        const creditScore = String(
            record.CreditScore ??
            record.credit_score ??
            record.creditScore ??
            ""
        ).toLowerCase();

        const loanAmount = String(
            record.LoanAmount ??
            record.loan_amount ??
            record.loanAmount ??
            ""
        ).toLowerCase();

        const matchesSearch =
            !search ||
            creditScore.includes(search) ||
            loanAmount.includes(search);

        const risk = getRiskType(record);
        const matchesRisk = riskFilter === "all" || risk === riskFilter;

        return matchesSearch && matchesRisk;
    });

    function getRecordTime(record) {
        const value =
            record.created_at ??
            record.timestamp ??
            record.date ??
            record.createdAt;

        const parsedDate = value ? new Date(value).getTime() : NaN;

        return Number.isFinite(parsedDate)
            ? parsedDate
            : Number(record.id ?? 0);
    }

    filteredRecords.sort((a, b) => {
        const difference = getRecordTime(a) - getRecordTime(b);
        return sortOrder === "oldest" ? difference : -difference;
    });

    return filteredRecords;
}

function renderHistory(records) {
    renderHistoryTable(getFilteredHistory(records));

    // Analytics always use all records, not filtered results.
    renderAnalytics(records);
}

/* =========================================
   DOWNLOAD PDF REPORT
========================================= */

async function downloadPDFReport(predictionId, button) {
    if (
        predictionId === undefined ||
        predictionId === null ||
        predictionId === ""
    ) {
        alert("Prediction ID is missing. Cannot download this report.");
        return;
    }

    const originalText = button.textContent;

    try {
        button.disabled = true;
        button.textContent = "Preparing...";

        const response = await fetch(
    `${API_BASE}/api/report/${encodeURIComponent(predictionId)}`,
    {
        credentials: "include"
    }
);

        if (!response.ok) {
            let message = "Unable to generate PDF report.";

            try {
                const errorData = await response.json();
                message = errorData.message || errorData.error || message;
            } catch {
                // Keep the default message if the response isn't JSON.
            }

            throw new Error(message);
        }

        const blob = await response.blob();

        const contentType = response.headers.get("content-type") || "";

        if (!contentType.includes("application/pdf")) {
            throw new Error("The server did not return a PDF file.");
        }

        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");

        link.href = url;
        link.download = `CredAI_Report_${predictionId}.pdf`;

        document.body.appendChild(link);
        link.click();
        link.remove();

        // Revoke the temporary URL after the browser has started downloading.
        setTimeout(() => URL.revokeObjectURL(url), 1000);

    } catch (error) {
        console.error("PDF download error:", error);
        alert(error.message || "PDF download failed.");

    } finally {
        button.disabled = false;
        button.textContent = originalText;
    }
}

/* =========================================
   HISTORY TABLE
========================================= */

function renderHistoryTable(records) {
    const tableBody = getElement("historyTableBody");

    if (!tableBody) return;

    if (!records.length) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="6" class="empty-row">
                    No predictions found. Submit a borrower assessment to begin.
                </td>
            </tr>
        `;

        return;
    }

    tableBody.innerHTML = records.map(record => {
        const timestamp =
            record.created_at ??
            record.timestamp ??
            record.date ??
            record.createdAt ??
            "";

        const creditScore =
            record.CreditScore ??
            record.credit_score ??
            record.creditScore ??
            "—";

        const loanAmount =
            record.LoanAmount ??
            record.loan_amount ??
            record.loanAmount;

        const probability = getProbability(record);
        const risk = getRiskType(record);
        const label = getRiskLabel(record);

        const predictionId = record.id ?? record.prediction_id;

        const riskClass =
            risk === "low"
                ? "low-risk"
                : risk === "high"
                    ? "high-risk"
                    : "";

        const downloadButton = predictionId !== undefined &&
            predictionId !== null
            ? `
                <button
                    type="button"
                    class="download-pdf-button"
                    data-prediction-id="${escapeHTML(predictionId)}"
                >
                    ↓ Download PDF
                </button>
            `
            : `<span>Unavailable</span>`;

        return `
            <tr>
                <td>${escapeHTML(formatDate(timestamp))}</td>
                <td>${escapeHTML(creditScore)}</td>
                <td>${escapeHTML(
                    loanAmount === undefined
                        ? "—"
                        : formatCurrency(loanAmount)
                )}</td>
                <td>
                    <span class="risk-label ${riskClass}">
                        ${escapeHTML(label)}
                    </span>
                </td>
                <td>
                    ${probability === null
                        ? "—"
                        : `${probability.toFixed(2)}%`}
                </td>
                <td>${downloadButton}</td>
            </tr>
        `;
    }).join("");

    // Attach PDF download actions to the generated buttons.
    tableBody.querySelectorAll(".download-pdf-button").forEach(button => {
        button.addEventListener("click", () => {
            downloadPDFReport(
                button.dataset.predictionId,
                button
            );
        });
    });
}

/* =========================================
   DASHBOARD ANALYTICS
========================================= */

function renderAnalytics(records) {
    const total = records.length;

    const lowCount = records.filter(
        record => getRiskType(record) === "low"
    ).length;

    const highCount = records.filter(
        record => getRiskType(record) === "high"
    ).length;

    setText("historyCount", total);
    setText("lowRiskCount", lowCount);
    setText("highRiskCount", highCount);

    setText("donutTotal", total);
    setText("lowRiskLegend", lowCount);
    setText("highRiskLegend", highCount);

    renderRiskDonut(lowCount, highCount);
    renderActivityChart(records);
}

/* =========================================
   RISK DISTRIBUTION DONUT
========================================= */

function renderRiskDonut(lowCount, highCount) {
    const donut = getElement("riskDonut");
    const emptyMessage = getElement("riskChartEmpty");

    if (!donut) return;

    const total = lowCount + highCount;

    if (total === 0) {
        donut.style.background =
            "conic-gradient(#e5eae5 0% 100%)";

        if (emptyMessage) emptyMessage.hidden = false;
        return;
    }

    if (emptyMessage) emptyMessage.hidden = true;

    const lowPercentage = (lowCount / total) * 100;

    donut.style.background = `
        conic-gradient(
            #32a66a 0% ${lowPercentage}%,
            #d84b4b ${lowPercentage}% 100%
        )
    `;

    donut.setAttribute(
        "aria-label",
        `Risk distribution: ${lowCount} low-risk and ${highCount} high-risk predictions`
    );
}

/* =========================================
   PREDICTION ACTIVITY CHART
========================================= */

function getRecordDate(record) {
    const value =
        record.created_at ??
        record.timestamp ??
        record.date ??
        record.createdAt;

    if (!value) return null;

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) return null;

    return date;
}

function renderActivityChart(records) {
    const chart = getElement("activityChart");
    const emptyMessage = getElement("activityChartEmpty");

    if (!chart) return;

    chart.innerHTML = "";

    const dailyCounts = new Map();

    records.forEach(record => {
        const date = getRecordDate(record);

        if (!date) return;

        const key = [
            date.getFullYear(),
            String(date.getMonth() + 1).padStart(2, "0"),
            String(date.getDate()).padStart(2, "0")
        ].join("-");

        dailyCounts.set(
            key,
            (dailyCounts.get(key) || 0) + 1
        );
    });

    const days = Array.from(dailyCounts.entries())
        .sort((a, b) => a[0].localeCompare(b[0]))
        .slice(-7);

    if (days.length === 0) {
        if (emptyMessage) emptyMessage.hidden = false;
        return;
    }

    if (emptyMessage) emptyMessage.hidden = true;

    const maximum = Math.max(
        1,
        ...days.map(([, count]) => count)
    );

    days.forEach(([dateKey, count]) => {
        const date = new Date(`${dateKey}T12:00:00`);

        const column = document.createElement("div");
        column.className = "activity-column";

        const value = document.createElement("span");
        value.className = "activity-value";
        value.textContent = count;

        const track = document.createElement("div");
        track.className = "activity-bar-track";

        const bar = document.createElement("div");
        bar.className = "activity-bar";

        const height = Math.max(3, (count / maximum) * 100);

        bar.style.height = `${height}%`;
        bar.title = `${count} prediction(s) on ${dateKey}`;

        track.appendChild(bar);

        const dateLabel = document.createElement("span");
        dateLabel.className = "activity-date";
        dateLabel.textContent = date.toLocaleDateString("en-IN", {
            day: "2-digit",
            month: "short"
        });

        column.appendChild(value);
        column.appendChild(track);
        column.appendChild(dateLabel);

        chart.appendChild(column);
    });

    chart.setAttribute(
        "aria-label",
        `Prediction activity chart displaying ${days.length} dates`
    );
}

/* =========================================
   EXPORT HISTORY TO CSV
========================================= */

function escapeCSVCell(value) {
    let text = String(value ?? "");

    if (/^[\s]*[=+\-@]/.test(text)) {
        text = "'" + text;
    }

    return `"${text.replace(/"/g, '""')}"`;
}

function exportHistoryToCSV() {
    if (!predictionHistory.length) {
        alert("No prediction history available to export.");
        return;
    }

    const headers = [
        "Date and Time",
        "Credit Score",
        "Loan Amount (INR)",
        "Risk Result",
        "Default Probability (%)"
    ];

    const rows = predictionHistory.map(record => {
        const timestamp =
            record.created_at ??
            record.timestamp ??
            record.date ??
            record.createdAt ??
            "";

        const creditScore =
            record.CreditScore ??
            record.credit_score ??
            record.creditScore ??
            "";

        const loanAmount =
            record.LoanAmount ??
            record.loan_amount ??
            record.loanAmount ??
            "";

        const probability = getProbability(record);

        return [
            timestamp,
            creditScore,
            loanAmount,
            getRiskLabel(record),
            probability === null ? "" : probability.toFixed(2)
        ];
    });

    const csvContent = [
        headers,
        ...rows
    ]
        .map(row => row.map(escapeCSVCell).join(","))
        .join("\r\n");

    const blob = new Blob(
        ["\uFEFF" + csvContent],
        { type: "text/csv;charset=utf-8;" }
    );

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    const today = new Date().toISOString().slice(0, 10);

    link.href = url;
    link.download = `CredAI_Prediction_History_${today}.csv`;

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* =========================================
   CLEAR PREDICTION HISTORY FROM DATABASE
========================================= */

async function handleClearHistory() {
    if (!predictionHistory.length) {
        alert("There are no prediction records to delete.");
        return;
    }

    const confirmed = window.confirm(
        "Are you sure you want to permanently delete ALL prediction history from the database? This action cannot be undone."
    );

    if (!confirmed) return;

    const clearButton = getElement("clearHistoryButton");
    const historyMessage = getElement("historyMessage");

    try {
        if (clearButton) {
            clearButton.disabled = true;
            clearButton.textContent = "Clearing...";
        }

        if (historyMessage) {
            historyMessage.textContent = "Deleting prediction history...";
            historyMessage.className = "history-message";
        }
        const response = await fetch(HISTORY_URL, {
    method: "DELETE",
    credentials: "include"
});

        const data = await response.json();

        if (!response.ok || data.status === "error") {
            throw new Error(
                data.message ||
                data.error ||
                "Could not delete prediction history."
            );
        }

        predictionHistory = [];
        renderHistory([]);

        if (historyMessage) {
            historyMessage.textContent =
                data.message ||
                "All prediction history deleted successfully.";
            historyMessage.className = "history-message";
        }

        alert("All prediction history has been deleted successfully.");

    } catch (error) {
        console.error("Clear history error:", error);

        if (historyMessage) {
            historyMessage.textContent =
                error.message ||
                "Failed to delete prediction history.";
            historyMessage.className = "history-message error";
        }

        alert(error.message || "Failed to delete prediction history.");

    } finally {
        if (clearButton) {
            clearButton.disabled = false;
            clearButton.textContent = "Clear history";
        }
    }
}

/* =========================================
   MODEL EXPLAINABILITY DASHBOARD
========================================= */

async function loadFeatureImportance() {

    const status = getElement("explainabilityStatus");
    const chart = getElement("featureImportanceChart");
    const message = getElement("featureImportanceMessage");

    if (!chart) return;

    if (status) {
        status.textContent = "Loading...";
    }

    chart.innerHTML = `
        <p class="feature-chart-message">
            Loading model feature importance...
        </p>
    `;

    if (message) {
        message.textContent = "";
    }

    try {

        const response = await fetch(FEATURE_IMPORTANCE_URL);

        if (!response.ok) {
            throw new Error(
                `Feature importance API returned HTTP ${response.status}.`
            );
        }

        const data = await response.json();

        if (
            data.status !== "success" ||
            !Array.isArray(data.features) ||
            data.features.length === 0
        ) {
            throw new Error(
                data.message || "No feature importance data available."
            );
        }

        renderFeatureImportance(data.features);

        if (status) {
            status.textContent = "Data loaded";
        }

        if (message) {
            message.textContent =
                "Feature importance successfully retrieved from the optimized Random Forest model.";
        }

    } catch (error) {

        console.error("Feature importance error:", error);

        chart.innerHTML = `
            <p class="feature-chart-message">
                Could not load feature importance.
                Please check your Flask backend.
            </p>
        `;

        if (status) {
            status.textContent = "Unavailable";
        }

        if (message) {
            message.textContent = error.message;
        }
    }
}


/* =========================================
   RENDER FEATURE IMPORTANCE CHART
========================================= */

function renderFeatureImportance(features) {

    const chart = getElement("featureImportanceChart");

    if (!chart) return;

    chart.innerHTML = "";

    const validFeatures = features.filter(feature =>
        typeof feature.feature === "string" &&
        Number.isFinite(Number(feature.importance)) &&
        Number(feature.importance) >= 0
    );

    if (validFeatures.length === 0) {

        chart.innerHTML = `
            <p class="feature-chart-message">
                No valid feature importance values found.
            </p>
        `;

        return;
    }

    const maximum = Math.max(
        ...validFeatures.map(feature =>
            Number(feature.importance)
        ),
        0.0001
    );

    validFeatures.forEach(feature => {

        const importance = Number(feature.importance);

        // Scale bars relative to the most important feature.
        const barWidth = (importance / maximum) * 100;

        const row = document.createElement("div");
        row.className = "feature-row";

        const name = document.createElement("span");
        name.className = "feature-name";
        name.textContent = feature.feature;

        const track = document.createElement("div");
        track.className = "feature-bar-track";

        const bar = document.createElement("div");
        bar.className = "feature-bar";

        bar.style.width = `${barWidth}%`;

        bar.setAttribute(
            "aria-label",
            `${feature.feature}: ${importance.toFixed(2)}% importance`
        );

        track.appendChild(bar);

        const value = document.createElement("span");
        value.className = "feature-value";
        value.textContent = `${importance.toFixed(2)}%`;

        row.appendChild(name);
        row.appendChild(track);
        row.appendChild(value);

        chart.appendChild(row);
    });
}

/* =========================================
   INITIALIZE DASHBOARD
========================================= */
/* =========================================
   MODEL PERFORMANCE DASHBOARD
========================================= */

const EVALUATION_URL = `${API_BASE}/api/evaluation`;

function formatMetric(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) return "—";

    const percentage = Math.abs(number) <= 1 ? number * 100 : number;
    return `${percentage.toFixed(2)}%`;
}

async function loadModelEvaluation() {
    const status = getElement("evaluationStatus");

    if (status) {
        status.textContent = "Loading metrics...";
    }

    try {
        const response = await fetch(EVALUATION_URL);

        if (!response.ok) {
            throw new Error(
                `Evaluation API returned HTTP ${response.status}.`
            );
        }

        const data = await response.json();

        if (data.status !== "success") {
            throw new Error(
                data.message || "Model evaluation failed."
            );
        }

        // Display performance metrics.
        setText("modelAccuracy", formatMetric(data.accuracy));
        setText("modelPrecision", formatMetric(data.precision));
        setText("modelRecall", formatMetric(data.recall));
        setText("modelF1", formatMetric(data.f1_score));
        setText("modelF2", formatMetric(data.f2_score));

        setText(
            "modelThreshold",
            Number(data.threshold).toFixed(4)
        );

        // Display confusion matrix.
        const matrix = data.confusion_matrix;

        if (
            !Array.isArray(matrix) ||
            matrix.length !== 2 ||
            !Array.isArray(matrix[0]) ||
            !Array.isArray(matrix[1])
        ) {
            throw new Error("Invalid confusion matrix received.");
        }

        const tn = matrix[0][0];
        const fp = matrix[0][1];
        const fn = matrix[1][0];
        const tp = matrix[1][1];

        setText("trueNegative", tn.toLocaleString("en-IN"));
        setText("falsePositive", fp.toLocaleString("en-IN"));
        setText("falseNegative", fn.toLocaleString("en-IN"));
        setText("truePositive", tp.toLocaleString("en-IN"));

        setText(
            "matrixExplanation",
            `The model correctly identified ${tp.toLocaleString("en-IN")} actual defaulters, missed ${fn.toLocaleString("en-IN")}, and incorrectly flagged ${fp.toLocaleString("en-IN")} non-defaulters.`
        );

        if (status) {
            status.textContent = "Evaluation successful";
        }

    } catch (error) {
        console.error("Model evaluation error:", error);

        if (status) {
            status.textContent = "Evaluation unavailable";
        }

        [
            "modelAccuracy",
            "modelPrecision",
            "modelRecall",
            "modelF1",
            "modelF2",
            "modelThreshold",
            "trueNegative",
            "falsePositive",
            "falseNegative",
            "truePositive"
        ].forEach(id => setText(id, "—"));

        setText(
            "matrixExplanation",
            "Unable to load evaluation results. Check your Flask backend."
        );
    }
}

/* =========================================
   DASHBOARD PAGE NAVIGATION
========================================= */


function showDashboardPage(pageId, updateAddress = true) {
    const pages = document.querySelectorAll(".page-view");
    const navLinks = document.querySelectorAll(
        ".sidebar-link[data-page-target]"
    );

    const target = document.getElementById(pageId);

    if (!target) return;

    // Display only the selected dashboard page.
    pages.forEach(page => {
        const isActive = page.id === pageId;

        page.classList.toggle("active", isActive);
        page.hidden = !isActive;
    });

    // Update sidebar navigation.
    navLinks.forEach(link => {
        const isActive = link.dataset.pageTarget === pageId;

        link.classList.toggle("active", isActive);

        if (isActive) {
            link.setAttribute("aria-current", "page");
        } else {
            link.removeAttribute("aria-current");
        }
    });

    if (updateAddress) {
        history.replaceState(null, "", `#${pageId}`);
    }

    // Restore saved CSV results when opening Assessment.
    if (pageId === "assessmentPage") {
        const savedResults = sessionStorage.getItem(
            "credaiAssessmentResults"
        );

        if (savedResults) {
            try {
                const saved = JSON.parse(savedResults);

                if (
                    Array.isArray(saved.results) &&
                    saved.results.length > 0
                ) {
                    renderCSVResults(
                        saved.results,
                        saved.fileName || ""
                    );
                }
            } catch (error) {
                console.error(
                    "Could not restore CSV results:",
                    error
                );
            }
        }
    }

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}


function initializePageNavigation() {
    const navLinks = document.querySelectorAll(".sidebar-link[data-page-target]");

    navLinks.forEach(link => {
        link.addEventListener("click", event => {
            event.preventDefault();
            showDashboardPage(link.dataset.pageTarget);
        });
    });

    const initialPage = window.location.hash.replace("#", "");
    const validPage = document.getElementById(initialPage)?.classList.contains("page-view")
        ? initialPage
        : "dashboardPage";

    showDashboardPage(validPage, false);

    window.addEventListener("hashchange", () => {
        const requestedPage = window.location.hash.replace("#", "");
        if (document.getElementById(requestedPage)?.classList.contains("page-view")) {
            showDashboardPage(requestedPage, false);
        }
    });
}

/* =========================================
   INITIALIZE DASHBOARD
========================================= */

document.addEventListener("DOMContentLoaded", () => {
    initializePageNavigation();

    const form = getElement("predictionForm");
    if (form) {
        form.addEventListener("submit", handlePrediction);
    }

    const clearButton = getElement("clearHistoryButton");
    if (clearButton) {
        clearButton.addEventListener("click", handleClearHistory);
    }

    const exportButton = getElement("exportHistoryButton");
    if (exportButton) {
        exportButton.addEventListener("click", exportHistoryToCSV);
    }

    // Search, filter and sort prediction history.
    const historySearch = getElement("historySearch");
    const riskFilter = getElement("riskFilter");
    const sortHistory = getElement("sortHistory");

    if (historySearch) {
        historySearch.addEventListener("input", () => {
            renderHistoryTable(getFilteredHistory(predictionHistory));
        });
    }

    if (riskFilter) {
        riskFilter.addEventListener("change", () => {
            renderHistoryTable(getFilteredHistory(predictionHistory));
        });
    }

    if (sortHistory) {
        sortHistory.addEventListener("change", () => {
            renderHistoryTable(getFilteredHistory(predictionHistory));
        });
    }

    checkAPI();
    loadHistory();
    loadModelEvaluation();
    loadFeatureImportance();
});

// =========================================
// INDIVIDUAL PREDICTION EXPLANATION
// =========================================

function renderIndividualExplanation(explanation) {
    const summary = document.getElementById("explanationSummary");
    const riskList = document.getElementById("riskFactorsList");
    const positiveList = document.getElementById("positiveFactorsList");

    if (!summary || !riskList || !positiveList) {
        return;
    }

    if (!explanation) {
        summary.textContent = "No explanation available.";
        return;
    }

    summary.textContent = explanation.summary || "No summary available.";

    riskList.innerHTML = "";
    positiveList.innerHTML = "";

    const addItems = (list, items, emptyMessage) => {
        if (!Array.isArray(items) || items.length === 0) {
            const li = document.createElement("li");
            li.textContent = emptyMessage;
            list.appendChild(li);
            return;
        }

        items.forEach(item => {
            const li = document.createElement("li");
            li.textContent = item;
            list.appendChild(li);
        });
    };

    addItems(
        riskList,
        explanation.risk_factors,
        "No risk indicators identified by the screening rules."
    );

    addItems(
        positiveList,
        explanation.positive_factors,
        "No positive indicators identified by the screening rules."
    );
}
async function logoutUser() {
    const confirmLogout = confirm("Are you sure you want to logout?");

    if (!confirmLogout) return;

    try {
        const response = await fetch(
            "http://127.0.0.1:5000/api/logout",
            {
                method: "POST",
                credentials: "include"
            }
        );

        const data = await response.json();

        if (response.ok && data.status === "success") {
            window.location.href = "login.html";
        } else {
            alert(data.message || "Logout failed.");
        }
    } catch (error) {
        console.error("Logout error:", error);
        alert("Could not connect to the server.");
    }
}

/* =========================================
   CREDAI CSV UPLOAD AND BATCH PREDICTION
========================================= */

const CSV_PREDICT_URL = `${API_BASE}/api/predict-csv`;

/* =========================================
   CSV BORROWER RISK EXPLANATION POPUP
========================================= */

let latestCSVResults = [];

let currentCSVPage = 1;

const CSV_PAGE_SIZE = 50;

function getCSVBorrowerData(record) {
    return {
        Age: record.Age ?? record.age,
        Income: record.Income ?? record.income,
        LoanAmount: record.LoanAmount ?? record.loan_amount ?? record.loanAmount,
        CreditScore: record.CreditScore ?? record.credit_score ?? record.creditScore,
        MonthsEmployed: record.MonthsEmployed ?? record.months_employed,
        NumCreditLines: record.NumCreditLines ?? record.num_credit_lines,
        InterestRate: record.InterestRate ?? record.interest_rate,
        LoanTerm: record.LoanTerm ?? record.loan_term,
        DTIRatio: record.DTIRatio ?? record.dti_ratio ?? record.dti,
        EmploymentType: record.EmploymentType ?? record.employment_type
    };
}

function addCSVExplanationStyles() {
    if (getElement("csvExplanationStyles")) return;

    const style = document.createElement("style");
    style.id = "csvExplanationStyles";

    style.textContent = `
        .csv-explain-button {
            border: 0;
            cursor: pointer;
            font: inherit;
            font-weight: 700;
            padding: 9px 15px;
            border-radius: 25px;
            transition: transform .2s, box-shadow .2s;
        }

        .csv-explain-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,.12);
        }

        .csv-explain-button.high-risk {
            background: #fff0f0;
            color: #d93636;
        }

        .csv-explain-button.low-risk {
            background: #e5f6ec;
            color: #087b48;
        }

        .csv-explanation-overlay {
            position: fixed;
            inset: 0;
            z-index: 99999;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background: rgba(8, 25, 18, .68);
            backdrop-filter: blur(5px);
        }

        .csv-explanation-overlay.show {
            display: flex;
        }

        .csv-explanation-modal {
            width: min(680px, 100%);
            max-height: 88vh;
            overflow-y: auto;
            background: #fff;
            border-radius: 20px;
            padding: 28px;
            box-shadow: 0 20px 65px rgba(0,0,0,.25);
            color: #20372b;
            position: relative;
        }

        .csv-explanation-close {
            position: absolute;
            top: 16px;
            right: 18px;
            border: 0;
            background: #f0f4f1;
            width: 34px;
            height: 34px;
            border-radius: 50%;
            font-size: 22px;
            cursor: pointer;
            color: #31483a;
        }

        .csv-modal-eyebrow {
            color: #07834e;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .csv-modal-heading {
            font-size: 25px;
            margin: 0 40px 20px 0;
        }

        .csv-modal-score-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 15px;
            padding: 20px;
            border-radius: 14px;
            background: #f5f8f6;
            margin-bottom: 22px;
        }

        .csv-modal-score {
            font-size: 36px;
            font-weight: 800;
            color: #d93636;
        }

        .csv-modal-score small {
            font-size: 14px;
            color: #718078;
        }

        .csv-modal-risk {
            background: #fff0f0;
            color: #d93636;
            padding: 8px 14px;
            border-radius: 20px;
            font-weight: 800;
        }

        .csv-modal-meter {
            height: 9px;
            background: #e6ece8;
            border-radius: 20px;
            overflow: hidden;
            margin-top: 12px;
        }

        .csv-modal-meter-fill {
            height: 100%;
            background: #df4b4b;
            border-radius: 20px;
        }

        .csv-modal-section {
            margin-top: 22px;
        }

        .csv-modal-section h3 {
            font-size: 16px;
            margin: 0 0 10px;
        }

        .csv-modal-section p {
            color: #607067;
            line-height: 1.65;
            margin: 0;
        }

        .csv-modal-list {
            padding-left: 22px;
            margin: 0;
        }

        .csv-modal-list li {
            padding: 5px 0;
            color: #52645a;
            line-height: 1.5;
        }

        .csv-modal-note {
            margin-top: 22px;
            padding: 13px 15px;
            border-left: 4px solid #22a66a;
            background: #eff8f2;
            color: #53665b;
            font-size: 12px;
            line-height: 1.6;
            border-radius: 7px;
        }

        @media(max-width: 600px) {
            .csv-explanation-modal {
                padding: 22px 18px;
            }

            .csv-modal-score-card {
                align-items: flex-start;
                flex-direction: column;
            }
        }
    `;

    document.head.appendChild(style);
}

function createCSVExplanationModal() {
    if (getElement("csvExplanationOverlay")) return;

    addCSVExplanationStyles();

    const overlay = document.createElement("div");
    overlay.id = "csvExplanationOverlay";
    overlay.className = "csv-explanation-overlay";

    overlay.innerHTML = `
        <div class="csv-explanation-modal"
             role="dialog"
             aria-modal="true"
             aria-labelledby="csvExplanationTitle">

            <button type="button"
                    class="csv-explanation-close"
                    id="csvExplanationClose"
                    aria-label="Close explanation">&times;</button>

            <div class="csv-modal-eyebrow">CredAI · Explainable AI</div>

            <h2 class="csv-modal-heading" id="csvExplanationTitle">
                Borrower Risk Explanation
            </h2>

            <div id="csvExplanationContent"></div>
        </div>
    `;

    document.body.appendChild(overlay);

    getElement("csvExplanationClose").addEventListener(
        "click",
        closeCSVExplanation
    );

    overlay.addEventListener("click", event => {
        if (event.target === overlay) {
            closeCSVExplanation();
        }
    });

    document.addEventListener("keydown", event => {
        if (event.key === "Escape") {
            closeCSVExplanation();
        }
    });
}

function closeCSVExplanation() {
    const overlay = getElement("csvExplanationOverlay");

    if (overlay) {
        overlay.classList.remove("show");
    }
}

function showCSVExplanation(index) {
    const record = latestCSVResults[index];

    if (!record) return;

    createCSVExplanationModal();

    const content = getElement("csvExplanationContent");
    if (!content) return;

    const probability = getProbability(record);
    const risk = getRiskType(record);
    const borrower = getCSVBorrowerData(record);

    // Use a backend-provided explanation when one is available.
    const suppliedExplanation =
        record.explanation ||
        record.explainability ||
        {};

    const suppliedFactors =
        record.risk_factors ||
        suppliedExplanation.risk_factors;

    const suppliedRecommendations =
        record.recommendations ||
        suppliedExplanation.recommendations;

    const insights = generateRiskInsights(borrower);

    const factors = Array.isArray(suppliedFactors)
        ? suppliedFactors
        : insights.factors;

    const recommendations = Array.isArray(suppliedRecommendations)
        ? suppliedRecommendations
        : insights.recommendations;

    const suppliedSummary =
        typeof suppliedExplanation === "string"
            ? suppliedExplanation
            : suppliedExplanation.summary;

    const summary = suppliedSummary ||
        (factors.length > 0 &&
         factors[0] !== "No configured screening indicators were triggered."
            ? "The borrower has one or more screening indicators that should be reviewed alongside the model's risk prediction."
            : "The model classified this borrower as high risk. The current batch API does not provide exact per-borrower model feature contributions, so the indicators below are screening checks rather than the model's internal explanation.");

    const borrowerName =
        record.Borrower ??
        record.borrower ??
        record.Name ??
        record.name ??
        record.LoanID ??
        record.loan_id ??
        `Borrower ${index + 1}`;

    const score = probability === null
        ? "—"
        : probability.toFixed(2);

    const factorHTML = factors.map(item =>
        `<li>${escapeHTML(item)}</li>`
    ).join("");

    const recommendationHTML = recommendations.map(item =>
        `<li>${escapeHTML(item)}</li>`
    ).join("");

    const meterWidth = probability === null
        ? 0
        : Math.max(0, Math.min(100, probability));

    content.innerHTML = `
        <div class="csv-modal-score-card">
            <div>
                <div style="font-size:13px;color:#718078;">
                    Default probability · ${escapeHTML(borrowerName)}
                </div>
                <div class="csv-modal-score">
                    ${score}<small>${probability === null ? "" : " / 100%"}</small>
                </div>
            </div>

            <span class="csv-modal-risk">
                ${escapeHTML(getRiskLabel(record))}
            </span>
        </div>

        <div class="csv-modal-meter">
            <div class="csv-modal-meter-fill"
                 style="width:${meterWidth}%"></div>
        </div>

        <div class="csv-modal-section">
            <h3>Why did this borrower receive this result?</h3>
            <p>${escapeHTML(summary)}</p>
        </div>

        <div class="csv-modal-section">
            <h3>Risk indicators to review</h3>
            <ul class="csv-modal-list">
                ${factorHTML}
            </ul>
        </div>

        <div class="csv-modal-section">
            <h3>Recommended actions</h3>
            <ul class="csv-modal-list">
                ${recommendationHTML}
            </ul>
        </div>

        <div class="csv-modal-note">
            <strong>Important:</strong> The probability is the model's
            prediction. The risk indicators and recommendations are
            rule-based screening guidance from the available borrower
            fields; they are not proof of the exact reasons behind the
            Random Forest prediction. Review the borrower's full financial
            information before making a lending decision.
        </div>
    `;

    const overlay = getElement("csvExplanationOverlay");
    overlay.classList.add("show");

    getElement("csvExplanationClose").focus();
}


const CSV_COLUMNS = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner"
];

// Download a CSV template.
function downloadCSVTemplate() {
    const csv = CSV_COLUMNS.join(",") + "\n";

    const blob = new Blob([csv], {
        type: "text/csv;charset=utf-8;"
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "CredAI_Borrower_Template.csv";

    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(url);
}

// Show uploaded filename.
function updateCSVFileName() {
    const input = getElement("csvFile");
    const name = getElement("csvFileName");

    if (!input || !name) return;

    name.textContent = input.files.length
        ? input.files[0].name
        : "No file selected";
}

// Display batch prediction results in the current assessment page.




function renderCSVResults(results, fileName = "") {
    const section = getElement("csvResultsSection");
    const body = getElement("csvResultsBody");
    const summary = getElement("csvResultSummary");

    if (!section || !body) {
        console.error("CSV results HTML elements were not found.");
        return;
    }

    if (!Array.isArray(results) || results.length === 0) {
        console.error("No valid CSV results received.");
        return;
    }

    latestCSVResults = results;
    currentCSVPage = 1;

    createCSVExplanationModal();

    section.hidden = false;
    section.style.display = "block";

    const assessmentPage = getElement("assessmentPage");

    if (assessmentPage) {
        assessmentPage.hidden = false;
        assessmentPage.classList.add("active");
    }

    if (summary) {
        summary.textContent =
            `${results.length.toLocaleString("en-IN")} borrower records assessed` +
            (fileName ? ` from ${fileName}.` : ".");
    }

    // Create pagination controls if they do not exist.
    let pagination = getElement("csvPagination");

    if (!pagination) {
        pagination = document.createElement("div");
        pagination.id = "csvPagination";
        pagination.className = "csv-pagination";

        section.appendChild(pagination);
    }

    renderCSVPage();

}


/* =========================================
   RENDER ONE CSV RESULTS PAGE
========================================= */

function renderCSVPage() {
    const body = getElement("csvResultsBody");
    const pagination = getElement("csvPagination");

    if (!body || !pagination) return;

    const totalRecords = latestCSVResults.length;

    const totalPages = Math.ceil(
        totalRecords / CSV_PAGE_SIZE
    );

    if (totalRecords === 0) {
        body.innerHTML = `
            <tr>
                <td colspan="5">No borrower records available.</td>
            </tr>
        `;

        pagination.innerHTML = "";
        return;
    }

    currentCSVPage = Math.max(
        1,
        Math.min(currentCSVPage, totalPages)
    );

    const startIndex =
        (currentCSVPage - 1) * CSV_PAGE_SIZE;

    const endIndex = Math.min(
        startIndex + CSV_PAGE_SIZE,
        totalRecords
    );

    const pageRecords = latestCSVResults.slice(
        startIndex,
        endIndex
    );

    // Build only the current page of table rows.
    const fragment = document.createDocumentFragment();

    pageRecords.forEach((record, pageIndex) => {

        const actualIndex = startIndex + pageIndex;

        const risk = getRiskType(record);
        const riskLabel = getRiskLabel(record);
        const probability = getProbability(record);

        const borrowerName =
            record.Borrower ??
            record.borrower ??
            record.Name ??
            record.name ??
            record.LoanID ??
            record.loan_id ??
            `Borrower ${actualIndex + 1}`;

        const creditScore =
            record.CreditScore ??
            record.credit_score ??
            record.creditScore ??
            "—";

        const loanAmount =
            record.LoanAmount ??
            record.loan_amount ??
            record.loanAmount;

        const row = document.createElement("tr");

        const nameCell = document.createElement("td");
        nameCell.textContent = borrowerName;

        const scoreCell = document.createElement("td");
        scoreCell.textContent = creditScore;

        const amountCell = document.createElement("td");
        amountCell.textContent =
            loanAmount === undefined
                ? "—"
                : formatCurrency(loanAmount);

        const riskCell = document.createElement("td");

        const badge = document.createElement("button");

        badge.type = "button";
        badge.className =
            `csv-explain-button ${risk}-risk`;

        badge.textContent = riskLabel;

        badge.dataset.csvExplainIndex = actualIndex;

        badge.title =
            "Click to view borrower risk explanation";

        riskCell.appendChild(badge);

        const probabilityCell = document.createElement("td");

        probabilityCell.textContent =
            probability === null
                ? "—"
                : `${probability.toFixed(2)}%`;

        row.appendChild(nameCell);
        row.appendChild(scoreCell);
        row.appendChild(amountCell);
        row.appendChild(riskCell);
        row.appendChild(probabilityCell);

        fragment.appendChild(row);
    });

    // Replace the previous page with the current page.
    body.replaceChildren(fragment);

// Connect Low Risk / High Risk buttons to the explanation popup.
body.querySelectorAll(".csv-explain-button").forEach(button => {

    button.addEventListener("click", function () {

        const index = Number(this.dataset.csvExplainIndex);

        showCSVExplanation(index);

    });

});


    // Pagination controls.
    pagination.innerHTML = `
        <div class="csv-pagination-info">
            Showing
            <strong>${(startIndex + 1).toLocaleString("en-IN")}</strong>
            –
            <strong>${endIndex.toLocaleString("en-IN")}</strong>
            of
            <strong>${totalRecords.toLocaleString("en-IN")}</strong>
            borrowers
        </div>

        <div class="csv-pagination-buttons">

            <button
                type="button"
                id="csvPreviousPage"
                ${currentCSVPage === 1 ? "disabled" : ""}
            >
                ← Previous
            </button>

            <span>
                Page ${currentCSVPage} of ${totalPages}
            </span>

            <button
                type="button"
                id="csvNextPage"
                ${currentCSVPage === totalPages ? "disabled" : ""}
            >
                Next →
            </button>

        </div>
    `;

    // Previous page.
    getElement("csvPreviousPage").addEventListener(
        "click",
        () => {
            if (currentCSVPage > 1) {
                currentCSVPage--;
                renderCSVPage();
            }
        }
    );

    // Next page.
    getElement("csvNextPage").addEventListener(
        "click",
        () => {
            if (currentCSVPage < totalPages) {
                currentCSVPage++;
                renderCSVPage();
            }
        }
    );

    // Scroll to the results table after changing pages.
    pagination.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
}



  


// Upload CSV and show the returned batch results on this page.
async function handleCSVUpload(event) {
    event.preventDefault();

    const input = getElement("csvFile");
    const button = getElement("csvUploadButton");
    const message = getElement("csvUploadMessage");

    if (!input || !input.files || !input.files.length) {
        if (message) {
            message.textContent = "Please select a CSV file first.";
            message.className = "form-message error";
        }
        return;
    }

    const file = input.files[0];
    console.log("========== NEW CSV FILE ==========");
console.log("File name:", file.name);
console.log("File size:", file.size);
console.log("Last modified:", new Date(file.lastModified));
console.log("==================================="); 
    // Clear previous CSV assessment results before processing a new file.
latestCSVResults = [];

sessionStorage.removeItem("credaiAssessmentResults");

const oldBody = getElement("csvResultsBody");
const oldSection = getElement("csvResultsSection");
const oldSummary = getElement("csvResultSummary");

if (oldBody) oldBody.innerHTML = "";
if (oldSummary) oldSummary.textContent = "";
if (oldSection) oldSection.hidden = true;

    if (!file.name.toLowerCase().endsWith(".csv")) {
        if (message) {
            message.textContent = "Please select a valid .csv file.";
            message.className = "form-message error";
        }
        return;
    }

    // Allow CSV files up to 100 MB.
// This must match the Flask backend upload limit.
const MAX_FILE_SIZE = 100 * 1024 * 1024;

if (file.size > MAX_FILE_SIZE) {
    if (message) {
        message.textContent =
            "CSV file is too large. Maximum allowed size is 100 MB.";

        message.className = "form-message error";
    }

    return;
}

    const formData = new FormData();
    formData.append("file", file);

    if (button) {
        button.disabled = true;
        button.textContent = "Assessing borrowers...";
    }

    if (message) {
        message.textContent = "Uploading and analyzing borrower records...";
        message.className = "form-message";
    }

    try {
        const response = await fetch(CSV_PREDICT_URL, {
            method: "POST",
            credentials: "include",
            body: formData
        });

        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error("The server returned an unreadable response. Check the Flask terminal.");
        }

        // Keep the complete response visible in DevTools for backend debugging.
        console.log("CSV API HTTP status:", response.status);
        console.log("CSV API response:", data);

        if (!response.ok || data.status === "error") {
            throw new Error(
                data.message ||
                data.error ||
                `CSV assessment failed (HTTP ${response.status}).`
            );
        }

        // Accept the common response formats used by Flask endpoints.
        // The backend may return results, predictions, borrowers, or a nested data object.
        let results = [];

        if (Array.isArray(data.results)) {
            results = data.results;
        } else if (Array.isArray(data.predictions)) {
            results = data.predictions;
        } else if (Array.isArray(data.borrowers)) {
            results = data.borrowers;
        } else if (Array.isArray(data.data)) {
            results = data.data;
        } else if (Array.isArray(data.result)) {
            results = data.result;
        } else if (data.data && typeof data.data === "object") {
            if (Array.isArray(data.data.results)) {
                results = data.data.results;
            } else if (Array.isArray(data.data.predictions)) {
                results = data.data.predictions;
            } else if (Array.isArray(data.data.borrowers)) {
                results = data.data.borrowers;
            }
        }

        console.log("Normalized borrower results:", results);

        if (results.length === 0) {
            console.error(
                "The CSV endpoint did not return a non-empty borrower array. Response keys:",
                data && typeof data === "object" ? Object.keys(data) : []
            );

            throw new Error(
                "The backend returned no borrower records. Open F12 > Console and check 'CSV API response'. If it contains no results/predictions/borrowers array, the Flask /api/predict-csv function must be corrected."
            );
        }

        // Render the results in the Assessment page.
        renderCSVResults(results, file.name);

        // Save the successful results so they remain visible when navigating away and back.
       try {
    sessionStorage.setItem(
        "credaiAssessmentResults",
        JSON.stringify({
            fileName: file.name,
            total: results.length,
            results: results,
            completedAt: new Date().toISOString()
        })
    );
} catch (storageError) {
    console.warn(
        "Results could not be saved in sessionStorage:",
        storageError
    );
}

        if (message) {
            message.textContent = `CSV assessment completed successfully. ${results.length} borrower(s) assessed.`;
            message.className = "form-message success";
        }

        await loadHistory();

    } catch (error) {
        console.error("CSV upload error:", error);

        if (message) {
            message.textContent = error.message || "CSV upload failed. Please try again.";
            message.className = "form-message error";
        }

    } finally {
        if (button) {
            button.disabled = false;
            button.innerHTML = "<span>Upload &amp; Assess Borrowers</span><span>↗</span>";
        }
    }
}

// CSV upload controls.
document.addEventListener("DOMContentLoaded", () => {
    const csvForm = getElement("csvUploadForm");
    const csvFile = getElement("csvFile");
    const templateButton = getElement("downloadCsvTemplateButton");

    if (csvForm) {
        csvForm.addEventListener("submit", handleCSVUpload);
    }

    if (csvFile) {
        csvFile.addEventListener("change", updateCSVFileName);
    }

    if (templateButton) {
        templateButton.addEventListener("click", downloadCSVTemplate);
    }
});

/* CREDAI DASHBOARD AI ASSISTANT */

(function () {
    const form = document.getElementById("assistant-form");
    const input = document.getElementById("assistant-input");
    const messages = document.getElementById("assistant-messages");

    if (!form || !input || !messages) return;

    function addMessage(message, type) {
        const div = document.createElement("div");

        div.className = type === "user"
            ? "assistant-user-message"
            : "assistant-bot-message";

        div.textContent = message;
        messages.appendChild(div);
    }

    function getResponse(question) {
        const q = question.toLowerCase();

        if (q.includes("credit risk")) {
            return "Credit risk is the possibility that a borrower may fail to repay a loan. CredAI uses machine learning to estimate this risk.";
        }

        if (q.includes("random forest")) {
            return "Random Forest is a machine learning algorithm that combines multiple decision trees to make predictions. CredAI uses it for loan default prediction.";
        }

        if (q.includes("default") || q.includes("loan")) {
            return "Loan default occurs when a borrower fails to make the required repayments. CredAI estimates the probability of default using borrower information.";
        }

        if (q.includes("credai")) {
            return "CredAI is an AI-based Credit Risk Assessment and Loan Default Prediction System designed to support informed lending decisions.";
        }

        if (q.includes("high risk")) {
            return "High risk indicates that the model estimates a higher probability of loan default. It is an estimate and should be reviewed alongside other financial information.";
        }

        if (q.includes("low risk")) {
            return "Low risk indicates a comparatively lower estimated probability of loan default. It does not guarantee repayment.";
        }

        if (q.includes("hello") || q.includes("hi")) {
            return "Hello! Welcome to CredAI. Ask me anything about credit risk or loan prediction.";
        }

        return "I can help explain CredAI, credit risk, Random Forest and loan default. Try asking one of these questions!";
    }

    function sendMessage(question) {
        if (!question.trim()) return;

        addMessage(question, "user");
        input.value = "";

        setTimeout(() => {
            addMessage(getResponse(question), "bot");
        }, 400);
    }

    form.addEventListener("submit", function (event) {
        event.preventDefault();
        sendMessage(input.value);
    });

    document.querySelectorAll(".assistant-question").forEach(button => {
        button.addEventListener("click", function () {
            sendMessage(this.textContent);
        });
    });
})();