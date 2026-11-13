
import requests

url = "http://127.0.0.1:5000/api/predict"

borrower = {
    "Age": 35,
    "Income": 60000,
    "LoanAmount": 20000,
    "CreditScore": 700,
    "MonthsEmployed": 48,
    "NumCreditLines": 3,
    "InterestRate": 10.5,
    "LoanTerm": 36,
    "DTIRatio": 0.3,
    "Education": "Bachelor's",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married",
    "HasMortgage": "No",
    "HasDependents": "Yes",
    "LoanPurpose": "Home",
    "HasCoSigner": "Yes"
}

response = requests.post(url, json=borrower)

print("HTTP Status:", response.status_code)
print("Prediction result:")
print(response.text)
