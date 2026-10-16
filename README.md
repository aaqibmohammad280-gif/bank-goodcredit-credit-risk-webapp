# Bank GoodCredit — Credit Risk Web App

A local Flask web application for presenting the scorecard validation results and scoring a model-ready customer portfolio.

The interface is based on the supplied validation notebook's champion model:

- Gradient Boosting classifier with a 100-feature input schema.
- Final held-out ROC-AUC: **0.7045**; Gini: **40.91 points** (above the 37.90-point benchmark).
- Batch-level risk deciles support manual-review prioritisation; scores are relative risk rankings, not probabilities of default.

## Run in PyCharm

1. In PyCharm, choose **Open** and select this `credit-risk-webapp` folder.
2. Configure a clean Python 3.11–3.13 project interpreter, then use PyCharm's Terminal:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. In the notebook, run the model-saving cell (cell 91). Copy its resulting file, `bank_goodcredit_champion_model.joblib`, into:

   ```text
   model_artifacts\bank_goodcredit_champion_model.joblib
   ```

4. Right-click `app.py` and select **Run 'app'**. Open the displayed local address, normally `http://127.0.0.1:5000`.

`scikit-learn==1.6.1` is intentional: the supplied champion-model artifact was saved with that version. Do not substitute a newer scikit-learn release without retraining and re-exporting the model.

## Scoring workflow

1. Once the model artifact is in place, open **Score a portfolio** and download the automatically generated input template.
2. Supply the required 100 *engineered* fields and keep the headings unchanged. You may include `customer_no` as an extra field to label results.
3. Upload the CSV. The app checks that each required field is present, applies the saved preprocessing pipeline, calculates the model-risk score, assigns within-batch risk deciles, and makes the result downloadable.

An identical blank template is also included as `credit_risk_input_template.csv`, so you can open it in Excel before starting the app.
`credit_risk_demo_upload.csv` contains ten deliberately blank, fictitious customer rows for testing the upload flow only; never use its results for a credit decision.

The application deliberately does **not** accept `Cust_demo.csv`, `Cust_account.csv`, and `CUST_ENQUIRY.csv` directly. Those are raw source files and must first pass through the notebook's aggregation, time-consistency, privacy, and feature-engineering steps. This prevents a misleading score from incomplete raw data.

## Project layout

```text
credit-risk-webapp/
├── app.py                         # Flask routes, upload validation, and scoring
├── model_artifacts/                # Put the exported .joblib model here
├── static/styles.css               # Responsive application design
├── templates/                      # Dashboard and scoring screens
└── requirements.txt
```

## Important governance note

The notebook correctly identifies the model output as a ranking score, not a calibrated probability of default. This demo therefore offers review-routing suggestions only. Production use needs policy approval, fairness testing, monitoring, secure access controls, audit logging, and out-of-time validation.
