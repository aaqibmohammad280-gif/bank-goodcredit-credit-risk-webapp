"""Local Flask application for the Bank GoodCredit credit-risk scorecard."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
from flask import Flask, abort, render_template, request, send_file, url_for


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # CSV uploads only; 15 MB limit.

MODEL_PATH = Path(app.root_path) / "model_artifacts" / "bank_goodcredit_champion_model.joblib"
RESULT_CACHE: dict[str, tuple[bytes, str]] = {}

REPORT_METRICS = {
    "roc_auc": 0.7045,
    "gini_points": 40.91,
    "benchmark": 37.90,
    "pr_auc": 0.0856,
    "top_decile_capture": 23.88,
    "top_two_decile_capture": 40.80,
}

DECILE_DATA = [
    {"decile": 10, "bad_rate": 10.04, "capture": 23.88},
    {"decile": 9, "bad_rate": 7.11, "capture": 16.92},
    {"decile": 8, "bad_rate": 5.86, "capture": 13.93},
    {"decile": 7, "bad_rate": 6.90, "capture": 16.42},
    {"decile": 6, "bad_rate": 3.97, "capture": 9.45},
    {"decile": 5, "bad_rate": 2.09, "capture": 4.98},
    {"decile": 4, "bad_rate": 2.93, "capture": 6.97},
    {"decile": 3, "bad_rate": 0.84, "capture": 1.99},
    {"decile": 2, "bad_rate": 1.46, "capture": 3.48},
    {"decile": 1, "bad_rate": 0.84, "capture": 1.99},
]


def load_artifact() -> tuple[dict | None, str | None]:
    """Return a validated model artifact and a safe status message."""
    if not MODEL_PATH.exists():
        return None, "Model artifact not installed"

    try:
        artifact = joblib.load(MODEL_PATH)
    except Exception:
        return None, "Model needs scikit-learn 1.6.1; recreate the PyCharm interpreter from requirements.txt"

    required_keys = {"model", "input_features"}
    if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
        return None, "Model artifact has an unsupported format"
    if not isinstance(artifact["input_features"], list):
        return None, "Model artifact does not contain an input schema"
    return artifact, None


def artifact_status() -> tuple[str, str]:
    artifact, error = load_artifact()
    if error:
        return "Offline", error
    return "Ready", f"Champion model ready · {len(artifact['input_features'])} input fields"


def numeric_input_columns(model) -> list[str]:
    """Read numeric source columns from the fitted preprocessing pipeline."""
    try:
        preprocessor = model.named_steps["preprocessor"]
        for name, _transformer, columns in preprocessor.transformers_:
            if name == "numeric":
                return list(columns)
    except (AttributeError, KeyError, TypeError):
        pass
    return []


def batch_deciles(scores: pd.Series) -> pd.Series:
    """Assign within-upload rank bands; 10 bands when the batch is large enough."""
    bands = min(10, len(scores))
    if bands < 2:
        return pd.Series([1], index=scores.index, dtype="int64")
    return pd.qcut(scores.rank(method="first"), q=bands, labels=False).add(1).astype(int)


def review_recommendation(decile: int, band_count: int) -> str:
    """Apply only the report's triage suggestions, never an automatic credit decision."""
    if band_count < 10:
        return "Use approved policy; batch is too small for standard decile triage"
    if decile == 10:
        return "Prioritise manual review"
    if decile == 9:
        return "Enhanced review"
    return "Standard policy route"


def prepare_scoring_frame(raw_data: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Validate a model-ready CSV and preserve expected feature dtypes where possible."""
    features = artifact["input_features"]
    missing = [column for column in features if column not in raw_data.columns]
    if missing:
        preview = ", ".join(missing[:12])
        suffix = " …" if len(missing) > 12 else ""
        raise ValueError(f"The upload is missing {len(missing)} required field(s): {preview}{suffix}")

    frame = raw_data.reindex(columns=features).copy()
    for column in numeric_input_columns(artifact["model"]):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


@app.route("/")
def dashboard():
    status, detail = artifact_status()
    return render_template(
        "dashboard.html",
        metrics=REPORT_METRICS,
        deciles=DECILE_DATA,
        status=status,
        status_detail=detail,
    )


@app.route("/score", methods=["GET", "POST"])
def score():
    status, detail = artifact_status()
    context = {"status": status, "status_detail": detail, "error": None, "summary": None, "rows": None, "download_url": None}

    if request.method == "POST":
        artifact, model_error = load_artifact()
        uploaded_file = request.files.get("portfolio")

        if model_error:
            context["error"] = "Install the champion-model artifact before scoring a portfolio."
        elif not uploaded_file or not uploaded_file.filename:
            context["error"] = "Choose a model-ready CSV file to score."
        elif not uploaded_file.filename.lower().endswith(".csv"):
            context["error"] = "Only CSV uploads are supported."
        else:
            try:
                raw_data = pd.read_csv(uploaded_file)
                if raw_data.empty:
                    raise ValueError("The uploaded CSV contains no customer rows.")
                scoring_frame = prepare_scoring_frame(raw_data, artifact)
                scores = pd.Series(
                    artifact["model"].predict_proba(scoring_frame)[:, 1],
                    index=raw_data.index,
                    name="Model risk score",
                )
                deciles = batch_deciles(scores)
                band_count = min(10, len(scores))

                identifier_column = next(
                    (name for name in ["customer_no", "customer_id", "Customer ID", "id"] if name in raw_data.columns),
                    None,
                )
                identifiers = raw_data[identifier_column] if identifier_column else pd.Series(
                    [f"Row {index + 1}" for index in range(len(raw_data))], index=raw_data.index
                )

                results = pd.DataFrame(
                    {
                        "Customer reference": identifiers,
                        "Model risk score": scores.round(6),
                        "Batch risk decile": deciles,
                        "Recommended route": [review_recommendation(value, band_count) for value in deciles],
                    }
                ).sort_values("Model risk score", ascending=False)

                result_key = uuid4().hex
                csv_bytes = results.to_csv(index=False).encode("utf-8")
                RESULT_CACHE[result_key] = (csv_bytes, f"credit_risk_scores_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.csv")

                context.update(
                    {
                        "summary": {
                            "customers": len(results),
                            "highest_score": scores.max(),
                            "review_count": int((deciles >= max(1, band_count - 1)).sum()) if band_count == 10 else None,
                            "bands": band_count,
                        },
                        "rows": results.head(12).to_dict(orient="records"),
                        "download_url": url_for("download_results", result_key=result_key),
                    }
                )
            except (ValueError, TypeError, KeyError) as error:
                context["error"] = str(error)
            except Exception:
                context["error"] = "Scoring could not be completed. Check that the CSV was prepared from the model's input template."

    return render_template("score.html", **context)


@app.route("/download-template")
def download_template():
    artifact, error = load_artifact()
    if error:
        abort(404)
    content = pd.DataFrame(columns=artifact["input_features"]).to_csv(index=False).encode("utf-8")
    return send_file(
        BytesIO(content),
        as_attachment=True,
        download_name="credit_risk_model_input_template.csv",
        mimetype="text/csv",
    )


@app.route("/download-results/<result_key>")
def download_results(result_key: str):
    saved_result = RESULT_CACHE.get(result_key)
    if not saved_result:
        abort(404)
    content, filename = saved_result
    return send_file(BytesIO(content), as_attachment=True, download_name=filename, mimetype="text/csv")


@app.errorhandler(413)
def upload_too_large(_error):
    status, detail = artifact_status()
    return render_template(
        "score.html",
        status=status,
        status_detail=detail,
        error="The upload exceeds the 15 MB limit.",
        summary=None,
        rows=None,
        download_url=None,
    ), 413


if __name__ == "__main__":
    app.run(debug=True)
