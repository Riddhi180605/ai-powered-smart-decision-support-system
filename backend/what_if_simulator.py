"""What-if scenario simulation utilities for ML predictions.

Required public functions:
- simulate_scenario(model, input_data, changes)
- compare_results(old_pred, new_pred)
- generate_explanation(...)
"""

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI


def _sanitize_api_key(raw_key: Optional[str]) -> str:
    if raw_key is None:
        return ""
    key = str(raw_key).strip()
    if not key:
        return ""
    if (key.startswith('"') and key.endswith('"')) or (key.startswith("'") and key.endswith("'")):
        key = key[1:-1].strip()
    return key


def _provider_api_env(provider: str) -> str:
    provider = (provider or "").strip().lower()
    if provider == "openai":
        return "OPENAI_API_KEY"
    return "GROQ_API_KEY"


def _resolve_provider(provider: Optional[str]) -> str:
    explicit = (provider or "").strip().lower()
    if explicit:
        return explicit

    env_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if env_provider:
        return env_provider

    return "groq"


def _load_api_key_from_env_files(key_names: List[str]) -> str:
    env_paths = [
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]

    key_set = {k.strip() for k in key_names if k and k.strip()}

    for env_path in env_paths:
        if not env_path.exists():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip().lstrip("\ufeff")
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("export "):
                    stripped = stripped[len("export ") :].strip()
                if "=" not in stripped:
                    continue
                name, value = stripped.split("=", 1)
                if name.strip() in key_set:
                    parsed = _sanitize_api_key(value.split(" #", 1)[0].strip())
                    if parsed:
                        return parsed
        except Exception:
            continue

    return ""


def _resolve_llm_api_key(api_key: Optional[str], provider: str) -> str:
    direct = _sanitize_api_key(api_key)
    if direct:
        return direct

    env_name = _provider_api_env(provider)
    from_env = _sanitize_api_key(os.getenv(env_name))
    if from_env:
        return from_env

    file_key = _load_api_key_from_env_files([env_name])
    if file_key:
        os.environ[env_name] = file_key
        return file_key

    return ""


def _resolve_base_url(provider: str) -> Optional[str]:
    explicit = os.getenv("LLM_BASE_URL", "").strip()
    if explicit:
        return explicit
    if provider == "groq":
        return "https://api.groq.com/openai/v1"
    return None


def _resolve_llm_models(provider: str) -> List[str]:
    configured = os.getenv("LLM_MODELS", "").strip()
    if configured:
        models = [m.strip() for m in configured.split(",") if m.strip()]
        if models:
            return models

    single = os.getenv("LLM_MODEL", "").strip()
    if single:
        return [single]

    if provider == "openai":
        return ["gpt-4o-mini", "gpt-4.1-mini"]

    return ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]


def _parse_change_value(old_value: Any, change: Any) -> Any:
    """Support absolute set, +/- deltas, and percentage changes like '+20%'."""
    if isinstance(change, dict):
        mode = str(change.get("mode", "set")).strip().lower()
        raw_val = change.get("value")

        if mode == "set":
            return raw_val

        try:
            if mode == "add":
                return float(old_value) + float(raw_val)
            if mode == "multiply":
                return float(old_value) * float(raw_val)
            if mode == "percent":
                return float(old_value) * (1.0 + float(raw_val) / 100.0)
        except Exception:
            # Non-numeric values should use set mode in what-if scenarios.
            return raw_val

        return raw_val

    if isinstance(change, str):
        cleaned = change.strip()

        if cleaned.endswith("%"):
            try:
                pct = float(cleaned[:-1])
                return float(old_value) * (1.0 + pct / 100.0)
            except Exception:
                return cleaned

        if cleaned.startswith("+") or cleaned.startswith("-"):
            # Treat signed numeric strings as additive deltas.
            try:
                return float(old_value) + float(cleaned)
            except Exception:
                return cleaned

        # Otherwise treat as direct set.
        try:
            return float(cleaned)
        except Exception:
            return cleaned

    return change


def _categorical_candidates(value: Any) -> List[str]:
    """Create robust categorical match candidates for exact and date-like values."""
    if value is None:
        return [""]

    raw = str(value).strip()
    if not raw:
        return [""]

    candidates: List[str] = [raw]

    # Normalize datetime/date-ish values to improve matching against encoder classes.
    parsed_dt = None
    try:
        if isinstance(value, pd.Timestamp):
            parsed_dt = value
        elif isinstance(value, (datetime, date)):
            parsed_dt = pd.Timestamp(value)
        else:
            parsed_dt = pd.to_datetime(raw, errors="coerce")
    except Exception:
        parsed_dt = None

    if parsed_dt is not None and not pd.isna(parsed_dt):
        candidates.extend(
            [
                parsed_dt.strftime("%Y-%m-%d"),
                parsed_dt.strftime("%Y-%m-%d %H:%M:%S"),
                str(parsed_dt.date()),
                parsed_dt.isoformat(),
            ]
        )

    # Also include case-folded variants for text categories.
    candidates.extend([c.lower() for c in list(candidates)])

    # Preserve order while removing duplicates.
    deduped: List[str] = []
    seen = set()
    for item in candidates:
        key = item.strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)

    return deduped


def _prepare_row_dataframe(
    input_data: Dict[str, Any],
    feature_columns: List[str],
    label_encoders: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    row = {}
    label_encoders = label_encoders or {}
    validation_errors: Dict[str, Any] = {}

    for col in feature_columns:
        val = input_data.get(col)

        if col in label_encoders:
            encoder = label_encoders[col]
            classes = [str(c) for c in encoder.classes_]
            class_to_index = {klass: idx for idx, klass in enumerate(classes)}
            class_lower_to_index = {klass.lower(): idx for idx, klass in enumerate(classes)}

            matched_index = None
            for candidate in _categorical_candidates(val):
                if candidate in class_to_index:
                    matched_index = class_to_index[candidate]
                    break
                lowered = candidate.lower()
                if lowered in class_lower_to_index:
                    matched_index = class_lower_to_index[lowered]
                    break

            if matched_index is not None:
                row[col] = int(matched_index)
            elif isinstance(val, (int, np.integer)) and 0 <= int(val) < len(classes):
                row[col] = int(val)
            else:
                validation_errors[col] = {
                    "provided": val,
                    "allowed_values": classes,
                    "message": f"Invalid category for '{col}'. Choose one of the allowed dataset values.",
                }
                # Keep shape valid; caller handles validation failure.
                row[col] = int(encoder.transform([classes[0]])[0])
        else:
            if val is None:
                row[col] = 0.0
            else:
                try:
                    row[col] = float(val)
                except Exception:
                    validation_errors[col] = {
                        "provided": val,
                        "message": f"Invalid numeric value for '{col}'.",
                    }
                    row[col] = 0.0

    return pd.DataFrame([row], columns=feature_columns), validation_errors


def _extract_prediction(
    model: Any,
    processed_df: pd.DataFrame,
    scaler: Optional[Any] = None,
    is_classification: bool = False,
) -> Dict[str, Any]:
    X_values = processed_df.values
    if scaler is not None:
        X_values = scaler.transform(X_values)

    pred_label = model.predict(X_values)[0]

    payload: Dict[str, Any] = {
        "raw_prediction": float(pred_label) if isinstance(pred_label, (int, float, np.number)) else str(pred_label)
    }

    if is_classification:
        score = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_values)[0]
            if len(probs) == 2:
                score = float(probs[1])
            else:
                score = float(np.max(probs))
            payload["class_probabilities"] = [float(p) for p in probs.tolist()]
        elif hasattr(model, "decision_function"):
            decision = float(np.ravel(model.decision_function(X_values))[0])
            score = float(1.0 / (1.0 + np.exp(-decision)))

        if score is None:
            try:
                score = float(pred_label)
            except Exception:
                score = 0.0

        payload["score"] = score

    return payload


def simulate_scenario(
    model: Any,
    input_data: Dict[str, Any],
    changes: Dict[str, Any],
    feature_columns: Optional[List[str]] = None,
    label_encoders: Optional[Dict[str, Any]] = None,
    scaler: Optional[Any] = None,
    is_classification: bool = False,
) -> Dict[str, Any]:
    """Run baseline and changed predictions and return both with applied changes."""
    if feature_columns is None:
        feature_columns = list(input_data.keys())

    baseline = {k: input_data.get(k) for k in feature_columns}
    updated = dict(baseline)

    applied_changes: Dict[str, Dict[str, Any]] = {}
    for feature, change in (changes or {}).items():
        if feature not in updated:
            continue
        old_val = updated[feature]
        new_val = _parse_change_value(old_val, change)
        updated[feature] = new_val
        applied_changes[feature] = {
            "old": old_val,
            "new": new_val,
            "change": change,
        }

    old_df, old_validation = _prepare_row_dataframe(baseline, feature_columns, label_encoders)
    new_df, new_validation = _prepare_row_dataframe(updated, feature_columns, label_encoders)

    old_pred = _extract_prediction(model, old_df, scaler=scaler, is_classification=is_classification)
    new_pred = _extract_prediction(model, new_df, scaler=scaler, is_classification=is_classification)

    return {
        "input_before": baseline,
        "input_after": updated,
        "applied_changes": applied_changes,
        "old_prediction": old_pred,
        "new_prediction": new_pred,
        "validation": {
            "old_input": old_validation,
            "new_input": new_validation,
        },
    }


def compare_results(old_pred: Dict[str, Any], new_pred: Dict[str, Any]) -> Dict[str, Any]:
    """Compare old vs new prediction and compute directional impact."""
    old_score = old_pred.get("score")
    new_score = new_pred.get("score")

    if old_score is None or new_score is None:
        return {
            "comparison_available": False,
            "summary": "Score comparison unavailable for this model output.",
        }

    delta = float(new_score - old_score)
    pct_change = float((delta / old_score) * 100.0) if old_score not in (0, None) else None

    direction = "increased" if delta > 0 else "decreased" if delta < 0 else "unchanged"
    magnitude = "significantly" if abs(delta) >= 0.1 else "moderately" if abs(delta) >= 0.03 else "slightly"

    return {
        "comparison_available": True,
        "old_score": float(old_score),
        "new_score": float(new_score),
        "delta": delta,
        "percent_change": pct_change,
        "direction": direction,
        "magnitude": magnitude,
        "summary": f"Prediction {direction} {magnitude}.",
    }


def _fallback_explanation(
    changes: Dict[str, Any],
    comparison: Dict[str, Any],
    target_label: str,
) -> str:
    if not comparison.get("comparison_available"):
        return (
            "I re-ran the scenario with your feature updates, but this model output does not expose a"
            " comparable risk score. You can still use the class/label change for directional guidance."
        )

    direction = comparison.get("direction", "changed")
    magnitude = comparison.get("magnitude", "")
    old_score = comparison.get("old_score", 0.0)
    new_score = comparison.get("new_score", 0.0)
    changed_keys = ", ".join(changes.keys()) if changes else "the provided features"

    return (
        f"After changing {changed_keys}, the predicted {target_label} moved from {old_score:.4f} to {new_score:.4f}. "
        f"This indicates the risk {direction} {magnitude}. Actionable next step: test smaller increments for the"
        f" same features and prioritize controls for the ones that produce the steepest change."
    )


def generate_explanation(
    changes: Dict[str, Any],
    old_pred: Dict[str, Any],
    new_pred: Dict[str, Any],
    comparison: Dict[str, Any],
    target_label: str = "target outcome",
    llm_provider: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> str:
    """Generate human-language explanation of how feature changes affect prediction."""
    provider = _resolve_provider(llm_provider)
    api_key = _resolve_llm_api_key(llm_api_key, provider)

    if not api_key:
        return _fallback_explanation(changes, comparison, target_label)

    base_url = _resolve_base_url(provider)
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    models = _resolve_llm_models(provider)

    prompt = (
        "Explain how feature changes affect prediction. "
        "Give a natural, business-friendly explanation and actionable insights.\n\n"
        f"Target label: {target_label}\n"
        f"Feature changes: {changes}\n"
        f"Old prediction: {old_pred}\n"
        f"New prediction: {new_pred}\n"
        f"Comparison: {comparison}\n\n"
        "Requirements:\n"
        "- Start with one clear conclusion sentence.\n"
        "- Explain why the prediction changed.\n"
        "- Highlight key changed features and their impact direction.\n"
        "- End with 2 practical action points.\n"
        "- Use markdown bullets and **bold** key feature names/metrics."
    )

    last_error = None
    for model_name in models:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a business analytics assistant. "
                            "Explain scenario simulation outputs in clear human language."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.35,
                max_tokens=400,
            )

            content = response.choices[0].message.content
            if content:
                return content.strip()
        except Exception as exc:
            last_error = str(exc)
            normalized = last_error.lower()
            if (
                "invalid api key" in normalized
                or "incorrect api key" in normalized
                or "authentication" in normalized
                or "insufficient_quota" in normalized
                or "quota" in normalized
                or "billing" in normalized
            ):
                break

    # Fall back to deterministic explanation if LLM call fails.
    return _fallback_explanation(changes, comparison, target_label)
