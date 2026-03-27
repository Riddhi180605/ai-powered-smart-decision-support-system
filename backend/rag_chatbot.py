"""RAG chatbot for business analytics queries.

This module implements a retrieval-augmented generation pipeline that uses:
- SentenceTransformers embeddings
- FAISS vector store
- OpenAI-compatible chat completion APIs (OpenAI, Groq)

Required public methods:
- create_vector_store(data)
- retrieve_context(query)
- generate_answer(query, context)
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BusinessAnalyticsRAG:
    """RAG system for natural language business analytics questions."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_provider: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_models: Optional[List[str]] = None,
    ):
        self.embedding_model_name = embedding_model
        self.embedder = SentenceTransformer(embedding_model)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_embeddings: Optional[np.ndarray] = None

        self.provider = self._resolve_provider(llm_provider)
        self.api_key = self._resolve_api_key(llm_api_key, self.provider)
        self.client = self._build_client(self.provider, self.api_key)
        self.llm_models = llm_models or self._resolve_llm_models(self.provider)

    @staticmethod
    def _sanitize_api_key(raw_key: Optional[str]) -> str:
        if raw_key is None:
            return ""
        key = str(raw_key).strip()
        if not key:
            return ""
        if (key.startswith('"') and key.endswith('"')) or (key.startswith("'") and key.endswith("'")):
            key = key[1:-1].strip()
        return key

    @staticmethod
    def _provider_env_var(provider: str) -> str:
        return "OPENAI_API_KEY" if provider == "openai" else "GROQ_API_KEY"

    @classmethod
    def _resolve_provider(cls, provider: Optional[str]) -> str:
        explicit = (provider or "").strip().lower()
        if explicit:
            return explicit
        env_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        if env_provider:
            return env_provider
        return "groq"

    @classmethod
    def _resolve_api_key(cls, api_key: Optional[str], provider: str) -> str:
        direct = cls._sanitize_api_key(api_key)
        if direct:
            return direct

        env_name = cls._provider_env_var(provider)
        from_env = cls._sanitize_api_key(os.getenv(env_name))
        if from_env:
            return from_env

        from_file = cls._load_key_from_env_files([env_name])
        if from_file:
            os.environ[env_name] = from_file
            return from_file

        return ""

    @classmethod
    def _load_key_from_env_files(cls, key_names: List[str]) -> str:
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
                        clean = cls._sanitize_api_key(value.split(" #", 1)[0].strip())
                        if clean:
                            return clean
            except Exception as exc:
                logger.warning("Failed reading env file %s: %s", env_path, exc)
        return ""

    @staticmethod
    def _resolve_llm_models(provider: str) -> List[str]:
        configured = os.getenv("LLM_MODELS", "").strip()
        if configured:
            parsed = [m.strip() for m in configured.split(",") if m.strip()]
            if parsed:
                return parsed

        single = os.getenv("LLM_MODEL", "").strip()
        if single:
            return [single]

        if provider == "openai":
            return ["gpt-4o-mini", "gpt-4.1-mini"]
        return ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]

    @staticmethod
    def _resolve_base_url(provider: str) -> Optional[str]:
        explicit = os.getenv("LLM_BASE_URL", "").strip()
        if explicit:
            return explicit
        if provider == "groq":
            return "https://api.groq.com/openai/v1"
        return None

    def _build_client(self, provider: str, api_key: str) -> Optional[OpenAI]:
        if not api_key:
            return None
        base_url = self._resolve_base_url(provider)
        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        return OpenAI(api_key=api_key)

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
        if not text:
            return []

        normalized = " ".join(text.split())
        if len(normalized) <= chunk_size:
            return [normalized]

        chunks: List[str] = []
        step = max(1, chunk_size - overlap)
        start = 0
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(normalized):
                break
            start += step

        return chunks

    @staticmethod
    def _to_json_safe(value: Any) -> Any:
        """Convert pandas/numpy values into JSON-serializable Python primitives."""
        if value is None:
            return None

        # Keep native JSON primitives as-is.
        if isinstance(value, (str, int, float, bool)):
            return value

        # Normalize timestamps and date-like values.
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return str(value)

        # Recursively normalize containers.
        if isinstance(value, dict):
            return {str(k): BusinessAnalyticsRAG._to_json_safe(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [BusinessAnalyticsRAG._to_json_safe(v) for v in value]

        # Handle pandas missing values and numpy scalar types.
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        if isinstance(value, np.generic):
            return value.item()

        # Last-resort conversion for unsupported objects.
        return str(value)

    @staticmethod
    def _df_to_text(df: pd.DataFrame, max_rows: int = 250) -> str:
        if df is None or df.empty:
            return "Historical dataset is empty."

        sample_df = df.head(max_rows).copy()
        schema = {col: str(dtype) for col, dtype in sample_df.dtypes.items()}

        numeric_summary = {}
        for col in sample_df.select_dtypes(include=[np.number]).columns:
            numeric_summary[col] = {
                "mean": float(sample_df[col].mean()) if pd.notna(sample_df[col].mean()) else None,
                "median": float(sample_df[col].median()) if pd.notna(sample_df[col].median()) else None,
                "min": float(sample_df[col].min()) if pd.notna(sample_df[col].min()) else None,
                "max": float(sample_df[col].max()) if pd.notna(sample_df[col].max()) else None,
            }

        categorical_summary = {}
        for col in sample_df.select_dtypes(exclude=[np.number]).columns:
            top_vals = sample_df[col].astype(str).value_counts().head(5).to_dict()
            categorical_summary[col] = top_vals

        records_preview = sample_df.head(60).to_dict(orient="records")

        payload = {
            "source": "historical_dataset",
            "rows": int(len(df)),
            "columns": list(df.columns),
            "schema": schema,
            "numeric_summary": numeric_summary,
            "categorical_summary": categorical_summary,
            "records_preview": records_preview,
        }
        return json.dumps(BusinessAnalyticsRAG._to_json_safe(payload), ensure_ascii=True)

    @staticmethod
    def _json_text(source_name: str, data: Any) -> str:
        payload = {"source": source_name, "content": data or {}}
        return json.dumps(BusinessAnalyticsRAG._to_json_safe(payload), ensure_ascii=True)

    def create_vector_store(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create embeddings and build FAISS vector index from provided data sources."""
        historical_dataset = data.get("historical_dataset")
        ml_outputs = data.get("ml_outputs") or {}
        shap_explanations = data.get("shap_explanations") or {}

        docs: List[Dict[str, Any]] = []

        if isinstance(historical_dataset, pd.DataFrame):
            docs.append({
                "source": "historical_dataset",
                "text": self._df_to_text(historical_dataset),
            })
        else:
            docs.append({
                "source": "historical_dataset",
                "text": "Historical dataset is not available.",
            })

        docs.append({
            "source": "ml_outputs",
            "text": self._json_text("ml_outputs", ml_outputs),
        })
        docs.append({
            "source": "shap_explanations",
            "text": self._json_text("shap_explanations", shap_explanations),
        })

        chunked_docs: List[Dict[str, Any]] = []
        for doc in docs:
            for idx, piece in enumerate(self._chunk_text(doc["text"])):
                chunked_docs.append(
                    {
                        "id": len(chunked_docs),
                        "source": doc["source"],
                        "chunk_index": idx,
                        "text": piece,
                    }
                )

        if not chunked_docs:
            raise ValueError("No data available to create vector store")

        texts = [item["text"] for item in chunked_docs]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = embeddings.astype("float32")

        dim = int(embeddings.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index
        self.chunk_embeddings = embeddings
        self.chunks = chunked_docs

        return {
            "success": True,
            "vector_db": "faiss",
            "embedding_model": self.embedding_model_name,
            "num_chunks": len(self.chunks),
            "sources": sorted(list({item["source"] for item in self.chunks})),
        }

    def retrieve_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve relevant chunks from FAISS based on query similarity."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if self.index is None or not self.chunks:
            raise ValueError("Vector store is not initialized. Run create_vector_store(data) first.")

        query_embedding = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        query_embedding = query_embedding.astype("float32")

        k = min(max(top_k, 1), len(self.chunks))
        distances, indices = self.index.search(query_embedding, k)

        contexts: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[int(idx)]
            contexts.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "source": chunk["source"],
                    "text": chunk["text"],
                }
            )

        context_text = "\n\n".join(
            [f"[{item['source']} | score={item['score']:.4f}] {item['text']}" for item in contexts]
        )

        return {
            "query": query,
            "contexts": contexts,
            "context_text": context_text,
        }

    def _normalize_chat_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize chatbot context into expected top-level keys."""
        if not isinstance(context, dict):
            return {
                "dataset_summary": {},
                "model_info": {},
                "shap_summary": {},
                "predictions_summary": {},
                "trend_analysis": {},
                "retrieved_context": [],
            }

        normalized = {
            "dataset_summary": context.get("dataset_summary") or {},
            "model_info": context.get("model_info") or {},
            "shap_summary": context.get("shap_summary") or {},
            "predictions_summary": context.get("predictions_summary") or {},
            "trend_analysis": context.get("trend_analysis") or {},
            "retrieved_context": context.get("retrieved_context") or [],
        }
        return self._to_json_safe(normalized)

    @staticmethod
    def _classify_question(query: str) -> str:
        text = (query or "").strip().lower()
        if not text:
            return "general"

        if "xgboost" in text and any(k in text for k in ["why", "selected", "choose", "chosen", "best"]):
            return "model_selection"

        if any(k in text for k in ["decrease", "decreased", "drop", "down", "decline"]) and any(
            k in text for k in ["profit", "revenue", "sales", "target", "performance"]
        ):
            return "trend_explanation"

        if any(k in text for k in ["feature", "important", "importance", "shap", "factor", "driver"]):
            return "feature_importance"

        return "general"

    @staticmethod
    def _has_context_data(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, dict):
            return any(BusinessAnalyticsRAG._has_context_data(v) for v in value.values())
        if isinstance(value, list):
            return any(BusinessAnalyticsRAG._has_context_data(v) for v in value)
        if isinstance(value, str):
            return bool(value.strip())
        return True

    def _context_sufficient_for_question(self, question_type: str, context: Dict[str, Any]) -> bool:
        dataset_summary = context.get("dataset_summary") or {}
        model_info = context.get("model_info") or {}
        shap_summary = context.get("shap_summary") or {}
        predictions_summary = context.get("predictions_summary") or {}
        trend_analysis = context.get("trend_analysis") or {}

        if question_type == "model_selection":
            return self._has_context_data(model_info)
        if question_type == "feature_importance":
            return self._has_context_data(shap_summary)
        if question_type == "trend_explanation":
            return self._has_context_data(trend_analysis) or self._has_context_data(predictions_summary)
        return any(
            self._has_context_data(item)
            for item in [dataset_summary, model_info, shap_summary, predictions_summary, trend_analysis]
        )

    def _build_chat_system_prompt(self, question_type: str) -> str:
        base_prompt = (
            "You are an expert data scientist and business analyst. "
            "Answer based ONLY on the provided project data. "
            "Do NOT give generic answers."
        )

        intent_rules = {
            "model_selection": "Focus on model comparison metrics and explain why a model was selected.",
            "trend_explanation": "Focus on trend signals in predictions and historical trend analysis.",
            "feature_importance": "Focus on SHAP and feature importance values.",
            "general": "Provide a concise, data-grounded answer.",
        }

        return (
            f"{base_prompt}\n"
            f"Question intent: {question_type}. {intent_rules.get(question_type, intent_rules['general'])}\n"
            "If the provided data cannot support the answer, respond exactly with: Not enough data available\n"
            "Response formatting requirements:\n"
            "1. Start with a direct 1-2 sentence answer\n"
            "2. Use ## Section headers for main topics\n"
            "3. Use bullet points (- item) for key findings:\n"
            "   - Bold important metrics with **bold text**\n"
            "   - Keep each point brief and actionable\n"
            "4. Include specific numbers and percentages from the data\n"
            "5. Use **bold** for all important metrics, feature names, and model names\n"
            "6. Organize information with headers and spacing for readability\n"
            "7. Do not mention external assumptions, generic industry advice, or unrelated best practices"
        )


    @staticmethod
    def _extract_sources(context: Dict[str, Any]) -> str:
        source_names = []
        for item in context.get("retrieved_context", []):
            source = str((item or {}).get("source", "")).strip()
            if source and source not in source_names:
                source_names.append(source)
        return ", ".join(source_names)

    def _build_rule_based_fallback(self, query: str, context: Dict[str, Any], question_type: str) -> str:
        """Generate deterministic context-grounded fallback when LLM is unavailable."""
        model_info = context.get("model_info") or {}
        shap_summary = context.get("shap_summary") or {}
        trend_analysis = context.get("trend_analysis") or {}

        if question_type == "model_selection":
            best_model = model_info.get("best_model")
            best_metrics = model_info.get("best_metrics") or {}
            if best_model:
                metrics_text = ", ".join(
                    [
                        f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                        for k, v in best_metrics.items()
                        if k in ["accuracy", "f1_score", "precision", "recall", "rmse", "r2_score", "cv_mean", "cv_rmse"]
                    ]
                )
                return (
                    "- The selected model is **{model}** because it outperformed alternatives on tracked metrics.\n"
                    "- Evidence: {metrics}.\n"
                    "- Reasoning: the selection is based on measured validation/test performance, not a default choice."
                ).format(model=best_model, metrics=metrics_text or "best available score in training results")

        if question_type == "feature_importance":
            feature_importance = shap_summary.get("feature_importance") or []
            if not feature_importance:
                feature_importance = model_info.get("derived_feature_importance") or []
            if feature_importance:
                lines = []
                for item in feature_importance[:5]:
                    name = item.get("feature", "unknown_feature")
                    score = item.get("importance")
                    if isinstance(score, (int, float)):
                        lines.append(f"- **{name}**: importance {float(score):.4f}")
                    else:
                        lines.append(f"- **{name}**: importance available")
                lines.append(
                    "- Reasoning: features with higher contribution values influence predictions more strongly."
                )
                return "\n".join(lines)

        if question_type == "trend_explanation":
            direction = trend_analysis.get("trend_direction")
            historical_avg = trend_analysis.get("historical_avg")
            current_avg = trend_analysis.get("current_avg")
            if direction or historical_avg is not None or current_avg is not None:
                return (
                    "- Trend direction: **{direction}**.\n"
                    "- Historical average: **{hist}**.\n"
                    "- Current predicted average: **{curr}**.\n"
                    "- Reasoning: the change between historical and current levels indicates the observed movement."
                ).format(
                    direction=direction or "not specified",
                    hist=f"{historical_avg:.4f}" if isinstance(historical_avg, (int, float)) else "not available",
                    curr=f"{current_avg:.4f}" if isinstance(current_avg, (int, float)) else "not available",
                )

        return "Not enough data available"

    def generate_chatbot_response(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate context-aware assistant response from structured ML project context."""
        if not query or not str(query).strip():
            return {"success": False, "answer": "Not enough data available"}

        normalized_context = self._normalize_chat_context(context)
        question_type = self._classify_question(query)

        if not self._context_sufficient_for_question(question_type, normalized_context):
            return {
                "success": True,
                "model": None,
                "answer": "Not enough data available",
            }

        system_prompt = self._build_chat_system_prompt(question_type)
        user_payload = {
            "query": query,
            "context": normalized_context,
        }

        if self.client is None:
            return {
                "success": True,
                "model": None,
                "answer": self._build_rule_based_fallback(query, normalized_context, question_type),
            }

        last_error = None
        payload_text = json.dumps(user_payload, ensure_ascii=True)
        for model_name in self.llm_models:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": payload_text},
                    ],
                    temperature=0.1,
                    max_tokens=550,
                )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    continue

                if re.search(r"not enough data", content, re.IGNORECASE):
                    return {"success": True, "model": model_name, "answer": "Not enough data available"}

                sources = self._extract_sources(normalized_context)
                answer = content
                if sources and "sources:" not in content.lower():
                    answer = f"{content}\n\n- Sources: {sources}"

                return {
                    "success": True,
                    "model": model_name,
                    "answer": answer,
                }
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Context-aware chatbot response failed on model %s: %s", model_name, last_error)
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

        fallback_answer = self._build_rule_based_fallback(query, normalized_context, question_type)
        if fallback_answer != "Not enough data available":
            return {
                "success": True,
                "model": None,
                "answer": fallback_answer,
            }

        return {
            "success": False,
            "answer": f"Unable to generate response due to LLM API error: {last_error or 'Unknown error'}",
        }

    def generate_answer(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate grounded answer using only retrieved context and explicit reasoning."""
        context_text = context.get("context_text", "")
        if not context_text.strip():
            return {
                "success": False,
                "answer": "I cannot answer because no retrieved context was provided.",
            }

        prompt = (
            "You are a business analytics assistant. "
            "Answer based only on retrieved context. "
            "If the context is insufficient, explicitly say what is missing. "
            "Write naturally like a human analyst: clear, practical, and conversational. "
            "Avoid robotic formatting and avoid making up facts.\n\n"
            f"User query: {query}\n\n"
            f"Retrieved context:\n{context_text}\n\n"
            "Response requirements:\n"
            "1. Start with a direct, concise answer in 1-2 sentences.\n"
            "2. Add a section header: ## Key Insights\n"
            "3. Break down the explanation into clear bullet points:\n"
            "   - Use **bold text** for important metrics and features\n"
            "   - Keep each point brief and actionable\n"
            "   - Add 2-3 main points explaining the answer\n"
            "4. If applicable, add a ## Recommendations section with bullet points\n"
            "5. End with: **Sources:** <comma-separated source names>\n"
            "6. Use proper spacing between sections\n"
            "7. If context is missing, clearly state what specific data is needed.\n"
            "8. Format all important metrics and feature names in **bold**."
        )

        if self.client is None:
            return {
                "success": False,
                "answer": "LLM API key is not configured. Set GROQ_API_KEY or OPENAI_API_KEY to generate answers.",
            }

        last_error = None
        for model_name in self.llm_models:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a business analytics assistant. "
                                "You must never use information outside retrieved context. "
                                "Your tone should be natural and human, not robotic."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=500,
                )

                content = response.choices[0].message.content
                if content:
                    return {
                        "success": True,
                        "model": model_name,
                        "answer": content.strip(),
                    }
                return {
                    "success": False,
                    "model": model_name,
                    "answer": "Model returned an empty response.",
                }
            except Exception as exc:
                last_error = str(exc)
                logger.warning("RAG answer generation failed on model %s: %s", model_name, last_error)
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

        return {
            "success": False,
            "answer": f"Unable to generate answer due to LLM API error: {last_error or 'Unknown error'}",
        }


def create_vector_store(data: Dict[str, Any], rag_system: BusinessAnalyticsRAG) -> Dict[str, Any]:
    """Convenience wrapper for required API shape."""
    return rag_system.create_vector_store(data)


def retrieve_context(query: str, rag_system: BusinessAnalyticsRAG) -> Dict[str, Any]:
    """Convenience wrapper for required API shape."""
    return rag_system.retrieve_context(query)


def generate_answer(query: str, context: Dict[str, Any], rag_system: BusinessAnalyticsRAG) -> Dict[str, Any]:
    """Convenience wrapper for required API shape."""
    return rag_system.generate_answer(query, context)


def generate_chatbot_response(
    query: str, context: Dict[str, Any], rag_system: BusinessAnalyticsRAG
) -> Dict[str, Any]:
    """Convenience wrapper for context-aware chatbot response API shape."""
    return rag_system.generate_chatbot_response(query, context)
