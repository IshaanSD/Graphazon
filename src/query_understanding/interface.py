from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import re
from typing import Any, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from src.data.schemas import QueryUnderstandingOutput

import logging
logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_FIRST_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def _read_prompt(name: str) -> str:
    prompt_path = Path(__file__).parent / "prompts" / name
    logger.debug("Loading prompt file: %s", prompt_path)
    return prompt_path.read_text(encoding="utf-8")

def _strip_code_fences(text: str) -> str:
    # Removes leading ```json and trailing ``` (if present)
    return _CODE_FENCE_RE.sub("", text).strip()

def _parse_json_strict(text: str) -> Dict[str, Any]:
    # Handle cases where model returns extra whitespace
    raw = text or ""
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: extract the first {...} JSON object from the text
        m = _FIRST_JSON_OBJ_RE.search(cleaned)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        raise

def _truncate(text: str, max_len: int = 500) -> str:
    if text is None:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "... [truncated]"

def _as_str_list(value: Any) -> List[str]:
    """
    Normalize a JSON value into a list[str].
    Accepts: None, string, list, tuple, set.
    """
    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for x in value:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    # Fallback: coerce single non-iterable to one-item list
    s = str(value).strip()
    return [s] if s else []

def build_query_understanding_chain(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    logger.info(
        "Initializing query understanding chain (model=%s, temperature=%.2f)",
        model_name,
        temperature,
    )
    load_dotenv()

    llm = ChatOpenAI(model=model_name, temperature=temperature)

    intent_prompt = PromptTemplate.from_template(_read_prompt("intent.txt"))
    attrs_prompt = PromptTemplate.from_template(_read_prompt("attributes.txt"))
    rewrite_prompt = PromptTemplate.from_template(_read_prompt("rewrite.txt"))

    return llm, intent_prompt, attrs_prompt, rewrite_prompt

def understand_query(
    *,
    query_id: str,
    query: str,
    locale: str,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> QueryUnderstandingOutput:
    logger.info(
        "Understanding query (query_id=%s, locale=%s): %s",
        query_id,
        locale,
        _truncate(query, 200),
    )
    llm, intent_prompt, attrs_prompt, rewrite_prompt = build_query_understanding_chain(
        model_name=model_name, temperature=temperature
    )

    # Intent
    intent_msg = intent_prompt.format(query=query, locale=locale)
    logger.debug("Intent prompt:\n%s", _truncate(intent_msg))
    intent_raw = llm.invoke(intent_msg).content
    logger.debug("Intent raw output:\n%s", _truncate(intent_raw))
    intent_json = _parse_json_strict(intent_raw)
    intent = intent_json["intent"]
    logger.info("Predicted intent=%s (query_id=%s)", intent, query_id)

    # Attributes
    attrs_msg = attrs_prompt.format(query=query, locale=locale)
    logger.debug("Attributes prompt:\n%s", _truncate(attrs_msg))
    attrs_raw = llm.invoke(attrs_msg).content
    logger.debug("Attributes raw output:\n%s", _truncate(attrs_raw))
    attrs = _parse_json_strict(attrs_raw)

    # Rewrite
    rewrite_msg = rewrite_prompt.format(query=query, locale=locale)
    logger.debug("Rewrite prompt:\n%s", _truncate(rewrite_msg))
    rewrite_raw = llm.invoke(rewrite_msg).content
    logger.debug("Rewrite raw output:\n%s", _truncate(rewrite_raw))
    rewrite_json = _parse_json_strict(rewrite_raw)
    rewritten_query = rewrite_json["rewritten_query"]
    logger.info(
        "Finished query understanding (query_id=%s, rewritten='%s')",
        query_id,
        _truncate(rewritten_query, 200),
    )
    product_type = product_type = attrs.get("product_type")
    if isinstance(product_type, str):
        product_type = product_type.strip() or None
    else:
        product_type = None if product_type is None else str(product_type)
    return QueryUnderstandingOutput(
        query_id=str(query_id),
        raw_query=query,
        locale=locale,
        intent=intent,
        rewritten_query=rewritten_query,
        product_type=product_type,
        brands=_as_str_list(attrs.get("brands")),
        colors=_as_str_list(attrs.get("colors")),
        constraints=_as_str_list(attrs.get("constraints")),
        compatibility=_as_str_list(attrs.get("compatibility")),
        model_name=model_name,
    )