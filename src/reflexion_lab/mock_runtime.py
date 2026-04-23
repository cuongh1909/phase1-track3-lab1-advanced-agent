from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_ENABLED = os.getenv("REFLEXION_RUNTIME", "openai").lower() == "openai"


@dataclass
class LLMCallStats:
    tokens: int
    latency_ms: int


def _build_context_text(example: QAExample) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(example.context, start=1):
        parts.append(f"[{i}] {chunk.title}\n{chunk.text}")
    return "\n\n".join(parts)


def _call_openai(system_prompt: str, user_prompt: str, json_mode: bool = False) -> tuple[str, LLMCallStats]:
    if OpenAI is None:
        raise RuntimeError("Package 'openai' is not installed. Run: pip install -r requirements.txt")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Please set it before running benchmark in openai mode.")
    client = OpenAI(api_key=api_key)
    started = time.perf_counter()
    params = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }
    if json_mode:
        params["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**params)
    latency_ms = int((time.perf_counter() - started) * 1000)
    text = (response.choices[0].message.content or "").strip()
    usage = response.usage
    total_tokens = int(usage.total_tokens) if usage and usage.total_tokens is not None else 0
    return text, LLMCallStats(tokens=total_tokens, latency_ms=latency_ms)


def _mock_actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if example.qid not in FIRST_ATTEMPT_WRONG:
        return example.gold_answer
    if agent_type == "react":
        return FIRST_ATTEMPT_WRONG[example.qid]
    if attempt_id == 1 and not reflection_memory:
        return FIRST_ATTEMPT_WRONG[example.qid]
    return example.gold_answer


def _mock_evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
    if normalize_answer(answer) == "london":
        return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
    return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])


def _mock_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
    return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)

def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    answer, _ = actor_answer_with_stats(example, attempt_id, agent_type, reflection_memory)
    return answer


def actor_answer_with_stats(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> tuple[str, LLMCallStats]:
    if not OPENAI_ENABLED:
        return _mock_actor_answer(example, attempt_id, agent_type, reflection_memory), LLMCallStats(tokens=0, latency_ms=0)
    memory_text = "\n".join(f"- {item}" for item in reflection_memory) if reflection_memory else "- (none)"
    user_prompt = (
        f"Question:\n{example.question}\n\n"
        f"Context:\n{_build_context_text(example)}\n\n"
        f"Agent type: {agent_type}\n"
        f"Attempt: {attempt_id}\n"
        f"Reflection memory:\n{memory_text}\n\n"
        "Return only the final short answer."
    )
    text, stats = _call_openai(ACTOR_SYSTEM, user_prompt, json_mode=False)
    return text.strip().strip('"'), stats

def evaluator(example: QAExample, answer: str) -> JudgeResult:
    result, _ = evaluator_with_stats(example, answer)
    return result


def evaluator_with_stats(example: QAExample, answer: str) -> tuple[JudgeResult, LLMCallStats]:
    if not OPENAI_ENABLED:
        return _mock_evaluator(example, answer), LLMCallStats(tokens=0, latency_ms=0)
    user_prompt = (
        f"Question:\n{example.question}\n\n"
        f"Gold answer:\n{example.gold_answer}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        f"Context:\n{_build_context_text(example)}"
    )
    text, stats = _call_openai(EVALUATOR_SYSTEM, user_prompt, json_mode=True)
    payload = json.loads(text)
    payload.setdefault("missing_evidence", [])
    payload.setdefault("spurious_claims", [])
    score = int(payload.get("score", 0))
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        score = 1
    elif score not in (0, 1):
        score = 0
    payload["score"] = score
    return JudgeResult.model_validate(payload), stats

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    reflection, _ = reflector_with_stats(example, attempt_id, judge)
    return reflection


def reflector_with_stats(example: QAExample, attempt_id: int, judge: JudgeResult) -> tuple[ReflectionEntry, LLMCallStats]:
    if not OPENAI_ENABLED:
        return _mock_reflector(example, attempt_id, judge), LLMCallStats(tokens=0, latency_ms=0)
    user_prompt = (
        f"Question:\n{example.question}\n\n"
        f"Context:\n{_build_context_text(example)}\n\n"
        f"Attempt id: {attempt_id}\n"
        f"Previous evaluator score: {judge.score}\n"
        f"Failure reason: {judge.reason}\n"
        f"Missing evidence: {judge.missing_evidence}\n"
        f"Spurious claims: {judge.spurious_claims}"
    )
    text, stats = _call_openai(REFLECTOR_SYSTEM, user_prompt, json_mode=True)
    payload = json.loads(text)
    payload["attempt_id"] = attempt_id
    return ReflectionEntry.model_validate(payload), stats
