from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .mock_runtime import (
    actor_answer_with_stats,
    evaluator_with_stats,
    reflector_with_stats,
)
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord


def _infer_failure_mode(reason: str) -> str:
    lowered = reason.lower()
    if "first hop" in lowered or "second hop" in lowered or "multi-hop" in lowered:
        return "incomplete_multi_hop"
    if "drift" in lowered or "wrong entity" in lowered:
        return "entity_drift"
    if "loop" in lowered:
        return "looping"
    return "wrong_final_answer"


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        final_reason = "No evaluation produced."
        for attempt_id in range(1, self.max_attempts + 1):
            answer, actor_stats = actor_answer_with_stats(example, attempt_id, self.agent_type, reflection_memory)
            judge, evaluator_stats = evaluator_with_stats(example, answer)
            token_estimate = actor_stats.tokens + evaluator_stats.tokens
            latency_ms = actor_stats.latency_ms + evaluator_stats.latency_ms
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)
            final_answer = answer
            final_score = judge.score
            final_reason = judge.reason
            if judge.score == 1:
                traces.append(trace)
                break

            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection, reflector_stats = reflector_with_stats(example, attempt_id, judge)
                reflections.append(reflection)
                trace.reflection = reflection
                reflection_memory.append(f"{reflection.lesson} Strategy: {reflection.next_strategy}")
                trace.token_estimate += reflector_stats.tokens
                trace.latency_ms += reflector_stats.latency_ms
            traces.append(trace)
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else _infer_failure_mode(final_reason)
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
