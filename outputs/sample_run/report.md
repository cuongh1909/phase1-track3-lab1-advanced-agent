# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: mock
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.96 | 1.0 | 0.04 |
| Avg attempts | 1 | 1.04 | 0.04 |
| Avg token estimate | 0 | 0 | 0 |
| Avg latency (ms) | 0 | 0 | 0 |

## Failure modes
```json
{
  "react": {
    "none": 96,
    "incomplete_multi_hop": 1,
    "wrong_final_answer": 3
  },
  "reflexion": {
    "none": 100
  },
  "overall": {
    "none": 196,
    "incomplete_multi_hop": 1,
    "wrong_final_answer": 3
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.
