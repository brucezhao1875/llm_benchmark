# Benchmark

This repository collects benchmark plans, execution notes, and reports for large language model serving workloads.

The current completed benchmark is:

| Model | Hardware | Runtime | Report |
|---|---|---|---|
| GLM-5.2-FP8 | 8 x NVIDIA H20 | vLLM | [GLM-5.2-FP8 on H20](models/glm-5.2-fp8-h20/README.md) |

## Repository layout

```text
benchmark/
├── README.md
├── methodology/
│   ├── llm-serving-benchmark-methodology.md
│   └── metrics-glossary.md
├── models/
│   └── glm-5.2-fp8-h20/
│       ├── README.md
│       ├── reports/
│       ├── plans/
│       └── notes/
└── archive/
    └── misc/
```

## Benchmark philosophy

LLM serving performance should not be summarized by a single QPS number.

The stable capacity metric is token throughput, especially:

- `output tokens/s` for decode capacity.
- `prefill tokens/s` for long-context input processing.
- `TTFT`, `TPOT`, `ITL`, and `E2E` percentiles for latency and user experience.
- `QPS` only after binding request shape and SLA.

Recommended QPS wording:

```text
Under <input tokens>/<output tokens> and SLA <TTFT/E2E/ITL/error rate>, the service can sustain <X req/s> and <Y output tokens/s>.
```

Do not use isolated QPS without request shape.

## How to add a new benchmark

Create a model-specific directory under `models/`:

```text
models/<model>-<hardware>/
├── README.md
├── reports/
├── plans/
└── notes/
```

Each benchmark should include:

1. Hardware and software environment.
2. Exact serving command or service unit.
3. Test plan and workload matrix.
4. Raw result directory or artifact references.
5. Final report with interpretation and limitations.
6. Production recommendations.
