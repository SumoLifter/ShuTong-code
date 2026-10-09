# Maze Medical Teaching Agent — Code Submission

## Overview

This package implements a medical teaching text-processing pipeline as a static
[Maze](https://github.com/maze-agent/Maze) workflow DAG of four prompt-driven LLM
tasks: seven-dimension teaching-alignment scoring, SOAP note generation, clinical
reasoning, and medical licensing-exam style question generation (with selective
self-reflection repair). Scoring, SOAP, and reasoning run in parallel; question
generation depends on the SOAP and reasoning outputs; a finalize task emits a run
summary. The SOAP, reasoning, and question tasks perform **read-only** retrieval
(RAG) against a pre-built Chroma index; the runtime never rebuilds or writes to
the index.

## Installation

```bash
cd medical_agent
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # Linux/macOS
```

This installs all dependencies including the Maze workflow framework
(`maze-agent>=1.0.2`, source: <https://github.com/maze-agent/Maze>).

## Configuration

The pipeline runs against any OpenAI-compatible chat model service.
`medical_agent/config/llm_config.yaml` ships with a single commented
`example` provider entry; replace it with your own endpoint, model, and the
name of the environment variable holding your API key (see
`medical_agent/.env.example`). Keys are supplied via environment variables or
`--api-key-file` and are never written into configuration, the index, or task
outputs.

Read-only retrieval settings (`MEDICAL_RAG_EMBEDDING_PROVIDER`,
`MEDICAL_RAG_EMBEDDING_MODEL`, `MEDICAL_RAG_EMBEDDING_DIMENSIONS`) must match
the pre-built index manifest, otherwise retrieval is refused at runtime.

## Data boundaries

- The raw knowledge-base materials and the pre-built vector index are **not
  distributed** with this code package.
- At runtime the retrieval side is strictly read-only: if the index
  (`data/rag/manifest.json` and `data/rag/local_chroma/` at the deployment root)
  is missing or corrupted, tasks fail with an error instead of rebuilding it.
- The read-only RAG runtime location is provided via the
  `MEDICAL_RAG_RUNTIME_ROOT` environment variable.
- The package contains no API keys, access tokens, or real patient case data.

## Running

Start the Maze head node:

```bash
cd medical_agent
maze start --head --port 8000
```

On Windows, Maze's ZeroMQ scheduling coroutines require the selector event
loop policy; set it before launching, e.g. run
`python -c "import asyncio; asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()); from maze.cli.cli import main; main()" start --head --port 8000`.

In a second terminal, submit an authorized UTF-8 case text:

```bash
python run_pipeline.py /path/to/authorized_case.txt \
  --server-url http://localhost:8000 \
  --provider example \
  --model your-chat-model \
  --output-dir outputs
```

`--provider` / `--model` select an entry from `config/llm_config.yaml`.

The workflow DAG is:

```text
case
 ├─ scoring
 ├─ soap
 └─ reasoning
       ↓
    question
       ↓
    finalize → run_summary.json
```

`scoring`, `soap`, and `reasoning` run in parallel; `question` waits for the
SOAP and reasoning outputs, records a degraded source if an upstream task
fails, and continues with whatever is available.

The package targets single-machine or shared-project-directory deployments.
For multi-machine deployments, each worker needs access to the project code,
the read-only vector index, and the model API key; Maze's artifact store can
be used for file distribution.

## Verification

```bash
python -m compileall -q medical_agent
```

`PROMPT_MANIFEST.md` records the SHA-256 digests of all fixed prompts, which
can be used to verify the integrity of the prompt files in this package.

This project is intended for teaching and text-processing assistance only and
does not replace clinical decision-making.
