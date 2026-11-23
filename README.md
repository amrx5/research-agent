# (M3.6) Multi-Tool Research Assistant Agent (draft)

An agent that answers multi-step technical questions by choosing among three tools (web search, a local paper corpus, and a Python sandbox), checking every tool result, and writing a cited answer. The code follows the M2 design: **plan → act → validate → synthesize**.

```
question ─► Planner ─► Executor loop ◄──────────────┐
            (LLM)      (LLM, max 8 calls, 90 s)      │ validated evidence [W1][P1][C1]
                          │ tool call                │ or typed error + suggested next step
                          ▼                          │
                       ToolRouter ─► web_search / paper_search / python ─► Validator
                          (args, dedupe, cache, timeout, retry+backoff, circuit breaker)
                          │
                          ▼ done / step limit
                       Synthesizer ─► answer + citations + code + gaps   (+ JSONL trace)
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env                       # add ANTHROPIC_API_KEY and TAVILY_API_KEY
docker build -t research-agent-sandbox sandbox/

# build the paper index (once)
mkdir -p data/papers && cp your_papers/*.pdf data/papers/
python scripts/build_corpus.py --pdf-dir data/papers --meta data/papers.csv

python -m agent.cli ask "In 'Attention Is All You Need', how much higher (%) is the EN-FR BLEU than EN-DE?"
streamlit run app/streamlit_app.py         # demo UI
pytest -q                                  # offline tests: no keys, network or Docker needed
```

`data/papers.csv` columns: `file,paper_id,title,year`.

## Layout

| Path | What it does | M2 component |
| --- | --- | --- |
| `agent/planner.py` | One LLM call that drafts an ordered tool plan (a hint, not a script) | Planner |
| `agent/executor.py` | Reason → call → observe loop; enforces tool-call and time budgets | Executor loop |
| `agent/router.py` | Validates arguments, blocks duplicate calls, caches, applies timeouts, retries with backoff, and disables a tool after 2 infrastructure failures; holds the evidence store (IDs W1/P1/C1) | Tool router |
| `agent/validator.py` | Checks schema, empty results, relevance threshold, code errors, NaN/inf; strips prompt-injection lines; truncates and de-duplicates | Validator |
| `agent/synthesizer.py` | Answer from evidence only; checks citations; regenerates once, then flags uncited numbers *unverified*; detects numeric conflicts between sources | Synthesizer |
| `agent/tools/` | `web_search` (Tavily, Brave fallback), `paper_search` (Chroma + BM25 + RRF), `python` (Docker sandbox) | Tools |
| `agent/llm.py` | Anthropic adapter; swap provider here | LLM API |
| `agent/trace.py` | One JSONL trace per question | Trace log |
| `scripts/build_corpus.py` | PDFs → page-tagged chunks → Chroma + BM25 | Corpus builder |
| `eval/run_eval.py`, `eval/faults.py` | Metrics, baselines (`llm_only`, `all_tools`, `agent_no_planner`), fault injection | Eval harness |
| `tests/` | 30 offline tests with a scripted LLM and fake tools | |

## How the agent handles failures

| What goes wrong | Where it is caught | What happens |
| --- | --- | --- |
| Timeout, 429, 5xx | Router | Up to 3 attempts with exponential backoff, then a typed error to the LLM |
| Same tool fails twice (infrastructure) | Router circuit breaker | Tool disabled for this question; LLM told to switch or report the gap |
| Bad arguments, unknown tool, repeated identical call | Router | Error note with the fix; no tool run |
| Empty / weakly relevant results | Validator | `EMPTY_RESULT` / `LOW_RELEVANCE` + hint to rephrase or switch tool |
| Code crash, no output, NaN/inf | Validator | `CODE_ERROR` (with traceback) / `EMPTY_RESULT` / `IMPLAUSIBLE_VALUE` |
| Instructions hidden in web or PDF text | Validator | Lines removed, warning shown to the LLM; tools are read-only anyway |
| Uncited numbers in the answer | Synthesizer | One regeneration, then flagged *unverified* and status lowered to `partial` |
| Two sources disagree by more than 5% | Synthesizer | Both values reported as a conflict, each cited |
| Budget or time limit hit | Executor | Stops, answers with what it has, lists the gap |

## Evaluation

```bash
cp eval/questions.example.jsonl eval/questions.jsonl   # then write the real 60 questions
python -m eval.run_eval --split dev
python -m eval.run_eval --mode llm_only --split test
python -m eval.run_eval --faults mixed --split test
CACHE_MODE=replay python -m eval.run_eval --split test   # rerun from cached web results
```
Each run writes a CSV (one row per question; text answers left for manual grading) and a Markdown table comparing results with the M2 targets.

## Status and known gaps (draft)

- Tested offline: the full loop, the router, the validator, synthesizer checks, fault injection, and the local sandbox mode. Chroma/BM25 indexing and retrieval, and the Anthropic adapter, were smoke-tested against the real libraries (stub embeddings, mocked HTTP).
- **Not yet run:** live API calls (Claude, Tavily, Brave), the sentence-transformers embedding model, and the Docker sandbox. Run these first.
- Change from M2: page text comes from Tavily's `include_raw_content` instead of httpx + trafilatura (less code, same result). Note this in the M3 design doc (M3.5).
- `eval/questions.example.jsonl` holds only 4 example questions; the team still writes the 60-question set.
- Fill in `PRICE_*_PER_MTOK` so cost metrics are real.
