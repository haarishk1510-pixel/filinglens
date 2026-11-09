# FilingLens

Ask any company filing. Get answers you can verify.

FilingLens downloads SEC 10-K annual reports, indexes them with hybrid search, and answers questions with citations that link to the exact table row or sentence.

## How it works

1. **Ingest:** Downloads filings from SEC EDGAR, splits them by 10-K section, and keeps financial tables intact.
2. **Index:** Creates local embeddings and stores them in SQLite.
3. **Retrieve:** Combines vector and BM25 search with reciprocal rank fusion, then reranks with a cross-encoder.
4. **Agent:** A LangGraph loop lets the model choose between four tools and shows each step on the website:
   - `financial_statement` for standard line items, as one clean table in millions.
   - `search_filings` for narrative detail, using the hybrid search above.
   - `query_financials` for exact figures, running read-only SQL over SEC XBRL data, limited to one table with row and time caps.
   - `calculate` for arithmetic, using a safe expression parser rather than `eval`.
5. **Answer:** A free local model (Qwen 3 8B through Ollama) cites numbered sources. It writes derived figures as formulas that code evaluates, and a verifier checks each figure against its cited source. Claude can be used instead by setting `LLM_PROVIDER=anthropic`.

## Run it

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # add your SEC contact email
brew install ollama && ollama serve &
ollama pull qwen3:8b && ollama create filinglens-llm -f scripts/Modelfile
.venv/bin/python scripts/ingest.py AAPL MSFT
.venv/bin/python -m uvicorn filinglens.api:app --port 8000
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. If no model is running, the app works in search-only mode. On a 16 GB M4 MacBook Air, answers take about 1 to 2.5 minutes.

## Test

```bash
cd backend
.venv/bin/python -m pytest tests
```

## Retrieval benchmark

Questions are labeled independently of the retrievers: financial figures from SEC XBRL values, and narrative questions from rules checked against the filing text. Results for one selected filing:

| Mode | Development Hit@5 (92 questions, 2 companies) | Holdout Hit@5 (117 figure questions, 4 companies) | Holdout MRR |
|---|---|---|---|
| Keyword (BM25) | 41.3% | 36.8% (28–46) | 0.201 |
| Vector | 60.9% | 53.0% (44–62) | 0.395 |
| Hybrid | 70.7% | 54.7% (46–63) | 0.428 |
| Hybrid + rerank | 70.7% | 63.2% (55–72) | 0.454 |

Parentheses show 95% bootstrap intervals. Reranking adds about one second per search.

Adding table titles to chunks was tested on holdout companies that played no part in designing it. Reranked Hit@1 rose from 28.2% to 32.5% and MRR from 0.425 to 0.454, but Hit@5 was essentially unchanged (65.0% to 63.2%). The narrative set rose from 86.4% to 95.5%, but it has only 22 questions and was used during development. The questions were written for this project, so treat these as internal results. Reproduce with `python scripts/eval_retrieval.py`; full output is in `backend/eval/results/`.

## Answer accuracy

FinanceBench numeric 10-K questions, answered by the free local model. Strict means the first number in the answer was within 1% of the gold value. Lenient means any number in the answer was. Parentheses show 95% bootstrap intervals.

| Question set | Agent strict | Agent lenient | Single-pass RAG strict | Single-pass RAG lenient |
|---|---|---|---|---|
| First run, 40 questions, before fixes | 17.5% (8–30) | 42.5% (28–60) | 20.0% (8–33) | 25.0% (13–40) |
| Same 40 after fixes (development) | 77.5% (63–90) | 87.5% (78–98) | 27.5% (15–43) | 27.5% (15–43) |
| 10 unseen questions | 60.0% (30–90) | 90.0% (70–100) | 40.0% (10–70) | 50.0% (20–80) |

The fixes came from reading the first-run failures: thinking used up the token limit, and the model picked wrong rows or misread long raw numbers from SQL. A `financial_statement` tool now returns standard line items in millions. The 40-question row is optimistic because those failures shaped the fixes. Only the 10 unseen questions are a clean test, and that sample is small. On them, the agent's strict lead over single-pass RAG has an interval of 0 to 50 points, while its lenient lead has 10 to 70 points.

The figure verifier confirms that numbers appear in cited sources, not that they are the right line item. In this run it did not separate right from wrong answers: on the 40 development questions, 11 of 14 fully matched answers and 23 of 25 flagged answers were correct. Strict grading can also under-credit correct answers, since a figure such as the "3" in "3M" or "3-year" counts as the first number. The grader was left unchanged after seeing results.

Reproduce with `python scripts/eval_answers.py --ingest --run --report --count 50`.

## Other models

| Task | Baseline | Result |
|---|---|---|
| Financial news sentiment (2,388 test items) | Original FinBERT 72.5% accuracy, macro F1 0.668. TF-IDF + logistic regression 80.8%, 0.747 | Fine-tuned FinBERT 88.8%, macro F1 0.854 (0.837–0.870) |
| Table extraction to JSON (23 test examples, 5 unseen companies) | Prompting Qwen2.5-3B: 13.0% valid JSON, field F1 0.192 | LoRA fine-tuned: 43.5% valid JSON, field F1 0.492 |
| Volatility jump after earnings (test from 2024) | Prior rate ROC AUC 0.586 | Logistic 0.670, XGBoost 0.642. The simpler model won |

The extraction model improved but is not reliable enough for production, and its test set is small.

## Roadmap

- [x] Filing ingestion and table-aware parsing
- [x] Hybrid retrieval with reranking
- [x] Cited, streaming answers and website
- [x] LangGraph agent with search, SQL, and calculator tools
- [x] Retrieval benchmark comparing four search modes, with a holdout set
- [x] Free local model with verified citations and code-computed figures
- [x] Answer accuracy evaluation with FinanceBench, comparing the agent with single-pass RAG
- [x] Post-earnings volatility model, compared with simpler baselines
- [x] Guardrails, MLflow tracking, Docker and CI files, dashboard, and evaluation page
- [x] Rupee dashboard and code support for Indian companies
- [x] Fine-tuned FinBERT sentiment model
- [x] LoRA fine-tuned model for structured extraction
- [x] Index Indian companies (Reliance, Infosys, Wipro, HDFC Bank, ICICI Bank, Dr. Reddy's)
- [ ] Publish the recorded demo
