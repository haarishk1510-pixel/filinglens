"use client";

import Link from "next/link";
import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { API, DEMO, apiUrl } from "./lib/api";

const MAX_SELECTED = 6;
const FEATURED_TICKERS = ["AAPL", "MSFT"];

type Filing = {
  id: string;
  ticker: string;
  company: string;
  form: string;
  fiscal_year: number;
  filing_date: string;
  url: string;
  chunk_count: number;
};

type Passage = {
  id: string;
  ticker: string;
  company: string;
  form: string;
  fiscal_year: number;
  section: string;
  kind: "text" | "table";
  text: string;
  url: string;
};

type Part =
  | { kind: "text"; text: string }
  | { kind: "citation"; passageIndex: number; citedText: string };

type CitationCheck = {
  claims: number;
  supported: number;
  uncited_sentences: number;
  unsupported_sentences: string[];
  invalid_citations: number;
};

type Meta = {
  retrievalMs?: number;
  cost?: number | null;
  model?: string;
  steps?: number;
  check?: CitationCheck;
};

type Step = {
  id: string;
  tool: string;
  input: Record<string, unknown>;
  status: "running" | "done" | "error";
  summary?: string;
  ms?: number;
};

type Turn = {
  question: string;
  parts: Part[];
  passages: Passage[];
  steps: Step[];
  notice?: string;
  error?: string;
  meta: Meta;
  streaming: boolean;
};

type Active = { turn: number; passage: number; citedText?: string };

type Recorded = { question: string; events: Record<string, never>[] };

const EXAMPLES = [
  "What were total net sales in the latest fiscal year?",
  "What supply chain risks does the company describe?",
  "How much was spent on research and development?",
];

const TOOL_LABELS: Record<string, string> = {
  search_filings: "Searched filings",
  query_financials: "Queried financial data",
  calculate: "Calculated",
};

function stepDetail(step: Step) {
  return String(step.input.query ?? step.input.sql ?? step.input.expression ?? "");
}

function AgentSteps({ steps }: { steps: Step[] }) {
  return (
    <ol className="mt-3 space-y-1.5 border-l-2 border-line pl-3" aria-label="Agent steps">
      {steps.map((step) => (
        <li key={step.id} className="text-xs leading-5">
          <span className="font-medium">{TOOL_LABELS[step.tool] ?? step.tool}</span>
          {step.status === "running" && <span className="text-muted"> · working…</span>}
          {step.status === "done" && (
            <span className="text-muted">
              {" "}
              · {step.summary} · {step.ms} ms
            </span>
          )}
          {step.status === "error" && <span className="text-error-ink"> · {step.summary}</span>}
          <code className="block break-words font-mono text-muted">{stepDetail(step)}</code>
        </li>
      ))}
    </ol>
  );
}

function normalize(text: string) {
  return text.replace(/\s+/g, " ").trim();
}

function Highlighted({ text, target }: { text: string; target?: string }) {
  // Cited sentences can come back joined differently, so spacing must not matter.
  const pattern = target
    ?.replace(/\s+/g, "")
    .split("")
    .map((char) => char.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("\\s*");
  const match = pattern ? new RegExp(pattern).exec(text) : null;
  if (!match) return <>{text}</>;
  return (
    <>
      {text.slice(0, match.index)}
      <mark className="rounded-sm bg-mark px-0.5 text-mark-ink">{match[0]}</mark>
      {text.slice(match.index + match[0].length)}
    </>
  );
}

function PassageBody({ passage, citedText }: { passage: Passage; citedText?: string }) {
  if (passage.kind === "text") {
    return (
      <p className="whitespace-pre-wrap text-sm leading-6">
        <Highlighted text={passage.text} target={citedText} />
      </p>
    );
  }

  const cited = citedText ? normalize(citedText).replace(/ /g, "") : "";
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs tabular-nums">
        <tbody>
          {passage.text.split("\n").map((row, rowIndex) => {
            const isCited = cited.length > 0 && cited.includes(normalize(row).replace(/ /g, ""));
            return (
              <tr
                key={rowIndex}
                className={`border-b border-line ${isCited ? "bg-mark text-mark-ink" : ""}`}
              >
                {row.split(" | ").map((cell, cellIndex) => (
                  <td
                    key={cellIndex}
                    className={`px-1.5 py-1 ${cellIndex === 0 ? "text-left" : "text-right whitespace-nowrap"}`}
                  >
                    {/^\d{7,}$/.test(cell) ? Number(cell).toLocaleString("en-US") : cell}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function Home() {
  const [filings, setFilings] = useState<Filing[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [apiDown, setApiDown] = useState(false);
  const [llmReady, setLlmReady] = useState(true);
  const [ticker, setTicker] = useState("");
  const [adding, setAdding] = useState(false);
  const [addMessage, setAddMessage] = useState<string>();
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [active, setActive] = useState<Active>();
  const [busy, setBusy] = useState(false);
  const [recorded, setRecorded] = useState<Recorded[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  const loadFilings = useCallback(async () => {
    const response = await fetch(apiUrl("/api/filings"));
    const data: Filing[] = await response.json();
    setFilings(data);
    return data;
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const health = await (await fetch(apiUrl("/api/health"))).json();
        if (DEMO) setRecorded(await (await fetch(apiUrl("/api/answers"))).json());
        setLlmReady(health.llm_configured);
        const data = await loadFilings();
        // Filings arrive newest first within each company.
        const featured = FEATURED_TICKERS.flatMap((ticker) => data.filter((f) => f.ticker === ticker).slice(0, 1)).map(
          (f) => f.id,
        );
        setSelected(featured.length ? featured : data.slice(0, MAX_SELECTED).map((f) => f.id));
      } catch {
        setApiDown(true);
      }
    })();
  }, [loadFilings]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [turns]);

  function updateTurn(index: number, change: (turn: Turn) => Turn) {
    setTurns((current) => current.map((turn, i) => (i === index ? change(turn) : turn)));
  }

  function applyEvent(index: number, event: Record<string, never>) {
    switch (event.type as string) {
      case "passages":
        updateTurn(index, (t) => ({
          ...t,
          passages: event.passages,
          meta: { ...t.meta, retrievalMs: event.retrieval_ms },
        }));
        setActive((current) =>
          current?.turn === index ? current : { turn: index, passage: 0 },
        );
        break;
      case "step":
        updateTurn(index, (t) => {
          const step = event as unknown as Step;
          const known = t.steps.some((s) => s.id === step.id);
          return {
            ...t,
            steps: known ? t.steps.map((s) => (s.id === step.id ? step : s)) : [...t.steps, step],
          };
        });
        break;
      case "text":
        updateTurn(index, (t) => {
          const last = t.parts[t.parts.length - 1];
          const parts: Part[] =
            last?.kind === "text"
              ? [...t.parts.slice(0, -1), { kind: "text", text: last.text + event.text }]
              : [...t.parts, { kind: "text", text: event.text }];
          return { ...t, parts };
        });
        break;
      case "citation":
        updateTurn(index, (t) => ({
          ...t,
          parts: [
            ...t.parts,
            { kind: "citation", passageIndex: event.passage_index, citedText: event.cited_text },
          ],
        }));
        break;
      case "notice":
        updateTurn(index, (t) => ({ ...t, notice: event.message }));
        break;
      case "error":
        updateTurn(index, (t) => ({ ...t, error: event.message }));
        break;
      case "done":
        updateTurn(index, (t) => ({
          ...t,
          meta: {
            ...t.meta,
            cost: event.cost_usd,
            model: event.model,
            steps: event.steps,
            check: event.citation_check,
          },
        }));
        break;
    }
  }

  async function ask(text: string) {
    const trimmed = text.trim();
    if (trimmed.length < 3 || busy || selected.length === 0) return;

    const index = turns.length;
    setTurns((current) => [
      ...current,
      { question: trimmed, parts: [], passages: [], steps: [], meta: {}, streaming: true },
    ]);
    setQuestion("");
    setBusy(true);

    try {
      if (DEMO) {
        recorded.find((run) => run.question === trimmed)?.events.forEach((event) => applyEvent(index, event));
        return;
      }
      const response = await fetch(`${API}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed, filing_ids: selected }),
      });
      if (!response.ok || !response.body) throw new Error(String(response.status));

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";
        for (const raw of events) {
          if (raw.startsWith("data: ")) applyEvent(index, JSON.parse(raw.slice(6)));
        }
      }
    } catch {
      updateTurn(index, (t) => ({ ...t, error: "The request failed. Check that the API is running." }));
    } finally {
      updateTurn(index, (t) => ({ ...t, streaming: false }));
      setBusy(false);
    }
  }

  async function addCompany(event: FormEvent) {
    event.preventDefault();
    const value = ticker.trim().toUpperCase();
    if (!value || adding) return;
    setAdding(true);
    setAddMessage(`Downloading and indexing ${value}. This can take about a minute.`);
    try {
      const response = await fetch(`${API}/api/filings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: value }),
      });
      const data = await response.json();
      if (!response.ok) {
        setAddMessage(typeof data.detail === "string" ? data.detail : "Enter a valid ticker, such as NVDA.");
        return;
      }
      await loadFilings();
      setSelected((current) =>
        [...new Set([...current, data[0].id])].slice(-MAX_SELECTED),
      );
      setTicker("");
      setAddMessage(undefined);
    } catch {
      setAddMessage("Could not reach the API.");
    } finally {
      setAdding(false);
    }
  }

  function toggle(id: string) {
    setSelected((current) => {
      if (current.includes(id)) return current.filter((value) => value !== id);
      return current.length < MAX_SELECTED ? [...current, id] : current;
    });
  }

  const evidenceTurn = active ? turns[active.turn] : undefined;

  return (
    <div className="grid min-h-dvh grid-cols-1 lg:h-dvh lg:grid-cols-[250px_minmax(0,1fr)_minmax(340px,420px)]">
      <aside className="flex flex-col gap-5 border-b border-line p-4 lg:overflow-y-auto lg:border-r lg:border-b-0">
        <header>
          <h1 className="font-display text-2xl">FilingLens</h1>
          <p className="mt-1 text-sm text-muted">Ask any company filing. Get answers you can verify.</p>
          <nav className="mt-3 flex gap-3 text-sm">
            <Link href="/dashboard" className="text-accent underline">Dashboard</Link>
            <Link href="/evaluation" className="text-accent underline">Evaluation</Link>
          </nav>
        </header>

        <section>
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">Filings</h2>
          {filings.length === 0 && !apiDown && (
            <p className="text-sm text-muted">No filings yet. Add a company below.</p>
          )}
          <ul className="space-y-1">
            {filings.map((filing) => (
              <li key={filing.id}>
                <label className="flex cursor-pointer items-start gap-2 rounded-md p-1.5 hover:bg-surface">
                  <input
                    type="checkbox"
                    className="mt-1 accent-[var(--accent)]"
                    checked={selected.includes(filing.id)}
                    onChange={() => toggle(filing.id)}
                  />
                  <span className="text-sm leading-5">
                    <span className="font-semibold">{filing.ticker}</span> {filing.form} · FY
                    {filing.fiscal_year}
                    <span className="block text-xs text-muted">{filing.company}</span>
                  </span>
                </label>
              </li>
            ))}
          </ul>
        </section>

        {!DEMO && <form onSubmit={addCompany}>
          <label htmlFor="ticker" className="mb-1 block text-xs font-semibold uppercase tracking-wider text-muted">
            Add a company
          </label>
          <div className="flex gap-2">
            <input
              id="ticker"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              placeholder="Ticker, e.g. NVDA"
              maxLength={10}
              className="w-full min-w-0 rounded-md border border-line bg-surface px-2 py-1.5 text-sm outline-accent"
            />
            <button
              disabled={adding || !ticker.trim()}
              className="rounded-md bg-accent px-3 text-sm font-medium text-accent-ink disabled:opacity-50"
            >
              Add
            </button>
          </div>
          {addMessage && <p className="mt-2 text-xs text-muted" role="status">{addMessage}</p>}
        </form>}

        <p className="text-xs text-muted lg:mt-auto">
          Summarizes public SEC filings. Not investment advice.
        </p>
      </aside>

      <main className="flex min-h-[60dvh] min-w-0 flex-col lg:h-dvh">
        {apiDown && (
          <p className="bg-error-soft px-4 py-2 text-sm text-error-ink">
            Can&apos;t reach the FilingLens API at {API}.
          </p>
        )}
        {DEMO && (
          <p className="bg-warn-soft px-4 py-2 text-sm text-warn-ink">
            Recorded demo. These answers were saved from real runs of the agent on a free local model.
          </p>
        )}
        {!llmReady && !apiDown && (
          <p className="bg-warn-soft px-4 py-2 text-sm text-warn-ink">
            Search-only mode. Start the local model with Ollama to turn on written answers.
          </p>
        )}

        <div className="flex-1 overflow-y-auto px-4 py-6 lg:px-10">
          <div className="mx-auto max-w-2xl space-y-8">
            {turns.length === 0 && (
              <div className="pt-[12dvh]">
                <h2 className="font-display text-3xl leading-tight">What do you want to know?</h2>
                <p className="mt-2 text-muted">Every answer links to the passage it came from.</p>
                <div className="mt-6 flex flex-col items-start gap-2">
                  {(DEMO ? recorded.map((run) => run.question) : EXAMPLES).map((example) => (
                    <button
                      key={example}
                      onClick={() => ask(example)}
                      className="rounded-md border border-line bg-surface px-3 py-2 text-left text-sm hover:border-accent"
                    >
                      {example}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {turns.map((turn, turnIndex) => (
              <article key={turnIndex}>
                <h3 className="font-display text-xl leading-snug">{turn.question}</h3>

                {turn.steps.length > 0 && <AgentSteps steps={turn.steps} />}

                {turn.streaming && turn.parts.length === 0 && (
                  <p className="mt-3 text-sm text-muted" role="status">
                    {turn.steps.length ? "Working through the evidence…" : "Planning the research…"}
                  </p>
                )}

                {turn.parts.length > 0 && (
                  <p className="mt-3 whitespace-pre-wrap leading-7">
                    {turn.parts.map((part, partIndex) =>
                      part.kind === "text" ? (
                        <span key={partIndex}>{part.text}</span>
                      ) : (
                        <button
                          key={partIndex}
                          onClick={() =>
                            setActive({
                              turn: turnIndex,
                              passage: part.passageIndex,
                              citedText: part.citedText,
                            })
                          }
                          aria-label={`Show source ${part.passageIndex + 1}`}
                          className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded bg-accent-soft px-1 align-text-top text-xs font-semibold text-accent"
                        >
                          {part.passageIndex + 1}
                        </button>
                      ),
                    )}
                  </p>
                )}

                {turn.notice && (
                  <p className="mt-3 rounded-md bg-warn-soft px-3 py-2 text-sm text-warn-ink">{turn.notice}</p>
                )}
                {turn.error && (
                  <p className="mt-3 rounded-md bg-error-soft px-3 py-2 text-sm text-error-ink" role="alert">
                    {turn.error}
                  </p>
                )}

                {!turn.streaming && turn.meta.check && turn.meta.check.claims > 0 && (
                  <p
                    className={`mt-3 rounded-md px-3 py-2 text-sm ${
                      turn.meta.check.supported === turn.meta.check.claims
                        ? "bg-accent-soft text-accent"
                        : "bg-warn-soft text-warn-ink"
                    }`}
                  >
                    {turn.meta.check.supported === turn.meta.check.claims
                      ? turn.meta.check.claims === 1
                        ? "The figure was found in its cited source. Confirm it's the right line item."
                        : `All ${turn.meta.check.claims} figures were found in the cited sources. Confirm they're the right line items.`
                      : `${turn.meta.check.claims - turn.meta.check.supported} of ${turn.meta.check.claims} figures could not be matched to a cited source. Check them before relying on this answer.`}
                  </p>
                )}

                {!turn.streaming && turn.passages.length > 0 && (
                  <p className="mt-3 text-xs text-muted">
                    {turn.passages.length} {turn.passages.length === 1 ? "source" : "sources"}
                    {turn.meta.retrievalMs !== undefined && ` in ${turn.meta.retrievalMs} ms`}
                    {turn.meta.steps
                      ? ` · ${turn.meta.steps} agent ${turn.meta.steps === 1 ? "step" : "steps"}`
                      : ""}
                    {typeof turn.meta.cost === "number" && ` · $${turn.meta.cost.toFixed(4)}`}
                    {turn.meta.model && ` · ${turn.meta.model}`}
                  </p>
                )}
              </article>
            ))}
            <div ref={bottomRef} />
          </div>
        </div>

        {DEMO ? (
          <div className="flex flex-wrap gap-2 border-t border-line p-3 lg:px-10">
            {turns.length > 0 &&
              recorded.map((run) => (
                <button
                  key={run.question}
                  disabled={busy}
                  onClick={() => ask(run.question)}
                  className="rounded-md border border-line bg-surface px-2 py-1 text-left text-xs hover:border-accent"
                >
                  {run.question}
                </button>
              ))}
          </div>
        ) : (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            ask(question);
          }}
          className="border-t border-line p-3 lg:px-10"
        >
          <div className="mx-auto flex max-w-2xl gap-2">
            <label htmlFor="question" className="sr-only">Your question</label>
            <input
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              maxLength={1000}
              placeholder={selected.length ? "Ask about the selected filings" : "Select a filing first"}
              className="w-full min-w-0 rounded-md border border-line bg-surface px-3 py-2 outline-accent"
            />
            <button
              disabled={busy || question.trim().length < 3 || selected.length === 0}
              className="rounded-md bg-accent px-4 font-medium text-accent-ink disabled:opacity-50"
            >
              Ask
            </button>
          </div>
        </form>
        )}
      </main>

      <aside className="border-t border-line bg-surface p-4 lg:overflow-y-auto lg:border-t-0 lg:border-l">
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted">Evidence</h2>
        {!evidenceTurn || evidenceTurn.passages.length === 0 ? (
          <p className="text-sm text-muted">Source passages will appear here.</p>
        ) : (
          <ol className="space-y-2">
            {evidenceTurn.passages.map((passage, index) => {
              const open = active?.passage === index;
              return (
                <li key={passage.id} className="rounded-md border border-line">
                  <button
                    aria-expanded={open}
                    onClick={() => setActive({ turn: active!.turn, passage: index })}
                    className="flex w-full items-start gap-2 p-2.5 text-left"
                  >
                    <span className="flex h-5 min-w-5 items-center justify-center rounded bg-accent-soft text-xs font-semibold text-accent">
                      {index + 1}
                    </span>
                    <span className="text-sm leading-5">
                      <span className="font-medium">
                        {passage.fiscal_year ? `${passage.ticker} FY${passage.fiscal_year}` : passage.company}
                      </span>
                      <span className="block text-xs text-muted">{passage.section}</span>
                    </span>
                  </button>
                  {open && (
                    <div className="border-t border-line p-3">
                      <PassageBody passage={passage} citedText={active?.citedText} />
                      <a
                        href={passage.url}
                        target="_blank"
                        rel="noreferrer"
                        className="mt-3 inline-block text-xs font-medium text-accent underline"
                      >
                        View the source on SEC.gov
                      </a>
                    </div>
                  )}
                </li>
              );
            })}
          </ol>
        )}
      </aside>
    </div>
  );
}
