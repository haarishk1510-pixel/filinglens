"use client";

import Link from "next/link";
import { ReactNode, useEffect, useState } from "react";
import { apiUrl } from "../lib/api";


// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Results = Record<string, any>;

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const interval = (range: number[]) => `${percent(range[0])} to ${percent(range[1])}`;

function Section({ title, note, children }: { title: string; note: string; children: ReactNode }) {
  return (
    <section className="mt-8">
      <h2 className="font-display text-xl">{title}</h2>
      <p className="mt-1 text-sm text-muted">{note}</p>
      <div className="mt-3 overflow-x-auto rounded-md border border-line bg-surface">{children}</div>
    </section>
  );
}

function Table({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <table className="w-full text-sm tabular-nums">
      <thead>
        <tr className="border-b border-line text-left text-muted">
          {head.map((cell) => (
            <th key={cell} className="p-3 font-medium">
              {cell}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={String(row[0])} className="border-b border-line last:border-0">
            {row.map((cell, index) => (
              <td key={index} className={`p-3 ${index === 0 ? "font-medium" : ""}`}>
                {cell}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const Pending = () => <p className="p-3 text-sm text-muted">This evaluation hasn&apos;t finished yet.</p>;

function AnswerTable({ systems }: { systems: Results }) {
  return (
    <Table
      head={["System", "Strict", "95% interval", "Any number correct", "Median seconds"]}
      rows={Object.entries<Results>(systems).map(([system, row]: [string, Results]) => [
        system.replace("_", " "),
        percent(row.strict.rate),
        interval(row.strict.ci95),
        percent(row.lenient.rate),
        Math.round(row.median_seconds),
      ])}
    />
  );
}

export default function Evaluation() {
  const [results, setResults] = useState<Results>();
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch(apiUrl("/api/evaluation"))
      .then((response) => response.json())
      .then(setResults)
      .catch(() => setError(true));
  }, []);

  const holdout = results?.retrieval_holdout?.scopes?.one_filing?.overall;
  const answers = results?.answers?.systems;

  return (
    <main className="mx-auto max-w-4xl px-4 py-8">
      <Link href="/" className="text-sm text-accent underline">
        Back to FilingLens
      </Link>
      <h1 className="mt-4 font-display text-3xl">Evaluation</h1>
      <p className="mt-1 text-muted">Measured results, including where simpler methods won.</p>
      {error && <p className="mt-4 rounded-md bg-error-soft px-3 py-2 text-sm text-error-ink">Can&apos;t reach the FilingLens API.</p>}

      <Section title="Search" note="Whether the right passage appears in the results, on companies not used in development.">
        {holdout ? (
          <Table
            head={["Mode", "Top 5", "95% interval", "Top 10", "MRR"]}
            rows={Object.entries<Results>(holdout).map(([mode, row]: [string, Results]) => [
              mode.replace("_", " + "),
              percent(row["hit@5"].value),
              interval(row["hit@5"].ci95),
              percent(row["hit@10"].value),
              row.mrr.toFixed(3),
            ])}
          />
        ) : (
          <Pending />
        )}
      </Section>

      <Section
        title="Answer accuracy on unseen questions"
        note="10 FinanceBench numeric 10-K questions held back from all agent changes, on a free local model. Strict means the first number stated was correct. With only 10 questions, the intervals are wide."
      >
        {results?.answers_holdout?.systems ? <AnswerTable systems={results.answers_holdout.systems} /> : <Pending />}
      </Section>

      <Section
        title="Answer accuracy on development questions"
        note="40 FinanceBench questions whose failures were used to improve the agent, so these scores are optimistic."
      >
        {answers ? <AnswerTable systems={answers} /> : <Pending />}
      </Section>

      <Section title="First run, before those improvements" note="The same 40 questions before the agent changes, kept for comparison.">
        {results?.answers_first_run?.systems ? <AnswerTable systems={results.answers_first_run.systems} /> : <Pending />}
      </Section>

      <Section title="Volatility after earnings" note="Whether volatility jumps after a release, tested on 2024 onward. A research model, not investment advice.">
        {results?.volatility ? (
          <Table
            head={["Model", "ROC AUC", "95% interval"]}
            rows={Object.entries<Results>(results.volatility.test).map(([name, row]: [string, Results]) => [
              name.replaceAll("_", " "),
              row.roc_auc.toFixed(3),
              `${row.roc_auc_ci95[0].toFixed(3)} to ${row.roc_auc_ci95[1].toFixed(3)}`,
            ])}
          />
        ) : (
          <Pending />
        )}
      </Section>

      <Section title="Sentiment fine-tuning" note="Financial news sentiment on a held-back test set.">
        {results?.finbert ? (
          <Table
            head={["Model", "Accuracy", "Macro F1", "95% interval"]}
            rows={Object.entries<Results>(results.finbert.test).map(([name, row]: [string, Results]) => [
              name.replaceAll("_", " "),
              percent(row.accuracy),
              row.macro_f1.toFixed(3),
              `${row.macro_f1_ci95[0].toFixed(3)} to ${row.macro_f1_ci95[1].toFixed(3)}`,
            ])}
          />
        ) : (
          <Pending />
        )}
      </Section>

      <Section title="Structured extraction" note="Pulling figures from tables as JSON, on companies never seen in training.">
        {results?.extraction ? (
          <Table
            head={["Approach", "Valid JSON", "Field F1", "All fields correct"]}
            rows={[results.extraction.prompting_only, results.extraction.fine_tuned].map((row, index) => [
              index ? "Fine-tuned" : "Prompting only",
              percent(row.valid_json_rate),
              row.f1.toFixed(3),
              percent(row.all_fields_correct_rate),
            ])}
          />
        ) : (
          <Pending />
        )}
      </Section>
    </main>
  );
}
