"use client";

import Link from "next/link";
import { apiUrl } from "../lib/api";
import { useEffect, useState } from "react";


type Company = { ticker: string; company: string };
type Overview = {
  ticker: string;
  company: string;
  currency: "INR" | "USD";
  years: number[];
  metrics: { name: string; values: (number | null)[] }[];
};

type Fx = { usd_inr: number; date: string };
type Currency = "INR" | "USD";

function money(value: number | null, currency: Currency, reported: Currency, fx?: Fx) {
  if (value === null) return "—";
  if (currency === "INR" && (reported === "INR" || fx)) {
    const rupees = reported === "INR" ? value : value * fx!.usd_inr;
    return `₹${(rupees / 1e7).toLocaleString("en-IN", { maximumFractionDigits: 0 })} cr`;
  }
  const dollars = reported === "USD" ? value : fx ? value / fx.usd_inr : null;
  if (dollars === null) return "—";
  return `$${(dollars / 1e9).toLocaleString("en-US", { maximumFractionDigits: 1 })}B`;
}

function change(values: (number | null)[]) {
  const [previous, latest] = values.slice(-2);
  if (previous == null || latest == null || previous === 0) return "—";
  const percent = ((latest - previous) / Math.abs(previous)) * 100;
  return `${percent >= 0 ? "+" : ""}${percent.toFixed(1)}%`;
}

export default function Dashboard() {
  const [companies, setCompanies] = useState<Company[]>([]);
  const [ticker, setTicker] = useState("");
  const [overview, setOverview] = useState<Overview>();
  const [error, setError] = useState<string>();
  const [currency, setCurrency] = useState<Currency>("INR");
  const [fx, setFx] = useState<Fx>();

  useEffect(() => {
    fetch(apiUrl("/api/fx"))
      .then((response) => (response.ok ? response.json() : Promise.reject()))
      .then(setFx)
      .catch(() => setCurrency("USD"));
  }, []);

  useEffect(() => {
    fetch(apiUrl("/api/companies"))
      .then((response) => response.json())
      .then((data: Company[]) => {
        setCompanies(data);
        if (data.length) setTicker(data[0].ticker);
      })
      .catch(() => setError("Can't reach the FilingLens API."));
  }, []);

  useEffect(() => {
    if (!ticker) return;
    fetch(apiUrl(`/api/companies/${ticker}`))
      .then((response) => {
        if (!response.ok) throw new Error();
        return response.json();
      })
      .then((data: Overview) => {
        setOverview(data);
        setError(undefined);
      })
      .catch(() => {
        setOverview(undefined);
        setError("No financial data is available for that company.");
      });
  }, [ticker]);

  return (
    <main className="mx-auto max-w-4xl px-4 py-8">
      <Link href="/" className="text-sm text-accent underline">
        Back to FilingLens
      </Link>
      <h1 className="mt-4 font-display text-3xl">Company dashboard</h1>
      <p className="mt-1 text-muted">Annual figures reported to the SEC.</p>

      <label htmlFor="company" className="mt-6 block text-xs font-semibold uppercase tracking-wider text-muted">
        Company
      </label>
      <select
        id="company"
        value={ticker}
        onChange={(event) => setTicker(event.target.value)}
        className="mt-1 w-full max-w-sm rounded-md border border-line bg-surface px-2 py-2"
      >
        {companies.map((company) => (
          <option key={company.ticker} value={company.ticker}>
            {company.ticker} · {company.company}
          </option>
        ))}
      </select>

      {fx && (
        <div className="mt-4">
          <div className="inline-flex overflow-hidden rounded-md border border-line" role="group" aria-label="Currency">
            {(["INR", "USD"] as Currency[]).map((option) => (
              <button
                key={option}
                aria-pressed={currency === option}
                onClick={() => setCurrency(option)}
                className={`px-3 py-1.5 text-sm ${currency === option ? "bg-accent text-accent-ink" : "bg-surface"}`}
              >
                {option === "INR" ? "₹ Rupees" : "$ US dollars"}
              </button>
            ))}
          </div>
          {currency === "INR" && (
            <p className="mt-2 text-xs text-muted">
              {overview?.currency === "INR"
                ? "Reported in Indian rupees."
                : `Reported in US dollars. Every year is converted at ₹${fx.usd_inr.toFixed(2)} per US$ (${fx.date}), so changes between years match the original figures.`}
            </p>
          )}
        </div>
      )}

      {error && <p className="mt-4 rounded-md bg-error-soft px-3 py-2 text-sm text-error-ink">{error}</p>}

      {overview && (
        <div className="mt-6 overflow-x-auto rounded-md border border-line bg-surface">
          <table className="w-full text-sm tabular-nums">
            <thead>
              <tr className="border-b border-line text-left text-muted">
                <th className="p-3 font-medium">Metric</th>
                {overview.years.map((year) => (
                  <th key={year} className="p-3 text-right font-medium">
                    FY{year}
                  </th>
                ))}
                <th className="p-3 text-right font-medium">Latest change</th>
              </tr>
            </thead>
            <tbody>
              {overview.metrics.map((metric) => (
                <tr key={metric.name} className="border-b border-line last:border-0">
                  <td className="p-3 font-medium">{metric.name}</td>
                  {metric.values.map((value, index) => (
                    <td key={index} className="whitespace-nowrap p-3 text-right">
                      {money(value, currency, overview.currency, fx)}
                    </td>
                  ))}
                  <td className="whitespace-nowrap p-3 text-right">{change(metric.values)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}
