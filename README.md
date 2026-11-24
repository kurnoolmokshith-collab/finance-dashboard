# Finance Dashboard

Two complementary tools for Indian financial analysis:

1. **Advanced Finance Screener Pro** (`index.html`) — an interactive, browser-based stock screener and dashboard.
2. **Bank & NBFC Credit Risk Assessment Model** (`bank_nbfc_credit_risk.py`) — an institution-level credit-risk engine that assesses the financial health of banks and NBFCs after credit deployment, scored against RBI benchmarks. It complements borrower-level PD/LGD/EAD modelling.

---

## Credit Risk Assessment Model

`bank_nbfc_credit_risk.py` parses annual-report / RBI report text, auto-detects key financial ratios, scores the institution CAMELS-style against RBI sector benchmarks and peer tiers, computes per-debt statutory provisions under RBI frameworks alongside Expected Loss, and stress-tests capital adequacy after slippage shocks.

It runs **100% offline**. Optional AI features activate automatically when API keys are present.

### Requirements

- Python 3.10+

Optional:

```bash
pip install pdfplumber   # better PDF table extraction
# or
pip install pypdf        # minimal PDF text extraction
```

Both are only needed if you parse PDF reports; `.txt` / `.md` reports need no dependencies.

### Quick start

```bash
# run the built-in demo (sample bank report, scoring, provisioning, stress test)
python bank_nbfc_credit_risk.py --demo

# parse every PDF/TXT/MD report in a folder and score each document
python bank_nbfc_credit_risk.py reports/

# score against NBFC peer benchmarks instead of bank benchmarks
python bank_nbfc_credit_risk.py reports/ --entity-type nbfc

# additionally export a machine-readable JSON summary (for the dashboard / other tooling)
python bank_nbfc_credit_risk.py reports/ --json out.json
```

### What it does

| Section | Capability |
|---|---|
| Ratio engine | Auto-detects GNPA, NNPA, PCR, CRAR, CET-1, Tier-1, ROA, ROE, NIM, CASA, credit-deposit ratio, gross advances, and credit growth from report text, with "as on" period detection |
| RBI benchmarks | Sector reference data from FSR Dec 2025 / Trend & Progress 2025 / NBFC-IRACP 2025 (SCB GNPA 2.1%, NNPA 0.5%, PCR ~77%, CRAR 17.2%, NBFC GNPA 2.9%, regulatory minimums) |
| Health scoring | Direction-aware CAMELS-style composite (0–100) with bands (`strong` / `satisfactory` / `adequate` / `weak`), separate peer tiers and weights for banks vs NBFCs, plus risk flags (credit exuberance, thin provisioning coverage, capital buffer stress) |
| Provisioning | Per-debt statutory provisions under bank IRACPs and legacy NBFC norms, including the Nov-2025 NBFC-IRACP glide path to bank parity, side-by-side with Expected Loss (PD × LGD × EAD) and under-provisioning flags |
| Stress testing | Slippage shocks (1% / 2.5% / 5% of advances) → GNPA migration → P&L provision hit → stressed CRAR vs the 9% RBI minimum and the 11.5% comfort line (9% + 2.5% CCB) |
| Security | Path validation and traversal blocking, 25 MB / 2M-char input caps, secret redaction, append-only audit log |

### Output

Each parsed report prints detected ratios (value, unit, period), a composite score with band, metric-level scores, and risk flags. With `--json`, you additionally get a structured summary per document:

```json
{
  "generated_at": "2026-...",
  "benchmarks_as_of": "2025-09-30 (FSR Dec 2025 / T&P 2025)",
  "entity_type": "bank",
  "documents": {
    "annual_report.txt": {
      "composite": 27.8,
      "band": "weak",
      "scores": {"gnpa": 33.3, "crar": 0.0, "...": 0.0},
      "flags": ["Credit growth 22% YoY: classic exuberance flag; ..."],
      "ratios": {"gnpa": {"value": 3.4, "unit": "pct", "period": "FY2024", "method": "regex"}}
    }
  }
}
```

### Optional AI integration (Gemini)

When `GEMINI_API_KEY` is set and the `google-genai` SDK is installed (`pip install google-genai`), an LLM pass refines/extends ratio extraction on messy tables, on top of the deterministic regex layer. Without a key the regex result stands unchanged.

```bash
export GEMINI_API_KEY="..."   # enables LLM-assisted table extraction
export HF_TOKEN="..."         # reserved for embedding-based search
```

Keys are read **only** from the environment or a gitignored `.env` file. Never hardcode or commit them.

### Refreshing benchmarks

Benchmark data lives in the `RBI_BENCHMARKS` dictionary (plus `PEER_TIERS` and the provision curves) near the top of the script. When RBI publishes new FSR / Trend & Progress data, update the values and the `as_of` field — no logic changes needed.

---

## Dashboard (`index.html`)

Open `index.html` directly in a browser — no build step or server required. It ships with a self-contained screener UI (Chart.js loaded from CDN for charting).

The credit-risk model's `--json` output is designed to be fed into dashboard tooling: per-document composite scores, bands, ratio details, and flags are all serialised in a stable structure.
