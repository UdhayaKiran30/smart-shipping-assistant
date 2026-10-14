# Smart Shipping Cost Assistant

An AI-assisted shipping cost analysis system that identifies unusual route-level transportation costs, compares them against historical and peer baselines, retrieves relevant operational context using semantic search, and generates evidence-grounded explanations.

## Project Overview

Shipping costs can fluctuate due to operational disruptions, fuel price changes, tolls, festivals, and other factors. This project analyzes shipment records to identify unusual cost patterns and determine whether available context notes explain those anomalies.

### Key Features

- Weekly route-level shipment cost aggregation.
- Cost-per-tonne-km calculation.
- Historical and similar-route baseline comparisons.
- Anomaly detection using configurable thresholds.
- Semantic retrieval of operational context notes using ChromaDB and Sentence Transformers.
- Evidence validation based on route applicability, dates, and cost-impact language.
- Gemini-powered natural-language Q&A.
- Automated evaluation scripts and regression tests.
- Q&A token-usage logging with optional API cost estimates.

## Technology Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Semantic retrieval:** Sentence Transformers, ChromaDB
- **LLM integration:** Google Gemini API
- **Testing:** Pytest

## Project Structure

```text
smart-shipping-assistant/
├── data/
│   ├── shipment_records.csv
│   ├── context_notes.csv
│   ├── sample_output_format_v2.csv
│   └── evaluation_cases.csv
├── src/
│   ├── analyze_shipments.py
│   ├── detect_anomalies.py
│   ├── build_vector_store.py
│   ├── retrieve_evidence.py
│   ├── build_final_output.py
│   ├── answer_questions.py
│   ├── evaluate_pipeline.py
│   ├── evaluate_evidence.py
│   ├── evaluate_rag.py
│   ├── evaluate_qa.py
│   └── evaluate_thresholds.py
├── tests/
├── outputs/
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Methodology

### 1. Weekly Shipment Analysis

Shipment records are grouped by origin, destination, route type, and week. Weeks run Monday through Sunday, and `week_of` represents the Monday starting that week.

The cost-per-tonne-km metric is calculated as:

\[
\text{Cost per tonne-km} =
\frac{\text{Total freight cost}}
{\sum(\text{Quantity in tonnes}\times\text{Distance in km})}
\]

This calculates a weighted route-level cost metric rather than taking a simple average of individual shipment costs.

### 2. Historical Baseline

Each route is compared against its trailing historical average. The baseline uses up to eight prior weekly observations and excludes the current week to avoid data leakage. When fewer than eight prior observations exist, all available prior observations are used.

### 3. Peer Baseline

Each route is compared with the average cost-per-tonne-km of other routes with the same route type in the same week. The current route is excluded from its peer baseline.

### 4. Anomaly Detection

The system flags route-weeks that exceed the configured relative-change threshold against their historical baseline or peer baseline. The current threshold is 10%.

A flagged route indicates an unusual cost pattern; it does not, by itself, establish the cause of the increase.

### 5. Retrieval-Augmented Evidence

Operational context notes are embedded using Sentence Transformers and stored in ChromaDB. For each flagged route-week, the system retrieves candidate notes and validates them against:

- Route applicability.
- Date or disruption-period overlap.
- Explicit language supporting a potential cost increase.

Only notes that pass these checks can be associated with a justified anomaly. The output includes the matched note ID where applicable. If no note passes validation, the anomaly is marked unexplained.

### 6. Natural-Language Q&A

The Q&A component uses calculated route results and validated evidence to answer questions about shipping costs. It is instructed not to invent explanations or treat unvalidated context notes as confirmed causes.

Gemini API usage is logged separately from the deterministic core pipeline. API cost estimates are available only when the corresponding pricing rates are configured.

## Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the installed dependencies.
- A Google Gemini API key for the Q&A component.

### Installation

Clone or extract the project and open a terminal in the project directory.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Never commit the actual `.env` file or expose the API key.

### Run the Pipeline

Execute the stages in order:

```powershell
python src/analyze_shipments.py
python src/detect_anomalies.py
python src/build_vector_store.py
python src/retrieve_evidence.py
python src/build_final_output.py
```

Run the core evaluation:

```powershell
python src/evaluate_pipeline.py
```

### Ask a Question

For example:

```powershell
python src/answer_questions.py "Why did Chennai-Bangalore shipping costs rise in February 2025?"
```

### Run Tests and Evaluations

```powershell
pytest -q
python src/evaluate_evidence.py
python src/evaluate_rag.py
python src/evaluate_qa.py
python src/evaluate_thresholds.py
```

The Q&A evaluation invokes the configured LLM and may incur API usage.

## Output Files

| File | Description |
|---|---|
| `outputs/weekly_route_costs.csv` | Aggregated weekly route metrics |
| `outputs/route_anomalies.csv` | Historical and peer comparisons with anomaly flags |
| `outputs/route_evidence.csv` | Evidence-validation outcomes |
| `outputs/final_output.csv` | Final output in the required schema |
| `outputs/evaluation_metrics.json` | Core pipeline metrics |
| `outputs/evidence_evaluation.csv` | Evidence evaluation results |
| `outputs/qa_evaluation.csv` | Q&A smoke-test results |
| `outputs/qa_usage_log.jsonl` | Q&A token usage and optional cost estimates |
| `outputs/threshold_sensitivity.csv` | Results at different anomaly thresholds |

The final CSV follows the required schema:

`route, week_of, cost_per_tonne_km, vs_own_history, vs_similar_routes, flagged, matched_note_id, reason`

## Evaluation Results

Results from the current evaluation run:

- Shipment records processed: **2,940**
- Weekly route records: **728**
- Flagged route-weeks: **90**
- Unflagged route-weeks: **638**
- Evidence-justified flags: **4**
- Unexplained flags: **86**
- Automated unit tests: **12 passed**
- Q&A smoke tests: **3 passed**
- Evidence evaluation: **3 of 3 selected cases passed**
- RAG retrieval: **Recall@5 of 4/4 on four test queries**

These are results from the current dataset and selected evaluation cases, not a guarantee of generalization to other datasets.

## Limitations

- The available context notes explain only a small portion of detected anomalies. An unexplained anomaly may still have a real operational cause that is absent from the notes.
- The current evidence validator uses deterministic rules and date/scope checks; it cannot independently establish causality.
- RAG retrieval evaluation covers four test queries, and evidence evaluation covers three selected cases. Larger, more diverse test sets are needed for stronger accuracy claims.
- Q&A smoke tests check execution and basic output behavior; they do not constitute a comprehensive factual-grounding evaluation.
- Results depend on data quality, the configured anomaly threshold, available context notes, and the selected model.
- The core pipeline is deterministic, while LLM-generated answers may vary between runs.

## Future Improvements

- Expand the context-note dataset and evaluation cases.
- Improve evidence coverage and temporal matching.
- Add broader automated