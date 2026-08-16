# Marginal

Marginal is an evidence-first research-paper reader. Upload a PDF and it renders the original paper beside a structured explanation, with every summary card connected to the passage, figure, or table that supports it.

The application uses a React frontend and a Python/FastAPI backend. Gemini acts as an orchestrator: it chooses which analysis tools a paper needs, records why each tool was called, and produces a traceable paper map rather than an ungrounded summary.

> [Open the standalone project and architecture overview](docs/architecture.html). When the app is running, it is also available at `/architecture`.

## Features

- PDF.js paper rendering with normalized source highlights
- PyMuPDF section, block, figure, table, and coordinate extraction
- Live gold-to-teal SVG beams connecting evidence to explanation
- Substantial plain-language summary cards
- Explicit versus inferred GitHub repository verdicts
- Inspectable agent decision trace
- Quota-aware batched Gemini calls and a useful local fallback
- Single-service Render deployment

## Architecture

```text
Research paper (PDF)
        │
        ▼
React upload ───────────────► POST /api/analyze
                                     │
                                     ▼
                           FastAPI request boundary
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
             PyMuPDF extraction                Gemini orchestrator
         sections · blocks · figures        chooses tools + gives reasons
                    │                                 │
                    └──────────────┬──────────────────┘
                                   ▼
                           Structured paper map
                  passages · boxes · summaries · repo · trace
                                   │
                                   ▼
                       React evidence-map renderer
                  PDF highlights ⇢ SVG beams ⇢ summary cards
```

The backend—not the browser—loads `GEMINI_API_KEY`. Uploaded PDFs are processed in memory and are not written to the repository.

### Agent tools

| Tool | Responsibility |
| --- | --- |
| `extract_sections()` | Finds section boundaries and text with page offsets. |
| `extract_figures()` | Detects figure/table captions and crops PDF regions. |
| `highlight_passages()` | Selects distinct source passages. |
| `summarize_passage()` | Batches substantial summaries to conserve quota. |
| `describe_figure()` | Uses Gemini vision to explain an important visual. |
| `find_repo()` | Scans for explicit `github.com/owner/repo` URLs. |
| `propose_repo()` | Suggests a repository only as an inferred result. |
| `finish()` | Declares the evidence map complete. |

Every tool call includes the orchestrator’s stated reason and is exposed in the frontend trace panel.

## Technology

| Layer | Technology |
| --- | --- |
| Frontend | React 19, TypeScript, Vite, plain CSS |
| PDF viewer | `pdfjs-dist` |
| API | Python, FastAPI, Uvicorn |
| PDF extraction | PyMuPDF |
| AI | Google Gemini with function calling |
| Deployment | Render Blueprint, one Python web service |

## Local setup

Requirements: Node.js 20+, Python 3.9+, and a [Gemini API key](https://aistudio.google.com/apikey).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm install
```

Create `.env` in the repository root:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash
MAX_GEMINI_CALLS_PER_PAPER=5
```

Start the API and frontend in separate terminals:

```bash
npm run dev:api
```

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). Vite proxies `/api` to FastAPI on port 8000.

## Production and API

```bash
npm run build
python -m backend
```

FastAPI serves the generated React build and these endpoints:

- `GET /api/health` — service and Gemini configuration status
- `POST /api/analyze` — multipart PDF upload in the `paper` field
- `GET /*` — React application

The included `render.yaml` creates a free Render Python service. Add `GEMINI_API_KEY` as a secret environment variable in Render.

## Project structure

```text
backend/main.py         API, extraction tools, orchestrator, static serving
docs/architecture.html  Standalone visual system overview
src/App.tsx             Upload and evidence-map interface
src/index.css           Marginal design system and responsive layout
src/types.ts            Frontend paper-map contract
render.yaml             Render Blueprint
requirements.txt        Python dependencies
vite.config.ts          React build and API proxy
```

## Quota behavior

Marginal selects passages locally and batches pending summaries into one Gemini call. A per-paper request budget prevents stalled uploads. If Gemini is unavailable or rate-limited, the backend returns a longer extractive evidence map and records the fallback reason in the trace.

## Privacy and confidence

- `.env` and `.env.local` are ignored by Git.
- The Gemini key is available only to Python.
- PDF uploads are limited to 20 MB.
- Only repository URLs present in the paper are labeled `found`; model suggestions are labeled `inferred`.
