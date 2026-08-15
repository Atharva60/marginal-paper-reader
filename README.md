# Marginal

Marginal is a React paper reader backed by a Python/FastAPI analysis agent.

## Local development

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm install
```

Keep the Gemini key in `.env` at the project root:

```env
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-3.5-flash
```

Start the API and frontend in two terminals:

```bash
npm run dev:api
npm run dev
```

Open `http://localhost:5173`. The Vite server proxies `/api` to FastAPI on port 8000.

## Production

`npm run build` creates `dist/`. FastAPI serves that React build and the API from one process:

```bash
python -m backend
```
