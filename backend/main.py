from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import API_KEY, MAX_FILE_SIZE, PaperContext, orchestrate, result_json

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Marginal API", version="3.0.0", description="LangGraph-powered evidence mapping for research papers")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "marginal-langgraph", "geminiConfigured": bool(API_KEY), "orchestrator": "langgraph"}


@app.post("/api/analyze")
async def analyze(paper: UploadFile = File(...)) -> Dict[str, Any]:
    if paper.content_type != "application/pdf":
        raise HTTPException(400, "Please upload a PDF.")
    data = await paper.read(MAX_FILE_SIZE + 1)
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, "PDFs are limited to 20 MB.")
    try:
        context = PaperContext(data, paper.filename or "paper.pdf")
        graph_state = orchestrate(context)
        response = result_json(context)
        response["graph"] = {
            "complete": graph_state["complete"],
            "repairAttempts": graph_state["repair_attempts"],
            "validationErrors": graph_state["validation_errors"],
            "plannedSections": graph_state["plan"].get("section_ids", []),
        }
        return response
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/architecture", include_in_schema=False)
def architecture_page():
    return FileResponse(ROOT / "docs" / "architecture.html")


DIST = ROOT / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def frontend(path: str):
        target = DIST / path
        return FileResponse(target if path and target.is_file() else DIST / "index.html")
