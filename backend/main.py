import base64
import contextvars
import difflib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MODEL_FALLBACKS = list(dict.fromkeys([MODEL, "gemini-3.5-flash", "gemini-3.6-flash"]))
ACTIVE_MODEL = MODEL
REQUEST_COUNT = contextvars.ContextVar("gemini_request_count", default=0)
MAX_GEMINI_CALLS_PER_PAPER = int(os.getenv("MAX_GEMINI_CALLS_PER_PAPER", "5"))
MAX_FILE_SIZE = 20 * 1024 * 1024
REPO_RE = re.compile(r"https?://(?:www\.)?github\.com/[\w.-]+/[\w.-]+", re.I)

app = FastAPI(title="Marginal API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def gemini_request(contents: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global ACTIVE_MODEL
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")
    current_count = REQUEST_COUNT.get()
    if current_count >= MAX_GEMINI_CALLS_PER_PAPER:
        raise RuntimeError("Gemini per-paper request budget reached; returning a local evidence map instead of waiting on free-tier rate limits.")
    REQUEST_COUNT.set(current_count + 1)
    body: Dict[str, Any] = {"contents": contents, "generationConfig": {"temperature": 0.2}}
    if tools:
        body["tools"] = [{"functionDeclarations": tools}]
        body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
    if schema:
        body["generationConfig"].update({"responseMimeType": "application/json", "responseSchema": schema})
    last_message = "No model was attempted"
    candidates = list(dict.fromkeys([ACTIVE_MODEL] + MODEL_FALLBACKS))
    for model_name in candidates:
        url = "https://generativelanguage.googleapis.com/v1beta/models/{0}:generateContent".format(model_name)
        response = requests.post(url, params={"key": API_KEY}, json=body, timeout=180)
        if response.ok:
            ACTIVE_MODEL = model_name
            return response.json()
        last_message = response.json().get("error", {}).get("message", response.text[:300])
        if not any(marker in last_message.lower() for marker in ("no longer available", "not found", "not supported")):
            break
    raise RuntimeError("Gemini request failed: {0}".format(last_message))


def response_text(payload: Dict[str, Any]) -> str:
    parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts)


def gemini_json(prompt: str, schema: Dict[str, Any], image: Optional[bytes] = None) -> Any:
    parts: List[Dict[str, Any]] = [{"text": prompt}]
    if image:
        parts.insert(0, {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(image).decode("ascii")}})
    payload = gemini_request([{"role": "user", "parts": parts}], schema=schema)
    return json.loads(response_text(payload))


def rect_json(rect: fitz.Rect, page_number: int, width: float, height: float) -> Dict[str, Any]:
    return {"page": page_number, "x": max(0, rect.x0 / width), "y": max(0, rect.y0 / height), "width": min(1, rect.width / width), "height": min(1, rect.height / height)}


class PaperContext:
    def __init__(self, data: bytes, filename: str):
        self.doc = fitz.open(stream=data, filetype="pdf")
        self.filename = filename
        self.pages = [page.get_text("text", sort=True) for page in self.doc]
        self.full_text = "\n".join("[PAGE {0}]\n{1}".format(i + 1, text) for i, text in enumerate(self.pages))
        self.sections = self._sections()
        self.figures = self._figures()
        self.trace: List[Dict[str, str]] = []
        self.selected: List[Dict[str, Any]] = []
        self.summaries: Dict[str, str] = {}
        self.figure_summaries: Dict[str, str] = {}
        self.repo: Dict[str, Any] = {"status": "none", "evidence": "No repository URL appears in the paper."}
        self.mode = "gemini" if API_KEY else "local"

    def log(self, tool: str, reason: str, result: str) -> None:
        self.trace.append({"tool": tool, "reason": reason, "result": result, "at": stamp()})

    def _sections(self) -> List[Dict[str, Any]]:
        headings: List[Dict[str, Any]] = []
        page_blocks: List[List[Any]] = []
        pattern = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z][^\n]{2,80})$")
        for page_index, page in enumerate(self.doc):
            blocks = page.get_text("blocks", sort=True)
            page_blocks.append(blocks)
            for block_index, block in enumerate(blocks):
                line = " ".join(block[4].split())
                match = pattern.match(line)
                if match and len(match.group(2).split()) <= 12:
                    headings.append({"label": match.group(1), "title": match.group(2).strip(), "page": page_index + 1, "block": block_index})
                elif page_index == 0 and line.lower() == "abstract":
                    headings.append({"label": "ABS", "title": "Abstract", "page": 1, "block": block_index})
        unique: List[Dict[str, Any]] = []
        for heading in headings:
            if not unique or (heading["label"], heading["title"]) != (unique[-1]["label"], unique[-1]["title"]):
                unique.append(heading)
        if not unique:
            unique = [{"label": "p.{0}".format(i + 1), "title": "Paper overview" if i == 0 else "Page {0}".format(i + 1), "page": i + 1} for i in range(min(8, len(self.pages)))]
            return [{"id": "s-{0}".format(i + 1), "label": item["label"], "title": item["title"], "pageStart": item["page"], "pageEnd": item["page"], "text": self.pages[item["page"] - 1][:18000]} for i, item in enumerate(unique)]
        result: List[Dict[str, Any]] = []
        for i, heading in enumerate(unique[:18]):
            next_heading = unique[i + 1] if i + 1 < len(unique) else None
            end_page = next_heading["page"] if next_heading else len(self.pages)
            chunks: List[str] = []
            for page_no in range(heading["page"], end_page + 1):
                start_block = heading["block"] if page_no == heading["page"] else 0
                stop_block = next_heading["block"] if next_heading and page_no == next_heading["page"] else len(page_blocks[page_no - 1])
                chunks.extend(block[4].strip() for block in page_blocks[page_no - 1][start_block:stop_block] if block[4].strip())
            text = "\n".join(chunks)[:18000]
            result.append({"id": "s-{0}".format(i + 1), "label": heading["label"], "title": heading["title"], "pageStart": heading["page"], "pageEnd": end_page, "text": text})
        return result

    def _figures(self) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []
        caption_re = re.compile(r"^(Figure|Fig\.|Table)\s*([\dA-Za-z.-]+)[:.\s]+(.+)", re.I)
        for page_no, page in enumerate(self.doc, 1):
            blocks = page.get_text("blocks", sort=True)
            for block in blocks:
                text = block[4].strip().replace("\n", " ")
                match = caption_re.match(text)
                if not match:
                    continue
                caption_rect = fitz.Rect(block[:4])
                crop = fitz.Rect(24, max(24, caption_rect.y0 - min(page.rect.height * .42, 280)), page.rect.width - 24, min(page.rect.height - 24, caption_rect.y1 + 12))
                pix = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), clip=crop, alpha=False)
                found.append({"id": "f-{0}".format(len(found) + 1), "page": page_no, "caption": text[:600], "kind": "table" if match.group(1).lower().startswith("table") else "figure", "box": rect_json(crop, page_no, page.rect.width, page.rect.height), "image": pix.tobytes("png")})
                if len(found) >= 8:
                    return found
        return found

    def locate(self, text: str, page_hint: Optional[int] = None) -> Dict[str, Any]:
        needle = " ".join(text.split())
        words = needle.split()
        candidates = [needle, " ".join(words[:14]), " ".join(words[:9]), " ".join(words[:6])]
        page_order = list(range(1, len(self.doc) + 1))
        if page_hint in page_order:
            page_order.remove(page_hint)
            page_order.insert(0, page_hint)
        for page_no in page_order:
            page = self.doc[page_no - 1]
            for candidate in candidates:
                rects = page.search_for(candidate, quads=False)
                if rects:
                    return {"page": page_no, "boxes": [rect_json(rect, page_no, page.rect.width, page.rect.height) for rect in rects[:8]]}
        best = (0.0, None, None)
        normalized = re.sub(r"\W+", " ", needle.lower()).strip()
        for page_no in page_order:
            page = self.doc[page_no - 1]
            for block in page.get_text("blocks", sort=True):
                block_text = re.sub(r"\W+", " ", block[4].lower()).strip()
                score = difflib.SequenceMatcher(None, normalized[:500], block_text[:700]).ratio()
                if normalized[:80] and normalized[:80] in block_text:
                    score += 0.5
                if score > best[0]:
                    best = (score, page_no, fitz.Rect(block[:4]))
        if best[0] >= 0.18 and best[1] and best[2]:
            page = self.doc[best[1] - 1]
            return {"page": best[1], "boxes": [rect_json(best[2], best[1], page.rect.width, page.rect.height)]}
        return {"page": page_hint or 1, "boxes": []}

    def metadata(self) -> Dict[str, str]:
        page = self.doc[0]
        spans = []
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                spans.extend(line.get("spans", []))
        spans = [span for span in spans if span.get("text", "").strip()]
        largest = sorted(spans, key=lambda span: span.get("size", 0), reverse=True)[:8]
        title = " ".join(span["text"].strip() for span in largest[:2]).strip() or Path(self.filename).stem
        return {"title": title[:220], "authors": "Uploaded research paper", "venue": "{0} pages · mapped with {1}".format(len(self.doc), ACTIVE_MODEL)}


TOOLS = [
    {"name": "extract_sections", "description": "Return section boundaries and text with page offsets. Call this before selecting passages.", "parameters": {"type": "OBJECT", "properties": {"reason": {"type": "STRING"}}, "required": ["reason"]}},
    {"name": "extract_figures", "description": "Detect figure and table regions. Call only when useful.", "parameters": {"type": "OBJECT", "properties": {"reason": {"type": "STRING"}}, "required": ["reason"]}},
    {"name": "highlight_passages", "description": "Select one or two exact source passages from a section.", "parameters": {"type": "OBJECT", "properties": {"section_id": {"type": "STRING"}, "reason": {"type": "STRING"}}, "required": ["section_id", "reason"]}},
    {"name": "summarize_passage", "description": "Write one plain-language bullet in original words for a selected passage.", "parameters": {"type": "OBJECT", "properties": {"passage_id": {"type": "STRING"}, "reason": {"type": "STRING"}}, "required": ["passage_id", "reason"]}},
    {"name": "describe_figure", "description": "Use vision to explain one detected figure or table in a single bullet.", "parameters": {"type": "OBJECT", "properties": {"figure_id": {"type": "STRING"}, "reason": {"type": "STRING"}}, "required": ["figure_id", "reason"]}},
    {"name": "find_repo", "description": "Scan the full paper for explicit GitHub repository URLs.", "parameters": {"type": "OBJECT", "properties": {"reason": {"type": "STRING"}}, "required": ["reason"]}},
    {"name": "propose_repo", "description": "Infer a likely official repository only if code is implied and no explicit URL was found.", "parameters": {"type": "OBJECT", "properties": {"reason": {"type": "STRING"}}, "required": ["reason"]}},
    {"name": "finish", "description": "Finish only after the passage-summary map and repository verdict are complete.", "parameters": {"type": "OBJECT", "properties": {"reason": {"type": "STRING"}}, "required": ["reason"]}},
]


def passage_candidates(text: str, limit: int = 2) -> List[str]:
    clean = " ".join(text.split())
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean) if 70 <= len(part.strip()) <= 650]
    signals = ("we propose", "we show", "we find", "our model", "results", "outperform", "architecture", "method", "however", "limitation", "attention", "algorithm")
    scored = []
    for index, sentence in enumerate(sentences):
        lowered = sentence.lower()
        score = sum(3 for signal in signals if signal in lowered) + min(len(sentence), 360) / 120 - index * .015
        scored.append((score, index))
    selected: List[str] = []
    for _, index in sorted(scored, reverse=True):
        parts = [sentences[index]]
        if index + 1 < len(sentences) and len(parts[0]) + len(sentences[index + 1]) < 900:
            parts.append(sentences[index + 1])
        passage = " ".join(parts)
        fingerprint = set(re.findall(r"\w+", passage.lower()))
        if any(len(fingerprint & set(re.findall(r"\w+", prior.lower()))) / max(1, len(fingerprint)) > .72 for prior in selected):
            continue
        selected.append(passage)
        if len(selected) >= limit:
            break
    return selected


def execute_tool(ctx: PaperContext, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    reason = str(args.get("reason", "No reason supplied"))
    if name == "extract_sections":
        result = [{**section, "text": section["text"][:9000]} for section in ctx.sections]
        ctx.log(name, reason, "{0} sections identified".format(len(result)))
        return {"sections": result}
    if name == "extract_figures":
        result = [{key: value for key, value in figure.items() if key != "image"} for figure in ctx.figures]
        ctx.log(name, reason, "{0} figure/table regions detected".format(len(result)))
        return {"figures": result}
    if name == "highlight_passages":
        section = next((item for item in ctx.sections if item["id"] == args.get("section_id")), None)
        if not section:
            return {"error": "Unknown section"}
        created = []
        for source_text in passage_candidates(section["text"], 2):
            if any(source_text == existing["text"] for existing in ctx.selected):
                continue
            passage = {"id": "p-{0}".format(len(ctx.selected) + 1), "sectionId": section["id"], "text": source_text, "pageHint": section["pageStart"]}
            ctx.selected.append(passage)
            created.append(passage)
        ctx.log(name, reason, "{0} source passages selected from {1}".format(len(created), section["title"]))
        return {"passages": created}
    if name == "summarize_passage":
        passage = next((item for item in ctx.selected if item["id"] == args.get("passage_id")), None)
        if not passage:
            return {"error": "Unknown passage"}
        if passage["id"] in ctx.summaries:
            ctx.log(name, reason, "Reused summary from the current paper's batch")
            return {"passage_id": passage["id"], "summary": ctx.summaries[passage["id"]], "cached": True}
        pending = [item for item in ctx.selected if item["id"] not in ctx.summaries]
        schema = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"id": {"type": "STRING"}, "summary": {"type": "STRING"}}, "required": ["id", "summary"]}}
        prompt = "Write one substantial plain-language bullet for every source passage below. Each bullet should be 45-80 words across 2-3 sentences: explain the claim, why it matters, and any important qualification. Use your own words and never quote the source. Return every id exactly once.\n\n" + "\n\n".join("{0}: {1}".format(item["id"], item["text"]) for item in pending)
        records = gemini_json(prompt, schema)
        for record in records:
            if any(item["id"] == record.get("id") for item in pending):
                ctx.summaries[record["id"]] = record["summary"]
        summary = ctx.summaries.get(passage["id"], "")
        ctx.log(name, reason, "Batched {0} passage summaries in one model call".format(len(records)))
        return {"passage_id": passage["id"], "summary": summary, "batched_ids": [record.get("id") for record in records]}
    if name == "describe_figure":
        figure = next((item for item in ctx.figures if item["id"] == args.get("figure_id")), None)
        if not figure:
            return {"error": "Unknown figure"}
        schema = {"type": "OBJECT", "properties": {"summary": {"type": "STRING"}}, "required": ["summary"]}
        summary = gemini_json("Explain what this research-paper visual shows in one plain-language bullet. Caption: {0}".format(figure["caption"]), schema, figure["image"])["summary"]
        ctx.figure_summaries[figure["id"]] = summary
        ctx.log(name, reason, summary)
        return {"figure_id": figure["id"], "summary": summary}
    if name == "find_repo":
        urls = list(dict.fromkeys(REPO_RE.findall(ctx.full_text)))
        if urls:
            ctx.repo = {"status": "found", "url": urls[0], "label": re.sub(r"^https?://", "", urls[0]), "evidence": "URL appears in the paper text."}
        ctx.log(name, reason, urls[0] if urls else "No explicit GitHub URL found")
        return {"urls": urls}
    if name == "propose_repo":
        schema = {"type": "OBJECT", "properties": {"url": {"type": "STRING"}, "confidence": {"type": "STRING", "enum": ["high", "low", "none"]}, "evidence": {"type": "STRING"}}, "required": ["url", "confidence", "evidence"]}
        answer = gemini_json("Infer the official GitHub repository for this paper only when highly confident. This result will be marked inferred. Use an empty URL otherwise.\n\n{0}".format(ctx.full_text[:10000]), schema)
        if answer.get("confidence") == "high" and REPO_RE.match(answer.get("url", "")):
            ctx.repo = {"status": "inferred", "url": answer["url"], "label": re.sub(r"^https?://", "", answer["url"]), "evidence": answer.get("evidence") or "Suggested by Gemini; not found in the paper."}
        ctx.log(name, reason, ctx.repo.get("url", "No confident candidate"))
        return answer
    if name == "finish":
        ctx.log(name, reason, "Orchestrator declared the evidence map complete")
        return {"complete": True}
    return {"error": "Unknown tool"}


def local_fallback(ctx: PaperContext) -> None:
    for item in ctx.selected:
        if item["id"] not in ctx.summaries:
            ctx.summaries[item["id"]] = "Key evidence from the paper: " + item["text"][:520] + ("…" if len(item["text"]) > 520 else "")
    used_sections = {item["sectionId"] for item in ctx.selected}
    for section in ctx.sections:
        if len(ctx.selected) >= 6:
            break
        if section["id"] in used_sections:
            continue
        candidates = passage_candidates(section["text"], 1)
        text = candidates[0] if candidates else " ".join(section["text"].split())[:520]
        if text and not any(text == existing["text"] for existing in ctx.selected):
            item = {"id": "p-{0}".format(len(ctx.selected) + 1), "sectionId": section["id"], "text": text, "pageHint": section["pageStart"]}
            ctx.selected.append(item)
            ctx.summaries[item["id"]] = "Key evidence from the paper: " + text
    execute_tool(ctx, "find_repo", {"reason": "Complete the repository verdict in local fallback mode."})


def orchestrate(ctx: PaperContext) -> None:
    if not API_KEY:
        ctx.log("extract_sections", "Use local extraction because GEMINI_API_KEY is unavailable.", "{0} sections identified".format(len(ctx.sections)))
        local_fallback(ctx)
        return
    prompt = """You are Marginal's orchestrator agent. Build a complete evidence-backed map of this research paper. You decide which tools to call and in what order. Always call extract_sections and find_repo. Select 4-8 passages across the most important sections, then call summarize_passage for every selected passage. Call extract_figures only if the paper has useful figures/tables and describe only the most explanatory visuals. Call propose_repo only when find_repo is empty and the paper clearly implies public software. Never treat an inferred repo as found. Every tool call must include your concise reason. Call finish only when the map and repo verdict are complete.

Batch independent tool calls together whenever possible and complete the plan within four orchestration turns to respect free-tier limits.

Paper: {0}
Pages: {1}
Opening text: {2}""".format(ctx.filename, len(ctx.doc), ctx.full_text[:5000])
    try:
        contents: List[Dict[str, Any]] = [{"role": "user", "parts": [{"text": prompt}]}]
        finished = False
        for _ in range(24):
            payload = gemini_request(contents, tools=TOOLS)
            content = payload.get("candidates", [{}])[0].get("content", {})
            contents.append(content)
            calls = [part["functionCall"] for part in content.get("parts", []) if "functionCall" in part]
            if not calls:
                break
            responses = []
            for call in calls:
                result = execute_tool(ctx, call["name"], call.get("args", {}))
                responses.append({"functionResponse": {"name": call["name"], "response": result}})
                if call["name"] == "finish":
                    finished = True
            contents.append({"role": "user", "parts": responses})
            if finished:
                break
    except RuntimeError as exc:
        ctx.mode = "local"
        ctx.log("fallback", "Preserve a usable result when the Gemini service cannot complete the run.", str(exc)[:300])
    if not ctx.selected or ctx.mode == "local":
        local_fallback(ctx)


def result_json(ctx: PaperContext) -> Dict[str, Any]:
    passages = []
    for item in ctx.selected:
        if item["id"] not in ctx.summaries:
            continue
        location = ctx.locate(item["text"], item.get("pageHint"))
        public_item = {key: value for key, value in item.items() if key != "pageHint"}
        passages.append({**public_item, "summary": ctx.summaries[item["id"]], "page": location["page"], "boxes": location["boxes"], "kind": "passage"})
    figures = [{"id": figure["id"], "page": figure["page"], "caption": figure["caption"], "summary": ctx.figure_summaries[figure["id"]], "box": figure["box"]} for figure in ctx.figures if figure["id"] in ctx.figure_summaries]
    passages.extend({"id": figure["id"], "sectionId": "figure", "text": figure["caption"], "summary": figure["summary"], "page": figure["page"], "boxes": [figure["box"]], "kind": "figure"} for figure in figures)
    passages.sort(key=lambda item: (item["page"], item["boxes"][0]["y"] if item["boxes"] else 2))
    return {**ctx.metadata(), "pageCount": len(ctx.doc), "sections": ctx.sections, "passages": passages, "figures": figures, "repo": ctx.repo, "trace": ctx.trace, "mode": ctx.mode}


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "marginal-python", "geminiConfigured": bool(API_KEY)}


@app.post("/api/analyze")
async def analyze(paper: UploadFile = File(...)) -> Dict[str, Any]:
    if paper.content_type != "application/pdf":
        raise HTTPException(400, "Please upload a PDF.")
    data = await paper.read(MAX_FILE_SIZE + 1)
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, "PDFs are limited to 20 MB.")
    try:
        REQUEST_COUNT.set(0)
        ctx = PaperContext(data, paper.filename or "paper.pdf")
        orchestrate(ctx)
        return result_json(ctx)
    except Exception as exc:
        raise HTTPException(500, str(exc))


DIST = ROOT / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def frontend(path: str):
        target = DIST / path
        return FileResponse(target if path and target.is_file() else DIST / "index.html")
