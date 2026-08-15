"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { PaperMap, Passage } from "./types";
import pdfWorker from "pdfjs-dist/build/pdf.worker.min.mjs?url";

type Beam = { id: string; d: string; x1: number; y1: number };

function PdfPages({ file, data, onRendered }: { file: File; data: PaperMap; onRendered: () => void }) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const pdfjs = await import("pdfjs-dist");
      pdfjs.GlobalWorkerOptions.workerSrc = pdfWorker;
      const doc = await pdfjs.getDocument({ data: new Uint8Array(await file.arrayBuffer()) }).promise;
      if (!host.current || cancelled) return;
      host.current.innerHTML = "";
      for (let n = 1; n <= doc.numPages; n++) {
        const page = await doc.getPage(n);
        const base = page.getViewport({ scale: 1 });
        const width = Math.min(760, host.current.clientWidth - 32);
        const viewport = page.getViewport({ scale: width / base.width });
        const shell = document.createElement("div"); shell.className = "pdf-page"; shell.dataset.page = String(n); shell.style.width = `${viewport.width}px`; shell.style.height = `${viewport.height}px`;
        const canvas = document.createElement("canvas"); canvas.width = viewport.width * devicePixelRatio; canvas.height = viewport.height * devicePixelRatio; canvas.style.width = `${viewport.width}px`; canvas.style.height = `${viewport.height}px`;
        shell.appendChild(canvas); host.current.appendChild(shell);
        const ctx = canvas.getContext("2d"); if (ctx) await page.render({ canvas, canvasContext: ctx, viewport, transform: devicePixelRatio === 1 ? undefined : [devicePixelRatio, 0, 0, devicePixelRatio, 0, 0] }).promise;
        data.passages.filter((p) => p.page === n).forEach((passage) => {
          const boxes = passage.boxes;
          const mark = document.createElement("button"); mark.type = "button"; mark.className = "source-mark"; mark.id = `source-${passage.id}`; mark.ariaLabel = `Source for summary: ${passage.summary}`;
          if (boxes.length) {
            const minX = Math.min(...boxes.map((b) => b.x)); const minY = Math.min(...boxes.map((b) => b.y)); const maxX = Math.max(...boxes.map((b) => b.x + b.width)); const maxY = Math.max(...boxes.map((b) => b.y + b.height));
            Object.assign(mark.style, { left: `${minX * 100}%`, top: `${Math.max(0, minY - .018) * 100}%`, width: `${Math.min(.94 - minX, maxX - minX) * 100}%`, height: `${Math.max(.035, maxY - minY + .035) * 100}%` });
          } else Object.assign(mark.style, { left: "8%", top: "12%", width: "84%", height: "5%" });
          mark.onclick = () => document.getElementById(`card-${passage.id}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
          shell.appendChild(mark);
        });
      }
      setTimeout(onRendered, 80);
    })();
    return () => { cancelled = true; };
  }, [file, data, onRendered]);
  return <div className="pdf-pages" ref={host} />;
}

function Reader({ file, data, reset }: { file: File; data: PaperMap; reset: () => void }) {
  const frame = useRef<HTMLDivElement>(null); const [beams, setBeams] = useState<Beam[]>([]); const [traceOpen, setTraceOpen] = useState(false);
  const draw = useCallback(() => {
    if (!frame.current || innerWidth < 900) return setBeams([]);
    const fr = frame.current.getBoundingClientRect();
    setBeams(data.passages.flatMap((p) => { const a = document.getElementById(`source-${p.id}`)?.getBoundingClientRect(); const b = document.getElementById(`card-${p.id}`)?.getBoundingClientRect(); if (!a || !b) return []; const x1 = a.right - fr.left, y1 = a.top - fr.top + a.height / 2, x2 = b.left - fr.left - 12, y2 = b.top - fr.top + 24, mx = x1 + (x2 - x1) * .55; return [{ id: p.id, x1, y1, d: `M ${x1} ${y1} C ${mx} ${y1}, ${x1 + (x2 - x1) * .3} ${y2}, ${x2} ${y2}` }]; }));
  }, [data]);
  useEffect(() => { const resize = () => draw(); addEventListener("resize", resize); addEventListener("scroll", resize, true); return () => { removeEventListener("resize", resize); removeEventListener("scroll", resize, true); }; }, [draw]);
  const sections = new Map(data.sections.map((s) => [s.id, s]));
  return <main className="reader">
    <header className="topbar"><div className="brand-mark"/><span className="brand-name">Marginal</span><span className="brand-sub">paper reader</span><div className="top-actions"><button onClick={() => setTraceOpen(true)}>{data.trace.length} agent steps</button><button onClick={reset}>New paper</button></div></header>
    <section className="paper-header"><div className="eyebrow">{data.venue}</div><h1>{data.title}</h1><div className="authors">{data.authors}</div></section>
    <div className="reader-frame" ref={frame}>
      <svg className="beams" aria-hidden="true"><defs><linearGradient id="beam-grad"><stop offset="0%" stopColor="#e0a83c" stopOpacity=".75"/><stop offset="100%" stopColor="#49c8b6" stopOpacity=".8"/></linearGradient><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#49c8b6"/></marker></defs>{beams.map((b) => <g key={b.id}><path d={b.d} stroke="url(#beam-grad)" strokeWidth="1.7" fill="none" markerEnd="url(#arrow)"/><circle cx={b.x1} cy={b.y1} r="3" fill="#e0a83c"/></g>)}</svg>
      <section className="paper-pane"><div className="pane-label"><span>Original paper</span><span>{data.pageCount} pages</span></div><PdfPages file={file} data={data} onRendered={draw}/></section>
      <aside className="summary-pane"><div className="pane-label"><span>Mapped summary</span><span>{data.passages.length} sources</span></div><div className="cards">{data.passages.map((p: Passage, i) => <article className="summary-card" id={`card-${p.id}`} key={p.id}><div className="card-meta"><span>{sections.get(p.sectionId)?.label || `p.${p.page}`}</span><span>source {String(i + 1).padStart(2, "0")}</span></div><p>{p.summary}</p><button onClick={() => document.getElementById(`source-${p.id}`)?.scrollIntoView({ behavior: "smooth", block: "center" })}>View passage <span>↗</span></button></article>)}</div>
        <div className={`repo-card ${data.repo.status}`}><div className="repo-icon">⌘</div><div><span className="repo-label">{data.repo.status === "found" ? "Repository found in paper" : data.repo.status === "inferred" ? "Possible repository · inferred" : "Repository verdict"}</span>{data.repo.url ? <a href={data.repo.url} target="_blank" rel="noreferrer">{data.repo.label} ↗</a> : <strong>No public repository identified</strong>}<small>{data.repo.evidence}</small></div></div>
      </aside>
    </div>
    {data.mode === "local" && <div className="mode-banner">Local preview mode — Gemini is unavailable, unconfigured, or rate-limited. Details are recorded in the agent trace.</div>}
    {traceOpen && <div className="modal-backdrop" onMouseDown={() => setTraceOpen(false)}><section className="trace-panel" onMouseDown={(e) => e.stopPropagation()}><header><div><span className="trace-kicker">Orchestrator trace</span><h2>How this summary was built</h2></div><button onClick={() => setTraceOpen(false)} aria-label="Close">×</button></header><div className="trace-list">{data.trace.map((step, i) => <div className="trace-step" key={`${step.tool}-${i}`}><span>{String(i + 1).padStart(2, "0")}</span><div><code>{step.tool}()</code><p>{step.reason}</p><small>{step.result}</small></div></div>)}</div></section></div>}
  </main>;
}

export default function MarginalApp() {
  const [file, setFile] = useState<File | null>(null); const [data, setData] = useState<PaperMap | null>(null); const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  async function analyze(next: File) { setFile(next); setBusy(true); setError(""); const form = new FormData(); form.append("paper", next); try { const res = await fetch("/api/analyze", { method: "POST", body: form }); const json = await res.json(); if (!res.ok) throw new Error(json.error || "Analysis failed"); setData(json); } catch (e) { setError(e instanceof Error ? e.message : "Analysis failed"); setFile(null); } finally { setBusy(false); } }
  if (file && data) return <Reader file={file} data={data} reset={() => { setFile(null); setData(null); }}/>
  return <main className="landing"><header className="topbar"><div className="brand-mark"/><span className="brand-name">Marginal</span><span className="brand-sub">paper reader</span></header><section className="hero"><div className="eyebrow">Evidence, kept attached</div><h1>Read the paper.<br/><em>See the point.</em></h1><p>Marginal maps every summary claim back to the exact passage, figure, or table it came from.</p><label className={`upload ${busy ? "busy" : ""}`}><input type="file" accept="application/pdf" disabled={busy} onChange={(e) => e.target.files?.[0] && analyze(e.target.files[0])}/><span className="upload-mark">{busy ? <i/> : "+"}</span><span><strong>{busy ? "Mapping your paper…" : "Choose a research paper"}</strong><small>{busy ? "The agent is deciding which tools it needs" : "PDF · up to 20 MB"}</small></span></label>{error && <p className="error">{error}</p>}<div className="promise"><span>01 <b>Upload</b></span><i/><span>02 <b>Agent maps evidence</b></span><i/><span>03 <b>Read with provenance</b></span></div></section><footer>Marginal — reads the paper, projects the point</footer></main>;
}
