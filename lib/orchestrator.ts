import type { PaperMap, Passage, RepoVerdict, TraceStep } from "./types";
import { extractPdf, extractSections, locatePassage, type ExtractedPage } from "./pdf";

const MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const now = () => new Date().toISOString();
const trace = (list: TraceStep[], tool: string, reason: string, result: string) => list.push({ tool, reason, result, at: now() });
const repoRegex = /https?:\/\/(?:www\.)?github\.com\/[\w.-]+\/[\w.-]+/gi;

async function gemini(prompt: string, responseSchema?: object) {
  const key = process.env.GEMINI_API_KEY;
  if (!key) throw new Error("NO_KEY");
  const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent?key=${key}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ contents: [{ role: "user", parts: [{ text: prompt }] }], generationConfig: { temperature: .2, responseMimeType: responseSchema ? "application/json" : "text/plain", ...(responseSchema ? { responseSchema } : {}) } }) });
  if (!res.ok) throw new Error(`Gemini request failed (${res.status})`);
  const json = await res.json();
  return json.candidates?.[0]?.content?.parts?.map((p: { text?: string }) => p.text || "").join("") || "";
}

async function summarizeWithGemini(sections: ReturnType<typeof extractSections>, pages: ExtractedPage[], fullText: string, log: TraceStep[]): Promise<PaperMap> {
  trace(log, "extract_sections", "Begin with the paper's structure so later calls can target only relevant sections.", `${sections.length} sections identified`);
  const repos = [...new Set(fullText.match(repoRegex) || [])];
  trace(log, "find_repo", "Check explicit paper evidence before considering an inferred repository.", repos.length ? repos[0] : "No explicit repository URL found");
  const chosen = sections.filter((s) => s.text.length > 300).slice(0, 7);
  const schema = { type: "array", items: { type: "object", properties: { sectionId: { type: "string" }, passage: { type: "string" }, summary: { type: "string" } }, required: ["sectionId", "passage", "summary"] } };
  trace(log, "highlight_passages", "Select the claims that best explain the contribution, method, evidence, and limitations.", `Reviewing ${chosen.length} substantive sections`);
  const raw = await gemini(`You are one tool inside a paper-reading agent. For each supplied section select 1-2 key passages, 4-10 total. The passage MUST be an exact contiguous quote from the supplied text, 1-3 sentences. Write a plain-language one-bullet summary in your own words, never quoting the passage. Return JSON only.\n\n${chosen.map((s) => `SECTION ${s.id} — ${s.title}\n${s.text}`).join("\n\n")}`, schema);
  const records = JSON.parse(raw) as Array<{ sectionId: string; passage: string; summary: string }>;
  const passages: Passage[] = records.slice(0, 10).map((r, i) => { const loc = locatePassage(r.passage, pages); return { id: `p-${i + 1}`, sectionId: r.sectionId, text: r.passage, summary: r.summary, page: loc.page, boxes: loc.boxes, kind: "passage" }; });
  passages.forEach((p) => trace(log, "summarize_passage", `Translate source passage ${p.id} without borrowing its wording.`, p.summary));
  let repo: RepoVerdict = repos[0] ? { status: "found", url: repos[0], label: repos[0].replace(/^https?:\/\//, ""), evidence: "URL appears in the paper text." } : { status: "none", evidence: "No repository URL appears in the extracted paper text." };
  if (!repos.length && /code|implementation|software|github|source available/i.test(fullText)) {
    trace(log, "propose_repo", "The paper implies software exists, so look for a cautious candidate while preserving uncertainty.", "Candidate search requested");
    const proposed = (await gemini(`Given this paper title and opening text, propose the most likely official GitHub repository URL only if highly likely. Return NONE otherwise. This is an inference, not paper evidence.\n${fullText.slice(0, 6000)}`)).trim();
    if (/^https?:\/\/github\.com\//i.test(proposed)) repo = { status: "inferred", url: proposed.split(/\s/)[0], label: proposed.split(/\s/)[0].replace(/^https?:\/\//, ""), evidence: "Suggested by Gemini; not found in the paper." };
  }
  trace(log, "extract_figures", "Inspect captions only when the paper contains figure or table signals.", /figure|table/i.test(fullText) ? "Caption signals detected; visual regions remain visible in the PDF pane" : "Skipped: no figure or table signals");
  const first = pages[0]?.text || "Untitled paper";
  const title = first.split(/\s{2,}| Abstract | ABSTRACT /)[0].slice(0, 180) || "Untitled paper";
  return { title, authors: "Uploaded research paper", venue: `${pages.length} pages · mapped with ${MODEL}`, pageCount: pages.length, sections, passages, figures: [], repo, trace: log, mode: "gemini" };
}

function localFallback(sections: ReturnType<typeof extractSections>, pages: ExtractedPage[], fullText: string, filename: string, log: TraceStep[]): PaperMap {
  trace(log, "extract_sections", "Map the document structure locally before selecting evidence.", `${sections.length} sections identified`);
  const passages: Passage[] = [];
  sections.slice(0, 6).forEach((section) => {
    const sentences = section.text.match(/[^.!?]+[.!?]+/g)?.filter((s) => s.trim().length > 100 && s.trim().length < 520) || [];
    const text = (sentences.sort((a, b) => b.length - a.length)[0] || section.text.slice(0, 360)).trim();
    if (!text) return;
    const loc = locatePassage(text, pages);
    passages.push({ id: `p-${passages.length + 1}`, sectionId: section.id, text, summary: text.length > 190 ? `${text.slice(0, 187).trim()}…` : text, page: loc.page, boxes: loc.boxes, kind: "passage" });
  });
  trace(log, "highlight_passages", "Use a deterministic local fallback because no Gemini key is configured.", `${passages.length} passages surfaced`);
  passages.forEach((p) => trace(log, "summarize_passage", `Create a local extractive preview for ${p.id}.`, "Preview only — add GEMINI_API_KEY for paraphrased summaries"));
  const found = [...new Set(fullText.match(repoRegex) || [])][0];
  trace(log, "find_repo", "Scan the full extracted text for explicit GitHub URLs.", found || "No explicit repository URL found");
  trace(log, "extract_figures", "Leave figures in the rendered PDF when vision analysis is unavailable.", "Skipped in local preview mode");
  return { title: pages[0]?.text.slice(0, 150) || filename.replace(/\.pdf$/i, ""), authors: "Uploaded research paper", venue: `${pages.length} pages · local preview`, pageCount: pages.length, sections, passages, figures: [], repo: found ? { status: "found", url: found, label: found.replace(/^https?:\/\//, ""), evidence: "URL appears in the paper text." } : { status: "none", evidence: "No repository URL appears in the paper." }, trace: log, mode: "local" };
}

export async function analyzePaper(bytes: Uint8Array, filename: string): Promise<PaperMap> {
  const log: TraceStep[] = [];
  const extracted = await extractPdf(bytes);
  const sections = extractSections(extracted.pages);
  if (!process.env.GEMINI_API_KEY) return localFallback(sections, extracted.pages, extracted.fullText, filename, log);
  return summarizeWithGemini(sections, extracted.pages, extracted.fullText, log);
}
