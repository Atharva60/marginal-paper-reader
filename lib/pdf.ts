import type { Section, SourceBox } from "./types";

type TextItem = { str: string; transform: number[]; width: number; height: number; hasEOL?: boolean };
export type ExtractedPage = { page: number; width: number; height: number; text: string; items: Array<{ text: string; box: SourceBox }> };

export async function extractPdf(bytes: Uint8Array) {
  const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
  const doc = await pdfjs.getDocument({ data: bytes }).promise;
  const pages: ExtractedPage[] = [];
  for (let pageNo = 1; pageNo <= doc.numPages; pageNo++) {
    const page = await doc.getPage(pageNo);
    const viewport = page.getViewport({ scale: 1 });
    const content = await page.getTextContent();
    const rawItems = (content.items as TextItem[]).filter((item) => "str" in item && item.str.trim());
    const items = rawItems.map((item) => ({
      text: item.str,
      box: { page: pageNo, x: item.transform[4] / viewport.width, y: 1 - item.transform[5] / viewport.height, width: item.width / viewport.width, height: Math.max(item.height / viewport.height, .012) },
    }));
    const text = rawItems.map((item) => `${item.str}${item.hasEOL ? "\n" : " "}`).join("").replace(/[ \t]+\n/g, "\n");
    pages.push({ page: pageNo, width: viewport.width, height: viewport.height, text, items });
  }
  return { pageCount: doc.numPages, pages, fullText: pages.map((p) => `\n[PAGE ${p.page}]\n${p.text}`).join("\n") };
}

export function extractSections(pages: ExtractedPage[]): Section[] {
  const candidates: Array<{ title: string; page: number; index: number }> = [];
  const heading = /^(\d+(?:\.\d+)*\s+[A-Z][A-Za-z][A-Za-z\s,&–—:-]{2,55})\s*$/gm;
  pages.forEach((page) => {
    for (const match of page.text.matchAll(heading)) candidates.push({ title: match[1].trim(), page: page.page, index: match.index ?? 0 });
  });
  const unique = candidates.filter((c, i) => i === 0 || c.title !== candidates[i - 1].title).slice(0, 16);
  if (!unique.length) return pages.slice(0, 8).map((p) => ({ id: `s-${p.page}`, label: `p.${p.page}`, title: p.page === 1 ? "Paper overview" : `Page ${p.page}`, pageStart: p.page, pageEnd: p.page, text: p.text }));
  return unique.map((c, i) => {
    const next = unique[i + 1];
    const end = next ? next.page : pages.length;
    return { id: `s-${i + 1}`, label: c.title.match(/^\d+(?:\.\d+)*/)?.[0] ?? `§${i + 1}`, title: c.title.replace(/^\d+(?:\.\d+)*\s*/, ""), pageStart: c.page, pageEnd: end, text: pages.slice(c.page - 1, end).map((p) => p.text).join(" ").slice(0, 14000) };
  });
}

export function locatePassage(text: string, pages: ExtractedPage[]): { page: number; boxes: SourceBox[] } {
  const words = text.replace(/\s+/g, " ").trim().split(" ").slice(0, 10);
  const needle = words.slice(0, 5).join(" ").toLowerCase();
  for (const page of pages) {
    const idx = page.text.toLowerCase().indexOf(needle);
    if (idx >= 0) {
      const tokens = words.map((w) => w.replace(/[^\p{L}\p{N}]/gu, "").toLowerCase()).filter(Boolean);
      const start = page.items.findIndex((item) => item.text.toLowerCase().includes(tokens[0]));
      if (start >= 0) return { page: page.page, boxes: page.items.slice(start, start + Math.min(28, tokens.length * 3)).map((i) => i.box) };
      return { page: page.page, boxes: [] };
    }
  }
  return { page: 1, boxes: [] };
}
