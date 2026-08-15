import { analyzePaper } from "@/lib/orchestrator";

export const runtime = "nodejs";
export const maxDuration = 300;

export async function POST(request: Request) {
  try {
    const form = await request.formData();
    const file = form.get("paper");
    if (!(file instanceof File) || file.type !== "application/pdf") return Response.json({ error: "Please upload a PDF." }, { status: 400 });
    if (file.size > 20 * 1024 * 1024) return Response.json({ error: "PDFs are limited to 20 MB." }, { status: 413 });
    const bytes = new Uint8Array(await file.arrayBuffer());
    return Response.json(await analyzePaper(bytes, file.name));
  } catch (error) {
    console.error(error);
    return Response.json({ error: error instanceof Error ? error.message : "Analysis failed." }, { status: 500 });
  }
}
