// Headless poster renderer.
// Boots the page in static mode and snapshots a single PNG.
//
// Usage:
//   node render_static.js sample        # quick low-res sample
//   node render_static.js full          # 7016x9933 (A3 600dpi)
//
// Custom: SAMPLE / FULL block at the bottom can be edited freely.

const { chromium } = require("playwright");
const path = require("path");
const fs = require("fs");
const http = require("http");
const { Resvg } = require("@resvg/resvg-js");
const sharp = require("sharp");

const ROOT = path.resolve(__dirname, "web");
const OUT_DIR = path.resolve(__dirname, "out");

function startServer() {
  const server = http.createServer((req, res) => {
    let p = decodeURIComponent(req.url.split("?")[0]);
    if (p === "/") p = "/index.html";
    const fp = path.join(ROOT, p);
    if (!fp.startsWith(ROOT)) { res.statusCode = 403; return res.end(); }
    fs.readFile(fp, (err, data) => {
      if (err) { res.statusCode = 404; return res.end(); }
      const ext = path.extname(fp).toLowerCase();
      const mime = {
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript",
        ".json": "application/json",
        ".css": "text/css",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
      }[ext] || "application/octet-stream";
      res.setHeader("Content-Type", mime);
      res.end(data);
    });
  });
  return new Promise(resolve => server.listen(0, "127.0.0.1", () => {
    resolve({ server, port: server.address().port });
  }));
}

async function render({ width, height, captions, semantic, outName, scale = 1 }) {
  // The page is rendered at 1/scale of the final size, then everything is
  // upsized to (width, height) on the way out. Effect: same final pixel
  // count, but 1.5x denser elements relative to the final canvas.
  const pageW = Math.round(width / scale);
  const pageH = Math.round(height / scale);
  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });
  const { server, port } = await startServer();
  console.log(`[server] http://127.0.0.1:${port}`);

  const browser = await chromium.launch({
    args: [
      // Tuned for M4 Pro + 48GB. Pump V8 heap, drop dev-shm cap, and let
      // Skia / blink hold huge layers without recycling them mid-paint.
      "--js-flags=--max-old-space-size=32768",
      "--disable-dev-shm-usage",
      "--no-sandbox",
      "--disable-gpu-vsync",
      "--disable-features=PartialRasterInvalidation,LazyFrameLoading,LazyImageLoading",
      "--force-device-scale-factor=1",
      "--max-active-webgl-contexts=1",
    ],
  });
  const ctx = await browser.newContext({
    viewport: { width: pageW, height: pageH },
    deviceScaleFactor: 1,
  });
  const page = await ctx.newPage();
  page.setDefaultTimeout(600000);
  page.setDefaultNavigationTimeout(600000);
  page.on("console", msg => console.log("[page]", msg.text()));
  page.on("pageerror", err => console.log("[error]", err.message));

  const url = `http://127.0.0.1:${port}/index.html?static=1&captions=${captions}&semantic=${semantic}`;
  console.log(`[render] ${url}  page=${pageW}x${pageH}  out=${width}x${height} (scale ${scale})`);
  await page.goto(url);

  // Wait for the page to signal it's done laying out.
  await page.waitForFunction(() => window.__STATIC_READY === true, { timeout: 600000 });
  // A small extra hold to let any final paint settle.
  await page.waitForTimeout(500);

  const stats = await page.evaluate(() => {
    const sv = document.getElementById("wires");
    const sb = document.getElementById("wires-bg");
    const words = document.querySelectorAll(".word[data-cat]");
    return {
      svgViewBox: sv?.getAttribute("viewBox"),
      pathsFg: sv?.querySelectorAll("path").length || 0,
      rectsFg: sv?.querySelectorAll("rect").length || 0,
      ellipsesFg: sv?.querySelectorAll("ellipse").length || 0,
      pathsBg: sb?.querySelectorAll("path").length || 0,
      rectsBg: sb?.querySelectorAll("rect").length || 0,
      words: words.length,
    };
  });
  console.log("[stats]", stats);

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");

  // Pull the SVG layers out of the page so resvg can rasterise them
  // out-of-process. Skia / Blink can't reliably paint 30k+ paths on a
  // 7016x9933 surface in one shot.
  const svgFg = await page.evaluate(() => window.__svgWiresFg);
  const svgBg = await page.evaluate(() => window.__svgWiresBg);
  fs.writeFileSync(path.join(OUT_DIR, `_dump_bg_${stamp}.svg`), svgBg);
  // First semantic path/rect for inspection.
  const semPath = svgBg.match(/<path[^>]*class="semantic"[^>]*>/);
  const semRect = svgBg.match(/<rect[^>]*class="semantic-box"[^>]*>/);
  console.log(`[debug] sem path: ${semPath ? semPath[0].slice(0,300) : "(none)"}`);
  console.log(`[debug] sem rect: ${semRect ? semRect[0].slice(0,300) : "(none)"}`);


  // Hide both SVGs and capture captions-only via Chromium. Captions and
  // their boxes are far fewer DOM nodes and don't trigger the limit.
  await page.evaluate(() => {
    document.getElementById("wires").style.display = "none";
    document.getElementById("wires-bg").style.display = "none";
  });
  await page.evaluate(async () => {
    const H = document.documentElement.scrollHeight;
    for (let y = 0; y < H; y += 1500) {
      window.scrollTo(0, y);
      await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    }
    window.scrollTo(0, 0);
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  });
  const captionsRawPng = path.join(OUT_DIR, `_captions_raw_${stamp}.png`);
  await page.screenshot({ path: captionsRawPng, type: "png", fullPage: true, omitBackground: true });
  console.log(`[layer] captions raw -> ${path.basename(captionsRawPng)}`);

  await browser.close();
  server.close();

  // Upscale captions to the final canvas size.
  async function prep(rawPath, finalName) {
    const out = path.join(OUT_DIR, `_${finalName}_${stamp}.png`);
    let img = sharp(rawPath).ensureAlpha();
    if (scale !== 1) img = img.resize(width, height, { kernel: "lanczos3", fit: "fill" });
    await img.png({ compressionLevel: 6 }).toFile(out);
    fs.unlinkSync(rawPath);
    return out;
  }
  const captionsPng = await prep(captionsRawPng, "captions");

  // Poem layer comes from a hand-authored PNG dropped into out/.
  // Expectations: white glyphs on a transparent canvas, sized exactly
  // to (width x height). If a per-orientation file isn't there we fall
  // back to the SVG-generated layer.
  const handPoem = path.join(OUT_DIR, "_poem_sample_black.png");
  let poemPng;
  if (fs.existsSync(handPoem)) {
    const meta = await sharp(handPoem).metadata();
    if (meta.width === width && meta.height === height) {
      poemPng = handPoem;
      console.log(`[layer] poem -> using ${path.basename(handPoem)}`);
    }
  }
  if (!poemPng) {
    console.log(`[layer] rasterise poem (server-side) ...`);
    poemPng = path.join(OUT_DIR, `_poem_${stamp}.png`);
    await renderPoemPng(width, height, poemPng);
  }

  // Vector layers: resvg renders directly at the final width, so wire
  // strokes stay crisp regardless of scale.
  console.log(`[layer] rasterise wires-bg (semantic) ...`);
  const bgPng = path.join(OUT_DIR, `_bg_${stamp}.png`);
  await rasterise(svgBg, width, height, bgPng);
  console.log(`[layer] rasterise wires-fg (chains) ...`);
  const fgPng = path.join(OUT_DIR, `_fg_${stamp}.png`);
  await rasterise(svgFg, width, height, fgPng);

  console.log(`[compose] stitching layers ...`);
  const finalPng = path.join(OUT_DIR, `${outName}_${stamp}.png`);
  await sharp({
    create: {
      width, height, channels: 4,
      background: { r: 255, g: 255, b: 255, alpha: 1 },
    },
  })
    .composite([
      { input: bgPng,       top: 0, left: 0 },
      { input: captionsPng, top: 0, left: 0 },
      { input: fgPng,       top: 0, left: 0 },
      // Poem layer applied with `difference`: black glyphs over white
      // become white inversion; over coloured pixels they invert that
      // colour. Same effect mix-blend-mode: difference does live.
      { input: poemPng,     top: 0, left: 0, blend: "difference" },
    ])
    .png({ compressionLevel: 6 })
    .toFile(finalPng);
  console.log(`[saved] ${finalPng}`);
}

// Server-side poem layer. Six fixed lines in Bebas Neue, justified
// edge-to-edge with a small horizontal inset so trailing punctuation
// never clips. The text is white because the final composite uses
// `difference` blend -- white on white = black (visible), white on a
// coloured pixel = that colour's complement.
const POEM_LINES = [
  "Ah Love! could thou and I with",
  "Fate conspire To grasp this",
  "sorry Scheme of Things entire,",
  "Would not we shatter it to",
  "bits - and then Re-mould it",
  "nearer to the Heart's Desire!",
];
const FONT_PATH = path.resolve(__dirname, "fonts/BebasNeue-Regular.ttf");

async function renderPoemPng(width, height, outPath, opts = {}) {
  const lines = POEM_LINES.length;
  // Uniform margin on all four sides.
  const margin = Math.round(Math.min(width, height) * 0.04);
  const innerW = width - margin * 2;
  const innerH = height - margin * 2;
  // Line height (== row pitch). Smaller scaleY pulls the rows further
  // apart visually; the row pitch itself stays innerH / lines.
  const lineH = innerH / lines;
  // Cap height fills only ~70% of the row, leaving generous leading.
  const fontSize = Math.round(lineH * 0.95);
  const scaleY = opts.scaleY ?? 1.0;
  const color = opts.color || "white";
  const escapedLines = POEM_LINES.map(l =>
    l.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  );
  const parts = [];
  parts.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`);
  parts.push(`<style>text { font-family: "Bebas Neue"; font-weight: 400; fill: ${color}; }</style>`);
  for (let i = 0; i < lines; i++) {
    const yBaseline = margin + lineH * (i + 0.78);
    parts.push(
      `<g transform="translate(0,${yBaseline.toFixed(1)}) scale(1,${scaleY})">` +
      `<text x="${margin}" y="0" ` +
      `font-size="${fontSize}" textLength="${innerW}" lengthAdjust="spacing">` +
      escapedLines[i] +
      `</text>` +
      `</g>`
    );
  }
  parts.push(`</svg>`);
  const svg = parts.join("\n");
  const r = new Resvg(svg, {
    fitTo: { mode: "original" },
    background: "rgba(0,0,0,0)",
    font: { fontFiles: [FONT_PATH], loadSystemFonts: false, defaultFontFamily: "Bebas Neue" },
  });
  fs.writeFileSync(outPath, r.render().asPng());
}

async function rasterise(svg, width, height, outPath) {
  const r = new Resvg(svg, {
    fitTo: { mode: "width", value: width },
    background: "rgba(0,0,0,0)",
  });
  const buf = r.render().asPng();
  // resvg's height can round 1 pixel away from `height` due to viewBox
  // ratio; force-resize to the exact canvas so sharp can composite.
  await sharp(buf)
    .resize(width, height, { kernel: "nearest", fit: "fill" })
    .png({ compressionLevel: 6 })
    .toFile(outPath);
}

const mode = process.argv[2] || "sample";

const A3_RATIO = 297 / 420;   // ≈ 0.7071

const PRESETS = {
  sample: {
    // Quarter of full: 1754 x 2483, easy to inspect.
    width: 1754, height: 2483,
    captions: 30, semantic: 2,
    outName: "poster_sample",
  },
  full: {
    // A3 portrait, 600 dpi. Page rendered at width/scale; everything
    // ends up 2.25x larger relative to the final canvas.
    width: 7016, height: 9933,
    captions: 1100, semantic: 80,
    scale: 2.25,
    outName: "poster_full",
  },
  full_landscape: {
    // A3 landscape, 600 dpi.
    width: 9933, height: 7016,
    captions: 1430, semantic: 30,
    scale: 2.25,
    outName: "poster_full_landscape",
  },
};

const cfg = PRESETS[mode];
if (!cfg) { console.error("unknown mode:", mode); process.exit(1); }
render(cfg).catch(err => { console.error(err); process.exit(1); });
