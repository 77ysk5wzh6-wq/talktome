const path = require("path");
const fs = require("fs");
const { Resvg } = require("@resvg/resvg-js");

const POEM_LINES = [
  "Ah Love! could thou and I with",
  "Fate conspire To grasp this",
  "sorry Scheme of Things entire,",
  "Would not we shatter it to",
  "bits - and then Re-mould it",
  "nearer to the Heart’s Desire!",
];
const FONT_PATH = path.resolve("fonts/BebasNeue-Regular.ttf");

function render(width, height, outPath, scaleY, leading) {
  const margin = Math.round(Math.min(width, height) * 0.01);
  const innerW = width - margin * 2;
  const innerH = height - margin * 2;
  const rowPitch = (innerH / POEM_LINES.length) * leading;
  const blockH = rowPitch * POEM_LINES.length;
  const topOffset = (innerH - blockH) / 2;
  const fontSize = Math.round(rowPitch * 0.7);
  const escaped = POEM_LINES.map(l =>
    l.replace(/&/g, "&amp;").replace(/</g, "&lt;")
  );
  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">`,
    `<rect width="${width}" height="${height}" fill="white"/>`,
    `<style>text { font-family:"Bebas Neue"; fill: black; }</style>`,
  ];
  for (let i = 0; i < POEM_LINES.length; i++) {
    const yBaseline = margin + topOffset + rowPitch * (i + 0.78);
    parts.push(
      `<g transform="translate(0,${yBaseline.toFixed(1)}) scale(1,${scaleY})">` +
      `<text x="${margin}" y="0" font-size="${fontSize}" ` +
      `textLength="${innerW}" lengthAdjust="spacing">${escaped[i]}</text>` +
      `</g>`
    );
  }
  parts.push(`</svg>`);
  const r = new Resvg(parts.join("\n"), {
    font: {
      fontFiles: [FONT_PATH],
      loadSystemFonts: false,
      defaultFontFamily: "Bebas Neue",
    },
  });
  fs.writeFileSync(outPath, r.render().asPng());
}

// scaleY 1.65 (+10% taller than prev 1.5), leading 0.7, margin 1%.
render(2000, 1400, "out/_poem_sample_black.png", 1.65, 0.7);
console.log("saved");
