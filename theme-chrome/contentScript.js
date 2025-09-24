// A single <style> we control:
let styleEl = null;
let isApplied = false;

// Utilities
function hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex.trim());
  if (!m) return null;
  return {
    r: parseInt(m[1], 16),
    g: parseInt(m[2], 16),
    b: parseInt(m[3], 16)
  };
}

function relLuminance({ r, g, b }) {
  // https://www.w3.org/TR/WCAG20-TECHS/G17.html#G17-tests
  const srgb = [r, g, b].map(v => v / 255).map(c => (c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)));
  return 0.2126 * srgb[0] + 0.7152 * srgb[1] + 0.0722 * srgb[2];
}

function contrastRatio(hexA, hexB) {
  const A = hexToRgb(hexA), B = hexToRgb(hexB);
  if (!A || !B) return 1;
  const L1 = relLuminance(A), L2 = relLuminance(B);
  const lighter = Math.max(L1, L2), darker = Math.min(L1, L2);
  return (lighter + 0.05) / (darker + 0.05);
}

function readableTextOn(bg, fallback = "#000000") {
  const whiteCR = contrastRatio(bg, "#ffffff");
  const blackCR = contrastRatio(bg, "#000000");
  // Prefer higher contrast; require >= 4.5 if possible
  if (whiteCR >= 4.5 || whiteCR > blackCR) return "#ffffff";
  if (blackCR >= 4.5 || blackCR > whiteCR) return "#000000";
  // fallback
  return fallback;
}

function ensure4Hexes(arr) {
  if (!Array.isArray(arr)) return null;
  const hexes = arr.filter(x => typeof x === "string" && /^#[0-9a-f]{6}$/.test(x.trim()));
  if (hexes.length !== 4) return null;
  return hexes.map(h => h.toLowerCase());
}

// Build CSS string using variables
// function buildCSS([c0, c1, c2, c3]) {
//   const textOn0 = readableTextOn(c0);
//   const textOn1 = readableTextOn(c1);
//   const textOn2 = readableTextOn(c2);
//   const textOn3 = readableTextOn(c3);

//   return `
// :root.palette-painter {
//   --pp-c0: ${c0};
//   --pp-c1: ${c1};
//   --pp-c2: ${c2};
//   --pp-c3: ${c3};
//   --pp-on-c0: ${textOn0};
//   --pp-on-c1: ${textOn1};
//   --pp-on-c2: ${textOn2};
//   --pp-on-c3: ${textOn3};
// }
// :root.palette-painter, :root.palette-painter body {
//   background-color: var(--pp-c0) !important;
//   color: var(--pp-on-c0) !important;
// }
// :root.palette-painter body * {
//   border-color: color-mix(in srgb, var(--pp-c3) 50%, var(--pp-c0)) !important;
// }

// /* links/buttons/inputs */
// :root.palette-painter a { color: var(--pp-c2) !important; }
// :root.palette-painter a:hover { color: var(--pp-c3) !important; }

// :root.palette-painter button,
// :root.palette-painter [role="button"],
// :root.palette-painter .btn,
// :root.palette-painter input[type="submit"],
// :root.palette-painter input[type="button"] {
//   background: var(--pp-c2) !important;
//   color: var(--pp-on-c2) !important;
//   border: 1px solid var(--pp-c3) !important;
// }

// :root.palette-painter input,
// :root.palette-painter textarea,
// :root.palette-painter select {
//   background: color-mix(in srgb, var(--pp-c1) 70%, white) !important;
//   color: var(--pp-on-c1) !important;
// }

// :root.palette-painter header, 
// :root.palette-painter nav, 
// :root.palette-painter footer,
// :root.palette-painter .navbar,
// :root.palette-painter .toolbar {
//   background: var(--pp-c1) !important;
//   color: var(--pp-on-c1) !important;
// }

// /* cards/panels */
// :root.palette-painter .card, 
// :root.palette-painter .panel, 
// :root.palette-painter .box,
// :root.palette-painter article,
// :root.palette-painter section {
//   background: color-mix(in srgb, var(--pp-c0) 75%, var(--pp-c1)) !important;
// }

// /* tables */
// :root.palette-painter table {
//   background: color-mix(in srgb, var(--pp-c0) 80%, var(--pp-c1)) !important;
//   color: var(--pp-on-c0) !important;
// }
// :root.palette-painter th {
//   background: var(--pp-c1) !important;
//   color: var(--pp-on-c1) !important;
// }
// :root.palette-painter tr:nth-child(even) td {
//   background: color-mix(in srgb, var(--pp-c0) 70%, var(--pp-c1)) !important;
// }

// /* focus/selection */
// :root.palette-painter ::selection {
//   background: var(--pp-c3) !important;
//   color: var(--pp-on-c3) !important;
// }

// /* try to avoid inverting images/videos */
// :root.palette-painter img,
// :root.palette-painter video,
// :root.palette-painter canvas {
//   filter: none !important;
//   background: transparent !important;
// }

// /* override some heavy frameworks */
// :root.palette-painter [class*="bg-"], 
// :root.palette-painter [style*="background"] {
//   background-image: none !important;
// }
// `.trim();
// }

// function applyPalette(palette) {
//   const colors = ensure4Hexes(palette);
//   if (!colors) {
//     console.warn("[Palette Painter] Invalid palette:", palette);
//     return;
//   }

//   if (!styleEl) {
//     styleEl = document.createElement("style");
//     styleEl.setAttribute("data-palette-painter", "true");
//     document.documentElement.appendChild(styleEl);
//   }

//   styleEl.textContent = buildCSS(colors);
//   document.documentElement.classList.add("palette-painter");
//   isApplied = true;
// }
function buildCSS([c0, c1, c2, c3]) {
  const textOn0 = readableTextOn(c0);
  const textOn1 = readableTextOn(c1);
  const textOn2 = readableTextOn(c2);
  const textOn3 = readableTextOn(c3);

  return `
:root.palette-painter {
  --pp-c0: ${c0};
  --pp-c1: ${c1};
  --pp-c2: ${c2};
  --pp-c3: ${c3};
  --pp-on-c0: ${textOn0};
  --pp-on-c1: ${textOn1};
  --pp-on-c2: ${textOn2};
  --pp-on-c3: ${textOn3};
}

/* ===== Global overrides for all elements ===== */
:root.palette-painter body,
:root.palette-painter body *:not(img):not(video):not(canvas):not(svg):not(path):not(iframe) {
  background-color: var(--pp-c0) !important;
  color: var(--pp-on-c0) !important;
  border-color: var(--pp-c3) !important;
  box-shadow: none !important;
  text-shadow: none !important;
}

/* Buttons & Inputs */
:root.palette-painter button,
:root.palette-painter [role="button"],
:root.palette-painter .btn,
:root.palette-painter input[type="submit"],
:root.palette-painter input[type="button"] {
  background: var(--pp-c2) !important;
  color: var(--pp-on-c2) !important;
  border: 1px solid var(--pp-c3) !important;
}

/* Links */
:root.palette-painter a {
  color: var(--pp-c3) !important;
}
:root.palette-painter a:hover {
  color: var(--pp-c2) !important;
}

/* Headings */
:root.palette-painter h1, 
:root.palette-painter h2, 
:root.palette-painter h3, 
:root.palette-painter h4, 
:root.palette-painter h5, 
:root.palette-painter h6 {
  color: var(--pp-on-c1) !important;
}

/* Form elements */
:root.palette-painter input,
:root.palette-painter textarea,
:root.palette-painter select {
  background: var(--pp-c1) !important;
  color: var(--pp-on-c1) !important;
  border: 1px solid var(--pp-c3) !important;
}

/* Tables */
:root.palette-painter table,
:root.palette-painter th,
:root.palette-painter td {
  background: var(--pp-c0) !important;
  color: var(--pp-on-c0) !important;
  border: 1px solid var(--pp-c3) !important;
}

/* Header/Nav/Footer */
:root.palette-painter header, 
:root.palette-painter nav, 
:root.palette-painter footer {
  background: var(--pp-c1) !important;
  color: var(--pp-on-c1) !important;
}

/* Cards, Panels, Sections */
:root.palette-painter .card, 
:root.palette-painter .panel, 
:root.palette-painter section,
:root.palette-painter article,
:root.palette-painter .box {
  background: var(--pp-c1) !important;
  color: var(--pp-on-c1) !important;
}

/* Selection highlight */
:root.palette-painter ::selection {
  background: var(--pp-c3) !important;
  color: var(--pp-on-c3) !important;
}

/* Avoid changing images/videos/svg paths */
:root.palette-painter img,
:root.palette-painter video,
:root.palette-painter canvas,
:root.palette-painter svg,
:root.palette-painter path,
:root.palette-painter iframe {
  background: transparent !important;
  filter: none !important;
}
`.trim();
}

function applyPalette(palette) {
  const colors = ensure4Hexes(palette);
  if (!colors) {
    console.warn("[Palette Painter] Invalid palette:", palette);
    return;
  }

  if (!styleEl) {
    styleEl = document.createElement("style");
    styleEl.setAttribute("data-palette-painter", "true");
    document.documentElement.appendChild(styleEl);
  }

  styleEl.textContent = buildCSS(colors);
  document.documentElement.classList.add("palette-painter");
  isApplied = true;
}

function resetPalette() {
  if (styleEl) styleEl.remove();
  styleEl = null;
  document.documentElement.classList.remove("palette-painter");
  isApplied = false;
}

// Messages from popup/background
chrome.runtime.onMessage.addListener((msg) => {
  if (msg?.type === "APPLY_PALETTE") {
    applyPalette(msg.palette);
  } else if (msg?.type === "RESET_PALETTE") {
    resetPalette();
  }
});
