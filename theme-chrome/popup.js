const descEl = document.getElementById("desc");
const generateBtn = document.getElementById("generate");
const resetBtn = document.getElementById("reset");
const statusEl = document.getElementById("status");
const swatchesEl = document.getElementById("swatches");

function isHexColor(s) {
  return typeof s === "string" && /^#[0-9a-f]{6}$/.test(s);
}

function luminance(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex.trim());
  if (!m) return 0;
  const [r,g,b] = [m[1], m[2], m[3]].map(x => parseInt(x,16) / 255);
  const f = (c) => (c <= 0.03928 ? c/12.92 : Math.pow((c+0.055)/1.055, 2.4));
  const [R,G,B] = [f(r), f(g), f(b)];
  return 0.2126*R + 0.7152*G + 0.0722*B;
}

function paintSwatches(colors) {
  swatchesEl.replaceChildren();
  colors.forEach(c => {
    const div = document.createElement("div");
    div.className = "swatch " + (luminance(c) < 0.5 ? "dark" : "light");
    div.style.background = c;
    div.textContent = c;
    swatchesEl.appendChild(div);
  });
  swatchesEl.hidden = false;
}

function setStatus(msg, ok = true) {
  statusEl.textContent = msg || "";
  statusEl.style.color = ok ? "#0a0" : "#c00";
}

generateBtn.addEventListener("click", () => {
  const description = (descEl.value || "").trim();
  if (!description) {
    setStatus("Please enter a description.", false);
    return;
  }
  setStatus("Generating...");
  chrome.runtime.sendMessage(
    { type: "FETCH_PALETTE", description },
    (resp) => {
      if (!resp?.ok) {
        setStatus(resp?.error || "Failed to call API", false);
        return;
      }
      const colors = (resp.data?.colors || []).map(String).map(s => s.toLowerCase());
      if (colors.length !== 4 || !colors.every(isHexColor)) {
        setStatus("API did not return 4 valid hex colors.", false);
        return;
      }
      paintSwatches(colors);
      setStatus("Applying…");
      chrome.runtime.sendMessage(
        { type: "APPLY_TO_ACTIVE_TAB", palette: colors, description },
        (applyResp) => {
          if (!applyResp?.ok) {
            setStatus(applyResp?.error || "Failed to apply colors", false);
          } else {
            setStatus("Applied!");
          }
        }
      );
    }
  );
});

resetBtn.addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "RESET_ACTIVE_TAB" }, (resp) => {
    if (!resp?.ok) {
      setStatus(resp?.error || "Failed to reset", false);
    } else {
      setStatus("Reset complete.");
      swatchesEl.hidden = true;
    }
  });
});
