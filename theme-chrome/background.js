const API_BASE = "http://localhost:8000";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "FETCH_PALETTE") {
    fetch(`${API_BASE}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        description: message.description,
        max_new_tokens: 64,
        temperature: 0.2,
        top_p: 0.9
      })
    })
      .then(async (r) => {
        if (!r.ok) {
          const text = await r.text().catch(() => "");
          throw new Error(`API ${r.status}: ${text || r.statusText}`);
        }
        return r.json();
      })
      .then((data) => {
        sendResponse({ ok: true, data });
      })
      .catch((err) => {
        sendResponse({ ok: false, error: err.message || String(err) });
      });
    return true; // keep the message channel open for async response
  }

  if (message?.type === "APPLY_TO_ACTIVE_TAB") {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tabId = tabs?.[0]?.id;
      if (!tabId) return sendResponse({ ok: false, error: "No active tab" });
      chrome.tabs.sendMessage(tabId, {
        type: "APPLY_PALETTE",
        palette: message.palette,
        description: message.description
      });
      sendResponse({ ok: true });
    });
    return true;
  }

  if (message?.type === "RESET_ACTIVE_TAB") {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tabId = tabs?.[0]?.id;
      if (!tabId) return sendResponse({ ok: false, error: "No active tab" });
      chrome.tabs.sendMessage(tabId, { type: "RESET_PALETTE" });
      sendResponse({ ok: true });
    });
    return true;
  }
});
