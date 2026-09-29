/* notebook-project — shared chat UI helpers.
 *
 * Used by both app.js (regular text chat) and voice.js (voice mode) so that
 * voice turns and text turns render with identical bubble styling, typing
 * indicators, and source citations.
 *
 * Citations render as ONE collapsed toggle line per answer
 * ("Sources (N of M) · X% read"); the full grouped list lives inside the
 * expandable panel so long citation lists never flood the chat window.
 */
(function () {
  "use strict";

  var ChatUI = {};

  ChatUI.appendMessage = function (role, text) {
    var messages = document.getElementById("chat-messages");
    if (!messages) return null;
    var div = document.createElement("div");
    div.className = "mb-2 " + (role === "user" ? "text-end" : "");
    var bubble = document.createElement("span");
    bubble.className =
      "d-inline-block px-3 py-2 rounded " +
      (role === "user"
        ? "bg-info text-dark"
        : "bg-body-tertiary border border-secondary");
    bubble.style.maxWidth = "80%";
    if (role === "user") {
      bubble.textContent = text;
    } else {
      bubble.className += " md-content";
      bubble.innerHTML = window.ChatMarkdown.render(text);
    }
    div.appendChild(bubble);
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return div;
  };

  ChatUI.appendTypingIndicator = function () {
    var messages = document.getElementById("chat-messages");
    if (!messages) return null;
    var div = document.createElement("div");
    div.className = "mb-2";
    var bubble = document.createElement("span");
    bubble.className =
      "d-inline-block px-3 py-2 rounded bg-body-tertiary border border-secondary";
    bubble.style.maxWidth = "80%";
    bubble.innerHTML =
      '<span class="typing-indicator"><span></span><span></span><span></span></span> Working...';
    div.appendChild(bubble);
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return { div: div, bubble: bubble };
  };

  function groupByFile(sources) {
    var groups = [];
    var index = {};
    sources.forEach(function (s) {
      var name = s.filename || "unknown";
      var g = index[name];
      if (!g) {
        g = { filename: name, pages: [], figures: [] };
        index[name] = g;
        groups.push(g);
      }
      if (s.page !== null && s.page !== undefined && g.pages.indexOf(s.page) === -1) {
        g.pages.push(s.page);
      }
      (s.figures || []).slice(0, 2).forEach(function (f) {
        if (g.figures.length < 2) g.figures.push(f);
      });
    });
    groups.forEach(function (g) {
      g.pages.sort(function (a, b) { return a - b; });
    });
    return groups;
  }

  ChatUI.appendSources = function (parentDiv, sources, total, ratio) {
    if (!parentDiv || !sources || sources.length === 0) return;
    var shown = sources.length;
    var full = typeof total === "number" && total > shown ? total : shown;

    var wrap = document.createElement("div");
    wrap.className = "small mt-1 cite-wrap";

    var toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "btn btn-link btn-sm p-0 cite-toggle text-decoration-none";
    toggle.setAttribute("aria-expanded", "false");

    var label = full > shown
      ? "Sources (" + shown + " of " + full + ")"
      : "Sources (" + shown + ")";
    if (typeof ratio === "number") {
      label += " · " + Math.round(ratio * 100) + "% read";
    }
    toggle.textContent = "▸ " + label;
    toggle.setAttribute(
      "title",
      "Show cited passages" +
        (full > shown ? " (top " + shown + " of " + full + ")" : "")
    );

    var panel = document.createElement("div");
    panel.className = "cite-panel d-none mt-1";

    groupByFile(sources).forEach(function (g) {
      var row = document.createElement("div");
      row.className = "cite-group";
      var name = document.createElement("span");
      name.className = "badge bg-secondary me-1 source-citation";
      name.textContent = g.filename;
      name.setAttribute("title", "Source: " + g.filename);
      row.appendChild(name);
      if (g.pages.length) {
        var pp = document.createElement("span");
        pp.className = "text-secondary";
        pp.textContent = g.pages.length === 1 ? "p. " + g.pages[0] : "pp. " + g.pages.join(", ");
        row.appendChild(pp);
      }
      panel.appendChild(row);
      g.figures.forEach(function (f) {
        var img = document.createElement("img");
        img.src = f.url;
        img.alt = f.caption || g.filename;
        img.title = f.caption || g.filename;
        img.className = "img-thumbnail mt-1 me-1";
        img.style.maxWidth = "160px";
        img.loading = "lazy";
        panel.appendChild(img);
      });
    });
    if (full > shown) {
      var more = document.createElement("div");
      more.className = "text-secondary mt-1";
      more.textContent = "+" + (full - shown) + " more cited passages (top " + shown + " shown)";
      panel.appendChild(more);
    }

    toggle.addEventListener("click", function () {
      var open = panel.classList.toggle("d-none") === false;
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      toggle.textContent = (open ? "▾ " : "▸ ") + label;
      var messages = document.getElementById("chat-messages");
      if (messages) messages.scrollTop = messages.scrollHeight;
    });

    wrap.appendChild(toggle);
    wrap.appendChild(panel);
    parentDiv.appendChild(wrap);
    var messages = document.getElementById("chat-messages");
    if (messages) messages.scrollTop = messages.scrollHeight;
  };

  window.ChatUI = ChatUI;
})();
