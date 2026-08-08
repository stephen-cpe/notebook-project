/* notebook-project — shared chat UI helpers.
 *
 * Used by both app.js (regular text chat) and voice.js (push-to-talk) so that
 * voice turns and text turns render with identical bubble styling, typing
 * indicators, and source citation badges.
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

  ChatUI.appendSources = function (parentDiv, sources) {
    if (!parentDiv || !sources || sources.length === 0) return;
    var src = document.createElement("div");
    src.className = "small mt-1";
    var tags = sources
      .map(function (s) {
        var label = s.filename + (s.page ? " p." + s.page : "");
        var tag = document.createElement("span");
        tag.className = "badge bg-secondary me-1 source-citation";
        tag.textContent = label;
        tag.setAttribute("title", "Source: " + label);
        return tag.outerHTML;
      })
      .join("");
    src.innerHTML = '<span class="text-secondary">Sources: </span>' + tags;
    parentDiv.appendChild(src);
    var messages = document.getElementById("chat-messages");
    if (messages) messages.scrollTop = messages.scrollHeight;
  };

  window.ChatUI = ChatUI;
})();