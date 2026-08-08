/* notebook-project — lightweight markdown-to-HTML renderer.
 *
 * Supports the subset of markdown that LLM responses commonly use:
 * - Bold (**text**), italic (*text*), inline code (`code`)
 * - Code blocks (```lang\n...\n```)
 * - Headings (##, ###, ####)
 * - Bullet lists (- or * at line start)
 * - Numbered lists (1. at line start)
 * - Links [text](url) — only http/https URLs (no javascript:)
 * - Blockquotes (> at line start)
 * - Paragraphs separated by blank lines
 * - Horizontal rules (---)
 *
 * Security: all text is HTML-escaped before markdown syntax is applied,
 * so user/LLM content cannot inject HTML. Links are restricted to http/https.
 *
 * No external dependencies. Designed for chat-bubble rendering.
 */
(function () {
  "use strict";

  function escapeHtml(text) {
    var div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function renderInline(text) {
    // Escape first so no raw HTML can slip through.
    var html = escapeHtml(text);

    // Inline code: `code` -> <code>code</code>
    html = html.replace(/`([^`]+)`/g, function (m, code) {
      return '<code class="md-code">' + code + "</code>";
    });

    // Bold: **text** -> <strong>text</strong>
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    // Italic: *text* or _text_ -> <em>text</em> (not touching ** already handled)
    html = html.replace(/(?<!\w)\*(.+?)\*(?!\w)/g, "<em>$1</em>");
    html = html.replace(/(?<!\w)_(.+?)_(?!\w)/g, "<em>$1</em>");

    // Links: [text](url) -> <a href="url">text</a> (http/https only)
    html = html.replace(
      /\[([^\]]*)\]\(([^)]*)\)/g,
      function (m, label, url) {
        if (/^https?:\/\//i.test(url)) {
          return (
            '<a href="' + url + '" target="_blank" rel="noopener noreferrer">' + label + "</a>"
          );
        }
        return m;
      }
    );

    return html;
  }

  function renderMarkdown(text) {
    if (!text) return "";

    var lines = text.split("\n");
    var htmlParts = [];
    var i = 0;

    while (i < lines.length) {
      var line = lines[i];

      // Code block: ```lang\n...\n```
      if (line.trimStart().startsWith("```")) {
        var lang = line.trim().slice(3);
        var codeLines = [];
        i++;
        while (i < lines.length && !lines[i].trimStart().startsWith("```")) {
          codeLines.push(lines[i]);
          i++;
        }
        i++; // skip closing ```
        htmlParts.push(
          '<pre class="md-code-block"><code>' +
            escapeHtml(codeLines.join("\n")) +
            "</code></pre>"
        );
        continue;
      }

      // Horizontal rule: --- (on its own line)
      if (/^---+\s*$/.test(line)) {
        htmlParts.push('<hr class="md-hr">');
        i++;
        continue;
      }

      // Headings: ##, ###, #### (avoid # which is rarely used in chat)
      var headingMatch = line.match(/^(#{2,4})\s+(.*)$/);
      if (headingMatch) {
        var level = headingMatch[1].length;
        htmlParts.push(
          "<h" + level + ' class="md-heading">' +
            renderInline(headingMatch[2]) +
            "</h" + level + ">"
        );
        i++;
        continue;
      }

      // Blockquote: > text
      if (line.trimStart().startsWith("> ")) {
        var quoteLines = [];
        while (i < lines.length && lines[i].trimStart().startsWith("> ")) {
          quoteLines.push(lines[i].trimStart().slice(2));
          i++;
        }
        htmlParts.push(
          '<blockquote class="md-blockquote">' + renderInline(quoteLines.join(" ")) + "</blockquote>"
        );
        continue;
      }

      // Bullet list: - or * at line start
      if (/^[\-\*]\s+/.test(line)) {
        var items = [];
        while (i < lines.length && /^[\-\*]\s+/.test(lines[i])) {
          items.push("<li>" + renderInline(lines[i].replace(/^[\-\*]\s+/, "")) + "</li>");
          i++;
        }
        htmlParts.push('<ul class="md-list">' + items.join("") + "</ul>");
        continue;
      }

      // Numbered list: 1. at line start
      if (/^\d+\.\s+/.test(line)) {
        var numItems = [];
        while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
          numItems.push("<li>" + renderInline(lines[i].replace(/^\d+\.\s+/, "")) + "</li>");
          i++;
        }
        htmlParts.push('<ol class="md-list">' + numItems.join("") + "</ol>");
        continue;
      }

      // Blank line — skip (paragraph separation)
      if (line.trim() === "") {
        i++;
        continue;
      }

      // Paragraph: collect consecutive non-empty, non-special lines
      var paraLines = [];
      while (
        i < lines.length &&
        lines[i].trim() !== "" &&
        !lines[i].trimStart().startsWith("```") &&
        !lines[i].trimStart().startsWith("> ") &&
        !/^#{2,4}\s/.test(lines[i]) &&
        !/^[\-\*]\s+/.test(lines[i]) &&
        !/^\d+\.\s+/.test(lines[i]) &&
        !/^---+\s*$/.test(lines[i])
      ) {
        paraLines.push(lines[i]);
        i++;
      }
      if (paraLines.length > 0) {
        htmlParts.push("<p>" + renderInline(paraLines.join(" ")) + "</p>");
      }
    }

    return htmlParts.join("");
  }

  window.ChatMarkdown = { render: renderMarkdown };
})();