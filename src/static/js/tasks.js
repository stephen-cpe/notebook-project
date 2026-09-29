/* notebook-project — background-task bell (singleflight tracker feed).
 *
 * The bell opens a dropdown listing recent Audio / Video Overview jobs.
 * Clicking an item opens the notebook it belongs to; "Mark all read"
 * clears the unread badge. Polling refreshes the list every 15s.
 */
(function () {
  "use strict";

  function csrfToken() {
    var meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute("content") : "";
  }

  function el(id) { return document.getElementById(id); }

  function relativeTime(seconds) {
    if (!seconds) return "";
    var delta = Math.max(0, Math.floor(Date.now() / 1000 - seconds));
    if (delta < 60) return "just now";
    if (delta < 3600) return Math.floor(delta / 60) + "m ago";
    if (delta < 86400) return Math.floor(delta / 3600) + "h ago";
    return Math.floor(delta / 86400) + "d ago";
  }

  var STATUS_META = {
    running: { icon: "bi-arrow-repeat", cls: "text-info", text: "Running" },
    ready: { icon: "bi-check-circle-fill", cls: "text-success", text: "Ready" },
    failed: { icon: "bi-exclamation-triangle-fill", cls: "text-danger", text: "Failed" },
  };

  function renderItem(task) {
    var meta = STATUS_META[task.status] || { icon: "bi-circle", cls: "text-secondary", text: task.status };
    var item = document.createElement("a");
    item.className = "tasks-item d-block px-3 py-2 text-decoration-none border-bottom border-secondary";
    if (!task.read && (task.status === "ready" || task.status === "failed")) {
      item.className += " tasks-item-unread";
    }
    item.href = task.notebook_id ? "/notebooks/" + task.notebook_id : "#";
    if (task.notebook_id) {
      item.addEventListener("click", function () {
        // Opening the notebook is the natural "read" action: clear this
        // item's unread highlight as we navigate away.
        if (!task.read) markRead(task.key);
      });
    }

    var top = document.createElement("div");
    top.className = "d-flex justify-content-between align-items-center gap-2";

    var label = document.createElement("span");
    label.className = "small text-body text-truncate";
    label.textContent = task.label || "Background task";

    var status = document.createElement("span");
    status.className = "small " + meta.cls + " text-nowrap";
    status.innerHTML = '<i class="bi ' + meta.icon + ' me-1"></i>' + meta.text;

    top.appendChild(label);
    top.appendChild(status);
    item.appendChild(top);

    var sub = document.createElement("div");
    sub.className = "small text-secondary";
    var bits = [];
    if (task.notebook_name) bits.push(task.notebook_name);
    if (task.status === "running") bits.push(task.progress + "%");
    var when = relativeTime(task.updated_at);
    if (when) bits.push(when);
    sub.textContent = bits.join(" · ");
    if (bits.length) item.appendChild(sub);
    return item;
  }

  function render(data) {
    var list = el("tasks-list");
    if (!list) return;
    list.textContent = "";
    var tasks = data.tasks || [];
    if (!tasks.length) {
      var empty = document.createElement("div");
      empty.className = "text-secondary small p-3 text-center";
      empty.textContent = "No background tasks yet.";
      list.appendChild(empty);
      return;
    }
    tasks.forEach(function (t) { list.appendChild(renderItem(t)); });
  }

  function markRead(key) {
    return fetch("/tasks/read", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": csrfToken(),
      },
      body: JSON.stringify({ key: key }),
    });
  }

  function markAllRead() {
    return fetch("/tasks/read", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": csrfToken(),
      },
      body: JSON.stringify({}),
    });
  }

  function poll() {
    if (!el("tasks-bell")) return Promise.resolve();
    return fetch("/tasks", { headers: { "X-Requested-With": "XMLHttpRequest" } })
      .then(function (r) {
        if (!r.ok) return null;
        return r.json();
      })
      .then(function (data) {
        if (!data) return;
        var badge = el("tasks-badge");
        if (badge) {
          var unread = data.unread || 0;
          badge.textContent = unread;
          badge.classList.toggle("d-none", unread === 0);
        }
        render(data);
      })
      .catch(function () {});
  }

  document.addEventListener("DOMContentLoaded", function () {
    if (!el("tasks-bell")) return;
    poll();
    setInterval(poll, 15000);
    var markAll = el("tasks-mark-all");
    if (markAll) {
      markAll.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        markAllRead()
          .catch(function () {})
          .then(poll);
      });
    }
  });
})();
