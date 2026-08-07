/* notebook-project — client-side app logic (upload, chat, audio). */
(function () {
  "use strict";

  // Read server config from a CSP-safe <script type="application/json"> block
  // (inline executable scripts are blocked by the script-src CSP, so we cannot
  // use the old `window.NOTEBOOK_ID = ...` pattern).
  var config = {};
  try {
    var cfgEl = document.getElementById("nb-config");
    if (cfgEl) config = JSON.parse(cfgEl.textContent || "{}");
  } catch (e) {
    config = {};
  }
  var NB_ID = config.notebook_id;
  var CSRF = config.csrf_token;

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  function fetchJSON(url, opts) {
    opts = opts || {};
    opts.headers = opts.headers || {};
    opts.headers["X-CSRFToken"] = CSRF;
    if (opts.body && !(opts.body instanceof FormData)) {
      opts.headers["Content-Type"] = "application/json";
    }
    return fetch(url, opts).then(function (r) {
      return r.json().then(function (data) {
        return { ok: r.ok, status: r.status, data: data };
      });
    });
  }

  function el(id) { return document.getElementById(id); }

  // -------------------------------------------------------------------------
  // Upload: drag-and-drop + multi-file with per-file progress
  // -------------------------------------------------------------------------

  var dropZone = el("drop-zone");
  var fileInput = el("file-input");
  var uploadBtn = el("upload-btn");
  var progressBox = el("upload-progress");
  var selectedFiles = [];

  dropZone.addEventListener("click", function () { fileInput.click(); });

  fileInput.addEventListener("change", function () {
    addFiles(fileInput.files);
  });

  ["dragenter", "dragover"].forEach(function (evt) {
    dropZone.addEventListener(evt, function (e) {
      e.preventDefault(); e.stopPropagation();
      dropZone.classList.add("drop-zone-active");
    });
  });
  ["dragleave", "drop"].forEach(function (evt) {
    dropZone.addEventListener(evt, function (e) {
      e.preventDefault(); e.stopPropagation();
      dropZone.classList.remove("drop-zone-active");
    });
  });
  dropZone.addEventListener("drop", function (e) {
    var files = e.dataTransfer.files;
    if (files && files.length) addFiles(files);
  });

  function addFiles(fileList) {
    var allowed = [".pdf", ".docx", ".pptx", ".txt", ".md"];
    for (var i = 0; i < fileList.length; i++) {
      var f = fileList[i];
      var ext = "." + f.name.split(".").pop().toLowerCase();
      if (allowed.indexOf(ext) === -1) {
        setUploadStatus("Unsupported type: " + f.name, "error");
        continue;
      }
      selectedFiles.push(f);
    }
    if (selectedFiles.length > 0) {
      uploadBtn.disabled = false;
      setUploadStatus(selectedFiles.length + " file(s) ready to upload. Click Upload.", "info");
    }
  }

  uploadBtn.addEventListener("click", function () {
    if (selectedFiles.length === 0) return;
    uploadBtn.disabled = true;
    dropZone.style.pointerEvents = "none";

    var total = selectedFiles.length;
    var done = 0, errors = 0;
    var html = '<div class="small mb-1 text-secondary">Uploading ' + total + ' file(s)...</div>';
    html += '<div class="progress mb-2" style="height: 20px;">';
    html += '<div id="upload-bar" class="progress-bar progress-bar-striped progress-bar-animated" ';
    html += 'role="progressbar" style="width: 0%;">0%</div></div>';
    html += '<div id="upload-log" class="small"></div>';
    progressBox.innerHTML = html;

    function uploadNext(idx) {
      if (idx >= total) {
        var bar = el("upload-bar");
        bar.classList.remove("progress-bar-animated");
        bar.style.width = "100%";
        bar.textContent = "Done";
        bar.className = "progress-bar bg-" + (errors > 0 ? "warning" : "success");
        appendLog(done + " succeeded, " + errors + " failed", errors > 0 ? "error" : "success");
        uploadBtn.disabled = false;
        dropZone.style.pointerEvents = "";
        if (errors === 0) {
          appendLog("Reloading page...", "info");
          setTimeout(function () { window.location.reload(); }, 1500);
        } else {
          selectedFiles = [];
          uploadBtn.disabled = true;
          appendLog('Click "Upload" again to retry failed files, or reload the page.', "info");
        }
        return;
      }
      var f = selectedFiles[idx];
      appendLog("Uploading " + f.name + "...", "info");
      updateBar(idx, total);

      var fd = new FormData();
      fd.append("file", f);

      fetchJSON("/notebooks/" + NB_ID + "/sources", { method: "POST", body: fd })
        .then(function (res) {
          done++;
          if (res.ok) {
            appendLog(f.name + " → " + (res.data.status || "ready"), "success");
          } else {
            errors++;
            appendLog(f.name + " → error: " + (res.data.error || "unknown"), "error");
          }
        })
        .catch(function (err) {
          errors++;
          appendLog(f.name + " → network error", "error");
        })
        .finally(function () { uploadNext(idx + 1); });
    }
    uploadNext(0);
  });

  function updateBar(idx, total) {
    var pct = Math.round((idx / total) * 100);
    var bar = el("upload-bar");
    if (bar) { bar.style.width = pct + "%"; bar.textContent = pct + "%"; }
  }

  function appendLog(msg, type) {
    var log = el("upload-log");
    if (!log) return;
    var color = type === "error" ? "danger" : type === "success" ? "success" : "secondary";
    var div = document.createElement("div");
    div.className = "text-" + color;
    div.textContent = msg;
    log.appendChild(div);
  }

  function setUploadStatus(msg, type) {
    var cls = type === "error" ? "danger" : type === "success" ? "success" : "info";
    progressBox.innerHTML = '<div class="alert alert-' + cls + ' py-1 px-2 small mb-0">' + msg + "</div>";
  }

  // -------------------------------------------------------------------------
  // Chat: SSE streaming with loading indicator
  // -------------------------------------------------------------------------

  var chatInput = el("chat-input");
  var chatSend = el("chat-send");
  var chatMessages = el("chat-messages");

  function sendChat() {
    var question = chatInput.value.trim();
    if (!question) return;
    chatInput.value = "";
    chatSend.disabled = true;
    ChatUI.appendMessage("user", question);

    // Assistant bubble with typing indicator.
    var typing = ChatUI.appendTypingIndicator();
    var assistantDiv = typing.div;
    var bubble = typing.bubble;

    var firstToken = true;

    fetch("/notebooks/" + NB_ID + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRFToken": CSRF },
      body: JSON.stringify({ question: question }),
    }).then(function (response) {
      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";

      function read() {
        reader.read().then(function (result) {
          if (result.done) {
            chatSend.disabled = false;
            return;
          }
          buffer += decoder.decode(result.value, { stream: true });
          var lines = buffer.split("\n");
          buffer = lines.pop();

          lines.forEach(function (line) {
            if (!line.startsWith("data: ")) return;
            var data;
            try { data = JSON.parse(line.slice(6)); } catch (e) { return; }

            if (data.token) {
              if (firstToken) {
                bubble.textContent = "";
                firstToken = false;
              }
              bubble.textContent += data.token;
              chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            if (data.done) {
              chatSend.disabled = false;
              ChatUI.appendSources(assistantDiv, data.sources);
            }
            if (data.error) {
              chatSend.disabled = false;
              bubble.textContent = "Error: " + data.error;
              bubble.classList.add("text-danger");
            }
          });
          read();
        });
      }
      read();
    }).catch(function (err) {
      chatSend.disabled = false;
      bubble.textContent = "Network error: " + err.message;
      bubble.classList.add("text-danger");
    });
  }

  chatSend.addEventListener("click", sendChat);
  chatInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });

  // -------------------------------------------------------------------------
  // Suggested questions: click to send
  // -------------------------------------------------------------------------

  document.addEventListener("click", function (e) {
    var link = e.target.closest(".suggested-question");
    if (!link) return;
    e.preventDefault();
    var q = link.getAttribute("data-question");
    if (!q) return;
    chatInput.value = q;
    sendChat();
  });

  // -------------------------------------------------------------------------
  // Clear chat history
  // -------------------------------------------------------------------------

  var clearBtn = el("chat-clear");
  if (clearBtn) {
    clearBtn.addEventListener("click", function () {
      if (!confirm("Clear all chat history for this notebook?")) return;
      clearBtn.disabled = true;
      fetchJSON("/notebooks/" + NB_ID + "/chat/clear", { method: "POST" })
        .then(function (res) {
          if (res.ok) {
            // Remove all chat messages but keep the summary/suggested-questions block.
            var welcome = chatMessages.querySelector(".text-secondary.small.mb-3");
            var suggested = chatMessages.querySelector("#suggested-questions");
            chatMessages.innerHTML = "";
            if (welcome) chatMessages.appendChild(welcome);
            if (suggested) chatMessages.appendChild(suggested);
          } else {
            alert("Failed to clear history.");
          }
          clearBtn.disabled = false;
        })
        .catch(function () {
          alert("Network error clearing history.");
          clearBtn.disabled = false;
        });
    });
  }

  // -------------------------------------------------------------------------
  // Source actions: view text, rename, delete
  // -------------------------------------------------------------------------

  var sourceTextModal = null;
  var sourceTextContent = el("source-text-content");
  var sourceTextTitle = el("source-text-title");
  if (sourceTextContent && window.bootstrap) {
    var modalEl = document.getElementById("sourceTextModal");
    if (modalEl) sourceTextModal = new bootstrap.Modal(modalEl);
  }

  document.addEventListener("click", function (e) {
    var viewBtn = e.target.closest(".source-view-btn");
    if (viewBtn) {
      e.preventDefault();
      openSourceText(viewBtn.getAttribute("data-source-id"));
      return;
    }
    var renameBtn = e.target.closest(".source-rename-btn");
    if (renameBtn) {
      e.preventDefault();
      renameSource(renameBtn.getAttribute("data-source-id"));
      return;
    }
    var delBtn = e.target.closest(".source-delete-btn");
    if (delBtn) {
      e.preventDefault();
      deleteSource(delBtn.getAttribute("data-source-id"), delBtn.getAttribute("data-source-name"));
      return;
    }
  });

  function openSourceText(sourceId) {
    if (!sourceTextModal || !sourceTextContent) return;
    sourceTextContent.textContent = "Loading...";
    sourceTextTitle.textContent = "Extracted Text";
    sourceTextModal.show();
    fetchJSON("/notebooks/" + NB_ID + "/sources/" + sourceId + "/text")
      .then(function (res) {
        if (res.ok) {
          sourceTextContent.textContent = res.data.text || "(empty)";
          var cc = res.data.char_count || 0;
          sourceTextTitle.textContent = "Extracted Text (" + cc + " chars)";
        } else {
          sourceTextContent.textContent = "Error: " + (res.data.error || "not found");
        }
      })
      .catch(function () {
        sourceTextContent.textContent = "Network error.";
      });
  }

  function renameSource(sourceId) {
    var row = document.querySelector('[data-source-id="' + sourceId + '"]');
    var currentName = "";
    if (row) {
      var nameSpan = row.querySelector(".source-filename");
      if (nameSpan) currentName = nameSpan.textContent.trim();
    }
    var newName = prompt("Rename source:", currentName);
    if (!newName || newName.trim() === currentName) return;
    fetchJSON("/notebooks/" + NB_ID + "/sources/" + sourceId + "/rename", {
      method: "PATCH",
      body: JSON.stringify({ filename: newName.trim() }),
    }).then(function (res) {
      if (res.ok) {
        if (nameSpan) nameSpan.textContent = res.data.filename || newName.trim();
      } else {
        alert("Rename failed: " + (res.data.error || "unknown"));
      }
    }).catch(function () {
      alert("Network error during rename.");
    });
  }

  function deleteSource(sourceId, sourceName) {
    if (!confirm("Delete source \"" + sourceName + "\"? This cannot be undone.")) return;
    fetchJSON("/notebooks/" + NB_ID + "/sources/" + sourceId, { method: "DELETE" })
      .then(function (res) {
        if (res.ok) {
          var row = document.querySelector('[data-source-id="' + sourceId + '"]');
          if (row) row.remove();
        } else {
          alert("Delete failed.");
        }
      })
      .catch(function () {
        alert("Network error during delete.");
      });
  }

  // -------------------------------------------------------------------------
  // Audio + Video Overview
  // -------------------------------------------------------------------------

  var audioBtn = el("audio-generate");
  var audioPlayer = el("audio-player");
  var videoBtn = el("video-generate");
  var videoPlayer = el("video-player");

  function getTopic() {
    return (el("overview-topic") || {}).value || "";
  }

  audioBtn.addEventListener("click", function () {
    audioBtn.disabled = true;
    audioBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';

    fetchJSON("/notebooks/" + NB_ID + "/audio", {
      method: "POST",
      body: JSON.stringify({ topic: getTopic() }),
    })
      .then(function (res) {
        if (res.data.status === "queued") {
          pollAudioStatus();
        } else {
          audioBtn.disabled = false;
          audioBtn.innerHTML = '<i class="bi bi-mic"></i> Audio';
          audioPlayer.innerHTML = '<p class="small text-danger">Failed: ' + (res.data.error || "unknown") + '</p>';
        }
      })
      .catch(function () {
        audioBtn.disabled = false;
        audioBtn.innerHTML = '<i class="bi bi-mic"></i> Audio';
        audioPlayer.innerHTML = '<p class="small text-danger">Network error.</p>';
      });
  });

  videoBtn.addEventListener("click", function () {
    videoBtn.disabled = true;
    videoBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';

    fetchJSON("/notebooks/" + NB_ID + "/video", {
      method: "POST",
      body: JSON.stringify({ topic: getTopic() }),
    })
      .then(function (res) {
        if (res.data.status === "queued") {
          pollVideoStatus();
        } else {
          videoBtn.disabled = false;
          videoBtn.innerHTML = '<i class="bi bi-camera-video"></i> Video';
          videoPlayer.innerHTML = '<p class="small text-danger">Failed: ' + (res.data.error || "unknown") + '</p>';
        }
      })
      .catch(function () {
        videoBtn.disabled = false;
        videoBtn.innerHTML = '<i class="bi bi-camera-video"></i> Video';
        videoPlayer.innerHTML = '<p class="small text-danger">Network error.</p>';
      });
  });

  function pollAudioStatus() {
    var attempts = 0;
    var max = 120;
    function poll() {
      if (attempts++ > max) {
        audioBtn.disabled = false;
        audioBtn.innerHTML = '<i class="bi bi-mic"></i> Audio';
        audioPlayer.innerHTML = '<p class="small text-danger">Timed out.</p>';
        return;
      }
      fetchJSON("/notebooks/" + NB_ID + "/audio/status").then(function (res) {
        var st = res.data.status;
        audioBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> ' + st + '...';
        if (st === "ready") {
          showAudioPlayer();
        } else if (st === "failed") {
          audioBtn.disabled = false;
          audioBtn.innerHTML = '<i class="bi bi-mic"></i> Audio';
          audioPlayer.innerHTML = '<p class="small text-danger">Failed.</p>';
        } else {
          setTimeout(poll, 2000);
        }
      });
    }
    poll();
  }

  function pollVideoStatus() {
    var attempts = 0;
    var max = 120;
    function poll() {
      if (attempts++ > max) {
        videoBtn.disabled = false;
        videoBtn.innerHTML = '<i class="bi bi-camera-video"></i> Video';
        videoPlayer.innerHTML = '<p class="small text-danger">Timed out.</p>';
        return;
      }
      fetchJSON("/notebooks/" + NB_ID + "/video/status").then(function (res) {
        var st = res.data.status;
        videoBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> ' + st + '...';
        if (st === "ready") {
          showVideoPlayer();
        } else if (st === "failed") {
          videoBtn.disabled = false;
          videoBtn.innerHTML = '<i class="bi bi-camera-video"></i> Video';
          videoPlayer.innerHTML = '<p class="small text-danger">Failed.</p>';
        } else {
          setTimeout(poll, 2000);
        }
      });
    }
    poll();
  }

  function showAudioPlayer() {
    audioPlayer.innerHTML =
      '<div class="overview-player-card mb-2">' +
      '<div class="overview-player-header"><i class="bi bi-mic-fill"></i> Audio Overview</div>' +
      '<audio controls class="w-100"><source src="/notebooks/' + NB_ID +
      '/audio/file" type="audio/mpeg"></audio>' +
      '<button id="audio-delete" class="btn btn-outline-danger btn-sm w-100 mt-1">' +
      '<i class="bi bi-trash"></i> Delete</button></div>';
    audioBtn.disabled = false;
    audioBtn.innerHTML = '<i class="bi bi-mic"></i> Regen Audio';
    wireAudioDelete();
  }

  function showVideoPlayer() {
    videoPlayer.innerHTML =
      '<div class="overview-player-card">' +
      '<div class="overview-player-header"><i class="bi bi-camera-video-fill"></i> Video Overview</div>' +
      '<video controls class="w-100"><source src="/notebooks/' + NB_ID +
      '/video/file" type="video/mp4"></video>' +
      '<button id="video-delete" class="btn btn-outline-danger btn-sm w-100 mt-1">' +
      '<i class="bi bi-trash"></i> Delete</button></div>';
    videoBtn.disabled = false;
    videoBtn.innerHTML = '<i class="bi bi-camera-video"></i> Regen Video';
    wireVideoDelete();
  }

  function wireAudioDelete() {
    var delBtn = el("audio-delete");
    if (!delBtn) return;
    delBtn.addEventListener("click", function () {
      if (!confirm("Delete this audio overview?")) return;
      delBtn.disabled = true;
      delBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';
      fetchJSON("/notebooks/" + NB_ID + "/audio", { method: "DELETE" })
        .then(function (res) {
          if (res.ok) {
            audioPlayer.innerHTML = "";
            audioBtn.innerHTML = '<i class="bi bi-mic"></i> Audio';
          } else {
            delBtn.disabled = false;
            delBtn.innerHTML = '<i class="bi bi-trash"></i> Delete';
          }
        })
        .catch(function () {
          delBtn.disabled = false;
          delBtn.innerHTML = '<i class="bi bi-trash"></i> Delete';
        });
    });
  }

  function wireVideoDelete() {
    var delBtn = el("video-delete");
    if (!delBtn) return;
    delBtn.addEventListener("click", function () {
      if (!confirm("Delete this video overview?")) return;
      delBtn.disabled = true;
      delBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';
      fetchJSON("/notebooks/" + NB_ID + "/video", { method: "DELETE" })
        .then(function (res) {
          if (res.ok) {
            videoPlayer.innerHTML = "";
            videoBtn.innerHTML = '<i class="bi bi-camera-video"></i> Video';
          } else {
            delBtn.disabled = false;
            delBtn.innerHTML = '<i class="bi bi-trash"></i> Delete';
          }
        })
        .catch(function () {
          delBtn.disabled = false;
          delBtn.innerHTML = '<i class="bi bi-trash"></i> Delete';
        });
    });
  }

  wireAudioDelete();
  wireVideoDelete();

})();
