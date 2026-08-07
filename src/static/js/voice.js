/* Voice conversation -- push-to-talk.
 *
 * - Press and hold the mic button (or Space/Enter when focused) to record.
 * - Release to send the complete recording to the HTTP /voice/turn endpoint,
 *   which handles STT -> RAG -> LLM -> TTS and returns the transcript, answer,
 *   sources, and reply audio URL.
 * - SocketIO stays connected for real-time status notifications but does NOT
 *   carry binary audio (the HTTP path is robust and runs in a proper request
 * - Esc or drag-away cancels the recording.
 */
(function () {
  "use strict";

  // Read server config from a CSP-safe <script type="application/json"> block.
  var config = {};
  try {
    var cfgEl = document.getElementById("nb-config");
    if (cfgEl) config = JSON.parse(cfgEl.textContent || "{}");
  } catch (e) {
    config = {};
  }
  if (!config.voice_enabled || !config.notebook_id) return;

  var micBtn = document.getElementById("voice-mic");
  var statusEl = document.getElementById("voice-status");
  var sendBtn = document.getElementById("chat-send");
  var maxSeconds = config.voice_max_recording_seconds || 60;

  if (!micBtn) return;

  var mediaRecorder = null;
  var chunks = [];
  var recording = false;
  var timer = null;
  var elapsed = 0;

  // --- SocketIO — status notifications only, no binary audio ---
  var socket = null;
  if (window.io) {
    socket = io("/voice");
    socket.on("voice:status", function (d) {
      if (d && d.state) setStatus(d.state.charAt(0).toUpperCase() + d.state.slice(1) + "...");
    });
    socket.on("voice:error", function (d) {
      setStatus("Voice error: " + ((d && d.error) || "error"));
    });
  }

  function setStatus(msg) {
    if (statusEl) statusEl.textContent = msg;
  }

  function isSecureContext() {
    return window.isSecureContext || location.hostname === "localhost" || location.hostname === "127.0.0.1";
  }

  function startRecording() {
    if (recording) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("Microphone not supported in this browser.");
      return;
    }
    if (!window.MediaRecorder) {
      setStatus("MediaRecorder is not supported in this browser.");
      return;
    }
    if (!isSecureContext()) {
      setStatus("Microphone requires HTTPS or localhost.");
      return;
    }
    chunks = [];
    elapsed = 0;
    navigator.mediaDevices
      .getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      })
      .then(function (stream) {
        var mime =
          ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"].find(function (m) {
            return MediaRecorder.isTypeSupported(m);
          }) || "";
        mediaRecorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
        mediaRecorder.ondataavailable = function (e) {
          if (e.data && e.data.size > 0) chunks.push(e.data);
        };
        mediaRecorder.onstop = function () {
          stream.getTracks().forEach(function (t) { t.stop(); });
          var blob = new Blob(chunks, { type: mime || "audio/webm" });
          if (blob.size === 0) {
            setStatus("No audio captured. Try again.");
            return;
          }
          sendVoiceTurn(blob);
        };
        mediaRecorder.start();
        recording = true;
        micBtn.classList.add("active", "btn-danger");
        micBtn.classList.remove("btn-outline-info");
        setStatus("Recording... release to send (0s)");
        timer = setInterval(function () {
          elapsed += 1;
          setStatus("Recording... release to send (" + elapsed + "s)");
          if (elapsed >= maxSeconds) stopRecording();
        }, 1000);
      })
      .catch(function (err) {
        setStatus("Microphone permission denied: " + err.message);
      });
  }

  function stopRecording() {
    if (!recording) return;
    recording = false;
    if (timer) { clearInterval(timer); timer = null; }
    micBtn.classList.remove("active", "btn-danger");
    micBtn.classList.add("btn-outline-info");
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
    setStatus("Processing...");
  }

  function cancelRecording() {
    if (!recording) return;
    recording = false;
    if (timer) { clearInterval(timer); timer = null; }
    micBtn.classList.remove("active", "btn-danger");
    micBtn.classList.add("btn-outline-info");
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.onstop = null;
      mediaRecorder.stream.getTracks().forEach(function (t) { t.stop(); });
    }
    setStatus("Cancelled.");
  }

  function sendVoiceTurn(blob) {
    var fd = new FormData();
    fd.append("audio", blob, "rec.webm");
    setStatus("Transcribing...");
    if (sendBtn) sendBtn.disabled = true;

    // Show typing indicator in the chat area while processing.
    var typing = ChatUI.appendTypingIndicator();

    fetch("/notebooks/" + config.notebook_id + "/voice/turn", {
      method: "POST",
      body: fd,
      credentials: "same-origin",
      headers: { "X-CSRFToken": config.csrf_token || "" },
    })
      .then(function (r) {
        return r.json().then(function (j) { return { status: r.status, json: j }; });
      })
      .then(function (r) {
        if (r.status === 200) {
          var j = r.json || {};
          // Remove typing indicator.
          if (typing && typing.div) typing.div.remove();
          // Render the transcript and answer with the same styling as regular chat.
          ChatUI.appendMessage("user", j.transcript || "");
          var assistantDiv = ChatUI.appendMessage("assistant", j.answer || "");
          if (assistantDiv && j.sources) ChatUI.appendSources(assistantDiv, j.sources);
          if (j.reply_audio_url) playReply(j.reply_audio_url);
          setStatus("");
        } else if (r.status === 422 && (r.json && r.json.error === "no_speech")) {
          if (typing && typing.div) typing.div.remove();
          setStatus("No speech detected. Try again.");
        } else {
          if (typing && typing.div) typing.div.remove();
          setStatus("Voice turn failed: " + ((r.json && r.json.error) || "error"));
        }
      })
      .catch(function (err) {
        if (typing && typing.div) typing.div.remove();
        setStatus("Network error: " + err.message);
      })
      .finally(function () {
        if (sendBtn) sendBtn.disabled = false;
      });
  }

  function playReply(url) {
    var audio = new Audio(url);
    audio.play().catch(function (err) { setStatus("Could not play reply: " + err.message); });
  }

  // Press-and-hold (mouse + touch + keyboard).
  micBtn.addEventListener("mousedown", startRecording);
  micBtn.addEventListener("mouseup", stopRecording);
  micBtn.addEventListener("mouseleave", cancelRecording);
  micBtn.addEventListener("touchstart", function (e) { e.preventDefault(); startRecording(); }, { passive: false });
  micBtn.addEventListener("touchend", stopRecording);
  micBtn.addEventListener("keydown", function (e) {
    if ((e.code === "Space" || e.code === "Enter") && !e.repeat) startRecording();
  });
  micBtn.addEventListener("keyup", function (e) {
    if (e.code === "Space" || e.code === "Enter") stopRecording();
  });
  document.addEventListener("keydown", function (e) {
    if (e.code === "Escape" && recording) cancelRecording();
  });
})();
