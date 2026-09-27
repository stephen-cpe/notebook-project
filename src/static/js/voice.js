/* Voice mode V2 -- toggle (not push-to-talk).
 *
 * - Click the mic button once to enter voice mode: the text input row hides
 *   and a voice panel (orb + state + End button) appears.
 * - While in voice mode the mic stays open hands-free: speak, pause, and the
 *   utterance auto-sends on ~1.2s of silence (client-side VAD via AnalyserNode).
 * - Each utterance POSTs to the existing HTTP /voice/turn endpoint
 *   (STT -> RAG -> LLM -> TTS), renders via ChatUI like a text turn, and
 *   auto-plays the reply audio. Then listening resumes automatically.
 * - Barge-in: talking while the reply plays stops the audio and starts a new
 *   utterance. Talking while "thinking" aborts the in-flight request.
 * - Click mic / End / Esc to leave voice mode and return to text chat.
 * - SocketIO is intentionally not used (status is local; audio goes over HTTP).
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
  var chatInputRow = document.getElementById("chat-input-row");
  var panel = document.getElementById("voice-mode-panel");
  var orb = document.getElementById("voice-orb");
  var stateEl = document.getElementById("voice-state");
  var hintEl = document.getElementById("voice-hint");
  var transcriptEl = document.getElementById("voice-transcript");
  var endBtn = document.getElementById("voice-end");
  var stopReplyBtn = document.getElementById("voice-stop-reply");
  var sendBtn = document.getElementById("chat-send");
  var maxSeconds = config.voice_max_recording_seconds || 60;

  if (!micBtn || !panel) return;

  // --- Tunables -----------------------------------------------------------
  var VAD_TICK_MS = 100;
  var SILENCE_MS = 1200; // pause length that ends an utterance
  var SPEECH_THRESHOLD = 0.02; // RMS above this counts as speech
  var BARGE_THRESHOLD = 0.03; // RMS above this counts as interruption
  var BARGE_MS = 350; // sustained loud speech while speaking => barge-in
  var THINK_BARGE_MS = 700; // sustained speech while thinking => abort + restart
  var ECHO_GUARD_MS = 500; // ignore mic right after reply starts (echo)

  // --- Session state -------------------------------------------------------
  var voiceMode = false;
  var state = "idle"; // idle | listening | thinking | speaking
  var generation = 0; // bumped on every exit/barge to invalidate callbacks
  var stream = null;
  var audioCtx = null;
  var analyser = null;
  var timeDomain = null;
  var mediaRecorder = null;
  var chunks = [];
  var recording = false;
  var vadTimer = null;
  var hasSpeech = false;
  var lastSpeechAt = 0;
  var utteranceStartedAt = 0;
  var bargeStreakMs = 0;
  var echoGuardUntil = 0;
  var currentAudio = null;
  var abortCtrl = null;
  var typing = null;

  function setStatus(msg) {
    if (statusEl) statusEl.textContent = msg || "";
  }

  function setVoiceState(s, hint) {
    if (stateEl) stateEl.textContent = s;
    if (hintEl && hint !== undefined) hintEl.textContent = hint;
    if (orb) {
      orb.classList.remove("listening", "thinking", "speaking");
      if (s.indexOf("Listen") === 0) orb.classList.add("listening");
      else if (s.indexOf("Think") === 0 || s.indexOf("Transcrib") === 0) orb.classList.add("thinking");
      else if (s.indexOf("Speak") === 0) orb.classList.add("speaking");
    }
    if (stopReplyBtn) {
      stopReplyBtn.classList.toggle("d-none", s.indexOf("Speak") !== 0);
    }
  }

  function setTranscript(msg) {
    if (transcriptEl) transcriptEl.textContent = msg || "";
  }

  function isSecureContextOk() {
    return (
      window.isSecureContext ||
      location.hostname === "localhost" ||
      location.hostname === "127.0.0.1"
    );
  }

  function pickMime() {
    var candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
    for (var i = 0; i < candidates.length; i++) {
      try {
        if (window.MediaRecorder && MediaRecorder.isTypeSupported(candidates[i])) {
          return candidates[i];
        }
      } catch (e) {
        /* ignore */
      }
    }
    return "";
  }

  function micLevel() {
    if (!analyser || !timeDomain) return 0;
    analyser.getByteTimeDomainData(timeDomain);
    var sum = 0;
    for (var i = 0; i < timeDomain.length; i++) {
      var v = (timeDomain[i] - 128) / 128;
      sum += v * v;
    }
    return Math.sqrt(sum / timeDomain.length);
  }

  // --- Mode toggle ----------------------------------------------------------

  function toggleVoiceMode() {
    if (voiceMode) exitVoiceMode();
    else enterVoiceMode();
  }

  function enterVoiceMode() {
    if (voiceMode) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("Microphone not supported in this browser.");
      return;
    }
    if (!window.MediaRecorder) {
      setStatus("MediaRecorder is not supported in this browser.");
      return;
    }
    if (!isSecureContextOk()) {
      setStatus("Microphone requires HTTPS or localhost.");
      return;
    }
    voiceMode = true;
    generation += 1;
    setStatus("");
    if (chatInputRow) chatInputRow.classList.add("d-none");
    panel.classList.remove("d-none");
    micBtn.classList.add("active", "btn-danger");
    micBtn.classList.remove("btn-outline-info");
    micBtn.setAttribute("aria-pressed", "true");
    setTranscript("");

    ensureStream()
      .then(function () {
        if (!voiceMode) return;
        startVadLoop();
        startUtterance();
      })
      .catch(function (err) {
        setStatus("Microphone permission denied: " + (err && err.message ? err.message : err));
        exitVoiceMode();
      });
  }

  function exitVoiceMode() {
    if (!voiceMode && state === "idle") return;
    voiceMode = false;
    generation += 1;
    state = "idle";
    stopVadLoop();
    stopRecorderSilently();
    stopReplyAudio();
    if (abortCtrl) {
      try {
        abortCtrl.abort();
      } catch (e) {
        /* ignore */
      }
      abortCtrl = null;
    }
    if (typing && typing.div) {
      typing.div.remove();
      typing = null;
    }
    if (stream) {
      stream.getTracks().forEach(function (t) {
        try {
          t.stop();
        } catch (e) {
          /* ignore */
        }
      });
      stream = null;
    }
    if (audioCtx) {
      var ctx = audioCtx;
      audioCtx = null;
      analyser = null;
      timeDomain = null;
      if (ctx.close) {
        ctx.close().catch(function () {});
      }
    }
    recording = false;
    hasSpeech = false;
    bargeStreakMs = 0;
    if (chatInputRow) chatInputRow.classList.remove("d-none");
    panel.classList.add("d-none");
    micBtn.classList.remove("active", "btn-danger");
    micBtn.classList.add("btn-outline-info");
    micBtn.setAttribute("aria-pressed", "false");
    setStatus("");
  }

  function ensureStream() {
    if (stream) return Promise.resolve(stream);
    return navigator.mediaDevices
      .getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      })
      .then(function (s) {
        stream = s;
        try {
          var AC = window.AudioContext || window.webkitAudioContext;
          if (AC) {
            audioCtx = new AC();
            var src = audioCtx.createMediaStreamSource(stream);
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 2048;
            src.connect(analyser);
            timeDomain = new Uint8Array(analyser.fftSize);
          }
        } catch (e) {
          analyser = null; // VAD falls back to "record until max seconds"
        }
        return stream;
      });
  }

  // --- VAD loop (one interval for the whole voice session) -------------------

  function startVadLoop() {
    stopVadLoop();
    vadTimer = setInterval(vadTick, VAD_TICK_MS);
  }

  function stopVadLoop() {
    if (vadTimer) {
      clearInterval(vadTimer);
      vadTimer = null;
    }
  }

  function vadTick() {
    if (!voiceMode) return;
    var now = Date.now();
    var level = micLevel();
    var loud = level > BARGE_THRESHOLD;
    var voiced = level > SPEECH_THRESHOLD;

    // Orb reacts to live level (cheap premium feel).
    if (orb && (state === "listening" || state === "speaking")) {
      var scale = 1 + Math.min(level * 4, 0.35);
      orb.style.transform = "scale(" + scale.toFixed(2) + ")";
    }

    if (state === "listening" && recording) {
      if (voiced) {
        hasSpeech = true;
        lastSpeechAt = now;
      }
      var elapsed = (now - utteranceStartedAt) / 1000;
      if (orb) {
        setVoiceState(
          hasSpeech ? "Listening... (pause to send)" : "Listening... speak now (" + Math.floor(elapsed) + "s)"
        );
      }
      if (hasSpeech && now - lastSpeechAt >= SILENCE_MS) {
        stopAndSend();
      } else if (elapsed >= maxSeconds) {
        stopAndSend();
      }
    } else if (state === "speaking") {
      if (now < echoGuardUntil) {
        bargeStreakMs = 0;
        return;
      }
      if (loud) bargeStreakMs += VAD_TICK_MS;
      else bargeStreakMs = 0;
      if (bargeStreakMs >= BARGE_MS) {
        bargeStreakMs = 0;
        bargeIntoNewUtterance();
      }
    } else if (state === "thinking") {
      if (loud) bargeStreakMs += VAD_TICK_MS;
      else bargeStreakMs = 0;
      if (bargeStreakMs >= THINK_BARGE_MS) {
        bargeStreakMs = 0;
        // User started talking while we think: drop the request, listen fresh.
        if (abortCtrl) {
          try {
            abortCtrl.abort();
          } catch (e) {
            /* ignore */
          }
          abortCtrl = null;
        }
        if (typing && typing.div) {
          typing.div.remove();
          typing = null;
        }
        var g = generation;
        state = "listening";
        setVoiceState("Listening... speak now");
        setTranscript("Heard you — listening again.");
        startUtterance(g);
      }
    } else {
      bargeStreakMs = 0;
    }
  }

  // --- Utterance lifecycle ----------------------------------------------------

  function startUtterance(expectedGeneration) {
    if (!voiceMode) return;
    if (expectedGeneration !== undefined && expectedGeneration !== generation) return;
    if (recording) return;
    if (!stream) return;
    chunks = [];
    hasSpeech = false;
    bargeStreakMs = 0;
    var mime = pickMime();
    try {
      mediaRecorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    } catch (e) {
      setVoiceState("Listening... unavailable", String((e && e.message) || e));
      return;
    }
    var myGen = generation;
    mediaRecorder.ondataavailable = function (e) {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };
    mediaRecorder.onstop = function () {
      recording = false;
      if (!voiceMode || myGen !== generation) return; // exited / barged meanwhile
      var blob = new Blob(chunks, { type: mime || "audio/webm" });
      if (blob.size === 0) {
        // Empty capture: just listen again.
        startUtterance(myGen);
        return;
      }
      sendVoiceTurn(blob, myGen);
    };
    try {
      mediaRecorder.start();
    } catch (e) {
      setVoiceState("Listening... unavailable", String((e && e.message) || e));
      return;
    }
    recording = true;
    state = "listening";
    utteranceStartedAt = Date.now();
    lastSpeechAt = utteranceStartedAt;
    setVoiceState("Listening... speak now");
  }

  function stopRecorderSilently() {
    if (mediaRecorder && recording && mediaRecorder.state !== "inactive") {
      try {
        mediaRecorder.onstop = null;
        mediaRecorder.stop();
      } catch (e) {
        /* ignore */
      }
    }
    recording = false;
  }

  function stopAndSend() {
    if (!recording) return;
    if (state !== "listening") return;
    state = "thinking"; // block VAD from double-sending while recorder stops
    setVoiceState("Thinking...", "Transcribing + answering from your sources.");
    try {
      if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
      else {
        state = "listening";
        startUtterance();
      }
    } catch (e) {
      state = "listening";
      startUtterance();
    }
  }

  function sendVoiceTurn(blob, myGen) {
    if (!voiceMode || myGen !== generation) return;
    state = "thinking";
    setVoiceState("Thinking...", "Transcribing + answering from your sources.");
    var fd = new FormData();
    fd.append("audio", blob, "rec.webm");
    if (sendBtn) sendBtn.disabled = true;
    typing = window.ChatUI ? window.ChatUI.appendTypingIndicator() : null;
    abortCtrl = new AbortController();

    fetch("/notebooks/" + config.notebook_id + "/voice/turn", {
      method: "POST",
      body: fd,
      credentials: "same-origin",
      headers: { "X-CSRFToken": config.csrf_token || "" },
      signal: abortCtrl.signal,
    })
      .then(function (r) {
        return r.json().then(function (j) {
          return { status: r.status, json: j };
        });
      })
      .then(function (r) {
        if (!voiceMode || myGen !== generation) return;
        if (typing && typing.div) {
          typing.div.remove();
          typing = null;
        }
        if (r.status === 200) {
          var j = r.json || {};
          if (window.ChatUI) {
            window.ChatUI.appendMessage("user", j.transcript || "");
            var assistantDiv = window.ChatUI.appendMessage("assistant", j.answer || "");
            if (assistantDiv && j.sources) window.ChatUI.appendSources(assistantDiv, j.sources);
          }
          setTranscript("Heard: " + (r.json.transcript || ""));
          if (j.reply_audio_url) playReply(j.reply_audio_url, myGen);
          else {
            state = "listening";
            startUtterance(myGen);
          }
        } else if (r.status === 422 && r.json && r.json.error === "no_speech") {
          setVoiceState("Listening... speak now", "Didn't catch that — try again.");
          state = "listening";
          startUtterance(myGen);
        } else {
          var code = (r.json && r.json.error) || "error";
          setVoiceState("Listening... speak now", "Voice turn failed (" + code + ") — try again.");
          state = "listening";
          startUtterance(myGen);
        }
      })
      .catch(function (err) {
        if (typing && typing.div) {
          typing.div.remove();
          typing = null;
        }
        if (err && err.name === "AbortError") return; // exit/barge owns the next step
        if (!voiceMode || myGen !== generation) return;
        setVoiceState("Listening... speak now", "Network error — try again.");
        state = "listening";
        startUtterance(myGen);
      })
      .finally(function () {
        if (sendBtn && !voiceMode) sendBtn.disabled = false;
        if (!voiceMode) abortCtrl = null;
      });
  }

  // --- Reply playback + barge-in -----------------------------------------------

  function playReply(url, myGen) {
    stopReplyAudio();
    if (!voiceMode || myGen !== generation) return;
    state = "speaking";
    setVoiceState("Speaking... (talk to interrupt)");
    var audio = new Audio(url);
    currentAudio = audio;
    echoGuardUntil = Date.now() + ECHO_GUARD_MS;
    bargeStreakMs = 0;
    audio.onended = function () {
      if (!voiceMode || myGen !== generation) return;
      if (currentAudio !== audio) return; // barged already
      currentAudio = null;
      state = "listening";
      startUtterance(myGen);
    };
    audio.onerror = function () {
      if (!voiceMode || myGen !== generation) return;
      currentAudio = null;
      setVoiceState("Listening... speak now", "Could not play reply — showing text.");
      state = "listening";
      startUtterance(myGen);
    };
    audio.play().catch(function () {
      if (!voiceMode || myGen !== generation) return;
      currentAudio = null;
      // Autoplay blocked: keep the text answer, resume listening.
      setVoiceState("Listening... speak now", "Browser blocked autoplay — answer is in chat.");
      state = "listening";
      startUtterance(myGen);
    });
  }

  function stopReplyAudio() {
    if (currentAudio) {
      var a = currentAudio;
      currentAudio = null;
      try {
        a.pause();
      } catch (e) {
        /* ignore */
      }
      try {
        a.src = "";
      } catch (e) {
        /* ignore */
      }
    }
  }

  function bargeIntoNewUtterance() {
    if (!voiceMode || state !== "speaking") return;
    var myGen = generation;
    stopReplyAudio();
    if (typing && typing.div) {
      typing.div.remove();
      typing = null;
    }
    state = "listening";
    setVoiceState("Listening... speak now", "Interrupted — I'm listening.");
    setTranscript("");
    startUtterance(myGen);
  }

  // --- Wiring -------------------------------------------------------------------

  micBtn.addEventListener("click", toggleVoiceMode);
  if (endBtn) endBtn.addEventListener("click", exitVoiceMode);
  if (stopReplyBtn) {
    stopReplyBtn.addEventListener("click", function () {
      if (state !== "speaking") return;
      var myGen = generation;
      stopReplyAudio();
      state = "listening";
      setVoiceState("Listening... speak now");
      startUtterance(myGen);
    });
  }
  document.addEventListener("keydown", function (e) {
    if (e.code === "Escape" && voiceMode) exitVoiceMode();
  });
  window.addEventListener("pagehide", exitVoiceMode);
})();
