import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Web Speech API wrapper for the search box.
 *
 * Support is feature-detected rather than inferred from the user agent, because
 * the support matrix is genuinely unstable right now: Chromium ships cloud
 * recognition, while Firefox exposes an on-device implementation that is
 * preference-gated and requires a language model to be downloaded first via
 * available()/install(). MDN still marks the interface "limited availability",
 * so a hardcoded browser list would be wrong within weeks.
 *
 * Everything here is strictly additive: when the API is absent the hook reports
 * `supported: false` and the caller renders no microphone affordance at all,
 * leaving the search path untouched.
 */

// Resolve the constructor once, tolerating the legacy webkit- prefix. Only a
// real function counts: a truthy non-callable value would make `supported` true
// and then throw on construction, which is exactly the degradation this feature
// must never cause.
export function getConstructor(scope) {
  const w = scope || (typeof window === "undefined" ? undefined : window);
  if (!w) return null;
  // Check each candidate independently so an invalid unprefixed value cannot
  // mask a usable prefixed one.
  for (const ctor of [w.SpeechRecognition, w.webkitSpeechRecognition]) {
    if (typeof ctor === "function") return ctor;
  }
  return null;
}

const ERROR_MESSAGES = {
  "no-speech": "No speech detected — try again.",
  "audio-capture": "No microphone found.",
  "not-allowed": "Microphone access was denied.",
  "service-not-allowed": "Speech service unavailable in this browser.",
  network: "Speech recognition needs a network connection.",
  aborted: "Speech recognition stopped.",
  "language-not-supported": "Speech recognition is not available for this language.",
};

export function describeError(code) {
  if (!code) return "Speech recognition failed.";
  return ERROR_MESSAGES[code] || `Speech recognition error: ${code}`;
}

export function useSpeechRecognition({ lang = "en-US", onFinal } = {}) {
  const Ctor = getConstructor();
  const supported = Boolean(Ctor);

  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const [status, setStatus] = useState(null); // { kind, text }
  const [needsModel, setNeedsModel] = useState(false);
  const [modelBusy, setModelBusy] = useState(false);

  const recognitionRef = useRef(null);
  // Guards against results arriving after stop()/unmount.
  const activeRef = useRef(false);
  const finalRef = useRef("");
  const onFinalRef = useRef(onFinal);
  onFinalRef.current = onFinal;

  // Build a fresh recognizer per session; reusing one after end() throws.
  const create = useCallback(() => {
    const recognition = new Ctor();
    recognition.lang = lang;
    // A single utterance is what a search box wants; continuous listening would
    // keep the microphone open after the user has finished speaking.
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      activeRef.current = true;
      finalRef.current = "";
      setInterim("");
      setStatus({ kind: "listening", text: "Listening…" });
    };

    recognition.onresult = (event) => {
      if (!activeRef.current) return;
      let live = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const text = result[0]?.transcript || "";
        if (result.isFinal) {
          finalRef.current += text;
        } else {
          live += text;
        }
      }
      setInterim(live);
      // Commit as soon as a final segment lands so the box fills progressively,
      // exactly as if the words had been typed.
      if (finalRef.current) {
        onFinalRef.current?.(finalRef.current.trim());
      }
    };

    recognition.onerror = (event) => {
      if (!activeRef.current) return;
      const code = event?.error;
      setStatus({ kind: "error", text: describeError(code) });
      if (code === "not-allowed" || code === "service-not-allowed") {
        setNeedsModel(false);
      }
    };

    recognition.onend = () => {
      activeRef.current = false;
      setListening(false);
      setInterim("");
      // Only clear the "listening" status if no error replaced it.
      setStatus((prev) => (prev && prev.kind === "listening" ? null : prev));
    };

    return recognition;
  }, [Ctor, lang]);

  // Abort any in-flight session if the component unmounts.
  useEffect(() => {
    return () => {
      activeRef.current = false;
      try {
        recognitionRef.current?.abort();
      } catch {
        /* already stopped */
      }
      recognitionRef.current = null;
    };
  }, []);

  const start = useCallback(async () => {
    if (!supported || listening) return;
    setStatus(null);

    // On implementations that only do on-device recognition, the model may not
    // be present yet. Ask before starting so the user gets a real message
    // rather than an opaque rejection from start().
    if (typeof Ctor?.available === "function") {
      try {
        const state = await Ctor.available({ langs: [lang], processLocally: true });
        if (state === "unavailable") {
          setNeedsModel(true);
          setStatus({
            kind: "error",
            text: "Speech recognition needs a one-time language download in this browser.",
          });
          return;
        }
        if (state === "downloading") {
          setStatus({ kind: "info", text: "Downloading speech model…" });
          return;
        }
      } catch {
        // available() is gated by Permissions-Policy on some builds; fall
        // through and let start() report the real problem.
      }
    }

    let recognition;
    try {
      // The constructor itself can throw (for example where the interface is
      // exposed but gated behind a pref or secure-context requirement), so it
      // must be inside the guarded region rather than producing an unhandled
      // rejection from the click handler.
      recognition = create();
      recognitionRef.current = recognition;
      recognition.start();
      setListening(true);
    } catch (err) {
      // start() throws synchronously for InvalidStateError (already running) and
      // NotAllowedError (blocked by browser AI controls or a denied permission).
      const name = err?.name || "";
      setListening(false);
      if (name === "NotAllowedError") {
        setStatus({ kind: "error", text: "Speech recognition is blocked in this browser." });
      } else if (name === "InvalidStateError") {
        setStatus({ kind: "error", text: "Already listening — stop first." });
      } else {
        setStatus({ kind: "error", text: describeError(null) });
      }
    }
  }, [Ctor, supported, listening, create, lang]);

  const stop = useCallback(() => {
    try {
      recognitionRef.current?.stop();
    } catch {
      /* nothing running */
    }
  }, []);

  const installModel = useCallback(async () => {
    if (typeof Ctor?.install !== "function") return;
    setModelBusy(true);
    setStatus({ kind: "info", text: "Downloading speech model…" });
    try {
      const ok = await Ctor.install({ langs: [lang], processLocally: true });
      if (ok) {
        setNeedsModel(false);
        setStatus({ kind: "info", text: "Speech model ready." });
      } else {
        setStatus({ kind: "error", text: "Speech model download failed." });
      }
    } catch {
      setStatus({ kind: "error", text: "Speech model download was blocked." });
    } finally {
      setModelBusy(false);
    }
  }, [Ctor, lang]);

  return {
    supported,
    listening,
    interim,
    status,
    needsModel,
    modelBusy,
    start,
    stop,
    installModel,
  };
}
