# PROMPT — Mahek (Module A: the conversation)

**Paste this entire file as your first message to Claude / ChatGPT / Cursor when you start
a session.** It gives your assistant everything it needs to write code that fits the rest
of the project instead of code you have to rewrite.

---

You are helping me build **Module A — the conversational multimodal history engine** of
Sanjeevani, our Smart India Hackathon 2026 project (Problem Statement 26047, *Patient
Case-Taking Software*, Ministry of Ayush / All India Institute of Ayurveda). Read this
whole brief before writing any code, and ask me before inventing a design decision it does
not cover.

## The problem

An Indian government hospital OPD registers **4,000–10,000 patients a day**. The
consultation lasts **2 to 5 minutes** — India's average primary-care consultation is just
over two minutes, among the shortest in the world. In that window a physician must elicit
the history, examine the patient, read whatever paper they brought, diagnose, counsel and
prescribe.

A good history gives the right diagnosis **70–80% of the time**, on its own. It is the
first thing sacrificed to the clock.

Sanjeevani is a **kiosk** the patient uses *before* the consultation. They give their
history by speaking in their own language or tapping pictures, their old paper records get
digitised, and the physician opens a complete structured summary the moment the patient
walks in.

Five people build it. Kashvi does documents and the summary (Modules B and C), Ronit does
consent/ABHA/ABDM (Module D), Meet builds the Flutter kiosk, Samridhi builds the clinician
console. **I build Module A — the interview itself.**

## What my module does

Conduct a structured clinical history interview through **voice and touch**, adaptively.

```
patient speaks (Hindi, English, …)  ──► ASR ──┐
                                              ├──► dialogue engine ──► next question
patient taps an icon ─────────────────────────┘         │              (prompt, input_type,
                                                        │               options, audio, progress)
                                          walks the clinical
                                          history ontology
                                                        │
                                                        ├──► red-flag detection ──► triage alert
                                                        └──► TTS ──► the question, spoken aloud
```

My folders: `ml/speech/` (ASR, TTS), `ml/dialogue/` (the engine), `ml/ayush/` (Dashavidha
Pariksha), `ml/redflags/`.

**Kashvi owns** `ocr/`, `extraction/`, `terminology/`, `timeline/`, `summary/`, `fhir/`.
Do not write code in her folders. My output — an `InterviewTranscript` of structured
answers plus the raw narration — is what her Module C summary generator consumes.

## The architectural rule everything hangs off

**The interview is server-driven. The clients render; they do not decide.**

We ship two kiosk clients — Meet's Flutter tablet app and Samridhi's web version. Neither
contains a list of medical questions. Neither decides what to ask next. They send an
answer, my engine decides, and they render whatever comes back.

The question set lives in **one file: `shared/clinical/history-ontology.json`**. My engine
walks it. Adding a question is a data change there, not code in three places.

Consequences for my code:

- The dialogue engine is a **traversal over that ontology**, not a pile of `if` statements
  about chest pain. If a branching rule cannot be expressed in the ontology, extend the
  ontology's schema and tell the team — do not hardcode it in Python.
- Every response carries `input_type`, `options`, an `audio_url` and `progress`. The client
  needs all of it to render a screen.
- There is one round trip per answer. Keep responses small, make the next question
  prefetchable, and never let the round trip block the microphone.

## Interaction mode: speak-only vs tap-first — patient chooses, switchable anytime

New requirement: the patient picks how they answer — **speak-only** (mic opens
automatically for every question, no touch needed) or **tap-first** (normal touch UI,
but if there's no answer within a timeout, the mic auto-opens and invites them to speak
instead). **The patient can switch between the two at any question, any time** — not a
one-time choice locked in at the start.

Design decision, so I don't duplicate work across three codebases: this is **entirely a
client-side concern**, not something my engine tracks.

- No new field on `Question`, `Answer`, or `InterviewTranscript` — `Answer.via_touch`
  already records, per answer, whether it came from a tap or from speech. That's the
  complete audit trail this feature needs.
- The **one** shared value both kiosk clients need is the tap-first timeout duration —
  how long to wait before auto-opening the mic. That goes into the ontology itself, as a
  small top-level default:
  ```json
  "interaction_defaults": { "tap_timeout_seconds": 10 }
  ```
  Reason: exactly the same "one file, not three places" principle that governs questions
  themselves — Meet's Flutter app and Samridhi's web kiosk both read this number instead
  of each hardcoding `10` separately and drifting apart later.
- My `ontology_loader.py` needs to expose this default alongside the sections/questions it
  already loads, so both clients can fetch it without guessing.
- Nothing about **what** gets asked changes based on mode — SOCRATES branching, red flags,
  AYUSH sections all behave identically whether the patient is speaking or tapping. Mode
  only changes how ONE answer gets captured, never the interview's logic.

## The four things that make this hard

1. **Noisy hospital, real accents, many languages.** Hindi and English first, then Marathi,
   Bengali, Tamil, Telugu. A corridor with 200 people in it. Use **Bhashini / AI4Bharat
   IndicWhisper** — free, Indian, trained on Indian accents. Not Western ASR.
2. **Zero training required.** A first-time, non-tech-savvy, low-literacy, elderly patient
   must finish alone. Every prompt is spoken aloud. Every question is answerable by voice
   **or** by tapping an icon — every one, not most. Someone who cannot read must be able to
   complete the entire interview by ear.
3. **Adaptive questioning.** On "chest pain" the engine probes site, onset, character,
   radiation, associations, time course, modifiers and severity — the **SOCRATES**
   framework, named explicitly in the problem statement. Ask what matters, skip what does
   not, finish before the patient gives up.
4. **Dashavidha Pariksha.** See below. This is what wins.

## Correlated questioning — how the engine actually reasons

The interview is not a flat questionnaire. Questions and answers are **correlated through
conditions**, via `shared/clinical/condition-map.json`:

- Every answer raises or lowers the **plausibility** of a set of conditions
  (`ConditionHypothesis` in `schemas.py`). "Chest" on the body map puts acute coronary
  syndrome AND acid reflux on the table; "burning" then lowers ACS and raises reflux;
  "cold sweating" does the reverse.
- **The next question is chosen to best separate whatever is still live** — highest
  expected information gain, ties broken toward the more urgent condition. That is what
  "mirroring a physician's clinical reasoning" (the PS's phrase) actually means in code.
- Kashvi's **document extraction feeds back into this**: a scanned prescription for an
  NSAID raises peptic ulcer; an old HbA1c of 8.1 raises diabetes. The interview gets
  smarter because the patient handed over their paper. That loop is unique to us.
- Scoring is deliberately **additive weights, clamped 0-1, fully inspectable** — a judge
  must be able to ask "why did it ask that?" and get a one-sentence answer. Do not replace
  it with an opaque model.

**The hard ethical line:** plausibility scores exist ONLY to order questions.

- Never shown to the patient, in any form.
- Never written into the record as a finding.
- The physician sees an `ExplorationReport` — what was explored, what was ruled down by
  explicit answers, and **what was not explored** — never "the patient probably has X".
- The field is called `plausibility`, not probability, on purpose: it is an unvalidated
  heuristic for question ordering, and calling it a probability invites someone to
  display it as one.

If you ever find yourself sorting the hypothesis list to present the top entry as an
answer, stop — that is the autonomous diagnosis the PS forbids.

The condition map is a SEED (10 conditions covering the commonest and most dangerous
Indian OPD presentations). Growing it is clinical work with a physician, and the
Ayurvedic correlation block is a placeholder that **must not** be populated by an LLM.

## AYUSH mode is the differentiator — treat it that way

The Ministry of Ayush and the All India Institute of Ayurveda are judging this. Every other
team will build allopathic intake and bolt AYUSH on at the end. **For us it is a
first-class mode.**

Ayurvedic history taking requires **Dashavidha Pariksha** — the ten-fold examination:
Prakriti, Vikriti, Sara, Samhanana, Pramana, Satmya, Sattva, Ahara Shakti, Vyayama Shakti,
Vaya — plus Agni, Koshtha, Ahara-Vihara, Nidana and Samprapti. It is far more extensive
than allopathic intake, which is exactly why an overloaded AYUSH OPD abbreviates it away.

**We are the only intake system that can capture it inside a queue.** That sentence is our
pitch.

**Rule: never let an LLM invent Ayurvedic content.** Prakriti scoring, Sara grading and
Koshtha classification are real clinical instruments with real textual basis (Charaka
Samhita, Vimanasthana 8/94). The seed content in the ontology is a starting point that
**must be reviewed by an AIIA practitioner** before submission. If you are unsure whether a
parameter or its options are correct, say so and stop — do not generate plausible-sounding
Sanskrit.

## Red flags — get the ethics right

The engine flags emergency symptoms (acute chest pain with breathlessness, stroke signs,
uncontrolled bleeding) and **alerts triage staff immediately** instead of leaving the
patient in a routine queue.

Three rules, all non-negotiable:

1. **A red flag never stops the interview.** It fires an alert and carries on.
2. **We never tell the patient they may be having an emergency.** That is a diagnosis, and
   we do not diagnose. The kiosk says a staff member will come to them shortly, calmly.
3. **False negatives are far worse than false positives here.** A patient sent to routine
   queue with an evolving MI is the failure mode that matters. When the rule is borderline,
   flag it.

## Rules that are not up for debate

1. **No cheating.** No hardcoded demo transcript, no "if the demo user says X then Y", no
   pretending the ASR worked. If a model is not wired up yet, the endpoint says so
   honestly.
2. **Everything works offline with no paid API.** Bhashini/AI4Bharat models run locally or
   against a free Indian government endpoint. Anything behind a paid key is an optional
   accuracy boost that must degrade cleanly when the key is absent — never on the happy
   path.
3. **Degrade, never fail.** No mic, no ASR, network dropped, unintelligible audio — every
   path falls back to touch input and the interview continues. A kiosk that dead-ends in a
   5,000-patient queue is worse than no kiosk.
4. **Honest confidence.** If ASR confidence is low, the engine asks for confirmation rather
   than silently recording a guess. A misheard drug allergy is a patient-safety event.
5. **No PHI in logs, ever.** Log question ids, timings, confidence scores, language codes.
   **Never** the transcript, the audio, or anything the patient said. Graded, and just
   correct.
6. **We elicit; the physician diagnoses.** The engine never offers an opinion on what is
   wrong, to the patient or in the data.
7. **Stay in my four folders.** Do not edit `ocr/`, `extraction/`, `terminology/`,
   `summary/`, `fhir/`, the backend, or either client. Changes to `shared/` need a heads-up
   to the whole team first.
8. **No client ever calls this service.** The backend calls me. It owns auth, consent and
   audit, so every AI output reaches a patient through it.

## How I want you to work

- Real, runnable Python with type hints and docstrings — not pseudocode, not stubs unless I
  ask for a skeleton.
- Every capability gets a `pytest` test. For ASR that means **real audio fixtures with
  Indian accents and background noise**, not a clean studio clip that proves nothing.
- Prefer a boring, debuggable, ontology-driven traversal over a clever LLM prompt I cannot
  explain to a judge in one sentence.
- Comment the *why*, especially for clinical or accessibility decisions.
- **When a clinical or Ayurvedic judgement comes up, stop and ask me.** Do not guess about
  patient safety or about Ayurveda.
- Tell me when a design of mine would strand a low-literacy patient or leak PHI. I would
  rather hear it now than from a judge.

## Where to look

- `docs/PS-26047.md` — the problem statement; Module A is section 4
- `docs/ARCHITECTURE.md` §3 — why the interview is server-driven
- `shared/clinical/history-ontology.json` — **the question set. Start here.**
- `shared/api/CONTRACT.md` — how the backend calls me
- `ml/README.md` — layout, run instructions, my task checklist

Run the service: `uvicorn sanjeevani_ml.service.main:app --reload --port 8100`
My endpoints: `POST /interview/next`, `POST /speech/transcribe`, `POST /speech/say`.

Now ask me what I want to work on today.
