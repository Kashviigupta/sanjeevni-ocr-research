# PROMPT — Kashvi (Modules B + C: documents and the summary)

**Paste this entire file as your first message to Claude / ChatGPT / Cursor when you start
a session.** It gives your assistant everything it needs to write code that fits the rest
of the project instead of code you have to rewrite.

---

You are helping me build **Module B (medical document digitisation) and Module C (the
structured history summary generator)** of Sanjeevani, our Smart India Hackathon 2026
project (Problem Statement 26047, *Patient Case-Taking Software*, Ministry of Ayush / All
India Institute of Ayurveda). Read this whole brief before writing any code, and ask me
before inventing a design decision it does not cover.

## The problem

An Indian government hospital OPD registers **4,000–10,000 patients a day**. The
consultation lasts **2 to 5 minutes**. In that window a physician must elicit the history,
examine the patient, **read whatever paper the patient brought**, diagnose, counsel and
prescribe.

That paper is the second half of the problem. Patients arrive carrying prescriptions, lab
reports, discharge summaries and imaging films from multiple prior providers — handwritten,
in several languages, in no particular order. The physician manually leafs through them
during the consultation, burning a large share of the two minutes.

Sanjeevani is a **kiosk** the patient uses *before* the consultation. They give their
history by voice or touch, they photograph their old records, and the physician opens **one
complete structured summary** the moment the patient walks in.

Five people build it. Mahek does the conversational interview (Module A), Ronit does
consent/ABHA/ABDM (Module D), Meet builds the Flutter kiosk, Samridhi builds the clinician
console. **I build Modules B and C.**

## What my modules do

**Module B — paper to structured clinical data.**

```
photo / PDF of a prescription, lab report or discharge summary
   → preprocess (deskew, denoise, contrast)
   → OCR (Tesseract, per-word confidence, multilingual)
   → extract entities (diagnoses, drugs with dosages, lab values with reference ranges,
                       procedures, dates, facility)
   → map to terminology (LOINC / SNOMED CT / RxNorm / ICD-11)
   → build FHIR R4
   → CHRONOLOGICAL TIMELINE + ABNORMAL-VALUE HIGHLIGHTING
```

**Module C — one physician-ready summary.**

Fuse Mahek's `InterviewTranscript` with my `DocumentBundle` into a single structured
summary in the standard clinical format:

> Chief complaint → History of present illness → Past medical & surgical → Drug & allergy →
> Family → Personal → Review of systems → Prior investigations summary

My folders: `ml/ocr/`, `ml/extraction/`, `ml/terminology/`, `ml/timeline/`, `ml/summary/`,
`ml/fhir/`.

**Mahek owns** `speech/`, `dialogue/`, `ayush/`, `redflags/`. Do not write code in her
folders. Her `InterviewTranscript` is an input to my summary generator — that shape is the
contract between us; do not change it without telling her.

## What already works — start from it, do not rewrite it

The OCR and extraction pipeline is **already built and green (23 tests)**:

- `ocr/` — Tesseract with per-word confidence, PDF rasterisation via PyMuPDF
- `extraction/` — lab rows, prescription lines, Indian `1-0-1` dosing, DD/MM/YYYY dates,
  facility names, vitals
- `terminology/` — curated LOINC / RxNorm / SNOMED tables in `ml/data/*.csv`
- `fhir/` — Observation, MedicationRequest, Condition, DiagnosticReport builders with
  provenance extensions
- `interactions/` — drug–drug checking across the full medication list

Read `ml/tests/test_pipeline.py` before you touch anything. **Keep it green — it is the
regression gate.**

## What is new, and where the marks are

1. **Handwriting.** The prescriptions patients actually carry are handwritten by doctors.
   Tesseract falls apart on them. Preprocessing first, a vision model only as an optional
   boost that degrades to the offline path.
2. **Chronological timeline.** A shoebox of undated, out-of-order documents becomes one
   coherent medical history in date order, de-duplicated. The PS calls this out explicitly,
   and it is what separates us from "a scanner".
3. **Abnormal-value highlighting.** Out-of-range labs flagged against their reference
   ranges, plus drug-interaction warnings, surfaced where a rushed physician will actually
   see them.
4. **Module C — the summary.** The single artefact the whole project is judged on. A
   physician should read it and conclude they would not have elicited more in five minutes.

## Module C — the rules that matter

1. **It is a draft, always.** The physician accepts, amends or rejects. The PS is explicit:
   *"never an autonomous diagnosis."* Never phrase output as a diagnosis, never rank
   differentials, never imply certainty we do not have.
2. **Every claim is traceable.** Each line says where it came from — patient's own words,
   or extracted from a named document with a confidence. A physician must always be able to
   ask "how do you know that?"
3. **Contradictions get surfaced, not resolved.** If the patient says they take no
   medicines and a scanned prescription says otherwise, show both. Silently picking one is
   a patient-safety bug.
4. **Concise or it does not get read.** A physician with two minutes will not read a page.
   Lead with chief complaint and red flags; detail below.
5. **Bilingual** — English/Hindi for the physician, spoken confirmation in the patient's
   language so they can correct it before it is submitted.

## Real-world constraints — our user is not the demo user

- **Brand names, not generics.** Indian prescriptions say "Glycomet", not "metformin".
  Brand→generic resolution is not polish; the interaction table is keyed on generics.
- **`1-0-1` dosing** — one tablet morning, none noon, one night. Dominant in India, almost
  absent from Western datasets.
- **DD/MM/YYYY dates.** Reading `03/04/2026` as March 4th is a clinical error.
- **Column-aligned lab printouts**, local abbreviations, missing units, unit chaos.
- **Hindi and mixed-script documents**, especially from government facilities.
- **A low-light phone photo of a creased page**, not a flatbed scan.

If a solution only works on a clean 300-DPI English PDF, it does not work.

## Rules that are not up for debate

1. **No cheating.** No hardcoded expected output for a demo document, no faked confidence,
   no pre-baked summary. If something is not implemented, the endpoint says so.
2. **Never invent a terminology code.** No table match → `code: None` plus the raw text. A
   plausible-but-wrong LOINC code propagates into a permanent record and into every system
   we exchange with. No code is a correct answer; a guessed code is a safety bug. "HDL" and
   "LDL" are one character apart and clinically opposite — that is why the fuzzy threshold
   is strict.
3. **Honest confidence, never inflated.** A wrong dose at 0.98 is dangerous; the same dose
   at 0.55 gets checked by a human.
4. **Never claim safety.** An empty interaction result means "nothing in our table", never
   "this combination is safe".
5. **Everything works offline with no paid API.** Rules plus curated tables are the
   baseline. A vision LLM is optional and must never be on the happy path.
6. **Provenance on everything.** Every resource names the engine that produced it and its
   confidence, as FHIR extensions.
7. **No PHI in logs, ever.** Counts, scores, engine names, timings. **Never** extracted
   text or a patient's name.
8. **Stay in my six folders.** Do not edit `speech/`, `dialogue/`, `ayush/`, `redflags/`,
   the backend, or either client. `shared/` changes need a heads-up first.
9. **No client ever calls this service.** The backend calls me.

## Curated tables are clinical work

`ml/data/*.csv` — every row is hand-checked. **A wrong code here becomes a wrong code in a
patient's permanent record**, so treat adding rows as clinical work, not data entry. Target
is ~200 SNOMED terms and ~300 drugs.

Why CSVs and not a terminology server: SNOMED CT needs a licence, UMLS needs registration,
and a kiosk in a district hospital must work with no network.

## How I want you to work

- Real, runnable Python with type hints and docstrings — not pseudocode, not stubs unless I
  ask for a skeleton.
- Every capability gets a `pytest` test with a **realistic document fixture** — an actual
  photo or a faithful transcription, not a synthetic string that happens to match the regex
  you just wrote.
- **Prefer boring, debuggable rules over a model I cannot explain to a judge in one
  sentence.** Every regex should be justifiable out loud.
- Comment the *why*, especially for clinical decisions.
- **When a medical judgement comes up — is this dose plausible, is this interaction major,
  is this value out of range — stop and ask me.** Do not guess about clinical safety.
- Tell me when something would leak PHI or overstate confidence.

## Where to look

- `docs/PS-26047.md` — the problem statement; Modules B and C are section 4
- `docs/ARCHITECTURE.md` — how my modules fit the whole
- `ml/tests/test_pipeline.py` — 23 passing tests; the regression gate
- `ml/src/sanjeevani_ml/schemas.py` — the shapes, including the handoff from Mahek
- `shared/api/CONTRACT.md` — how the backend calls me

Run the service: `uvicorn sanjeevani_ml.service.main:app --reload --port 8100`
My endpoints: `POST /documents/ingest`, `POST /documents/timeline`, `POST /summary/generate`,
`POST /interactions`.

Now ask me what I want to work on today.
