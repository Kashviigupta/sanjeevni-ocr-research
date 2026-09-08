# ml/ — Modules A, B and C

**Owners: Mahek (Module A) & Kashvi (Modules B + C)**

The AI core of Sanjeevani (SIH PS 26047): the conversation that elicits the history, the
pipeline that digitises the patient's paper, and the generator that fuses both into one
physician-ready summary.

> New here? You each have your own LLM brief — paste it into your AI assistant at the
> start of every session:
> **Mahek → [`PROMPT-MAHEK.md`](PROMPT-MAHEK.md)** · **Kashvi → [`PROMPT-KASHVI.md`](PROMPT-KASHVI.md)**
>
> Then read [`../docs/PS-26047.md`](../docs/PS-26047.md) and
> [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).

---

## Split of work

| | **Mahek — Module A** | **Kashvi — Modules B + C** |
|---|---|---|
| One sentence | The conversation | The paperwork, and the summary |
| Folders | `speech/`, `dialogue/`, `ayush/`, `redflags/` | `ocr/`, `extraction/`, `terminology/`, `timeline/`, `summary/`, `fhir/`, `interactions/` |
| Owns | ASR (Bhashini/AI4Bharat), TTS, the dialogue engine over the ontology, correlated questioning via the condition map, Dashavidha Pariksha mode, red-flag detection | OCR incl. handwriting, entity extraction, LOINC/SNOMED/RxNorm coding, chronological timeline, abnormal values, drug interactions, the Module C summary, FHIR |
| Produces | `InterviewTranscript` | `DocumentBundle`, then `HistorySummary` from both |
| Done when | a first-time patient finishes the interview by voice, in Hindi, in noise; chest pain + breathlessness raises a triage alert in seconds | a physician reads the summary and says they would not have elicited more in five minutes |

The handoff shapes live in `src/sanjeevani_ml/schemas.py` —
`InterviewTranscript` is the seam. Neither of you changes it without telling the other.

`service/main.py` (shared) is the HTTP surface. **Only the backend calls it** — no kiosk
or console ever talks to this service directly.

---

## The two files that drive the interview

Both in [`../shared/clinical/`](../shared/clinical/), both owned by Mahek, both reviewed
by everyone:

- **`history-ontology.json`** — every question, its input type, its Hindi/English text,
  its icon, its red-flag rules. Clients render these; they never hardcode one.
- **`condition-map.json`** — what makes the interview *correlated*: answers raise and
  lower condition hypotheses, and the engine asks whatever best separates what is still
  live. Plausibility exists **only to order questions** — never shown to the patient,
  never recorded as a finding. That is the line between adaptive history-taking (required)
  and autonomous diagnosis (forbidden).

Clinical content in both files is a SEED. Growing it is clinical work with a physician —
and the Ayurvedic content **must** be reviewed by an AIIA practitioner, never generated
by an LLM.

---

## Run it

```bash
cd ml
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
pip install -e .
uvicorn sanjeevani_ml.service.main:app --reload --port 8100
```

Open http://127.0.0.1:8100/docs.

**Tesseract is a system binary, not a pip package** (Kashvi's half needs it):

- Windows: https://github.com/UB-Mannheim/tesseract/wiki → set `TESSERACT_CMD` in `.env`
- Linux: `sudo apt install tesseract-ocr tesseract-ocr-hin`

The service starts and reports `degraded` without it, so nobody is blocked on anyone's
local setup.

```bash
pytest        # 23 green — the regression gate. Keep it green.
```

---

## Layout

```
ml/
├── PROMPT-MAHEK.md               ← Mahek: paste into your LLM first
├── PROMPT-KASHVI.md              ← Kashvi: paste into your LLM first
├── data/                         curated terminology tables (CSV, hand-checked)
├── tests/                        test_pipeline.py — the regression gate
└── src/sanjeevani_ml/
    ├── schemas.py                every shape, including the A→C handoff
    ├── config.py
    ├── speech/                   MAHEK — ASR (Bhashini/AI4Bharat), TTS
    ├── dialogue/                 MAHEK — walks the ontology + condition map
    ├── ayush/                    MAHEK — Dashavidha Pariksha
    ├── redflags/                 MAHEK — triage alerts
    ├── ocr/                      KASHVI — Tesseract, preprocessing, handwriting
    ├── extraction/               KASHVI — entities out of OCR text
    ├── terminology/              KASHVI — LOINC/SNOMED/RxNorm coding
    ├── timeline/                 KASHVI — chronological document ordering
    ├── interactions/             KASHVI — drug-drug checks
    ├── summary/                  KASHVI — Module C: the physician-ready summary
    ├── fhir/                     KASHVI — FHIR R4 builders
    └── service/main.py           the HTTP surface — backend-only
```

---

## Build order

**Week 1 — one thin slice, end to end.**

- [ ] Mahek: dialogue engine walks the ontology; `POST /interview/next` returns real
      questions and branches on answers
- [ ] Mahek: ASR on one Hindi audio fixture; TTS for one prompt
- [ ] Kashvi: timeline builder — 3 dated fixture documents in, ordered `DocumentBundle` out
- [ ] Kashvi: first `HistorySummary` from a hand-written `InterviewTranscript` fixture +
      that bundle
- [ ] Both: agree `InterviewTranscript` is final enough to build against

**Week 2 — the full interview, the full pipeline.**

- [ ] Mahek: condition-map correlation live — chest-pain demo separates ACS from reflux
      question-by-question; red flags fire
- [ ] Mahek: Dashavidha mode complete; **AIIA practitioner review booked**
- [ ] Kashvi: handwriting path; abnormal-value flagging; document signals feeding the
      condition map
- [ ] Kashvi: summary provenance — every line traceable; contradictions surfaced

**Week 3 — make it survive reality.**

- [ ] Mahek: noisy-audio fixtures, barge-in, low-confidence → confirm loop
- [ ] Kashvi: bilingual summary output + spoken patient confirmation text
- [ ] Both: latency budget — next-question under 500 ms, summary under 10 s

---

## Hard rules (both of you)

1. **No cheating** — no hardcoded demo transcripts, no pre-baked summaries, no faked
   confidence. Endpoints that are not implemented say so.
2. **Offline-first, nothing paid.** Bhashini/AI4Bharat are free and Indian. Anything
   behind a key is an optional boost that degrades cleanly.
3. **We elicit; the physician diagnoses.** Nothing this service emits is ever phrased as
   a diagnosis. `HistorySummary.status` is `"draft"` by type, on purpose.
4. **Honest confidence.** Under-claim, never inflate, and ask for confirmation when ASR
   or OCR is unsure.
5. **Never invent a code, never invent Ayurveda.** No table match → `code: None`. No
   practitioner review → the AYUSH block stays a placeholder.
6. **No PHI in logs.** Ids, counts, scores, timings. Never transcripts, audio, text or
   names.
7. **This service is called; it does not call.** No requests to the backend or a client.
