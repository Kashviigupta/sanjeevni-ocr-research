"""Server-driven dialogue engine for Sanjeevani Module A."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from sanjeevani_ml.dialogue.condition_map import ConditionMap
from sanjeevani_ml.dialogue.conditions import condition_matches
from sanjeevani_ml.dialogue.ontology_loader import flatten_questions, load_ontology, section_items
from sanjeevani_ml.redflags.rules import evaluate_answer
from sanjeevani_ml.schemas import Answer, InterviewMode, InterviewTranscript, Question, QuestionOption

_CAPTURE_MODES = {"patient_reported", "derived", "examiner_required", "excluded"}

@dataclass
class _SessionState:
    session_id: str
    language: str
    mode: InterviewMode
    answers: dict[str, object] = field(default_factory=dict)
    narration: list[str] = field(default_factory=list)
    red_flags: list[Any] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

class DialogueEngine:
    """Walk the ontology, adapt ordering with the condition map, and never diagnose."""
    def __init__(self, ontology_path: str, condition_map_path: str | None = None):
        self._ontology = load_ontology(ontology_path)
        self._flat_questions = flatten_questions(self._ontology)
        self._sections = sorted(self._ontology.get("sections", []), key=lambda s: s.get("order", 0))
        self._condition_map = ConditionMap(condition_map_path) if condition_map_path else None
        self._follow_up_targets = {
            q["follow_up_on_yes"] for s in self._sections for q in section_items(s)
            if "follow_up_on_yes" in q
        }
        self._sessions: dict[str, _SessionState] = {}

    def start_interview(self, session_id: str, language: str = "hi", mode: InterviewMode = "allopathic") -> Question:
        self._sessions[session_id] = _SessionState(session_id, language, mode)
        return self._next_question_response(session_id)

    start_session = start_interview

    def next_question(self, session_id: str) -> Question | None:
        state = self._require_session(session_id)
        qid = self._compute_next_question_id(state)
        return None if qid is None else self._build_question(state, qid)

    def submit_answer(self, session_id: str, answer: Answer) -> Question | None:
        state = self._require_session(session_id)
        expected = self._compute_next_question_id(state)
        if expected is None:
            raise ValueError(f"Session '{session_id}' has no more questions -- interview already complete.")
        if answer.question_id != expected:
            raise ValueError(f"Answer is for '{answer.question_id}' but the engine is waiting on '{expected}'. Client and engine have gone out of sync.")
        state.answers[answer.question_id] = answer.value
        if answer.transcript:
            state.narration.append(answer.transcript)
        state.red_flags.extend(evaluate_answer(
            question_id=answer.question_id,
            value=answer.value,
            all_answers=state.answers,
            flattened_questions=self._flat_questions,
        ))
        return self.next_question(session_id)

    def is_complete(self, session_id: str) -> bool:
        return self._compute_next_question_id(self._require_session(session_id)) is None

    def get_pending_examiner_fields(self, session_id: str) -> list[dict[str, Any]]:
        return self._build_ayush_status(self._require_session(session_id)).get("pending_examiner", [])

    def get_transcript(self, session_id: str) -> InterviewTranscript:
        state = self._require_session(session_id)
        return InterviewTranscript(
            session_id=state.session_id,
            mode=state.mode,
            language=state.language,
            answers=dict(state.answers),
            narration=list(state.narration),
            red_flags=list(state.red_flags),
            completed=self.is_complete(session_id),
            unanswered=[q for q in self._eligible_question_sequence(state) if q not in state.answers],
            ayush=self._build_ayush_status(state),
        )

    def get_session_answers(self, session_id: str) -> dict[str, object]:
        return dict(self._require_session(session_id).answers)

    def _eligible_sections(self, state: _SessionState) -> list[dict]:
        result = []
        for section in self._sections:
            section_mode = section.get("mode")
            if section_mode is not None and section_mode != state.mode:
                continue
            if section.get("always"):
                result.append(section); continue
            trigger = section.get("trigger")
            if trigger:
                if condition_matches(trigger, state.answers): result.append(section)
                continue
            if section_mode is not None:
                result.append(section)
        return result

    @staticmethod
    def _capture_mode(entry: dict[str, Any]) -> str:
        mode = entry.get("capture_mode")
        if mode in _CAPTURE_MODES: return mode
        if entry.get("input_type") == "derived": return "derived"
        return "patient_reported"

    @classmethod
    def _is_askable(cls, entry: dict[str, Any]) -> bool:
        return cls._capture_mode(entry) == "patient_reported"

    def _eligible_question_sequence(self, state: _SessionState) -> list[str]:
        sequence = []
        for section in self._eligible_sections(state):
            for q in section_items(section):
                qid = q["id"]
                if not self._is_askable(q) or qid in self._follow_up_targets:
                    continue
                sequence.append(qid)
                follow = q.get("follow_up_on_yes")
                if follow and state.answers.get(qid) == "yes":
                    target = self._flat_questions.get(follow)
                    if target and self._is_askable(target): sequence.append(follow)
        return list(dict.fromkeys(sequence))

    def _compute_next_question_id(self, state: _SessionState) -> str | None:
        unanswered = [q for q in self._eligible_question_sequence(state) if q not in state.answers]
        if not unanswered: return None
        if self._condition_map is None: return unanswered[0]
        ranked = self._condition_map.rank_questions(candidates=unanswered, answers=state.answers)
        return ranked[0] if ranked else unanswered[0]

    def _next_question_response(self, session_id: str) -> Question:
        q = self.next_question(session_id)
        if q is None: raise ValueError("Ontology produced zero eligible patient questions.")
        return q

    def _build_question(self, state: _SessionState, question_id: str) -> Question:
        entry = self._flat_questions[question_id]
        section = self._find_section(question_id)
        options = [QuestionOption(id=o["id"], label=o.get("label", {}), icon=o.get("icon"), exclusive=o.get("exclusive", False)) for o in entry.get("options", [])]
        if "skippable" in entry: skippable = bool(entry["skippable"])
        elif "required" in entry: skippable = not bool(entry["required"])
        else: skippable = True
        hints = {k: entry[k] for k in ("regions", "min", "max", "face_scale", "max_seconds", "units") if k in entry}
        if "regions" in hints:
            labels = self._ontology.get("$region_labels", {})
            hints["region_labels"] = {r: labels[r] for r in hints["regions"] if r in labels}
        sequence = self._eligible_question_sequence(state)
        return Question(
            question_id=question_id,
            section_id=section["id"],
            input_type=entry["input_type"],
            prompt=entry.get("prompt", {}),
            audio_url=entry.get("audio_url"),
            options=options,
            skippable=skippable,
            hints=hints,
            answered=sum(q in state.answers for q in sequence),
            estimated_total=len(sequence),
            selection_reason="condition_map" if self._condition_map else "ontology_order",
        )

    def _build_ayush_status(self, state: _SessionState) -> dict[str, Any]:
        if state.mode != "ayush": return {}
        derived, pending, patient = [], [], []
        for section in self._sections:
            if section.get("mode") != "ayush": continue
            for p in section.get("parameters", []):
                mode = self._capture_mode(p)
                if mode == "derived":
                    derived.append({"parameter_id": p.get("id"), "capture_mode": mode, "derived_from": p.get("derived_from"), "provisional": True})
                elif mode == "examiner_required":
                    pending.append({"parameter_id": p.get("id"), "capture_mode": mode, "status": "pending"})
            for q in section_items(section):
                mode = self._capture_mode(q); qid = q.get("id")
                if mode == "derived":
                    derived.append({"parameter_id": qid, "capture_mode": mode, "derived_from": q.get("derived_from"), "provisional": True})
                elif mode == "examiner_required":
                    pending.append({"parameter_id": qid, "capture_mode": mode, "status": "pending"})
                elif mode == "patient_reported" and qid in state.answers:
                    patient.append(qid)
        return {"derived": derived, "pending_examiner": pending, "patient_reported": patient, "provisional": bool(derived)}

    def _find_section(self, question_id: str) -> dict:
        for section in self._sections:
            if any(q.get("id") == question_id for q in section_items(section)): return section
        raise KeyError(f"Question '{question_id}' is not in any ontology section.")

    def _require_session(self, session_id: str) -> _SessionState:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown interview session '{session_id}'. Call start_interview() first.")
        return self._sessions[session_id]
