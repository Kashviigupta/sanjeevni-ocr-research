"""Unit tests for DialogueEngine. Clinical correctness belongs in clinician-reviewed fixtures."""
from __future__ import annotations
import json
import pytest
from sanjeevani_ml.dialogue.engine import DialogueEngine
from sanjeevani_ml.schemas import Answer

@pytest.fixture
def ontology_file(tmp_path):
    data = {"sections": [
        {"id":"cc","order":1,"always":True,"questions":[
            {"id":"cc.area","input_type":"single_choice","prompt":{"en":"Where?","hi":"कहाँ?"},"required":True,"options":[{"id":"chest","label":{"en":"Chest","hi":"सीना"}}]},
            {"id":"cc.primary","input_type":"single_choice","prompt":{"en":"What?","hi":"क्या?"},"options":[{"id":"pain","label":{"en":"Pain","hi":"दर्द"}}]},
        ]},
        {"id":"ay","order":2,"mode":"ayush","questions":[
            {"id":"ay.patient","capture_mode":"patient_reported","input_type":"single_choice","prompt":{"en":"Patient?","hi":"रोगी?"},"options":[{"id":"yes","label":{"en":"Yes","hi":"हाँ"}}]},
            {"id":"ay.derived","capture_mode":"derived","input_type":"derived","derived_from":"proxies","prompt":{"en":"Never ask","hi":"न पूछें"}},
            {"id":"ay.exam","capture_mode":"examiner_required","input_type":"single_choice","prompt":{"en":"Never ask","hi":"न पूछें"}},
        ],"parameters":[
            {"id":"dp.prakriti","input_type":"derived","derived_from":"prakriti_assessment"},
            {"id":"dp.sara","input_type":"single_choice","capture_mode":"examiner_required"}
        ]}
    ]}
    path=tmp_path/"history-ontology.json"; path.write_text(json.dumps(data),encoding="utf-8"); return path

def engine(path): return DialogueEngine(str(path))

def test_start_and_answer(ontology_file):
    e=engine(ontology_file); q=e.start_interview("s1")
    assert q.question_id=="cc.area"
    q=e.submit_answer("s1",Answer(question_id="cc.area",value="chest"))
    assert q.question_id=="cc.primary"

def test_derived_and_examiner_entries_are_never_questions(ontology_file):
    e=engine(ontology_file); q=e.start_interview("s1",mode="ayush")
    assert q.question_id=="cc.area"
    e.submit_answer("s1",Answer(question_id="cc.area",value="chest"))
    q=e.next_question("s1")
    assert q.question_id=="cc.primary"
    e.submit_answer("s1",Answer(question_id="cc.primary",value="pain"))
    q=e.next_question("s1")
    assert q.question_id=="ay.patient"
    e.submit_answer("s1",Answer(question_id="ay.patient",value="yes"))
    assert e.next_question("s1") is None

def test_ayush_transcript_has_provisional_and_pending_exam(ontology_file):
    e=engine(ontology_file); e.start_interview("s1",mode="ayush")
    t=e.get_transcript("s1")
    assert t.ayush["provisional"] is True
    assert "dp.prakriti" in {x["parameter_id"] for x in t.ayush["derived"]}
    assert "dp.sara" in {x["parameter_id"] for x in t.ayush["pending_examiner"]}
    assert "dp.sara" not in t.unanswered

def test_wrong_question_rejected(ontology_file):
    e=engine(ontology_file); e.start_interview("s1")
    with pytest.raises(ValueError,match="gone out of sync"):
        e.submit_answer("s1",Answer(question_id="wrong",value="x"))

def test_alias_and_missing_session(ontology_file):
    e=engine(ontology_file); q=e.start_session("s1")
    assert e.next_question("s1").question_id==q.question_id
    with pytest.raises(KeyError,match="Unknown interview session"):
        e.next_question("missing")
