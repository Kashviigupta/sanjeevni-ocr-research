"""English option labels for the ontology's fixed-choice questions, copied verbatim
from shared/clinical/history-ontology.json.

This is NOT generated or inferred text — every string here is copied character-
for-character from the ontology file Mahek's dialogue engine actually walks. That
matters: it means a patient's answer id (e.g. "avara") renders as exactly the
same words the ontology itself would show them ("Weak"), not a paraphrase this
module invented. If the ontology's wording changes, this table goes stale and
should be regenerated from the source file — it is a copy, not an independent
source of truth.

Only questions actually used by `interview_sections.py` are covered. Extending
coverage (e.g. to render Dashavidha Pariksha answers) is a deliberate follow-up,
not an oversight — see the module docstring in `interview_sections.py` for why
that section is still a gap.
"""
from __future__ import annotations

CC_PRIMARY_LABELS: dict[str, str] = {
    "pain": "Pain", "fever": "Fever", "cough_breath": "Cough or breathing trouble",
    "digestive": "Stomach or digestion", "weakness": "Weakness or tiredness",
    "skin": "Skin problem", "checkup": "Routine check-up", "other": "Something else",
}

BODY_REGION_LABELS: dict[str, str] = {
    "head": "head", "eye": "eye", "ear": "ear", "mouth": "mouth", "throat": "throat",
    "chest": "chest", "abdomen": "abdomen", "back": "back", "arm": "arm", "leg": "leg",
    "skin": "skin", "whole_body": "whole body", "other": "another area",
}

SOC_CHARACTER_LABELS: dict[str, str] = {
    "sharp": "sharp, stabbing", "dull": "a dull ache", "burning": "burning",
    "cramping": "cramping, twisting", "pressure": "heaviness, pressure", "throbbing": "throbbing",
}

SOC_ASSOCIATION_LABELS: dict[str, str] = {
    "breathless": "breathlessness", "sweating": "cold sweating", "nausea": "nausea or vomiting",
    "fever": "fever", "dizzy": "dizziness or fainting", "none": "nothing else",
}

SOC_MODIFIER_LABELS: dict[str, str] = {
    "worse_movement": "worse on moving", "worse_eating": "worse after eating",
    "worse_night": "worse at night", "better_rest": "better on resting",
    "better_medicine": "better with medicine",
}

PX_CONDITION_LABELS: dict[str, str] = {
    "diabetes": "Diabetes (sugar)", "hypertension": "High blood pressure", "asthma": "Asthma",
    "heart": "Heart disease", "thyroid": "Thyroid problem", "tb": "Tuberculosis (TB)",
    "kidney": "Kidney problem", "none": "None",
}

PF_FAMILY_LABELS: dict[str, str] = {
    "diabetes": "Diabetes", "hypertension": "High blood pressure", "heart": "Heart disease",
    "cancer": "Cancer", "none": "None",
}

PF_HABITS_LABELS: dict[str, str] = {
    "tobacco_smoke": "Smoking", "tobacco_chew": "Chewing tobacco, gutkha",
    "alcohol": "Alcohol", "none": "None",
}

ROS_SCREEN_LABELS: dict[str, str] = {
    "weight_loss": "losing weight without trying", "night_sweats": "night sweats",
    "persistent_fever": "fever for more than 2 weeks",
    "blood": "blood in cough, stool or urine", "none": "none of these",
}
