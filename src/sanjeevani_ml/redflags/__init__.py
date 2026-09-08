"""Red-flag detection - MAHEK (Module A).

Emergency symptoms trigger an immediate triage alert instead of routine queueing.

Three rules: a red flag never stops the interview; we never tell the patient they may
be having an emergency (that is a diagnosis); and a false negative is far worse than a
false positive - when borderline, flag it.
"""
