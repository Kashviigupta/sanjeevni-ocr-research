"""Curated English -> Hindi translations for the FIXED template strings used to build
physician-facing summary sections.

This is deliberately NOT machine translation. The sentences `document_sections.py`
produces come from a small, known set of templates ("Abnormal results:", "No
medications or allergy records found...") — not free-form text — so translating
them is a hand-checked lookup table, the same pattern as `ml/data/*.csv`, not an
NLP problem. Zero network calls, zero API cost, fully offline.

**Every string below needs verification by someone who reads Hindi fluently before
this is treated as final** — the same discipline the platform requires for AYUSH
content. Nothing here has been reviewed by a clinician or native speaker yet; this
table is a first draft.

Clinical content that should NEVER be translated, in either direction: drug names,
lab/analyte names, dates, numeric values, units, and terminology codes. A physician
reading a Hindi summary still expects to see "Glycomet 500mg", not a translated
drug name — translating clinical nouns is how a summary becomes actively wrong
rather than just imperfectly worded.
"""
from __future__ import annotations

#: English template -> Hindi translation. Keys must match the fixed strings in
#: document_sections.py and gaps.py EXACTLY (not a fuzzy match) — a translation
#: table entry that silently stops matching because the English wording changed
#: is worse than an obvious missing-translation gap, so we want an exact-key
#: lookup that fails loudly (KeyError-free via .get(), falls back to English)
#: rather than a similarity match that could pair the wrong sentence.
TEMPLATES: dict[str, str] = {
    "Abnormal results:": "असामान्य परिणाम:",
    "All investigations on file:": "उपलब्ध सभी जाँच रिपोर्ट:",
    "Drug interaction warnings:": "दवा पारस्परिक प्रभाव चेतावनी:",
    "No medications or allergy records found among the documents provided.":
        "प्रदान किए गए दस्तावेज़ों में कोई दवा या एलर्जी रिकॉर्ड नहीं मिला।",
    "No prior investigation reports found among the documents provided.":
        "प्रदान किए गए दस्तावेज़ों में कोई पिछली जाँच रिपोर्ट नहीं मिली।",
    "(as read; low confidence)": "(जैसा पढ़ा गया; कम विश्वसनीयता)",
}

#: Connector words used when assembling a translated sentence out of a template
#: plus a data value (a date, a drug name) that itself is never translated.
CONNECTORS: dict[str, str] = {
    "reference": "संदर्भ सीमा",
    "undated": "तिथि रहित",
}


def translate_template(english: str) -> str:
    """Look up the Hindi version of a fixed template string.

    Falls back to the English original when no curated entry exists — an
    untranslated line the physician can still read is a correct degradation;
    a guessed or machine-translated line is not. This mirrors the
    never-invent-a-terminology-code rule: no confident match is a correct,
    honest outcome, not a failure to paper over.
    """
    return TEMPLATES.get(english, english)
