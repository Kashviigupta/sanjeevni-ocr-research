"""Dialogue engine - MAHEK (Module A).

Walks shared/clinical/history-ontology.json and decides the next question. This is a
TRAVERSAL over the ontology, not a pile of if-statements about chest pain: if a
branching rule cannot be expressed in the ontology, extend the ontology schema rather
than hardcoding it here.

The backend calls this; clients render what it returns.
"""
