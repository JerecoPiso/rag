import re
import datetime


def parse_tag_value_string(raw: str, value_key: str = "measure") -> list[dict[str, str]]:
    """Parse a 'tag = X | <value_key> = Y | unit = Z' comma-separated string into a list of dicts.

    Commas that appear inside a value (e.g. 'COMATOSE, AWAKE AND ALERT') are preserved
    because the split anchors on commas followed by 'tag ='.
    """
    entries = re.split(r",(?=tag\s*=)", raw)
    result = []
    for entry in entries:
        parts: dict[str, str] = {}
        for segment in entry.split("|"):
            if "=" in segment:
                k, _, v = segment.partition("=")
                parts[k.strip()] = v.strip()
        result.append(parts)
    return result


def _patient_full_name(row: dict) -> str:
    parts = [
        row.get("patient_firstname") or "",
        row.get("patient_middlename") or "",
        row.get("patient_lastname") or "",
        row.get("patient_suffix") or "",
    ]
    return " ".join(p for p in parts if str(p).strip())


def _get(row: dict, key: str, *, timedelta_as_time: bool = False) -> str:
    v = row.get(key)
    if v is None or str(v).strip() == "":
        return "N/A"
    if timedelta_as_time and isinstance(v, datetime.timedelta):
        total = int(v.total_seconds())
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"
    return str(v)


def _build(*lines: str) -> str:
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-view formatters
# ---------------------------------------------------------------------------

def format_doctors_note(row: dict, source: str) -> str:
    g = lambda key: _get(row, key)
    return _build(
        "=== EMR_RECORD ===",
        "Type: Doctors Order / Doctors Notes",
        f"SOURCE_VIEW: {source}",
        "",
        f"Patient: {_patient_full_name(row)}",
        f"Type: {g('patient_type')}",
        f"Case Classification: {g('case_classification')}",
        f"OPD / ER: {g('opd_er')}",
        f"Station: {g('station')}",
        "",
        f"Doctors Order Date: {g('doctors_note_date')}",
        f"Doctors Order Time: {g('doctors_note_time')}",
        f"Doctor: {g('process_by')}",
        "",
        "Chief Complaint:",
        g("complaint"),
        "",
        "Initial Diagnosis:",
        g("initial_diagnosis"),
        "",
        "Final Diagnosis:",
        g("final_diagnosis"),
        "",
        "Doctors Order / Doctors Notes:",
        g("doctors_note_order"),
        "",
        "Progress Notes:",
        g("doctors_note_notes"),
        "",
        "========================",
    )


def format_vital_vw(row: dict, source: str) -> str:
    g = lambda key: _get(row, key)

    findings: list[str] = []
    raw = row.get("vital") or ""
    if raw:
        for entry in parse_tag_value_string(str(raw), value_key="measure"):
            tag   = entry.get("tag", "").strip()
            value = entry.get("measure", "").strip()
            if tag:
                findings.append(f"- {tag}: {value}")

    return _build(
        "=== EMR_RECORD ===",
        "TYPE: CLINICAL_ASSESSMENT",
        f"SOURCE_VIEW: {source}",
        "",
        f"PATIENT_NAME: {_patient_full_name(row)}",
        f"PATIENT_TYPE: {g('patient_type')}",
        f"CASE_CLASSIFICATION: {g('case_classification')}",
        f"OPD_ER: {g('opd_er')}",
        f"STATION: {g('station')}",
        f"FORM_TYPE: {g('form_type')}",
        "",
        f"DOCTOR: {g('process_by')}",
        f"CAPTURED: {g('vital_capture_timestamp')}",
       "",
        "Chief Complaint:",
        g("complaint"),
        "",
        "Initial Diagnosis:",
        g("initial_diagnosis"),
        "",
        "Final Diagnosis:",
        g("final_diagnosis"),
        "",
        "CLINICAL_FINDINGS:",
        *findings,
        "",
        "========================",
    )


def format_diet_vw(row: dict, source: str) -> str:
    g = lambda key: _get(row, key, timedelta_as_time=True)
    return _build(
        "=== EMR_RECORD ===",
        "TYPE: DIET_ORDER",
        f"SOURCE_VIEW: {source}",
        "",
        f"PATIENT_NAME: {_patient_full_name(row)}",
        f"PATIENT_TYPE: {g('patient_type')}",
        f"CASE_CLASSIFICATION: {g('case_classification')}",
        f"OPD_ER: {g('opd_er')}",
        f"STATION: {g('station')}",
        "",
        f"ORDERED_BY: {g('process_by')}",
        f"DIET_DATE: {g('diet_date')}",
        f"DIET_TIME: {g('diet_time')}",
        "",
        "CHIEF_COMPLAINT:",
        g("complaint"),
        "",
        "INITIAL_DIAGNOSIS:",
        g("initial_diagnosis"),
        "",
        "DIET_ORDER:",
        g("diet"),
        "",
        "========================",
    )


def format_nurses_note_vw(row: dict, source: str) -> str:
    g = lambda key: _get(row, key)
    return _build(
        "=== EMR_RECORD ===",
        "TYPE: NURSE_NOTE",
        f"SOURCE_VIEW: {source}",
        "",
        f"PATIENT_NAME: {_patient_full_name(row)}",
        f"PATIENT_TYPE: {g('patient_type')}",
        f"CASE_CLASSIFICATION: {g('case_classification')}",
        f"OPD_ER: {g('opd_er')}",
        f"STATION: {g('station')}",
        "",
        f"NOTE_DATE: {g('nurses_note_date')}",
        f"NOTE_TIME: {g('nurses_note_time')}",
        f"NURSE: {g('process_by')}",
        f"NOTE_TYPE: {g('nurses_note_type')}",
        f"FORM_TYPE: {g('nurses_note_form_type')}",
        "",
        "Chief Complaint:",
        g("complaint"),
        "",
        "Initial Diagnosis:",
        g("initial_diagnosis"),
        "",
        "Final Diagnosis:",
        g("final_diagnosis"),
        "",
        "FOCUS:",
        g("nurses_note_focus"),
        "",
        "NOTES:",
        g("nurses_note_notes"),
        "",
        "REMARKS:",
        g("nurses_note_remarks"),
        "",
        "========================",
    )


def format_animal_bite_vw(row: dict, source: str) -> str:
    g = lambda key: _get(row, key)

    details: list[str] = []
    raw = row.get("value") or ""
    if raw:
        for entry in parse_tag_value_string(str(raw), value_key="value"):
            tag   = entry.get("tag", "").strip()
            value = entry.get("value", "").strip()
            if tag:
                details.append(f"- {tag}: {value}")

    return _build(
        "=== EMR_RECORD ===",
        "TYPE: ANIMAL_BITE",
        f"SOURCE_VIEW: {source}",
        "",
        f"PATIENT_NAME: {_patient_full_name(row)}",
        f"PATIENT_TYPE: {g('patient_type')}",
        f"CASE_CLASSIFICATION: {g('case_classification')}",
        f"OPD_ER: {g('opd_er')}",
        f"STATION: {g('station')}",
        "",
        f"PROCESSED_BY: {g('process_by')}",
        f"CAPTURED: {g('vital_capture_timestamp')}",
       "",
        "Chief Complaint:",
        g("complaint"),
        "",
        "Initial Diagnosis:",
        g("initial_diagnosis"),
        "",
        "Final Diagnosis:",
        g("final_diagnosis"),
        "",
        "ANIMAL_BITE_DETAILS:",
        *details,
        "",
        "REMARKS:",
        g("remarks"),
        "",
        "========================",
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_FORMATTERS: dict[str, object] = {
    "_patient_case_doctors_note_vw": format_doctors_note,
    "_patient_case_vital_vw":        format_vital_vw,
    "_patient_case_diet_vw":         format_diet_vw,
    "_patient_case_nurses_note_vw":  format_nurses_note_vw,
    "_patient_animal_bite_vw":       format_animal_bite_vw,
}


def format_record(row: dict, source: str) -> str:
    formatter = _FORMATTERS.get(source)
    if formatter:
        return formatter(row, source)
    lines = ["=== EMR_RECORD ===", f"SOURCE_VIEW: {source}", ""]
    for k, v in row.items():
        lines.append(f"{k.upper()}: {v}")
    lines += ["", "========================"]
    return "\n".join(lines)
