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
        "=== Patient Chart ===",
        "TYPE: Doctors Order / Doctors Notes",
        f"SOURCE_VIEW: {source}",
        "",
        f"Patient: {_patient_full_name(row)}",
        f"Patient Type: {g('patient_type')}",
        f"Case Classification: {g('case_classification')}",
        f"OPD/ER: {g('opd_er')}",
        f"Station: {g('station')}",
        f"Doctor: {g('process_by')}",
        f"Date / Seen By Doctor: {g('doctors_note_date')} {g('doctors_note_time')}",
        f"Chief Complaint: {g('complaint')}",
        f"Initial Diagnosis: {g('initial_diagnosis')}",
        f"Final Diagnosis: {g('final_diagnosis')}",
        f"Doctors Orders / Doctors Notes: {g('doctors_note_order')}",
        f"Progress Notes: {g('doctors_note_notes')}",
    )


def format_clinical_assessements_vw(row: dict, source: str) -> str:
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
        "=== Patient Chart ===",
        "TYPE: Clinical Assessment / Physical Examinations",
        f"SOURCE_VIEW: {source}",
        "",
        f"Patient: {_patient_full_name(row)}",
        f"Patient Type: {g('patient_type')}",
        f"Case Classification: {g('case_classification')}",
        f"OPD/ER: {g('opd_er')}",
        f"Station: {g('station')}",
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


def format_tpr_vital_vw(row: dict, source: str) -> str:
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
        "=== Patient Chart ===",
        "TYPE: TPR (Temperature, Pulse, Rate) Ward Vital Signs",
        f"SOURCE_VIEW: {source}",
        "",
        f"Patient: {_patient_full_name(row)}",
        f"Patient Type: {g('patient_type')}",
        f"Case Classification: {g('case_classification')}",
        f"OPD/ER: {g('opd_er')}",
        f"Station: {g('station')}",
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


def format_opr_vital_vw(row: dict, source: str) -> str:
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
        "=== Patient Chart ===",
        "TYPE: Out Patient Record (Temperature, Pulse, Rate) Vital Signs",
        f"SOURCE_VIEW: {source}",
        "",
        f"Patient: {_patient_full_name(row)}",
        f"Patient Type: {g('patient_type')}",
        f"Case Classification: {g('case_classification')}",
        f"OPD/ER: {g('opd_er')}",
        f"Station: {g('station')}",
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

def format_monitor_vital_vw(row: dict, source: str) -> str:
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
        "=== Patient Chart ===",
        "TYPE: Monitoring Vital Signs",
        f"SOURCE_VIEW: {source}",
        "",
        f"Patient: {_patient_full_name(row)}",
        f"Patient Type: {g('patient_type')}",
        f"Case Classification: {g('case_classification')}",
        f"OPD/ER: {g('opd_er')}",
        f"Station: {g('station')}",
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


def format_fluid_intake_outout_vw(row: dict, source: str) -> str:
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
        "=== Patient Chart ===",
        "TYPE: Fluid Intake and Output (FIAO)",
        f"SOURCE_VIEW: {source}",
        "",
        f"Patient: {_patient_full_name(row)}",
        f"Patient Type: {g('patient_type')}",
        f"Case Classification: {g('case_classification')}",
        f"OPD/ER: {g('opd_er')}",
        f"Station: {g('station')}",
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
        "=== Patient Chart ===",
        "TYPE: DIET_ORDER",
        f"SOURCE_VIEW: {source}",
        "",
        f"Patient: {_patient_full_name(row)}",
        f"Patient Type: {g('patient_type')}",
        f"Case Classification: {g('case_classification')}",
        f"OPD/ER: {g('opd_er')}",
        f"Station: {g('station')}",
        "",
        f"ORDERED_BY: {g('process_by')}",
        f"DIET_DATE: {g('diet_date')}",
        f"DIET_TIME: {g('diet_time')}",
        "",
        f"Chief Complaint: {g('complaint')}",
        f"Initial Diagnosis: {g('initial_diagnosis')}",
        "",
        "DIET_ORDER:",
        g("diet"),
        "",
        "========================",
    )

def format_diagnostic_vw(row: dict, source: str) -> str:
    g = lambda key: _get(row, key, timedelta_as_time=True)
    return _build(
        "=== Patient Chart ===",
        "TYPE: Diagnostics Laboratories and Radiology",
        f"SOURCE_VIEW: {source}",
        "",
        f"Patient: {_patient_full_name(row)}",
        f"Patient Type: {g('patient_type')}",
        f"Case Classification: {g('case_classification')}",
        f"OPD/ER: {g('opd_er')}",
        f"Station: {g('station')}",
        "",
        "Result:",
        g("diagnostic_result"),
        "",
        "========================",
    )

def format_nurses_note_vw(row: dict, source: str) -> str:
    g = lambda key: _get(row, key)
    return _build(
        "=== Patient Chart ===",
        "TYPE: NURSE_NOTE",
        f"SOURCE_VIEW: {source}",
        "",
        f"Patient: {_patient_full_name(row)}",
        f"Patient Type: {g('patient_type')}",
        f"Case Classification: {g('case_classification')}",
        f"OPD/ER: {g('opd_er')}",
        f"Station: {g('station')}",
        "",
        f"NOTE_DATE: {g('nurses_note_date')}",
        f"NOTE_TIME: {g('nurses_note_time')}",
        f"NURSE: {g('process_by')}",
        f"NOTE_TYPE: {g('nurses_note_type')}",
        f"FORM_TYPE: {g('nurses_note_form_type')}",
        "",
        f"Chief Complaint: {g('complaint')}",
        f"Initial Diagnosis: {g('initial_diagnosis')}",
        f"Final Diagnosis: {g('final_diagnosis')}",
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
        "=== format_doctors_note ===",
        "TYPE: ANIMAL_BITE",
        f"SOURCE_VIEW: {source}",
        "",
        f"Patient: {_patient_full_name(row)}",
        f"Patient Type: {g('patient_type')}",
        f"Case Classification: {g('case_classification')}",
        f"OPD/ER: {g('opd_er')}",
        f"Station: {g('station')}",
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


def format_case_status_vw(row: dict, source: str) -> str:
    g  = lambda key: _get(row, key)
    gt = lambda key: _get(row, key, timedelta_as_time=True)
    return _build(
        "=== Patient Chart ===",
        "TYPE: CASE_STATUS",
        f"SOURCE_VIEW: {source}",
        "",
        f"Patient: {_patient_full_name(row)}",
        f"Patient Type: {g('patient_type')}",
        f"Case Classification: {g('case_classification')}",
        f"Patient Status: {g('patient_status')}",
        f"OPD/ER: {g('opd_er')}",
        f"Station: {g('station')}",
        "",
        f"Chief Complaint: {g('complaint')}",
        f"Initial Diagnosis: {g('initial_diagnosis')}",
        f"Final Diagnosis: {g('final_diagnosis')}",
        "",
        "Discharge:",
        f"  Type: {g('discharge_type')}",
        f"  Date: {g('discharge_date')}",
        f"  Time: {gt('discharge_time')}",
        f"  Remarks: {g('discharge_remarks')}",
        "",
        "========================",
    )


def format_medical_consumption_vw(row: dict, source: str) -> str:
    g  = lambda key: _get(row, key)
    gt = lambda key: _get(row, key, timedelta_as_time=True)
    return _build(
        "=== Patient Chart ===",
        "TYPE: MEDICAL_CONSUMPTION (Ivf, Oxygen Consumption, CBG)",
        f"SOURCE_VIEW: {source}",
        "",
        f"Patient: {_patient_full_name(row)}",
        f"Patient Type: {g('patient_type')}",
        f"Case Classification: {g('case_classification')}",
        f"OPD/ER: {g('opd_er')}",
        f"Station: {g('station')}",
        f"Provider: {g('provider_name')}",
        "",
        f"Chief Complaint: {g('complaint')}",
        f"Initial Diagnosis: {g('initial_diagnosis')}",
        f"Final Diagnosis: {g('final_diagnosis')}",
        "",
        "Medicine:",
        f"  Name: {g('medicine_name')}",
        f"  Generic Name: {g('medicine_generic_name')}",
        f"  Brand Name: {g('medicine_brand_name')}",
        "",
        "Consumption:",
        f"  Type: {g('medical_comsumption_type')}",
        f"  Date: {g('medical_comsumption_date')}",
        f"  Start Time: {gt('medical_comsumption_start_time')}",
        f"  End Time: {gt('medical_comsumption_end_time')}",
        f"  Initial Reading: {g('medical_comsumption_initial_reading')}",
        f"  Last Reading: {g('medical_comsumption_last_reading')}",
        f"  Room: {g('medical_comsumption_room')}",
        f"  Reference: {g('medical_comsumption_reference')}",
        "",
        "========================",
    )


def format_medicine_vw(row: dict, source: str) -> str:
    g = lambda key: _get(row, key)
    return _build(
        "=== Patient Chart ===",
        "TYPE: MEDICINE_ORDER",
        f"SOURCE_VIEW: {source}",
        "",
        f"Patient: {_patient_full_name(row)}",
        f"Patient Type: {g('patient_type')}",
        f"Case Classification: {g('case_classification')}",
        f"OPD/ER: {g('opd_er')}",
        f"Station: {g('station')}",
        f"Provider: {g('provider_name')}",
        f"Form Type: {g('form_type')}",
        "",
        f"Chief Complaint: {g('complaint')}",
        f"Initial Diagnosis: {g('initial_diagnosis')}",
        f"Final Diagnosis: {g('final_diagnosis')}",
        "",
        "Medicine:",
        f"  Name: {g('medicine_name')}",
        f"  Generic Name: {g('medicine_generic_name')}",
        f"  Brand Name: {g('medicine_brand_name')}",
        f"  Schedule: {g('medicine_medication_schedule')}",
        f"  Preparation: {g('medicine_preparation')}",
        f"  Route: {g('medicine_route')}",
        f"  Status: {g('type')}",
        "",
        "========================",
    )


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

_SOURCE_RECORD_TYPE = {
    "_patient_case_doctors_note_vw": "DOCTOR_ORDER",
    "_patient_case_vital_vw":        "CLINICAL_ASSESSMENT",
    "_patient_case_diet_vw":         "DIET_ORDER",
    "_patient_case_nurses_note_vw":  "NURSE_NOTE",
    "_patient_animal_bite_vw":       "ANIMAL_BITE",
    "_patient_case_medicine_vw":              "MEDICINE_ORDER",
    "_patient_case_medical_consumption_vw":   "MEDICAL_CONSUMPTION",
    "_patient_case_status_vw":               "CASE_STATUS",
    "_patient_tpr_vw":             "Ward Vital Signs",
    "_patient_opr_vw":             "Out Patient Vital Signs",
    "_patient_monitor_vw":             "Monitoring Vital Signs",
    "_patient_fluid_intake_and_output_vw": "Fluid Intake and Output (FIAO)",
    "_patient_fluid_intake_and_output_vw": "Fluid Intake and Output (FIAO)"

}

_SOURCE_DATE_FIELD = {
    "_patient_case_doctors_note_vw":          "doctors_note_date",
    "_patient_case_vital_vw":                 "vital_capture_timestamp",
    "_patient_case_diet_vw":                  "diet_date",
    "_patient_case_nurses_note_vw":           "nurses_note_date",
    "_patient_animal_bite_vw":                "vital_capture_timestamp",
    "_patient_case_medicine_vw":              "medicine_date",
    "_patient_case_medical_consumption_vw":   "medical_comsumption_date",
    "_patient_case_status_vw":                "discharge_date",
    "_patient_tpr_vw":                        "vital_capture_timestamp",
    "_patient_opr_vw":                        "vital_capture_timestamp",
    "_patient_monitor_vw":                    "vital_capture_timestamp",
    "_patient_fluid_intake_and_output_vw":    "vital_capture_timestamp",
    "_patient_diagnostics_vw":                "vital_capture_timestamp",
}


def build_metadata(row: dict, source: str) -> dict:
    date_field = _SOURCE_DATE_FIELD.get(source, "")
    base = {
        "source":       source,
        "record_type":  _SOURCE_RECORD_TYPE.get(source, "UNKNOWN"),
        "patient_name": _patient_full_name(row),
        "doctor":       str(row.get("process_by") or ""),
        "date":         str(row.get(date_field, "") or ""),
    }
    base.update({k: str(v) for k, v in row.items()})
    return base


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_FORMATTERS: dict[str, object] = {
    "_patient_case_doctors_note_vw": format_doctors_note,
    "_patient_case_vital_vw":        format_clinical_assessements_vw,
    "_patient_case_diet_vw":         format_diet_vw,
    "_patient_case_nurses_note_vw":  format_nurses_note_vw,
    "_patient_animal_bite_vw":       format_animal_bite_vw,
    "_patient_case_medicine_vw":            format_medicine_vw,
    "_patient_case_medical_consumption_vw": format_medical_consumption_vw,
    "_patient_case_status_vw":             format_case_status_vw,
    "_patient_tpr_vw":             format_tpr_vital_vw,
    "_patient_opr_vw":             format_opr_vital_vw,
    "_patient_monitor_vw":             format_monitor_vital_vw,
    "_patient_fluid_intake_and_output_vw": format_fluid_intake_outout_vw,
    "_patient_diagnostics_vw": format_diagnostic_vw

}


def format_record(row: dict, source: str) -> str:
    formatter = _FORMATTERS.get(source)
    if formatter:
        return formatter(row, source)
    lines = ["=== PATIENT CHART ===", f"SOURCE_VIEW: {source}", ""]
    for k, v in row.items():
        lines.append(f"{k.upper()}: {v}")
    lines += ["", "========================"]
    return "\n".join(lines)
