"""
Database Layer for Lumina Health AI Voice Agent
Handles database connections, schema setup, queries, and transactions using SQLite.
Includes sensitive hospital records and customer/patient health data.
"""

import sqlite3
import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Database file path
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DATABASE_PATH", os.path.join(DB_DIR, "lumina_health.db"))


def get_db_connection() -> sqlite3.Connection:
    """Creates a thread-safe connection to the SQLite database with row factory enabled and WAL mode."""
    conn = sqlite3.connect(DB_PATH, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for high-performance concurrent reads & writes
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db():
    """Initializes all database tables with proper indexes and relational constraints."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Hospitals / Clinics Table (Contains sensitive administrative & licensing information)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS hospitals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        branch TEXT DEFAULT 'Main Campus',
        registration_number TEXT NOT NULL UNIQUE,  -- Sensitive: State/Federal Medical Council Registration
        tax_id_ein TEXT NOT NULL,                  -- Sensitive: Taxpayer ID / EIN
        dea_license_number TEXT NOT NULL,          -- Sensitive: Drug Enforcement Administration License
        hipaa_compliance_id TEXT NOT NULL,         -- Sensitive: HIPAA Audit Certificate ID
        emergency_hotline TEXT NOT NULL,
        direct_phone TEXT NOT NULL,
        email TEXT NOT NULL,
        address TEXT NOT NULL,
        billing_gateway_key TEXT,                 -- Sensitive: Payment merchant key
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. Doctors Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS doctors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hospital_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        specialization TEXT NOT NULL,
        license_number TEXT NOT NULL UNIQUE,       -- Sensitive: Doctor Medical Practitioner License
        consultation_fee REAL NOT NULL,
        available_days TEXT NOT NULL,              -- e.g. "Mon,Tue,Wed,Thu,Fri"
        available_hours TEXT NOT NULL,             -- e.g. "09:00 - 17:00"
        contact_email TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (hospital_id) REFERENCES hospitals (id) ON DELETE CASCADE
    );
    """)

    # 3. Services / Departments Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS services (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hospital_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        duration_minutes INTEGER NOT NULL DEFAULT 30,
        price REAL NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (hospital_id) REFERENCES hospitals (id) ON DELETE CASCADE
    );
    """)

    # 4. Patients / Customers Table (Contains Sensitive PII and Protected Health Information PHI)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        phone TEXT NOT NULL UNIQUE,
        email TEXT,
        date_of_birth TEXT,                        -- Sensitive: YYYY-MM-DD
        medical_record_number TEXT UNIQUE,        -- Sensitive: MRN (e.g. MRN-84920-X)
        national_id_ssn_last4 TEXT,                -- Sensitive: Last 4 digits of SSN / National ID
        blood_group TEXT,                          -- Sensitive: Blood type (A+, O-, etc.)
        allergies TEXT,                            -- Sensitive: Medical allergies (e.g. Penicillin, Latex)
        chronic_conditions TEXT,                   -- Sensitive: e.g. Hypertension, Diabetes Type 2
        current_medications TEXT,                  -- Sensitive: Prescribed active medications
        insurance_provider TEXT,                   -- Sensitive: Primary Health Insurance Provider
        insurance_policy_number TEXT,              -- Sensitive: Insurance ID / Policy #
        emergency_contact_name TEXT,
        emergency_contact_phone TEXT,
        confidential_notes TEXT,                   -- Sensitive: Doctor's private clinical history notes
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 5. Appointments / Bookings Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        booking_code TEXT NOT NULL UNIQUE,         -- Reference code like LUM-49281
        patient_id INTEGER NOT NULL,
        doctor_id INTEGER,
        service_id INTEGER,
        service_name TEXT NOT NULL,
        appointment_date TEXT NOT NULL,            -- YYYY-MM-DD
        appointment_time TEXT NOT NULL,            -- HH:MM (24-hour)
        status TEXT NOT NULL DEFAULT 'CONFIRMED',  -- CONFIRMED, COMPLETED, CANCELLED, PENDING
        booked_via TEXT DEFAULT 'AI_VOICE_AGENT',  -- AI_VOICE_AGENT, WEB_PORTAL, RECEPTION
        patient_notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE,
        FOREIGN KEY (doctor_id) REFERENCES doctors (id) ON DELETE SET NULL,
        FOREIGN KEY (service_id) REFERENCES services (id) ON DELETE SET NULL
    );
    """)

    # 6. Call Logs Table (Records voice agent conversations and telemetry)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS call_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL UNIQUE,
        caller_phone TEXT,
        caller_name TEXT,
        service_requested TEXT,
        appointment_id INTEGER,
        call_duration_seconds INTEGER DEFAULT 0,
        transcript TEXT,
        status TEXT DEFAULT 'COMPLETED',           -- COMPLETED, INTERRUPTED, ABANDONED
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (appointment_id) REFERENCES appointments (id) ON DELETE SET NULL
    );
    """)

    # Create Indexes for fast lookup
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_patients_phone ON patients(phone);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_appointments_slot ON appointments(appointment_date, appointment_time);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_appointments_code ON appointments(booking_code);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_call_logs_session ON call_logs(session_id);")

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully at %s", DB_PATH)


# --- Helper CRUD Operations ---

def get_or_create_patient(name: str, phone: str, **kwargs) -> Dict[str, Any]:
    """Finds an existing patient by phone number or creates a new patient record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Normalize phone
    clean_phone = "".join(filter(str.isdigit, phone)) if phone else "0000000000"
    
    cursor.execute("SELECT * FROM patients WHERE phone = ? OR phone = ?", (phone, clean_phone))
    row = cursor.fetchone()
    
    if row:
        patient = dict(row)
        # Update name if previously unknown/blank
        if name and (not patient["name"] or patient["name"] == "Unknown"):
            cursor.execute("UPDATE patients SET name = ? WHERE id = ?", (name, patient["id"]))
            conn.commit()
            patient["name"] = name
        conn.close()
        return patient

    # Generate MRN for new patient
    import random
    mrn = f"MRN-{random.randint(10000, 99999)}-LUM"

    cursor.execute("""
    INSERT INTO patients (
        name, phone, email, date_of_birth, medical_record_number,
        national_id_ssn_last4, blood_group, allergies, chronic_conditions,
        current_medications, insurance_provider, insurance_policy_number,
        emergency_contact_name, emergency_contact_phone, confidential_notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        name or "New Patient",
        phone,
        kwargs.get("email", f"{name.lower().replace(' ', '.') if name else 'patient'}@example.com"),
        kwargs.get("date_of_birth", "1990-01-01"),
        mrn,
        kwargs.get("national_id_ssn_last4", str(random.randint(1000, 9999))),
        kwargs.get("blood_group", "O+"),
        kwargs.get("allergies", "No known drug allergies (NKDA)"),
        kwargs.get("chronic_conditions", "None documented"),
        kwargs.get("current_medications", "None"),
        kwargs.get("insurance_provider", "Standard Healthcare"),
        kwargs.get("insurance_policy_number", f"POL-{random.randint(100000, 999999)}"),
        kwargs.get("emergency_contact_name", "Family Contact"),
        kwargs.get("emergency_contact_phone", phone),
        kwargs.get("confidential_notes", "Registered via Lumina AI Voice Agent.")
    ))
    conn.commit()
    new_id = cursor.lastrowid
    
    cursor.execute("SELECT * FROM patients WHERE id = ?", (new_id,))
    new_patient = dict(cursor.fetchone())
    conn.close()
    return new_patient


def check_slot_available(date: str, time: str, doctor_id: Optional[int] = None) -> bool:
    """Checks if a date and time slot is free in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if doctor_id:
        cursor.execute("""
        SELECT COUNT(*) as count FROM appointments
        WHERE appointment_date = ? AND appointment_time = ? AND doctor_id = ? AND status != 'CANCELLED'
        """, (date, time, doctor_id))
    else:
        # Check if 2 or more doctors are booked at this exact time (assuming multi-doctor clinic)
        cursor.execute("""
        SELECT COUNT(*) as count FROM appointments
        WHERE appointment_date = ? AND appointment_time = ? AND status != 'CANCELLED'
        """, (date, time))
        
    row = cursor.fetchone()
    count = row["count"] if row else 0
    conn.close()
    
    # Allow max 2 simultaneous appointments if doctor not specified, or 0 if specific
    max_concurrent = 2 if not doctor_id else 1
    return count < max_concurrent


def match_doctor_for_service(service_name: str) -> Optional[Dict[str, Any]]:
    """Finds an appropriate doctor based on service department match."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "%" + service_name.lower() + "%"
    cursor.execute("""
    SELECT d.* FROM doctors d
    JOIN services s ON s.department = d.department
    WHERE LOWER(s.name) LIKE ? OR LOWER(d.department) LIKE ? OR LOWER(d.specialization) LIKE ?
    LIMIT 1
    """, (query, query, query))
    
    row = cursor.fetchone()
    if not row:
        # Fallback to general physician
        cursor.execute("SELECT * FROM doctors WHERE department = 'General Medicine' LIMIT 1")
        row = cursor.fetchone()
        
    doctor = dict(row) if row else None
    conn.close()
    return doctor


def match_service_by_name(service_name: str) -> Optional[Dict[str, Any]]:
    """Finds service by keyword."""
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "%" + service_name.lower() + "%"
    cursor.execute("SELECT * FROM services WHERE LOWER(name) LIKE ? OR LOWER(department) LIKE ? LIMIT 1", (query, query))
    row = cursor.fetchone()
    service = dict(row) if row else None
    conn.close()
    return service


def create_appointment(
    service_name: str,
    date: str,
    time: str,
    patient_name: str,
    patient_phone: str,
    booked_via: str = "AI_VOICE_AGENT",
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """Persists a new appointment with patient linking and booking code generation."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Get or create patient record
    patient = get_or_create_patient(patient_name, patient_phone)
    
    # 2. Match service and doctor
    service = match_service_by_name(service_name)
    doctor = match_doctor_for_service(service_name)
    
    import random
    booking_code = f"LUM-{random.randint(10000, 99999)}"
    
    cursor.execute("""
    INSERT INTO appointments (
        booking_code, patient_id, doctor_id, service_id,
        service_name, appointment_date, appointment_time,
        status, booked_via, patient_notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'CONFIRMED', ?, ?)
    """, (
        booking_code,
        patient["id"],
        doctor["id"] if doctor else None,
        service["id"] if service else None,
        service_name,
        date,
        time,
        booked_via,
        notes or f"Booked via Voice Agent for {patient_name}"
    ))
    conn.commit()
    appointment_id = cursor.lastrowid
    
    cursor.execute("""
    SELECT a.*, p.name as patient_name, p.phone as patient_phone, 
           p.medical_record_number as patient_mrn, p.insurance_provider,
           d.name as doctor_name, d.department as doctor_department
    FROM appointments a
    JOIN patients p ON a.patient_id = p.id
    LEFT JOIN doctors d ON a.doctor_id = d.id
    WHERE a.id = ?
    """, (appointment_id,))
    
    appointment = dict(cursor.fetchone())
    conn.close()
    return appointment


def save_call_log(
    session_id: str,
    caller_phone: Optional[str] = None,
    caller_name: Optional[str] = None,
    service_requested: Optional[str] = None,
    appointment_id: Optional[int] = None,
    call_duration_seconds: int = 0,
    transcript: Optional[str] = None,
    status: str = "COMPLETED"
) -> int:
    """Logs voice agent call transcripts and outcome."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    INSERT INTO call_logs (
        session_id, caller_phone, caller_name, service_requested,
        appointment_id, call_duration_seconds, transcript, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(session_id) DO UPDATE SET
        caller_phone = excluded.caller_phone,
        caller_name = excluded.caller_name,
        service_requested = excluded.service_requested,
        appointment_id = excluded.appointment_id,
        call_duration_seconds = excluded.call_duration_seconds,
        transcript = excluded.transcript,
        status = excluded.status
    """, (
        session_id, caller_phone, caller_name, service_requested,
        appointment_id, call_duration_seconds, transcript, status
    ))
    conn.commit()
    log_id = cursor.lastrowid
    conn.close()
    return log_id


def get_all_hospitals() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM hospitals ORDER BY id").fetchall()
    result = [dict(r) for r in rows]
    conn.close()
    return result


def get_all_doctors() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute("""
    SELECT d.*, h.name as hospital_name 
    FROM doctors d 
    JOIN hospitals h ON d.hospital_id = h.id 
    ORDER BY d.id
    """).fetchall()
    result = [dict(r) for r in rows]
    conn.close()
    return result


def get_all_services() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM services ORDER BY department, name").fetchall()
    result = [dict(r) for r in rows]
    conn.close()
    return result


def get_all_patients(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM patients ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    result = [dict(r) for r in rows]
    conn.close()
    return result


def get_all_appointments(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute("""
    SELECT a.*, p.name as patient_name, p.phone as patient_phone, 
           p.medical_record_number, p.blood_group, p.allergies,
           d.name as doctor_name, d.department as doctor_department
    FROM appointments a
    JOIN patients p ON a.patient_id = p.id
    LEFT JOIN doctors d ON a.doctor_id = d.id
    ORDER BY a.appointment_date DESC, a.appointment_time DESC
    LIMIT ?
    """, (limit,)).fetchall()
    result = [dict(r) for r in rows]
    conn.close()
    return result


def get_all_call_logs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute("""
    SELECT cl.*, a.booking_code, a.appointment_date, a.appointment_time
    FROM call_logs cl
    LEFT JOIN appointments a ON cl.appointment_id = a.id
    ORDER BY cl.created_at DESC
    LIMIT ?
    """, (limit,)).fetchall()
    result = [dict(r) for r in rows]
    conn.close()
    return result


def get_db_summary() -> Dict[str, Any]:
    """Returns total counts for dashboard overview."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    counts = {}
    for table in ["hospitals", "doctors", "services", "patients", "appointments", "call_logs"]:
        cursor.execute(f"SELECT COUNT(*) as cnt FROM {table}")
        counts[table] = cursor.fetchone()["cnt"]
        
    conn.close()
    return {
        "status": "connected",
        "database_file": DB_PATH,
        "counts": counts,
        "timestamp": datetime.now().isoformat()
    }
