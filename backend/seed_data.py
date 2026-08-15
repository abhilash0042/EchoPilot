"""
Database Seeding Script for Lumina Health AI Voice Agent
Populates realistic sample data including sensitive hospital records and customer/patient health records.
"""

import sys
import os
import random
from datetime import datetime, timedelta

# Ensure backend directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import init_db, get_db_connection


def seed_database(force_refresh: bool = False):
    """Initializes and fills sample data into the database."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    if force_refresh:
        print("[DB Seed] Clearing existing records for fresh seed...")
        cursor.execute("DELETE FROM call_logs;")
        cursor.execute("DELETE FROM appointments;")
        cursor.execute("DELETE FROM patients;")
        cursor.execute("DELETE FROM services;")
        cursor.execute("DELETE FROM doctors;")
        cursor.execute("DELETE FROM hospitals;")
        conn.commit()

    # Check if data already exists
    cursor.execute("SELECT COUNT(*) as cnt FROM hospitals")
    if cursor.fetchone()["cnt"] > 0 and not force_refresh:
        print("[DB Seed] Database already contains data. Skipping seed. (Use force_refresh=True to overwrite)")
        conn.close()
        return

    print("[DB Seed] Seeding Hospital records (with sensitive licensing & compliance info)...")
    
    # 1. Seed Hospital (Sensitive: Tax ID, DEA License, HIPAA cert, Private Hotline, Merchant Key)
    cursor.execute("""
    INSERT INTO hospitals (
        name, branch, registration_number, tax_id_ein,
        dea_license_number, hipaa_compliance_id, emergency_hotline,
        direct_phone, email, address, billing_gateway_key
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "Lumina Health Multi-Speciality Medical Center",
        "Downtown Medical Plaza - Main Campus",
        "REG-MED-2024-88492-CA",                        # Sensitive: State Medical Board Registration
        "XX-9824109",                                   # Sensitive: Federal Tax EIN
        "DEA-MD-849201948",                             # Sensitive: DEA Controlled Substance License
        "HIPAA-AUDIT-CERT-9481-2025-VALID",            # Sensitive: HIPAA Compliance Audit Key
        "+1 (800) 555-9110 (Direct ER Triage)",
        "+1 (555) 234-5678",
        "admin.secure@luminahealthclinic.org",
        "742 Evergreen Medical Blvd, Suite 400, San Francisco, CA 94102",
        "sec_live_94819a8fbc2901c8e77402f1a"           # Sensitive: Merchant Billing API Token
    ))
    hospital_id = cursor.lastrowid

    print("[DB Seed] Seeding Doctors & Medical Specialists...")
    # 2. Seed Doctors (with sensitive Medical License Numbers)
    doctors_data = [
        (
            hospital_id,
            "Dr. Sarah Lin, MD, FACC",
            "Cardiology",
            "Cardiovascular Disease & Hypertension",
            "LIC-CA-MED-94821",                         # Sensitive: State Medical License
            250.00,
            "Mon,Tue,Wed,Thu,Fri",
            "09:00 - 17:00",
            "dr.sarah.lin@luminahealthclinic.org"
        ),
        (
            hospital_id,
            "Dr. Rajesh Mehta, MD",
            "General Medicine",
            "Internal Medicine & Preventive Health",
            "LIC-CA-MED-73629",
            150.00,
            "Mon,Tue,Wed,Thu,Fri,Sat",
            "08:30 - 18:00",
            "dr.rajesh.mehta@luminahealthclinic.org"
        ),
        (
            hospital_id,
            "Dr. Emily Chen, MD, FAAP",
            "Pediatrics",
            "Pediatric Primary Care & Immunization",
            "LIC-CA-MED-59201",
            175.00,
            "Mon,Wed,Thu,Fri",
            "09:00 - 16:30",
            "dr.emily.chen@luminahealthclinic.org"
        ),
        (
            hospital_id,
            "Dr. Marcus Vance, MD",
            "Orthopedics",
            "Joint Reconstruction & Sports Medicine",
            "LIC-CA-MED-84910",
            275.00,
            "Tue,Thu,Fri",
            "10:00 - 18:00",
            "dr.marcus.vance@luminahealthclinic.org"
        ),
        (
            hospital_id,
            "Dr. Priya Sharma, MD, FAAD",
            "Dermatology",
            "Clinical Dermatology & Skin Diagnostics",
            "LIC-CA-MED-62849",
            200.00,
            "Mon,Tue,Thu,Sat",
            "09:00 - 17:30",
            "dr.priya.sharma@luminahealthclinic.org"
        ),
        (
            hospital_id,
            "Dr. Alexander Ross, DDS",
            "Dental",
            "Preventive & Restorative Dentistry",
            "LIC-CA-DDS-41920",
            140.00,
            "Mon,Tue,Wed,Thu,Fri",
            "08:00 - 16:00",
            "dr.alex.ross@luminahealthclinic.org"
        )
    ]

    cursor.executemany("""
    INSERT INTO doctors (
        hospital_id, name, department, specialization,
        license_number, consultation_fee, available_days,
        available_hours, contact_email
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, doctors_data)

    print("[DB Seed] Seeding Hospital Services & Treatment Offerings...")
    # 3. Seed Services
    services_data = [
        (hospital_id, "General Checkup", "General Medicine", 30, 120.00, "Comprehensive annual physical examination, vital signs, and wellness consultation."),
        (hospital_id, "Cardiology Consultation", "Cardiology", 45, 250.00, "In-depth cardiac evaluation, ECG review, and blood pressure optimization."),
        (hospital_id, "Pediatric Wellness Screening", "Pediatrics", 30, 150.00, "Routine developmental screening and childhood vaccination review."),
        (hospital_id, "Orthopedic Evaluation", "Orthopedics", 45, 220.00, "Assessment of joint pain, sports injuries, and mobility disorders."),
        (hospital_id, "Dermatology Skin Exam", "Dermatology", 30, 180.00, "Full body skin examination, mole check, and acne/rash treatment."),
        (hospital_id, "Dental Cleaning & Checkup", "Dental", 45, 130.00, "Routine ultrasonic scaling, plaque removal, and oral cavity exam."),
        (hospital_id, "Blood Test & Lab Work", "General Medicine", 20, 85.00, "Diagnostic blood draw: Complete Blood Count (CBC), Lipid Panel, HbA1c."),
        (hospital_id, "Flu Vaccination & Immunity Shot", "General Medicine", 15, 45.00, "Seasonal influenza shot and booster immunizations.")
    ]

    cursor.executemany("""
    INSERT INTO services (
        hospital_id, name, department, duration_minutes, price, description
    ) VALUES (?, ?, ?, ?, ?, ?)
    """, services_data)

    print("[DB Seed] Seeding Patients with Sensitive Medical Records & Insurance PHI...")
    # 4. Seed Patients with Sensitive Data (MRN, SSN, Allergies, Medications, Insurance, Confidential Notes)
    patients_data = [
        (
            "Johnathan Davis",
            "+1 (555) 432-8921",
            "j.davis84@example.com",
            "1984-04-12",
            "MRN-84920-LUM",                             # Sensitive: Hospital Medical Record Number
            "8492",                                      # Sensitive: SSN Last 4
            "O+",                                        # Sensitive: Blood Group
            "Penicillin (Anaphylaxis), Amoxicillin",      # Sensitive: Medical Allergies
            "Type 2 Diabetes Mellitus, Essential HTN",   # Sensitive: Chronic Conditions
            "Metformin 1000mg BID, Lisinopril 20mg daily", # Sensitive: Active Prescriptions
            "BlueCross BlueShield Premier Choice",       # Sensitive: Insurance Carrier
            "BCBS-94829103-01",                          # Sensitive: Insurance Policy #
            "Eleanor Davis (Spouse)",
            "+1 (555) 432-8922",
            "Patient has high compliance. Requires kidney function panel check every 6 months. History of mild sleep apnea."
        ),
        (
            "Sophia Martinez",
            "+1 (555) 871-3490",
            "sophia.martinez@example.com",
            "1992-09-28",
            "MRN-39102-LUM",
            "1294",
            "A-",
            "Sulfa Drugs, Aspirin",
            "Mild Persistent Asthma, Seasonal Rhinitis",
            "Albuterol HFA Inhaler PRN, Fluticasone Nasal Spray",
            "Aetna Choice POS II",
            "AETNA-74920194-02",
            "Carlos Martinez (Brother)",
            "+1 (555) 871-3499",
            "Asthma well-controlled with inhaler. Last peak flow 450 L/min. Avoid beta-blockers due to bronchospasm risk."
        ),
        (
            "Michael Chang",
            "+1 (555) 239-6612",
            "mchang.tech@example.com",
            "1978-11-05",
            "MRN-58291-LUM",
            "7301",
            "B+",
            "No Known Drug Allergies (NKDA)",
            "Hyperlipidemia, Chronic Lower Back Strain (L4-L5)",
            "Atorvastatin 20mg at bedtime",
            "UnitedHealthcare Choice Plus",
            "UHC-84910294-A",
            "Grace Chang (Wife)",
            "+1 (555) 239-6615",
            "Referred to physical therapy for lumbar strain. Scheduled routine lipid profile follow-up."
        ),
        (
            "Samantha Reed",
            "+1 (555) 902-1144",
            "sam.reed.designer@example.com",
            "1995-03-17",
            "MRN-19482-LUM",
            "4419",
            "AB+",
            "Latex, Codeine",
            "Eczema, Contact Dermatitis",
            "Hydrocortisone 2.5% Cream, Cetirizine 10mg",
            "Cigna Open Access Plus",
            "CIGNA-3891048-01",
            "David Reed (Father)",
            "+1 (555) 902-1140",
            "Patch testing completed in 2024. Severe sensitivity to fragrance and nickel."
        ),
        (
            "Abhilash Rao",
            "+1 (555) 789-0123",
            "abhilash.rao@example.com",
            "1996-07-22",
            "MRN-77391-LUM",
            "9932",
            "O+",
            "NKDA (No known drug allergies)",
            "None documented",
            "Multivitamin daily",
            "Kaiser Permanente HMO",
            "KP-5829104-09",
            "Pooja Rao (Sister)",
            "+1 (555) 789-0129",
            "Annual health maintenance and preventive screening visit."
        )
    ]

    cursor.executemany("""
    INSERT INTO patients (
        name, phone, email, date_of_birth, medical_record_number,
        national_id_ssn_last4, blood_group, allergies, chronic_conditions,
        current_medications, insurance_provider, insurance_policy_number,
        emergency_contact_name, emergency_contact_phone, confidential_notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, patients_data)

    print("[DB Seed] Seeding Sample Appointments...")
    # Fetch inserted doctors, services and patients
    cursor.execute("SELECT id, name, department FROM doctors")
    doctors = {d["department"]: d["id"] for d in cursor.fetchall()}

    cursor.execute("SELECT id, name FROM services")
    services_map = {s["name"]: s["id"] for s in cursor.fetchall()}

    cursor.execute("SELECT id, name, phone FROM patients")
    patients_map = {p["name"]: p["id"] for p in cursor.fetchall()}

    today = datetime.now()
    tomorrow = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    day_after = (today + timedelta(days=2)).strftime("%Y-%m-%d")
    next_week = (today + timedelta(days=5)).strftime("%Y-%m-%d")

    appointments_data = [
        (
            "LUM-10492",
            patients_map.get("Johnathan Davis"),
            doctors.get("Cardiology"),
            services_map.get("Cardiology Consultation"),
            "Cardiology Consultation",
            tomorrow,
            "10:00",
            "CONFIRMED",
            "AI_VOICE_AGENT",
            "Follow-up for blood pressure medication review. Patient requested morning slot."
        ),
        (
            "LUM-10493",
            patients_map.get("Sophia Martinez"),
            doctors.get("General Medicine"),
            services_map.get("General Checkup"),
            "General Checkup",
            tomorrow,
            "14:00",
            "CONFIRMED",
            "AI_VOICE_AGENT",
            "Annual health assessment and routine prescription refill."
        ),
        (
            "LUM-10494",
            patients_map.get("Michael Chang"),
            doctors.get("Orthopedics"),
            services_map.get("Orthopedic Evaluation"),
            "Orthopedic Evaluation",
            day_after,
            "11:30",
            "CONFIRMED",
            "WEB_PORTAL",
            "Lower back stiffness after sports activity."
        ),
        (
            "LUM-10495",
            patients_map.get("Samantha Reed"),
            doctors.get("Dermatology"),
            services_map.get("Dermatology Skin Exam"),
            "Dermatology Skin Exam",
            next_week,
            "15:00",
            "CONFIRMED",
            "AI_VOICE_AGENT",
            "Flare-up on forearms, needs dermatologist consultation."
        )
    ]

    cursor.executemany("""
    INSERT INTO appointments (
        booking_code, patient_id, doctor_id, service_id,
        service_name, appointment_date, appointment_time,
        status, booked_via, patient_notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, appointments_data)

    print("[DB Seed] Seeding Call Logs & Telemetry...")
    cursor.execute("SELECT id, booking_code FROM appointments")
    appts = {a["booking_code"]: a["id"] for a in cursor.fetchall()}

    # 5. Seed Call Logs
    call_logs_data = [
        (
            "sess_voice_948210491",
            "+1 (555) 432-8921",
            "Johnathan Davis",
            "Cardiology Consultation",
            appts.get("LUM-10492"),
            42,
            "User: Hi, I'd like to book a cardiology appointment.\nAssistant: Hey there! I'm Lumina from the health clinic...\nUser: Tomorrow at 10 AM.\nAssistant: Awesome! Can I get your name please?\nUser: Johnathan Davis, phone 555-432-8921.\nAssistant: Booking confirmed for Cardiology with Dr. Sarah Lin!",
            "COMPLETED"
        ),
        (
            "sess_voice_847192048",
            "+1 (555) 871-3490",
            "Sophia Martinez",
            "General Checkup",
            appts.get("LUM-10493"),
            38,
            "User: I need a general checkup tomorrow afternoon.\nAssistant: Got it! What time would you like to come in?\nUser: 2 PM please. Sophia Martinez.\nAssistant: Confirmed for 2 PM!",
            "COMPLETED"
        )
    ]

    cursor.executemany("""
    INSERT INTO call_logs (
        session_id, caller_phone, caller_name, service_requested,
        appointment_id, call_duration_seconds, transcript, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, call_logs_data)

    conn.commit()
    conn.close()
    print("[DB Seed] Database successfully seeded with full healthcare and sensitive customer records!")


if __name__ == "__main__":
    seed_database(force_refresh=True)
