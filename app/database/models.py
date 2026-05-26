from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, func
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from datetime import datetime  # noqa: F401
from app.database.db import Base
from sqlalchemy import Boolean


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(150), nullable=False)
    email = Column(String(150), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    specialization = Column(String(150))
    hospital_name = Column(String(150))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    date_of_birth = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    patients = relationship("Patient", back_populates="doctor")
    diagnoses = relationship("Diagnosis", back_populates="doctor")
    appointments = relationship("Appointment", back_populates="doctor")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=True)

    full_name = Column(String(150), nullable=False)
    age = Column(Integer)
    gender = Column(String(20))
    phone = Column(String(50))
    national_id = Column(String(100))
    email = Column(String(255), nullable=True)
    password_hash = Column(String(255), nullable=True)
    address = Column(String(255), nullable=True)
    date_of_birth = Column(String(50), nullable=True)
    medical_notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    doctor = relationship("Doctor", back_populates="patients")
    diagnoses = relationship("Diagnosis", back_populates="patient")
    appointments = relationship("Appointment", back_populates="patient")

    is_deleted = Column(Boolean, default=False)


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id = Column(Integer, primary_key=True, index=True)

    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)

    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=True)
    doctor_name = Column(String(150), nullable=True)

    scan_type = Column(String(50))
    image_path = Column(Text)
    heatmap_path = Column(Text)

    image_prediction = Column(String(100))
    image_confidence = Column(Float)

    symptom_prediction = Column(String(100))
    symptom_confidence = Column(Float)

    final_prediction = Column(String(100))
    final_confidence = Column(Float)
    risk_level = Column(String(50))

    image_scores_json = Column(Text)
    symptom_scores_json = Column(Text)
    final_scores_json = Column(Text)
    selected_symptoms_json = Column(Text)

    doctor_notes = Column(Text)
    assistant_explanation = Column(Text)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    doctor = relationship("Doctor", back_populates="diagnoses")
    patient = relationship("Patient", back_populates="diagnoses")
    report = relationship("Report", back_populates="diagnosis", uselist=False)


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)

    diagnosis_id = Column(Integer, ForeignKey("diagnoses.id"), nullable=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=True)

    report_path = Column(Text)
    report_status = Column(String(50), default="generated")

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    diagnosis = relationship("Diagnosis", back_populates="report")


class PatientAlert(Base):
    __tablename__ = "patient_alerts"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)

    severity = Column(String(50), default="Medium")
    message = Column(Text, nullable=True)
    symptoms_json = Column(Text, nullable=True)

    status = Column(String(50), default="New")
    doctor_reply = Column(Text, nullable=True)
    patient_seen = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)

    requested_date = Column(String(50), nullable=False)
    requested_time = Column(String(50), nullable=False)
    reason = Column(Text, nullable=True)
    symptoms_json = Column(Text, nullable=True)

    status = Column(String(50), default="Pending")
    doctor_response = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="appointments")
    doctor = relationship("Doctor", back_populates="appointments")
