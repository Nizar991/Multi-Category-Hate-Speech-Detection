import sqlite3
from pathlib import Path
from datetime import datetime


# ============================================
# DATABASE LOCATION
# ============================================

DATABASE_PATH = Path(__file__).parent / "feedback.db"


# ============================================
# DATABASE CONNECTION
# ============================================

def get_connection():
    return sqlite3.connect(DATABASE_PATH)


# ============================================
# INITIALIZE DATABASE
# ============================================

def initialize_database():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence TEXT NOT NULL,
            predicted_category TEXT NOT NULL,
            correct_category TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


# ============================================
# SAVE CORRECTION
# ============================================

def save_correction(
    sentence,
    predicted_category,
    correct_category,
    confidence
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO corrections (
            sentence,
            predicted_category,
            correct_category,
            confidence,
            created_at
        )
        VALUES (?, ?, ?, ?, datetime('now'))
    """, (
        sentence,
        predicted_category,
        correct_category,
        confidence
    ))

    # Get the newly created correction ID
    feedback_id = cursor.lastrowid

    conn.commit()
    conn.close()

    # Send the ID back to app.py
    return feedback_id