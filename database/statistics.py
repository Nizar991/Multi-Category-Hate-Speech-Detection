import sqlite3
from pathlib import Path


# ============================================
# DATABASE LOCATION
# ============================================

DATABASE_PATH = Path(__file__).parent / "feedback.db"


# ============================================
# GET DATABASE CONNECTION
# ============================================

def get_connection():
    return sqlite3.connect(DATABASE_PATH)


# ============================================
# GET BASIC STATISTICS
# ============================================

def get_statistics():

    connection = get_connection()
    cursor = connection.cursor()

    # Total number of corrections
    cursor.execute("""
        SELECT COUNT(*)
        FROM corrections
    """)

    total_corrections = cursor.fetchone()[0]


    # Number of corrections by predicted category
    cursor.execute("""
        SELECT
            predicted_category,
            COUNT(*)
        FROM corrections
        GROUP BY predicted_category
        ORDER BY COUNT(*) DESC
    """)

    corrections_by_prediction = cursor.fetchall()


    # Number of corrections where each category
    # was the actual correct category
    cursor.execute("""
        SELECT
            correct_category,
            COUNT(*)
        FROM corrections
        GROUP BY correct_category
        ORDER BY COUNT(*) DESC
    """)

    correct_category_counts = cursor.fetchall()


    # Most frequently corrected prediction
    most_corrected = None

    if corrections_by_prediction:

        most_corrected = corrections_by_prediction[0]


    connection.close()


    return {
        "total_corrections": total_corrections,
        "corrections_by_prediction": corrections_by_prediction,
        "correct_category_counts": correct_category_counts,
        "most_corrected": most_corrected
    }

def get_confusion_statistics():
    
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            predicted_category,
            correct_category,
            COUNT(*) AS correction_count
        FROM corrections
        GROUP BY predicted_category, correct_category
        ORDER BY correction_count DESC
    """)

    rows = cursor.fetchall()

    connection.close()

    return rows


# ============================================
# GET CORRECTION HISTORY
# ============================================

def get_correction_history():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            sentence,
            predicted_category,
            correct_category,
            confidence,
            created_at
        FROM corrections
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    connection.close()

    return rows

def get_correction_chart_data():
    
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            predicted_category,
            COUNT(*)
        FROM corrections
        GROUP BY predicted_category
        ORDER BY COUNT(*) DESC
    """)

    rows = cursor.fetchall()

    connection.close()

    categories = [
        "Insult",
        "Neutral",
        "Racism",
        "Sexual Abuse",
        "Threat"
    ]

    data = []

    for category in categories:

        count = 0

        for row_category, row_count in rows:

            if row_category == category:
                count = row_count
                break

        data.append({
            "Category": category,
            "Corrections": count
        })

    return data