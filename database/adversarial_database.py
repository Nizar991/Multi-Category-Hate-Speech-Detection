import sqlite3
from pathlib import Path
from datetime import datetime


DATABASE_PATH = Path(__file__).parent / "adversarial.db"


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def initialize_adversarial_database():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS adversarial_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            feedback_id INTEGER,

            original_sentence TEXT NOT NULL,
            generated_sentence TEXT NOT NULL,
            transformation_type TEXT NOT NULL,

            correct_category TEXT NOT NULL,

            roberta_prediction TEXT NOT NULL,
            roberta_confidence REAL NOT NULL,
            result TEXT NOT NULL,

            model_b_prediction TEXT,
            model_b_confidence REAL,
            model_b_reward REAL,

            used_for_training INTEGER NOT NULL DEFAULT 0,

            created_at TEXT NOT NULL
        )
    """)


    # --------------------------------------------------
    # Automatically upgrade existing database
    # --------------------------------------------------

    cursor.execute(
        "PRAGMA table_info(adversarial_cases)"
    )

    existing_columns = {
        row[1]
        for row in cursor.fetchall()
    }


    new_columns = {

        "feedback_id":
            "INTEGER",

        "model_b_prediction":
            "TEXT",

        "model_b_confidence":
            "REAL",

        "model_b_reward":
            "REAL",

        "used_for_training":
            "INTEGER NOT NULL DEFAULT 0"
    }


    for column, column_type in new_columns.items():

        if column not in existing_columns:

            cursor.execute(
                f"""
                ALTER TABLE adversarial_cases
                ADD COLUMN {column} {column_type}
                """
            )


    # --------------------------------------------------
    # Prevent duplicate variants for same feedback
    # --------------------------------------------------

    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_feedback_transformation
        ON adversarial_cases(
            feedback_id,
            transformation_type
        )
    """)


    conn.commit()
    conn.close()


# ============================================================
# SAVE ROBERTA RESULT
# ============================================================

def save_adversarial_case(
    feedback_id,
    original_sentence,
    generated_sentence,
    transformation_type,
    correct_category,
    roberta_prediction,
    roberta_confidence,
    result
):

    conn = get_connection()
    cursor = conn.cursor()

    created_at = datetime.now().isoformat(
        timespec="seconds"
    )


    cursor.execute("""
        INSERT INTO adversarial_cases (

            feedback_id,
            original_sentence,
            generated_sentence,
            transformation_type,
            correct_category,

            roberta_prediction,
            roberta_confidence,
            result,

            created_at
        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

        ON CONFLICT(
            feedback_id,
            transformation_type
        )

        DO UPDATE SET

            original_sentence =
                excluded.original_sentence,

            generated_sentence =
                excluded.generated_sentence,

            correct_category =
                excluded.correct_category,

            roberta_prediction =
                excluded.roberta_prediction,

            roberta_confidence =
                excluded.roberta_confidence,

            result =
                excluded.result,

            created_at =
                excluded.created_at
    """, (

        feedback_id,
        original_sentence,
        generated_sentence,
        transformation_type,
        correct_category,

        roberta_prediction,
        roberta_confidence,
        result,

        created_at
    ))


    conn.commit()
    conn.close()


# ============================================================
# SAVE MODEL B RESULT
# ============================================================

def save_model_b_result(
    feedback_id,
    transformation_type,
    prediction,
    confidence,
    reward
):

    conn = get_connection()
    cursor = conn.cursor()


    cursor.execute("""
        UPDATE adversarial_cases

        SET
            model_b_prediction = ?,
            model_b_confidence = ?,
            model_b_reward = ?

        WHERE feedback_id = ?
        AND transformation_type = ?
    """, (

        prediction,
        confidence,
        reward,

        feedback_id,
        transformation_type
    ))


    conn.commit()
    conn.close()


# ============================================================
# GET CASES FOR ONE FEEDBACK REPORT
# ============================================================

def get_cases_for_feedback(
    feedback_id
):

    conn = get_connection()
    cursor = conn.cursor()


    cursor.execute("""
        SELECT
            id,
            feedback_id,
            original_sentence,
            generated_sentence,
            transformation_type,
            correct_category,

            roberta_prediction,
            roberta_confidence,
            result,

            model_b_prediction,
            model_b_confidence,
            model_b_reward

        FROM adversarial_cases

        WHERE feedback_id = ?

        ORDER BY id
    """, (
        feedback_id,
    ))


    rows = cursor.fetchall()

    conn.close()

    return rows


# ============================================================
# GET ALL CASES
# ============================================================

def get_all_cases():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM adversarial_cases
        ORDER BY feedback_id, id
    """)

    rows = cursor.fetchall()

    conn.close()

    return rows


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    initialize_adversarial_database()

    print(
        "Adversarial database ready for "
        "RoBERTa + Model B automation."
    )