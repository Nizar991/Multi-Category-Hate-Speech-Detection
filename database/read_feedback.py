import sqlite3
from pathlib import Path

FEEDBACK_DB = Path(__file__).parent / "feedback.db"


def get_verified_corrections():
    conn = sqlite3.connect(FEEDBACK_DB)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, sentence, correct_category
        FROM corrections
        ORDER BY id
    """)

    rows = cursor.fetchall()

    conn.close()

    return rows


if __name__ == "__main__":
    corrections = get_verified_corrections()

    print("Total corrections found:", len(corrections))

    for row in corrections:
        print(row)