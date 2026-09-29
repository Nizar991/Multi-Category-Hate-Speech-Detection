import sqlite3
from pathlib import Path


DATABASE_PATH = Path(__file__).parent / "feedback.db"


connection = sqlite3.connect(DATABASE_PATH)

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


print("\n===== SAVED CORRECTIONS =====\n")

if not rows:

    print("No corrections found.")

else:

    for row in rows:

        print("ID:", row[0])
        print("Sentence:", row[1])
        print("Predicted:", row[2])
        print("Correct:", row[3])
        print("Confidence:", f"{row[4]:.2%}")
        print("Created:", row[5])
        print("-" * 50)


connection.close()