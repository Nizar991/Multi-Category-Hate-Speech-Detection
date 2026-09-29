import sqlite3
from pathlib import Path

from openpyxl import load_workbook


# ============================================================
# PATHS
# ============================================================

BASE_PATH = Path(__file__).parent.parent

DATASET_PATH = (
    BASE_PATH
    / "Model final dataset.xlsx"
)

DATABASE_PATH = (
    BASE_PATH
    / "database"
    / "adversarial.db"
)


# ============================================================
# LABEL MAPPING
# ============================================================

CATEGORY_TO_LABEL = {
    "Insult": 0,
    "Neutral": 1,
    "Racism": 2,
    "Sexual Abuse": 3,
    "Threat": 4
}


# ============================================================
# LOAD FAILED ADVERSARIAL CASES
# ============================================================

def get_failed_cases():

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            generated_sentence,
            correct_category,
            transformation_type
        FROM adversarial_cases

        WHERE result = 'FAIL'

        AND transformation_type != 'Original'

        ORDER BY id
    """)

    rows = cursor.fetchall()

    conn.close()

    return rows


# ============================================================
# MERGE FAILED CASES INTO DATASET
# ============================================================

def merge():

    # --------------------------------------------------------
    # Check dataset exists
    # --------------------------------------------------------

    if not DATASET_PATH.exists():

        print(
            "Dataset not found:"
        )

        print(
            DATASET_PATH
        )

        return


    # --------------------------------------------------------
    # Open existing dataset
    # --------------------------------------------------------

    workbook = load_workbook(
        DATASET_PATH
    )

    sheet = workbook.active


    dataset_size_before = (
        sheet.max_row - 1
    )


    print()
    print(
        "Dataset rows before merge:",
        dataset_size_before
    )


    # --------------------------------------------------------
    # Read sentences already inside dataset
    # --------------------------------------------------------

    existing_sentences = set()


    for row_number in range(
        2,
        sheet.max_row + 1
    ):

        sentence = sheet.cell(
            row=row_number,
            column=1
        ).value


        if sentence is None:
            continue


        sentence = str(
            sentence
        ).strip()


        if sentence:

            existing_sentences.add(
                sentence.lower()
            )


    # --------------------------------------------------------
    # Read failed adversarial variants from DB2
    # --------------------------------------------------------

    failed_cases = (
        get_failed_cases()
    )


    print(
        "Generated FAIL variants found:",
        len(failed_cases)
    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    added = 0

    skipped_duplicates = 0

    skipped_unknown_categories = 0


    added_by_category = {

        category: 0

        for category
        in CATEGORY_TO_LABEL
    }


    # --------------------------------------------------------
    # Append failed variants to bottom of dataset
    # --------------------------------------------------------

    for (
        case_id,
        sentence,
        correct_category,
        transformation
    ) in failed_cases:


        if sentence is None:
            continue


        sentence = str(
            sentence
        ).strip()


        if not sentence:
            continue


        sentence_key = (
            sentence.lower()
        )


        # ----------------------------------------------------
        # Skip sentences already inside dataset
        # ----------------------------------------------------

        if sentence_key in existing_sentences:

            skipped_duplicates += 1

            continue


        # ----------------------------------------------------
        # Make sure category is valid
        # ----------------------------------------------------

        if (
            correct_category
            not in CATEGORY_TO_LABEL
        ):

            print(
                "Unknown category skipped:",
                correct_category
            )

            skipped_unknown_categories += 1

            continue


        # ----------------------------------------------------
        # Convert category name to numeric label
        # ----------------------------------------------------

        label = CATEGORY_TO_LABEL[
            correct_category
        ]


        # ----------------------------------------------------
        # Append to Excel
        #
        # Column A = Sentence
        # Column B = Label
        # ----------------------------------------------------

        sheet.append(
            [
                sentence,
                label
            ]
        )


        # ----------------------------------------------------
        # Record sentence so it cannot be added twice
        # during the same run
        # ----------------------------------------------------

        existing_sentences.add(
            sentence_key
        )


        added += 1


        added_by_category[
            correct_category
        ] += 1


    # --------------------------------------------------------
    # Save directly to SAME dataset
    # --------------------------------------------------------

    workbook.save(
        DATASET_PATH
    )


    final_dataset_size = (
        sheet.max_row - 1
    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 60)

    print(
        "ADVERSARIAL MERGE COMPLETE"
    )

    print("=" * 60)


    print(
        "Dataset rows before:",
        dataset_size_before
    )


    print(
        "New sentences added:",
        added
    )


    print(
        "Duplicates skipped:",
        skipped_duplicates
    )


    print(
        "Unknown categories skipped:",
        skipped_unknown_categories
    )


    print(
        "Final dataset rows:",
        final_dataset_size
    )


    print()
    print(
        "Added by category:"
    )


    for (
        category,
        count
    ) in added_by_category.items():

        print(
            f"{category}: {count}"
        )


    print()
    print(
        "Updated dataset:"
    )

    print(
        DATASET_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    merge()