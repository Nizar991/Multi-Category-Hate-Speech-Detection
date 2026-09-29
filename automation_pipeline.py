import sys
import random
import sqlite3
from pathlib import Path

import torch
from openpyxl import load_workbook
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# PATHS
# ============================================================

BASE_PATH = Path(__file__).parent

DATABASE_FOLDER = BASE_PATH / "database"
MODEL_B_FOLDER = BASE_PATH / "model_b"

ROBERTA_MODEL_PATH = BASE_PATH / "model"

FEEDBACK_DB_PATH = (
    DATABASE_FOLDER
    / "feedback.db"
)

DATASET_PATH = (
    BASE_PATH
    / "Model final dataset.xlsx"
)


# Allow imports from database/ and model_b/
sys.path.append(str(DATABASE_FOLDER))
sys.path.append(str(MODEL_B_FOLDER))


from adversarial_generator import generate_variants

from adversarial_database import (
    initialize_adversarial_database,
    save_adversarial_case
)

from live_model_b import (
    process_feedback_with_model_b
)


# ============================================================
# LABELS
# ============================================================

LABEL_MAP = {
    0: "Insult",
    1: "Neutral",
    2: "Racism",
    3: "Sexual Abuse",
    4: "Threat"
}


CATEGORY_TO_LABEL = {
    "Insult": 0,
    "Neutral": 1,
    "Racism": 2,
    "Sexual Abuse": 3,
    "Threat": 4
}


# ============================================================
# LOAD ROBERTA ONCE
# ============================================================

print(
    "Loading RoBERTa for automated feedback pipeline..."
)

tokenizer = AutoTokenizer.from_pretrained(
    ROBERTA_MODEL_PATH,
    local_files_only=True
)

roberta_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        ROBERTA_MODEL_PATH,
        local_files_only=True
    )
)

roberta_model.to("cpu")
roberta_model.eval()

print(
    "RoBERTa automation model loaded."
)


# ============================================================
# READ ONE CORRECTION FROM DB1
# ============================================================

def get_feedback_correction(feedback_id):

    conn = sqlite3.connect(
        FEEDBACK_DB_PATH
    )

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            sentence,
            correct_category
        FROM corrections
        WHERE id = ?
    """, (
        feedback_id,
    ))

    row = cursor.fetchone()

    conn.close()

    return row


# ============================================================
# ROBERTA PREDICTION
# ============================================================

def predict_roberta(text):

    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = roberta_model(
            **encoded
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )


    prediction_id = (
        prediction.item()
    )

    predicted_category = (
        LABEL_MAP[
            prediction_id
        ]
    )

    confidence_percent = (
        confidence.item()
        * 100
    )

    return (
        predicted_category,
        confidence_percent
    )


# ============================================================
# APPEND NEW FAIL CASES TO DATASET
# ============================================================

def append_failed_variants_to_dataset(
    feedback_id,
    model_b_results
):

    if not DATASET_PATH.exists():

        print(
            "Dataset not found:"
        )

        print(
            DATASET_PATH
        )

        return 0


    workbook = load_workbook(
        DATASET_PATH
    )

    sheet = workbook.active


    # --------------------------------------------------------
    # Existing sentences
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

        if sentence:

            existing_sentences.add(
                str(sentence)
                .strip()
                .lower()
            )


    added = 0


    # --------------------------------------------------------
    # Add only RoBERTa FAIL adversarial variants
    #
    # Do not add Original again.
    # --------------------------------------------------------

    for result in model_b_results:

        transformation = result[
            "transformation"
        ]

        roberta_result = result[
            "roberta_result"
        ]



        if roberta_result != "FAIL":
            continue


        sentence = (
            result["sentence"]
            .strip()
        )

        sentence_key = (
            sentence.lower()
        )


        if sentence_key in existing_sentences:
            continue


        correct_category = result[
            "correct_category"
        ]


        if (
            correct_category
            not in CATEGORY_TO_LABEL
        ):
            continue


        label = CATEGORY_TO_LABEL[
            correct_category
        ]


        # Column A = sentence
        # Column B = label
        sheet.append(
            [
                sentence,
                label
            ]
        )


        existing_sentences.add(
            sentence_key
        )

        added += 1


    workbook.save(
        DATASET_PATH
    )


    return added


# ============================================================
# MAIN AUTOMATED PIPELINE
# ============================================================

def run_feedback_pipeline(
    feedback_id
):

    random.seed(
        feedback_id
    )


    print()
    print("=" * 65)
    print(
        f"AUTOMATED PIPELINE - "
        f"FEEDBACK ID {feedback_id}"
    )
    print("=" * 65)


    # --------------------------------------------------------
    # STEP 1
    # Read corrected sentence from DB1
    # --------------------------------------------------------

    feedback = (
        get_feedback_correction(
            feedback_id
        )
    )


    if feedback is None:

        raise ValueError(
            f"Feedback ID {feedback_id} "
            "was not found in feedback.db."
        )


    (
        db_feedback_id,
        original_sentence,
        correct_category
    ) = feedback


    print(
        "Correction loaded from DB1."
    )


    # --------------------------------------------------------
    # STEP 2
    # Generate original + 4 variants
    # --------------------------------------------------------

    variants = generate_variants(
        original_sentence
    )


    print(
        "Adversarial variants generated."
    )


    # --------------------------------------------------------
    # STEP 3
    # RoBERTa tests all variants
    # --------------------------------------------------------

    initialize_adversarial_database()


    roberta_results = []


    for (
        transformation_type,
        generated_sentence
    ) in variants.items():


        if generated_sentence is None:
            continue


        (
            prediction,
            confidence
        ) = predict_roberta(
            generated_sentence
        )


        if (
            prediction
            == correct_category
        ):

            result = "PASS"

        else:

            result = "FAIL"


        save_adversarial_case(

            feedback_id=
                db_feedback_id,

            original_sentence=
                original_sentence,

            generated_sentence=
                generated_sentence,

            transformation_type=
                transformation_type,

            correct_category=
                correct_category,

            roberta_prediction=
                prediction,

            roberta_confidence=
                confidence,

            result=
                result
        )


        roberta_results.append({

            "transformation":
                transformation_type,

            "prediction":
                prediction,

            "confidence":
                confidence,

            "result":
                result
        })


    print(
        "RoBERTa results saved to DB2."
    )


    # --------------------------------------------------------
    # STEP 4
    # Model B predicts + receives reward/penalty
    # --------------------------------------------------------

    model_b_results = (
        process_feedback_with_model_b(
            db_feedback_id
        )
    )


    print(
        "Model B prediction and "
        "learning completed."
    )


    # --------------------------------------------------------
    # STEP 5
    # Add RoBERTa FAIL variants to dataset
    # --------------------------------------------------------

    added_to_dataset = (
        append_failed_variants_to_dataset(
            db_feedback_id,
            model_b_results
        )
    )


    print(
        "New FAIL variants added "
        f"to dataset: {added_to_dataset}"
    )


    # --------------------------------------------------------
    # CREATE SIMPLE RESULT FOR INTERFACE
    # --------------------------------------------------------

    total_cases = len(
        model_b_results
    )


    roberta_passes = sum(
        1
        for item in model_b_results
        if item["roberta_result"]
        == "PASS"
    )


    roberta_fails = (
        total_cases
        - roberta_passes
    )


    model_b_rewards = sum(
        1
        for item in model_b_results
        if item["reward"] == 1.0
    )


    model_b_penalties = (
        total_cases
        - model_b_rewards
    )


    summary = {

        "feedback_id":
            db_feedback_id,

        "correct_category":
            correct_category,

        "total_cases":
            total_cases,

        "roberta_passes":
            roberta_passes,

        "roberta_fails":
            roberta_fails,

        "model_b_rewards":
            model_b_rewards,

        "model_b_penalties":
            model_b_penalties,

        "added_to_dataset":
            added_to_dataset,

        "details":
            model_b_results
    }


    print()
    print(
        "Automated feedback pipeline completed."
    )


    return summary