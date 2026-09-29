import random
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from read_feedback import get_verified_corrections
from adversarial_generator import generate_variants
from adversarial_database import (
    initialize_adversarial_database,
    save_adversarial_case
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent.parent / "model"

MAX_LENGTH = 128
BATCH_SIZE = 8

# Final project label mapping
LABEL_MAP = {
    0: "Insult",
    1: "Neutral",
    2: "Racism",
    3: "Sexual Abuse",
    4: "Threat"
}


# --------------------------------------------------
# LOAD ROBERTA
# --------------------------------------------------

def load_roberta():

    print("Loading RoBERTa from:")
    print(MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    # Explicitly use CPU
    device = torch.device("cpu")

    model.to(device)
    model.eval()

    print("RoBERTa loaded successfully.")
    print("Device: CPU")

    return tokenizer, model, device


# --------------------------------------------------
# PREDICT A BATCH
# --------------------------------------------------

def predict_batch(
    texts,
    tokenizer,
    model,
    device
):

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        outputs = model(**encoded)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidences, predictions = torch.max(
            probabilities,
            dim=1
        )

    results = []

    for prediction, confidence in zip(
        predictions,
        confidences
    ):

        label_id = prediction.item()

        predicted_category = LABEL_MAP[
            label_id
        ]

        # Store confidence as percentage
        confidence_percent = (
            confidence.item() * 100
        )

        results.append(
            (
                predicted_category,
                confidence_percent
            )
        )

    return results


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def run_tests():

    random.seed(42)
    torch.manual_seed(42)

    initialize_adversarial_database()

    tokenizer, model, device = load_roberta()

    corrections = get_verified_corrections()

    print()
    print(
        "Verified corrections:",
        len(corrections)
    )

    # ----------------------------------------------
    # Generate all test cases first
    # ----------------------------------------------

    all_cases = []

    for (
        feedback_id,
        sentence,
        correct_category
    ) in corrections:

        variants = generate_variants(
            sentence
        )

        for (
            transformation_type,
            generated_sentence
        ) in variants.items():

            if generated_sentence is None:
                continue

            all_cases.append({
                "feedback_id": feedback_id,
                "original_sentence": sentence,
                "generated_sentence": generated_sentence,
                "transformation_type": transformation_type,
                "correct_category": correct_category
            })

    print(
        "Total test cases generated:",
        len(all_cases)
    )

    print()
    print("Starting RoBERTa testing...")
    print()


    # ----------------------------------------------
    # Statistics
    # ----------------------------------------------

    total_pass = 0
    total_fail = 0

    transformation_stats = defaultdict(
        lambda: {
            "pass": 0,
            "fail": 0
        }
    )


    # ----------------------------------------------
    # Process in CPU-friendly batches
    # ----------------------------------------------

    for start in range(
        0,
        len(all_cases),
        BATCH_SIZE
    ):

        batch_cases = all_cases[
            start:start + BATCH_SIZE
        ]

        batch_texts = [
            case["generated_sentence"]
            for case in batch_cases
        ]

        predictions = predict_batch(
            batch_texts,
            tokenizer,
            model,
            device
        )

        for case, prediction_data in zip(
            batch_cases,
            predictions
        ):

            (
                predicted_category,
                confidence
            ) = prediction_data

            correct_category = case[
                "correct_category"
            ]

            if predicted_category == correct_category:

                result = "PASS"
                total_pass += 1

                transformation_stats[
                    case["transformation_type"]
                ]["pass"] += 1

            else:

                result = "FAIL"
                total_fail += 1

                transformation_stats[
                    case["transformation_type"]
                ]["fail"] += 1


            # --------------------------------------
            # Save result into adversarial.db
            # --------------------------------------

            save_adversarial_case(
                feedback_id=case["feedback_id"],
                original_sentence=case[
                    "original_sentence"
                ],
                generated_sentence=case[
                    "generated_sentence"
                ],
                transformation_type=case[
                    "transformation_type"
                ],
                correct_category=correct_category,
                roberta_prediction=predicted_category,
                roberta_confidence=confidence,
                result=result
            )


            # --------------------------------------
            # Print individual result
            # --------------------------------------

            print(
                f'ID {case["feedback_id"]} | '
                f'{case["transformation_type"]} | '
                f'True: {correct_category} | '
                f'Predicted: {predicted_category} | '
                f'Confidence: {confidence:.2f}% | '
                f'{result}'
            )


    # ----------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------

    total = total_pass + total_fail

    if total > 0:

        accuracy = (
            total_pass / total
        ) * 100

    else:
        accuracy = 0


    print()
    print("=" * 70)
    print("ROBERTA ADVERSARIAL TEST SUMMARY")
    print("=" * 70)

    print(
        "Total cases:",
        total
    )

    print(
        "PASS:",
        total_pass
    )

    print(
        "FAIL:",
        total_fail
    )

    print(
        f"Overall accuracy: {accuracy:.2f}%"
    )


    print()
    print("Results by transformation:")
    print()


    for transformation in [
        "Original",
        "Leetspeak",
        "Spacing",
        "Punctuation",
        "Typo"
    ]:

        passed = transformation_stats[
            transformation
        ]["pass"]

        failed = transformation_stats[
            transformation
        ]["fail"]

        transformation_total = (
            passed + failed
        )

        if transformation_total > 0:

            transformation_accuracy = (
                passed /
                transformation_total
            ) * 100

        else:
            transformation_accuracy = 0


        print(
            f"{transformation}: "
            f"{passed} PASS | "
            f"{failed} FAIL | "
            f"{transformation_accuracy:.2f}%"
        )


    print()
    print(
        "All results saved to "
        "database/adversarial.db"
    )


# --------------------------------------------------
# START
# --------------------------------------------------

if __name__ == "__main__":
    run_tests()