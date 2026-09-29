import sqlite3
import random
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from sklearn.feature_extraction.text import HashingVectorizer


# ============================================================
# PATHS
# ============================================================

BASE_PATH = Path(__file__).parent.parent

DATABASE_PATH = (
    BASE_PATH
    / "database"
    / "adversarial.db"
)

OUTPUT_PATH = Path(__file__).parent


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
EPOCHS = 100
LEARNING_RATE = 0.005
INPUT_SIZE = 2048


LABEL_TO_ID = {
    "Insult": 0,
    "Neutral": 1,
    "Racism": 2,
    "Sexual Abuse": 3,
    "Threat": 4
}

ID_TO_LABEL = {
    value: key
    for key, value in LABEL_TO_ID.items()
}


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(2)


# ============================================================
# TEXT REPRESENTATION
# ============================================================

vectorizer = HashingVectorizer(
    n_features=INPUT_SIZE,
    alternate_sign=False,
    norm="l2",
    ngram_range=(1, 2)
)


def vectorize(text):

    features = vectorizer.transform(
        [text]
    )

    dense = features.toarray().astype(
        np.float32
    )

    return torch.tensor(
        dense,
        dtype=torch.float32
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_cases():

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            feedback_id,
            generated_sentence,
            transformation_type,
            correct_category
        FROM adversarial_cases
        ORDER BY feedback_id, id
    """)

    rows = cursor.fetchall()

    conn.close()

    cases = []

    for (
        feedback_id,
        sentence,
        transformation,
        category
    ) in rows:

        if category not in LABEL_TO_ID:
            continue

        cases.append({
            "feedback_id": feedback_id,
            "sentence": sentence,
            "transformation": transformation,
            "category": category,
            "label": LABEL_TO_ID[category]
        })

    return cases


# ============================================================
# MODEL B
# ============================================================

class ContextualBandit(nn.Module):

    def __init__(
        self,
        input_size=INPUT_SIZE,
        num_actions=5
    ):

        super().__init__()

        self.policy = nn.Linear(
            input_size,
            num_actions
        )


    def forward(self, x):

        logits = self.policy(x)

        return torch.softmax(
            logits,
            dim=1
        )


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_fold(train_cases, fold_seed):

    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)

    model = ContextualBandit()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    for epoch in range(EPOCHS):

        model.train()

        order = list(
            range(len(train_cases))
        )

        random.shuffle(order)


        for index in order:

            case = train_cases[index]

            x = vectorize(
                case["sentence"]
            )

            probabilities = model(x)

            distribution = Categorical(
                probabilities
            )

            action = distribution.sample()


            if action.item() == case["label"]:
                reward = 1.0
            else:
                reward = -1.0


            reward_tensor = torch.tensor(
                reward,
                dtype=torch.float32
            )

            log_probability = (
                distribution.log_prob(
                    action
                )
            )

            entropy = (
                distribution.entropy()
            )

            loss = (
                -log_probability
                * reward_tensor
                - 0.01 * entropy
            ).mean()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    return model


# ============================================================
# TEST ONE FOLD
# ============================================================

def test_fold(
    model,
    test_cases
):

    model.eval()

    predictions = []


    with torch.no_grad():

        for case in test_cases:

            x = vectorize(
                case["sentence"]
            )

            probabilities = model(x)

            prediction_id = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = (
                torch.max(probabilities).item()
                * 100
            )


            predicted_category = (
                ID_TO_LABEL[prediction_id]
            )


            if prediction_id == case["label"]:
                result = "PASS"
            else:
                result = "FAIL"


            predictions.append({
                "feedback_id":
                    case["feedback_id"],

                "sentence":
                    case["sentence"],

                "transformation":
                    case["transformation"],

                "true_category":
                    case["category"],

                "predicted_category":
                    predicted_category,

                "confidence":
                    confidence,

                "result":
                    result
            })


    return predictions


# ============================================================
# MAIN LOGO EVALUATION
# ============================================================

def evaluate_logo():

    cases = load_cases()

    feedback_ids = sorted(
        set(
            case["feedback_id"]
            for case in cases
        )
    )


    print(
        "Total adversarial cases:",
        len(cases)
    )

    print(
        "Total source sentences:",
        len(feedback_ids)
    )

    print()

    print("=" * 70)
    print(
        "LEAVE-ONE-FEEDBACK-SOURCE-OUT EVALUATION"
    )
    print("=" * 70)


    all_predictions = []


    # --------------------------------------------------------
    # One source sentence held out at a time
    # --------------------------------------------------------

    for fold_number, held_out_id in enumerate(
        feedback_ids,
        start=1
    ):

        train_cases = [
            case
            for case in cases
            if case["feedback_id"]
            != held_out_id
        ]


        test_cases = [
            case
            for case in cases
            if case["feedback_id"]
            == held_out_id
        ]


        true_category = (
            test_cases[0]["category"]
        )


        model = train_fold(
            train_cases,
            SEED + fold_number
        )


        fold_predictions = test_fold(
            model,
            test_cases
        )


        correct = sum(
            1
            for item in fold_predictions
            if item["result"] == "PASS"
        )


        total = len(
            fold_predictions
        )


        accuracy = (
            correct / total * 100
        )


        print(
            f"Fold {fold_number:2d} | "
            f"Feedback ID {held_out_id:2d} | "
            f"True: {true_category:12s} | "
            f"{correct}/{total} correct | "
            f"{accuracy:6.2f}%"
        )


        all_predictions.extend(
            fold_predictions
        )


    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    total_correct = sum(
        1
        for item in all_predictions
        if item["result"] == "PASS"
    )


    total_predictions = len(
        all_predictions
    )


    overall_accuracy = (
        total_correct
        / total_predictions
        * 100
    )


    # ========================================================
    # RESULTS BY TRANSFORMATION
    # ========================================================

    transformation_stats = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0
        }
    )


    for item in all_predictions:

        transformation = item[
            "transformation"
        ]

        transformation_stats[
            transformation
        ]["total"] += 1


        if item["result"] == "PASS":

            transformation_stats[
                transformation
            ]["correct"] += 1


    # ========================================================
    # RESULTS BY TRUE CLASS
    # ========================================================

    class_stats = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0
        }
    )


    for item in all_predictions:

        category = item[
            "true_category"
        ]

        class_stats[
            category
        ]["total"] += 1


        if item["result"] == "PASS":

            class_stats[
                category
            ]["correct"] += 1


    # ========================================================
    # MAJORITY BASELINE
    # ========================================================

    overall_class_counts = Counter(
        case["category"]
        for case in cases
    )


    majority_category = (
        overall_class_counts
        .most_common(1)[0][0]
    )


    majority_correct = sum(
        1
        for case in cases
        if case["category"]
        == majority_category
    )


    majority_accuracy = (
        majority_correct
        / len(cases)
        * 100
    )


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL MODEL B LOGO RESULTS")
    print("=" * 70)


    print(
        "Total unseen predictions:",
        total_predictions
    )

    print(
        "Correct predictions:",
        total_correct
    )

    print(
        f"Unseen-source accuracy: "
        f"{overall_accuracy:.2f}%"
    )


    print()
    print(
        f"Majority-class baseline "
        f"({majority_category}): "
        f"{majority_accuracy:.2f}%"
    )


    print()
    print(
        "Results by transformation:"
    )


    transformation_results = {}


    for transformation in [
        "Original",
        "Leetspeak",
        "Spacing",
        "Punctuation",
        "Typo"
    ]:

        values = (
            transformation_stats[
                transformation
            ]
        )

        correct = values["correct"]
        total = values["total"]


        accuracy = (
            correct / total * 100
            if total > 0
            else 0
        )


        transformation_results[
            transformation
        ] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        }


        print(
            f"{transformation}: "
            f"{correct}/{total} | "
            f"{accuracy:.2f}%"
        )


    print()
    print(
        "Results by class:"
    )


    class_results = {}


    for category in LABEL_TO_ID.keys():

        values = class_stats[
            category
        ]

        correct = values["correct"]

        total = values["total"]


        accuracy = (
            correct / total * 100
            if total > 0
            else None
        )


        class_results[
            category
        ] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        }


        if total == 0:

            print(
                f"{category}: "
                "No feedback-derived cases"
            )

        else:

            print(
                f"{category}: "
                f"{correct}/{total} | "
                f"{accuracy:.2f}%"
            )


    # ========================================================
    # SAVE REPORTABLE RESULTS
    # ========================================================

    summary = {

        "evaluation_method":
            "Leave-One-Feedback-Source-Out",

        "total_source_sentences":
            len(feedback_ids),

        "total_cases":
            len(cases),

        "total_unseen_predictions":
            total_predictions,

        "correct_unseen_predictions":
            total_correct,

        "unseen_source_accuracy":
            overall_accuracy,

        "majority_baseline_category":
            majority_category,

        "majority_baseline_accuracy":
            majority_accuracy,

        "transformation_results":
            transformation_results,

        "class_results":
            class_results,

        "epochs_per_fold":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "input_features":
            INPUT_SIZE,

        "reward_correct":
            1,

        "penalty_incorrect":
            -1,

        "random_seed_base":
            SEED
    }


    with open(
        OUTPUT_PATH
        / "logo_evaluation_summary.json",
        "w"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )


    with open(
        OUTPUT_PATH
        / "logo_predictions.json",
        "w"
    ) as file:

        json.dump(
            all_predictions,
            file,
            indent=4
        )


    print()
    print(
        "LOGO evaluation summary saved."
    )

    print(
        "All unseen predictions saved."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    evaluate_logo()