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
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# PATHS
# ============================================================

BASE_PATH = Path(__file__).parent.parent

DATABASE_PATH = (
    BASE_PATH
    / "database"
    / "adversarial.db"
)

MODEL_B_PATH = Path(__file__).parent

MODEL_B_PATH.mkdir(
    exist_ok=True
)


# ============================================================
# LABELS
# ============================================================

LABEL_TO_ID = {
    "Insult": 0,
    "Neutral": 1,
    "Racism": 2,
    "Sexual Abuse": 3,
    "Threat": 4
}

ID_TO_LABEL = {
    value: key
    for key, value
    in LABEL_TO_ID.items()
}


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

EPOCHS = 100

TEST_SIZE = 0.20

INPUT_SIZE = 2048

LEARNING_RATE = 0.005


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(2)


# ============================================================
# LOAD CASES FROM DB2
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
# GROUPED TRAIN / TEST SPLIT
# ============================================================

def grouped_split(cases):

    indices = np.arange(
        len(cases)
    )

    groups = np.array([
        case["feedback_id"]
        for case in cases
    ])

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    train_indices, test_indices = next(
        splitter.split(
            indices,
            groups=groups
        )
    )

    train_cases = [
        cases[i]
        for i in train_indices
    ]

    test_cases = [
        cases[i]
        for i in test_indices
    ]

    return (
        train_cases,
        test_cases
    )


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

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        return probabilities


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate(model, cases):

    model.eval()

    correct = 0

    predictions = []


    with torch.no_grad():

        for case in cases:

            x = vectorize(
                case["sentence"]
            )

            probabilities = model(x)

            predicted_id = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = torch.max(
                probabilities
            ).item() * 100


            if predicted_id == case["label"]:

                correct += 1
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
                    ID_TO_LABEL[predicted_id],

                "confidence":
                    confidence,

                "result":
                    result
            })


    total = len(cases)

    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )


    return (
        accuracy,
        correct,
        predictions
    )


# ============================================================
# TRANSFORMATION RESULTS
# ============================================================

def transformation_results(predictions):

    stats = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0
        }
    )


    for item in predictions:

        transformation = item[
            "transformation"
        ]

        stats[
            transformation
        ]["total"] += 1

        if item["result"] == "PASS":

            stats[
                transformation
            ]["correct"] += 1


    results = {}


    for transformation, values in stats.items():

        correct = values["correct"]

        total = values["total"]

        accuracy = (
            correct / total * 100
            if total > 0
            else 0
        )


        results[transformation] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        }


    return results


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def class_distribution(cases):

    counter = Counter(
        case["category"]
        for case in cases
    )

    return dict(counter)


# ============================================================
# TRAINING
# ============================================================

def train():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    all_cases = load_cases()


    print(
        "Total cases loaded from DB2:",
        len(all_cases)
    )


    if len(all_cases) == 0:

        print(
            "No adversarial cases found."
        )

        return


    # --------------------------------------------------------
    # GROUPED SPLIT
    # --------------------------------------------------------

    train_cases, test_cases = grouped_split(
        all_cases
    )


    train_feedback_ids = sorted(
        set(
            case["feedback_id"]
            for case in train_cases
        )
    )


    test_feedback_ids = sorted(
        set(
            case["feedback_id"]
            for case in test_cases
        )
    )


    print()
    print("=" * 65)
    print("GROUPED DATA SPLIT")
    print("=" * 65)


    print(
        "Training source sentences:",
        len(train_feedback_ids)
    )

    print(
        "Testing source sentences:",
        len(test_feedback_ids)
    )

    print(
        "Training cases:",
        len(train_cases)
    )

    print(
        "Testing cases:",
        len(test_cases)
    )


    print()
    print(
        "Training feedback IDs:",
        train_feedback_ids
    )

    print(
        "Testing feedback IDs:",
        test_feedback_ids
    )


    print()
    print(
        "Training class distribution:",
        class_distribution(
            train_cases
        )
    )

    print(
        "Testing class distribution:",
        class_distribution(
            test_cases
        )
    )


    # --------------------------------------------------------
    # CHECK FOR LEAKAGE
    # --------------------------------------------------------

    overlap = (
        set(train_feedback_ids)
        &
        set(test_feedback_ids)
    )


    if overlap:

        raise RuntimeError(
            "DATA LEAKAGE DETECTED: "
            "feedback IDs appear in "
            "both train and test sets."
        )


    print()
    print(
        "Group leakage check: PASSED"
    )

    print(
        "No source sentence appears "
        "in both training and testing."
    )


    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = ContextualBandit()


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # --------------------------------------------------------
    # STARTING ACCURACY
    # --------------------------------------------------------

    starting_train_accuracy, _, _ = (
        evaluate(
            model,
            train_cases
        )
    )


    starting_test_accuracy, _, _ = (
        evaluate(
            model,
            test_cases
        )
    )


    print()
    print("=" * 65)
    print("BEFORE REWARD LEARNING")
    print("=" * 65)


    print(
        f"Starting training accuracy: "
        f"{starting_train_accuracy:.2f}%"
    )

    print(
        f"Starting unseen test accuracy: "
        f"{starting_test_accuracy:.2f}%"
    )


    print()
    print(
        "Starting reward-driven learning..."
    )

    print()


    # --------------------------------------------------------
    # TRAIN ONLY ON TRAINING GROUPS
    # --------------------------------------------------------

    history = []


    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()


        order = list(
            range(
                len(train_cases)
            )
        )

        random.shuffle(order)


        total_reward = 0


        for index in order:

            case = train_cases[
                index
            ]


            x = vectorize(
                case["sentence"]
            )


            true_label = case[
                "label"
            ]


            probabilities = model(
                x
            )


            distribution = Categorical(
                probabilities
            )


            # Model B chooses one category
            action = distribution.sample()


            # --------------------------------------------
            # REWARD
            # --------------------------------------------

            if action.item() == true_label:

                reward = 1.0

            else:

                reward = -1.0


            total_reward += reward


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


        # ----------------------------------------------------
        # Measure training progress only
        # ----------------------------------------------------

        train_accuracy, _, _ = evaluate(
            model,
            train_cases
        )


        average_reward = (
            total_reward
            / len(train_cases)
        )


        history.append({
            "epoch":
                epoch,

            "total_reward":
                total_reward,

            "average_reward":
                average_reward,

            "training_accuracy":
                train_accuracy
        })


        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"Epoch {epoch:3d} | "
                f"Reward: "
                f"{total_reward:6.1f} | "
                f"Avg reward: "
                f"{average_reward:6.3f} | "
                f"Train accuracy: "
                f"{train_accuracy:6.2f}%"
            )


    # ========================================================
    # FINAL TRAINING PERFORMANCE
    # ========================================================

    (
        final_train_accuracy,
        train_correct,
        _
    ) = evaluate(
        model,
        train_cases
    )


    # ========================================================
    # FINAL UNSEEN TEST
    # ========================================================

    (
        final_test_accuracy,
        test_correct,
        test_predictions
    ) = evaluate(
        model,
        test_cases
    )


    test_transformation_results = (
        transformation_results(
            test_predictions
        )
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 65)
    print("MODEL B GROUPED EVALUATION")
    print("=" * 65)


    print(
        f"Starting training accuracy: "
        f"{starting_train_accuracy:.2f}%"
    )

    print(
        f"Final training accuracy: "
        f"{final_train_accuracy:.2f}%"
    )


    print()

    print(
        f"Unseen test cases: "
        f"{len(test_cases)}"
    )

    print(
        f"Correct unseen predictions: "
        f"{test_correct}"
    )

    print(
        f"Final unseen test accuracy: "
        f"{final_test_accuracy:.2f}%"
    )


    print()
    print(
        "Unseen test results "
        "by transformation:"
    )

    print()


    transformation_order = [
        "Original",
        "Leetspeak",
        "Spacing",
        "Punctuation",
        "Typo"
    ]


    for transformation in transformation_order:

        if (
            transformation
            not in
            test_transformation_results
        ):

            continue


        result = (
            test_transformation_results[
                transformation
            ]
        )


        print(
            f"{transformation}: "
            f'{result["correct"]}/'
            f'{result["total"]} correct | '
            f'{result["accuracy"]:.2f}%'
        )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_file = (
        MODEL_B_PATH
        / "contextual_bandit_grouped.pt"
    )


    torch.save(
        model.state_dict(),
        model_file
    )


    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    history_file = (
        MODEL_B_PATH
        / "grouped_training_history.json"
    )


    with open(
        history_file,
        "w"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )


    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    predictions_file = (
        MODEL_B_PATH
        / "grouped_test_predictions.json"
    )


    with open(
        predictions_file,
        "w"
    ) as file:

        json.dump(
            test_predictions,
            file,
            indent=4
        )


    # ========================================================
    # SAVE COMPLETE EVALUATION SUMMARY
    # ========================================================

    evaluation_summary = {

        "method":
            "Grouped train-test split by feedback_id",

        "random_seed":
            SEED,

        "total_source_sentences":
            len(
                set(
                    case["feedback_id"]
                    for case in all_cases
                )
            ),

        "total_cases":
            len(all_cases),

        "training_source_sentences":
            len(train_feedback_ids),

        "testing_source_sentences":
            len(test_feedback_ids),

        "training_feedback_ids":
            train_feedback_ids,

        "testing_feedback_ids":
            test_feedback_ids,

        "training_cases":
            len(train_cases),

        "testing_cases":
            len(test_cases),

        "training_class_distribution":
            class_distribution(
                train_cases
            ),

        "testing_class_distribution":
            class_distribution(
                test_cases
            ),

        "starting_training_accuracy":
            starting_train_accuracy,

        "starting_unseen_test_accuracy":
            starting_test_accuracy,

        "final_training_accuracy":
            final_train_accuracy,

        "final_unseen_test_accuracy":
            final_test_accuracy,

        "correct_unseen_predictions":
            test_correct,

        "test_transformation_results":
            test_transformation_results,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "reward_correct":
            1,

        "penalty_incorrect":
            -1,

        "input_features":
            INPUT_SIZE
    }


    summary_file = (
        MODEL_B_PATH
        / "grouped_evaluation_summary.json"
    )


    with open(
        summary_file,
        "w"
    ) as file:

        json.dump(
            evaluation_summary,
            file,
            indent=4
        )


    # ========================================================
    # SAVE LABEL MAP
    # ========================================================

    labels_file = (
        MODEL_B_PATH
        / "labels.json"
    )


    with open(
        labels_file,
        "w"
    ) as file:

        json.dump(
            ID_TO_LABEL,
            file,
            indent=4
        )


    print()
    print("=" * 65)

    print(
        "Grouped Model B saved to:"
    )

    print(
        model_file
    )


    print()
    print(
        "Evaluation summary saved to:"
    )

    print(
        summary_file
    )


    print(
        "Test predictions saved."
    )

    print(
        "Training history saved."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    train()