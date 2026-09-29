import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from sklearn.feature_extraction.text import HashingVectorizer


# ============================================================
# PATHS
# ============================================================

BASE_PATH = Path(__file__).parent.parent

DATABASE_FOLDER = BASE_PATH / "database"

sys.path.append(
    str(DATABASE_FOLDER)
)

from adversarial_database import (
    get_cases_for_feedback,
    save_model_b_result
)


MODEL_FOLDER = Path(__file__).parent


# Original evaluated Model B
# This file will NEVER be overwritten by the live system.
BASE_MODEL_PATH = (
    MODEL_FOLDER
    / "contextual_bandit_grouped.pt"
)


# Continuously learning Model B
LIVE_MODEL_PATH = (
    MODEL_FOLDER
    / "contextual_bandit_live.pt"
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
    0: "Insult",
    1: "Neutral",
    2: "Racism",
    3: "Sexual Abuse",
    4: "Threat"
}


# ============================================================
# TEXT FEATURES
# ============================================================

INPUT_SIZE = 2048


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
# LOAD MODEL B
# ============================================================

def load_model_b():

    model = ContextualBandit()


    # --------------------------------------------------------
    # If the live model already exists,
    # continue learning from it.
    # --------------------------------------------------------

    if LIVE_MODEL_PATH.exists():

        state_dict = torch.load(
            LIVE_MODEL_PATH,
            map_location="cpu"
        )

        model.load_state_dict(
            state_dict
        )

        print(
            "Loaded existing live Model B."
        )


    # --------------------------------------------------------
    # First ever live run:
    # start from evaluated grouped Model B.
    # --------------------------------------------------------

    elif BASE_MODEL_PATH.exists():

        state_dict = torch.load(
            BASE_MODEL_PATH,
            map_location="cpu"
        )

        model.load_state_dict(
            state_dict
        )

        print(
            "Live Model B initialized "
            "from grouped Model B."
        )


    # --------------------------------------------------------
    # Fallback if neither model exists
    # --------------------------------------------------------

    else:

        print(
            "WARNING: No saved Model B found."
        )

        print(
            "Starting Model B with "
            "random weights."
        )


    model.train()

    return model


# ============================================================
# PROCESS ONE FEEDBACK REPORT
# ============================================================

def process_feedback_with_model_b(
    feedback_id
):

    # --------------------------------------------------------
    # Get the 5 cases created for this correction:
    #
    # Original
    # Leetspeak
    # Spacing
    # Punctuation
    # Typo
    # --------------------------------------------------------

    cases = get_cases_for_feedback(
        feedback_id
    )


    if not cases:

        print(
            "No adversarial cases found "
            f"for feedback ID {feedback_id}."
        )

        return []


    model = load_model_b()


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.005
    )


    results = []


    # ========================================================
    # PROCESS EACH CASE
    # ========================================================

    for row in cases:

        (
            case_id,
            db_feedback_id,
            original_sentence,
            generated_sentence,
            transformation_type,
            correct_category,

            roberta_prediction,
            roberta_confidence,
            roberta_result,

            old_model_b_prediction,
            old_model_b_confidence,
            old_model_b_reward

        ) = row


        # ----------------------------------------------------
        # Convert sentence to numeric representation
        # ----------------------------------------------------

        x = vectorize(
            generated_sentence
        )


        # ----------------------------------------------------
        # MODEL B PREDICTION
        #
        # Prediction is made BEFORE learning from this case.
        # ----------------------------------------------------

        probabilities = model(x)


        prediction_id = torch.argmax(
            probabilities,
            dim=1
        ).item()


        confidence = (
            torch.max(
                probabilities
            ).item()
            * 100
        )


        prediction_category = (
            ID_TO_LABEL[
                prediction_id
            ]
        )


        true_label = (
            LABEL_TO_ID[
                correct_category
            ]
        )


        # ----------------------------------------------------
        # REWARD / PENALTY
        # ----------------------------------------------------

        if prediction_id == true_label:

            reward = 1.0

        else:

            reward = -1.0


        # ----------------------------------------------------
        # POLICY UPDATE
        # ----------------------------------------------------

        distribution = Categorical(
            probabilities
        )


        action_tensor = torch.tensor(
            prediction_id,
            dtype=torch.long
        )


        log_probability = (
            distribution.log_prob(
                action_tensor
            )
        )


        entropy = (
            distribution.entropy()
        )


        reward_tensor = torch.tensor(
            reward,
            dtype=torch.float32
        )


        loss = (
            -log_probability
            * reward_tensor
            - 0.01
            * entropy
        ).mean()


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # SAVE MODEL B RESULT TO DB2
        # ----------------------------------------------------

        save_model_b_result(
            feedback_id=db_feedback_id,
            transformation_type=transformation_type,
            prediction=prediction_category,
            confidence=confidence,
            reward=reward
        )


        # ----------------------------------------------------
        # RETURNABLE RESULT
        # ----------------------------------------------------

        results.append({

            "transformation":
                transformation_type,

            "sentence":
                generated_sentence,

            "correct_category":
                correct_category,

            "roberta_prediction":
                roberta_prediction,

            "roberta_confidence":
                roberta_confidence,

            "roberta_result":
                roberta_result,

            "model_b_prediction":
                prediction_category,

            "model_b_confidence":
                confidence,

            "reward":
                reward
        })


    # ========================================================
    # SAVE CONTINUALLY UPDATED LIVE MODEL
    # ========================================================

    torch.save(
        model.state_dict(),
        LIVE_MODEL_PATH
    )


    print(
        f"Model B processed "
        f"{len(results)} cases."
    )

    print(
        "Live Model B updated and saved."
    )


    return results


# ============================================================
# OPTIONAL MANUAL TEST
# ============================================================

if __name__ == "__main__":

    feedback_id = input(
        "Enter feedback ID: "
    )


    results = (
        process_feedback_with_model_b(
            int(feedback_id)
        )
    )


    print()


    for result in results:

        print(
            result["transformation"],
            "| RoBERTa:",
            result["roberta_prediction"],
            "| Model B:",
            result["model_b_prediction"],
            "| Reward:",
            result["reward"]
        )