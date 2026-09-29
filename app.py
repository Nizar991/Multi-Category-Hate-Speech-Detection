import torch
import gradio as gr
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

from database.database import initialize_database, save_correction
from database.statistics import get_statistics
from database.statistics import get_confusion_statistics
from database.statistics import get_correction_history
from database.statistics import get_correction_chart_data
from automation_pipeline import run_feedback_pipeline

MODEL_PATH = "./model"

LABELS = {
    0: "Insult",
    1: "Neutral",
    2: "Racism",
    3: "Sexual Abuse",
    4: "Threat"
}


CSS_PATH = Path("styles/style.css")

with open(CSS_PATH, "r", encoding="utf-8") as f:
    custom_css = f.read()


initialize_database()


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

device = torch.device("cpu")

model.to(device)
model.eval()

print("\nModel loaded successfully!")
print("Device:", device)
print("Number of labels:", model.config.num_labels)
print("Labels:", LABELS)

def predict(sentence):

    if not sentence or not sentence.strip():
        return (
            "Please enter a sentence.",
            "0.00%",
            "No prediction available."
        )

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    category = LABELS[predicted_class]

    explanation = (
        f"The model classified this sentence as "
        f"**{category}** with "
        f"**{confidence:.2%} confidence**."
    )

    return (
        category,
        f"{confidence:.2%}",
        explanation
    )


def show_feedback_panel():
    return gr.update(visible=True)


def hide_feedback_panel():
    return gr.update(visible=False)


def show_correction_fields(feedback_type):

    if feedback_type == "Incorrect prediction":

        return (
            gr.update(visible=True),
            gr.update(visible=True)
        )

    return (
        gr.update(visible=False),
        gr.update(visible=False)
    )


def submit_feedback(
    feedback_type,
    correct_category,
    sentence,
    predicted_category,
    confidence
):

    if feedback_type == "Incorrect prediction":

        if not sentence or not sentence.strip():
            return (
                "Please analyse a sentence before "
                "submitting a correction."
            )

        if not correct_category:
            return "Please select the correct category."

        try:
            confidence_value = float(
                confidence.replace("%", "")
            ) / 100

            feedback_id = save_correction(
                sentence=sentence,
                predicted_category=predicted_category,
                correct_category=correct_category,
                confidence=confidence_value
            )

  
            pipeline_result = run_feedback_pipeline(
                feedback_id
            )

    
            model_b_rows = ""

            for result in pipeline_result["details"]:

                if result["reward"] == 1.0:
                    reward_text = "+1 ✅"
                else:
                    reward_text = "-1 ❌"

                model_b_rows += (
                    f"| {result['transformation']} | "
                    f"{result['roberta_prediction']} | "
                    f"{result['roberta_result']} | "
                    f"{result['model_b_prediction']} | "
                    f"{reward_text} |\n"
                )

            return f"""
✅ **Correction saved and automated learning completed.**

**Feedback ID:** {feedback_id}

**Correct category:** {correct_category}

**RoBERTa:**  
{pipeline_result["roberta_passes"]} PASS / {pipeline_result["roberta_fails"]} FAIL

**Model B:**  
{pipeline_result["model_b_rewards"]} rewards / {pipeline_result["model_b_penalties"]} penalties

**New adversarial sentences added to training dataset:**  
{pipeline_result["added_to_dataset"]}

| Version | RoBERTa | Result | Model B | Reward |
|---|---|---|---|---|
{model_b_rows}
"""

        except Exception as error:
            return (
                f"❌ Automated feedback pipeline error: "
                f"{error}"
            )

    elif feedback_type == "Other issue":
        return "Thank you for your feedback."

    return "Please select a feedback type."


def display_statistics():
    
    stats = get_statistics()
    confusion_data = get_confusion_statistics()
    history_data = get_correction_history()

    total = stats["total_corrections"]

    most_corrected = stats["most_corrected"]

    if most_corrected:
        most_corrected_text = (
            f"{most_corrected[0]} "
            f"({most_corrected[1]} corrections)"
        )
    else:
        most_corrected_text = "No corrections yet."


    prediction_counts = stats[
        "corrections_by_prediction"
    ]

    counts = {
        category: 0
        for category in LABELS.values()
    }

    for category, count in prediction_counts:

        if category in counts:
            counts[category] = count


    confusion_rows = ""

    for predicted, correct, count in confusion_data:

        confusion_rows += (
            f"| {predicted} | "
            f"{correct} | "
            f"{count} |\n"
        )

    if not confusion_rows:

        confusion_rows = (
            "| No corrections yet | - | 0 |\n"
        )


    history_rows = ""

    for (
        correction_id,
        sentence,
        predicted,
        correct,
        confidence,
        created_at
    ) in history_data:

        history_rows += (
            f"| {correction_id} | "
            f"{sentence} | "
            f"{predicted} | "
            f"{correct} | "
            f"{confidence:.2%} | "
            f"{created_at} |\n"
        )

    if not history_rows:

        history_rows = (
            "| - | No corrections yet | - | - | - | - |\n"
        )


    return f"""
### 📊 Correction Statistics

**Total corrections:** {total}

**Most frequently corrected prediction:**

{most_corrected_text}

#### Corrections by predicted category

| Category | Corrections |
|---|---:|
| Insult | {counts["Insult"]} |
| Neutral | {counts["Neutral"]} |
| Racism | {counts["Racism"]} |
| Sexual Abuse | {counts["Sexual Abuse"]} |
| Threat | {counts["Threat"]} |

#### 🔄 Prediction → Correct Category

| Model predicted | Correct category | Corrections |
|---|---|---:|
{confusion_rows}

#### 📝 Correction History

| ID | Sentence | Model predicted | Correct | Confidence | Date |
|---:|---|---|---|---:|---|
{history_rows}
"""
    
    stats = get_statistics()

    confusion_data = get_confusion_statistics()

    history_data = get_correction_history()

    total = stats["total_corrections"]

    most_corrected = stats["most_corrected"]

    if most_corrected:

        most_corrected_text = (
            f"{most_corrected[0]} "
            f"({most_corrected[1]} corrections)"
        )

    else:

        most_corrected_text = "No corrections yet."

        confusion_rows = ""

        history_rows = ""

        for (
            correction_id,
            sentence,
            predicted,
            correct,
            confidence,
            created_at
        ) in history_data:

            history_rows += (
                f"| {correction_id} | "
                f"{sentence} | "
                f"{predicted} | "
                f"{correct} | "
                f"{confidence:.2%} | "
                f"{created_at} |\n"
            )

        if not history_rows:

            history_rows = (
                "| - | No corrections yet | - | - | - | - |\n"
            )

        for predicted, correct, count in confusion_data:

            confusion_rows += (
                f"| {predicted} | {correct} | {count} |\n"
            )

        if not confusion_rows:

            confusion_rows = (
                "| No corrections yet | - | 0 |\n"
            )

    prediction_counts = stats[
        "corrections_by_prediction"
    ]

    counts = {
        category: 0
        for category in LABELS.values()
    }

    for category, count in prediction_counts:

        if category in counts:
            counts[category] = count

    return f"""
### 📊 Correction Statistics

**Total corrections:** {total}

**Most frequently corrected prediction:**

{most_corrected_text}

#### Corrections by predicted category

| Category | Corrections |
|---|---:|
| Insult | {counts["Insult"]} |
| Neutral | {counts["Neutral"]} |
| Racism | {counts["Racism"]} |
| Sexual Abuse | {counts["Sexual Abuse"]} |
| Threat | {counts["Threat"]} |

#### 🔄 Prediction → Correct Category

| Model predicted | Correct category | Corrections |
|---|---|---:|
{confusion_rows}

#### 📝 Correction History

| ID | Sentence | Model predicted | Correct | Confidence | Date |
|---:|---|---|---|---:|---|
{history_rows}

"""


def show_statistics():
    
    chart_data = get_correction_chart_data()

    chart_dataframe = pd.DataFrame(
        chart_data
    )

    return (
        gr.update(visible=True),
        display_statistics(),
        chart_dataframe
    )


def hide_statistics():

    return gr.update(visible=False)


theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate"
)


with gr.Blocks(
    title="Harmful Text Classifier",
    theme=theme,
    css=custom_css
) as app:


    gr.Markdown(
        """
        # 🛡️ Multi-Category Harmful Text Classifier
        """,
        elem_id="main-title"
    )

    gr.Markdown(
        """
        **RoBERTa-based text classification system**

        Analyse a sentence and identify its predicted
        category across five harmful-text classes.
        """,
        elem_id="subtitle"
    )


    with gr.Group(
        elem_classes="input-card"
    ):

        gr.Markdown(
            "### 📝 Enter Text"
        )

        sentence_input = gr.Textbox(
            label="Sentence",
            placeholder="Type or paste a sentence here...",
            lines=5
        )

        analyse_button = gr.Button(
            "🔍 Analyse Sentence",
            variant="primary",
            elem_id="analyse-button"
        )


    with gr.Group(
        elem_classes="output-card"
    ):

        gr.Markdown(
            "### 🎯 Prediction"
        )

        with gr.Row():

            category_output = gr.Textbox(
                label="Predicted Category",
                interactive=False,
                elem_classes="category-output"
            )

            confidence_output = gr.Textbox(
                label="Confidence",
                interactive=False,
                elem_classes="confidence-output"
            )

        explanation_output = gr.Markdown(
            "Enter a sentence and click "
            "**Analyse Sentence**."
        )


    gr.Markdown("---")

    with gr.Row():

        feedback_button = gr.Button(
            "💬 Report / Feedback"
        )

        statistics_button = gr.Button(
            "📊 Statistics"
        )


    with gr.Group(
        visible=False
    ) as statistics_panel:

        gr.Markdown(
            "### 📊 Correction Statistics"
        )

        statistics_output = gr.Markdown(
            "No statistics available yet."
        )

        correction_chart = gr.BarPlot(
            x="Category",
            y="Corrections",
            title="Corrections by Predicted Category"
        )

        close_statistics_button = gr.Button(
            "Close"
        )


    with gr.Group(
        visible=False
    ) as feedback_panel:

        gr.Markdown(
            """
            ### 💬 Report / Feedback

            Help improve the system by reporting an issue
            with the prediction.
            """
        )

        feedback_type = gr.Radio(
            choices=[
                "Incorrect prediction",
                "Other issue"
            ],
            label="What would you like to report?"
        )

        correct_category = gr.Dropdown(
            choices=list(LABELS.values()),
            label="What is the correct category?",
            visible=False
        )

        submit_feedback_button = gr.Button(
            "Submit Feedback",
            variant="primary"
        )

        close_feedback_button = gr.Button(
            "Close"
        )

        feedback_status = gr.Markdown()


    with gr.Accordion(
        "ℹ️ About the Classifier",
        open=False
    ):

        gr.Markdown(
            """
            ### Model

            This application uses a fine-tuned
            **RoBERTa sequence-classification model**.

            ### Categories

            - **Insult**
            - **Neutral**
            - **Racism**
            - **Sexual Abuse**
            - **Threat**

            ### Evaluation

            The selected RoBERTa model achieved approximately
            **99.6% accuracy and 99.6% macro F1** on the
            held-out test dataset.

            The model runs locally using **CPU inference**.

            > The confidence score represents the model's
            > predicted probability for its selected category
            > and should not be interpreted as absolute certainty.
            """
        )


    gr.Markdown(
        """
        Multi-Category Harmful Text Detection System
        """,
        elem_classes="footer"
    )


    analyse_button.click(
        fn=predict,
        inputs=sentence_input,
        outputs=[
            category_output,
            confidence_output,
            explanation_output
        ]
    )


    feedback_button.click(
        fn=show_feedback_panel,
        inputs=None,
        outputs=feedback_panel
    )


    feedback_type.change(
        fn=show_correction_fields,
        inputs=feedback_type,
        outputs=[
            correct_category,
            submit_feedback_button
        ]
    )


    submit_feedback_button.click(
        fn=submit_feedback,
        inputs=[
            feedback_type,
            correct_category,
            sentence_input,
            category_output,
            confidence_output
        ],
        outputs=feedback_status
    )


    statistics_button.click(
        fn=show_statistics,
        inputs=None,
        outputs=[
            statistics_panel,
            statistics_output,
            correction_chart
        ]
    )


    close_statistics_button.click(
        fn=hide_statistics,
        inputs=None,
        outputs=statistics_panel
    )


    close_feedback_button.click(
        fn=hide_feedback_panel,
        inputs=None,
        outputs=feedback_panel
    )


print(
    "\nStarting Gradio interface..."
)

app.launch()