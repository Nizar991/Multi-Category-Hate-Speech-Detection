import random
import re

from read_feedback import get_verified_corrections


# --------------------------------------------------
# 1. LEETSPEAK ATTACK
# --------------------------------------------------

def leetspeak_variant(sentence):
    replacements = {
        "a": "4",
        "e": "3",
        "i": "1",
        "o": "0",
        "s": "5"
    }

    possible_positions = [
        i for i, char in enumerate(sentence)
        if char.lower() in replacements
    ]

    if not possible_positions:
        return None

    # Change around 30% of eligible characters
    number_to_change = max(
        1,
        int(len(possible_positions) * 0.30)
    )

    positions = random.sample(
        possible_positions,
        min(number_to_change, len(possible_positions))
    )

    characters = list(sentence)

    for position in positions:
        characters[position] = replacements[
            characters[position].lower()
        ]

    return "".join(characters)


# --------------------------------------------------
# 2. SPACING ATTACK
# --------------------------------------------------

def spacing_variant(sentence):
    # Choose a word with at least 4 letters
    matches = list(
        re.finditer(r"\b[A-Za-z]{4,}\b", sentence)
    )

    if not matches:
        return None

    match = random.choice(matches)

    original_word = match.group()
    spaced_word = " ".join(original_word)

    return (
        sentence[:match.start()]
        + spaced_word
        + sentence[match.end():]
    )


# --------------------------------------------------
# 3. PUNCTUATION ATTACK
# --------------------------------------------------

def punctuation_variant(sentence):
    words = sentence.split()

    if len(words) < 2:
        return sentence + " !!!"

    position = random.randint(
        1,
        len(words) - 1
    )

    punctuation_choices = [
        "...",
        "!!!",
        "???",
        "---",
        "***"
    ]

    noise = random.choice(
        punctuation_choices
    )

    words.insert(position, noise)

    return " ".join(words)


# --------------------------------------------------
# 4. TYPO ATTACK
# --------------------------------------------------

def typo_variant(sentence):
    matches = list(
        re.finditer(r"\b[A-Za-z]{4,}\b", sentence)
    )

    # Find words that contain at least one pair
    # of different neighbouring letters
    valid_options = []

    for match in matches:
        word = match.group()

        for position in range(1, len(word) - 1):

            if word[position] != word[position + 1]:
                valid_options.append(
                    (match, position)
                )

    if not valid_options:
        return None

    match, position = random.choice(
        valid_options
    )

    word = match.group()
    characters = list(word)

    characters[position], characters[position + 1] = (
        characters[position + 1],
        characters[position]
    )

    typo_word = "".join(characters)

    return (
        sentence[:match.start()]
        + typo_word
        + sentence[match.end():]
    )

    # Find words long enough to modify safely
    matches = list(
        re.finditer(r"\b[A-Za-z]{4,}\b", sentence)
    )

    if not matches:
        return None

    match = random.choice(matches)

    word = match.group()

    if len(word) < 4:
        return None

    characters = list(word)

    # Swap two neighbouring internal characters
    # First and last characters remain unchanged
    position = random.randint(
        1,
        len(characters) - 2
    )

    characters[position], characters[position + 1] = (
        characters[position + 1],
        characters[position]
    )

    typo_word = "".join(characters)

    return (
        sentence[:match.start()]
        + typo_word
        + sentence[match.end():]
    )


# --------------------------------------------------
# GENERATE ALL VERSIONS
# --------------------------------------------------

def generate_variants(sentence):

    variants = {
        "Original": sentence,
        "Leetspeak": leetspeak_variant(sentence),
        "Spacing": spacing_variant(sentence),
        "Punctuation": punctuation_variant(sentence),
        "Typo": typo_variant(sentence)
    }

    return variants


# --------------------------------------------------
# TEST GENERATOR
# --------------------------------------------------

if __name__ == "__main__":

    # Same results every time for reproducibility
    random.seed(42)

    corrections = get_verified_corrections()

    print(
        "Total verified corrections:",
        len(corrections)
    )

    for correction_id, sentence, correct_category in corrections:

        print("\n" + "=" * 70)

        print(
            "Correction ID:",
            correction_id
        )

        print(
            "Correct category:",
            correct_category
        )

        variants = generate_variants(
            sentence
        )

        for transformation, text in variants.items():

            if text is None:
                print(
                    f"{transformation}: "
                    "[Variant could not be generated]"
                )

            else:
                print(
                    f"{transformation}: {text}"
                )