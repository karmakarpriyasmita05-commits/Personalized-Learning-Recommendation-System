"""Command-line interface for personalized learning recommendations."""

from pathlib import Path

from recommender import (
    train_and_evaluate,
    recommend,
    MODEL_PATH,
)


# =========================================================
# CONFIGURATION
# =========================================================

VALID_INTERESTS = [
    "Python",
    "Data Science",
    "Machine Learning",
    "Web Development",
    "Databases",
]

VALID_PREFERENCES = [
    "Video",
    "Reading",
    "Practice",
    "Interactive",
]

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "last_recommendations.csv"


# =========================================================
# INPUT VALIDATION
# =========================================================

def get_int(prompt, minimum, maximum):
    """
    Accept an integer within the specified range.

    Invalid text and out-of-range values are rejected.
    The user is prompted again until valid input is received.
    """

    while True:
        raw = input(prompt).strip()

        try:
            value = int(raw)

            if minimum <= value <= maximum:
                return value

            print(
                f"Enter a number from {minimum} to {maximum}."
            )

        except ValueError:
            print(
                "Invalid input. Please enter a whole number."
            )


def choose(prompt, options):
    """
    Display a numbered list and return the selected option.
    """

    while True:
        print("\n" + prompt)

        for index, option in enumerate(options, start=1):
            print(f"  {index}. {option}")

        choice = get_int(
            "Choose an option number: ",
            1,
            len(options),
        )

        return options[choice - 1]


# =========================================================
# DISPLAY RECOMMENDATIONS
# =========================================================

def display_recommendations(results):
    """
    Display the personalized learning recommendations.
    """

    print("\n" + "=" * 58)
    print("PERSONALIZED RECOMMENDATIONS")
    print("=" * 58)

    if results.empty:
        print("\nNo recommendations were generated.")
        return

    for index, row in results.iterrows():

        print(
            f"\n{index + 1}. "
            f"{row['topic']} — {row['resource_type']}"
        )

        print(
            f"   Category: {row['resource_category']} "
            f"| Difficulty: {row['difficulty']}"
        )

        print(
            f"   Estimated time: "
            f"{row['estimated_minutes']} minutes"
        )

        print(
            f"   Model suitability score: "
            f"{row['suitability_score']:.1%}"
        )

        print(
            f"   Why it may help: {row['description']}"
        )

        # Updated label: these are now resource links,
        # not placeholder links.
        print(
            f"   Learning resource: {row['url']}"
        )


# =========================================================
# SAVE RECOMMENDATIONS
# =========================================================

def save_recommendations(results):
    """
    Save the generated recommendations to a CSV file.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nRecommendations saved to: {OUTPUT_PATH}"
    )


# =========================================================
# MAIN APPLICATION
# =========================================================

def main():

    print("=" * 58)
    print(
        " GR-1: PERSONALIZED LEARNING RECOMMENDATION SYSTEM"
    )
    print("=" * 58)

    try:

        # -------------------------------------------------
        # STEP 1: Prepare the machine learning model
        # -------------------------------------------------

        if not MODEL_PATH.exists():

            print(
                "\nPreparing the model for first use..."
            )

            _, metrics = train_and_evaluate()

            print(
                "Model ready. "
                f"Test F1-score: {metrics['f1_score']:.3f}"
            )

        # -------------------------------------------------
        # STEP 2: Collect student information
        # -------------------------------------------------

        performance = get_int(
            "\nEnter recent performance score (0-100): ",
            0,
            100,
        )

        interest = choose(
            "Select your main interest:",
            VALID_INTERESTS,
        )

        preference = choose(
            "Select your preferred learning style:",
            VALID_PREFERENCES,
        )

        # -------------------------------------------------
        # STEP 3: Collect completed topics
        # -------------------------------------------------

        completed_raw = input(
            "\nEnter completed topics separated by commas "
            "(or press Enter for none): "
        ).strip()

        completed = [
            topic.strip()
            for topic in completed_raw.split(",")
            if topic.strip()
        ]

        # -------------------------------------------------
        # STEP 4: Select recommendation count
        # -------------------------------------------------

        top_n = get_int(
            "How many recommendations? (1-10): ",
            1,
            10,
        )

        # -------------------------------------------------
        # STEP 5: Generate personalized recommendations
        # -------------------------------------------------

        results = recommend(
            performance,
            interest,
            preference,
            completed,
            top_n,
        )

        # -------------------------------------------------
        # STEP 6: Display the results
        # -------------------------------------------------

        display_recommendations(results)

        # -------------------------------------------------
        # STEP 7: Save recommendations
        # -------------------------------------------------

        if not results.empty:
            save_recommendations(results)

        # -------------------------------------------------
        # STEP 8: Display disclaimer
        # -------------------------------------------------

        print(
            "\nNote: Suitability scores are model estimates, "
            "not guaranteed learning outcomes."
        )

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
    ) as exc:

        print(
            f"\nCould not complete the recommendation: {exc}"
        )

        print(
            "Check that both CSV files exist and contain "
            "the required columns."
        )


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
