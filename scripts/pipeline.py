import pandas as pd
from pathlib import Path

from preprocessing import TextPreprocessor
from modelling import TopicModel
from utils import get_sentiment
import matplotlib.pyplot as plt



# -----------------------------
# Configuration
# -----------------------------

RAW_DATA_PATH = Path("data/raw_data/sample_raw_data.csv")
PROCESSED_DATA_PATH = Path("data/processed/sample_clean_data.csv")
MODEL_PATH = Path("models/lda_model.pkl")


# -----------------------------
# Pipeline
# -----------------------------

def run_pipeline():
    print("Starting NLP Topic Modelling Pipeline...\n")

    # Step 1: Load raw data
    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)

    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column")

    # Step 2: Preprocess text
    print("Preprocessing text...")
    preprocessor = TextPreprocessor()
    df["clean_text"] = preprocessor.preprocess_series(df["text"])
    print("Calculating sentiment scores...")
    df["sentiment"] = df["clean_text"].apply(get_sentiment)

    # Step 3: Save processed data
    print("Saving processed data...")
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[["id", "clean_text"]].to_csv(PROCESSED_DATA_PATH, index=False)

    # Step 4: Train topic model
    print("Training topic model...")
    model = TopicModel(n_topics=5)
    model.fit(df["clean_text"].tolist())

    # Step 5: Assign topics
    print("Assigning topics...")
    df["topic"] = model.transform(df["clean_text"].tolist())
    print("Saving topic assignments...")
    OUTPUTS_PATH = Path("outputs")
    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

    df[["id", "text", "clean_text", "topic", "sentiment"]].to_csv(
        OUTPUTS_PATH / "topic_assignments.csv",
        index=False
    )

    # Step 6: Save model
    print("Saving trained model...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    # Step 7: Output results
    print("\n--- Topic Keywords ---")
    for topic_id, keywords in model.get_topics():
        print(f"Topic {topic_id}: {', '.join(keywords)}")

    print("\n--- Topic Distribution ---")
    print(df["topic"].value_counts().sort_index())
    print("Generating topic frequency chart...")

    topic_counts = df["topic"].value_counts().sort_index()

    plt.figure()
    topic_counts.plot(kind="bar")
    plt.xlabel("Topic")
    plt.ylabel("Number of Records")
    plt.title("Topic Frequency Distribution")
    plt.tight_layout()

    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUTS_PATH / "topic_frequency.png")
    plt.close()

    print("\nPipeline completed successfully.")


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    run_pipeline()
