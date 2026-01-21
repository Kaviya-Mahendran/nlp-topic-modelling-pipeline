**Applied NLP Pipeline: Topic Modelling, Sentiment & Frequency Analysis**

**1. Overview**

Unstructured text data such as customer feedback, CRM notes, or internal summaries often contains valuable behavioural signals, yet it is rarely analysed systematically. The main challenge is that this data is messy, inconsistent, and difficult to translate into insights that decision makers can trust.

This repository implements an end to end NLP analytics pipeline that transforms raw text into structured, interpretable outputs using topic modelling and sentiment analysis. The system is designed to work with realistic, real world text while remaining lightweight, explainable, and privacy aware.

The goal of this project is not to maximise model complexity, but to demonstrate how applied NLP can be embedded into an analytics workflow that supports prioritisation, monitoring, and evidence based decision making.

Although implementations vary across organisations, these principles apply broadly to most data analytics environments.

**2. Architecture Diagram**

Location: diagrams/architecture.png

High level architecture:

Raw Text Data (CSV)
        ↓
Preprocessing & Normalisation
        ↓
Vectorisation (Bag of Words)
        ↓
Topic Modelling (LDA)
        ↓
Sentiment Scoring
        ↓
Structured Outputs & Visualisations


The pipeline is deliberately modular. Each stage has a single responsibility, making the system easier to test, extend, and reason about.

**3. Pipeline / System Design**
Step 1: Ingestion

Raw text data is loaded from a CSV file located in data/raw/.
The pipeline assumes the data is noisy and unstructured, similar to real CRM or feedback data.

Step 2: Preprocessing (Privacy Aware)

Text is cleaned and normalised using a dedicated preprocessing module:

lowercasing

punctuation and noise removal

stopword filtering

lemmatisation

removal of obvious PII patterns (e.g. emails, numeric identifiers)

This step ensures that modelling is performed on consistent, non identifiable text.

Step 3: Transformation

Cleaned text is converted into numerical representations using a bag of words approach.
This provides a transparent and interpretable input to the topic model.

Step 4: Topic Modelling

Latent Dirichlet Allocation (LDA) is used to identify recurring themes within the text.
The number of topics is chosen based on interpretability rather than purely statistical optimisation.

Each record is assigned a dominant topic, enabling downstream aggregation and analysis.

Step 5: Sentiment Analysis

Sentiment polarity is calculated on the cleaned text using TextBlob.
This provides an additional signal that captures how people feel about each topic, not just what they are discussing.

Step 6: Outputs & Validation

The pipeline produces:

a structured CSV with topic and sentiment assignments

aggregated topic frequencies

a saved, reusable model artifact

Outputs are stored in the outputs/ directory for inspection and reuse.

**4. Code Highlights**
Preprocessing logic
def clean_text(self, text):
    text = text.lower()
    text = self._remove_pii(text)
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join(
        self.lemmatizer.lemmatize(t)
        for t in text.split()
        if t not in self.stop_words
    )


This ensures consistent text representation while explicitly addressing privacy concerns.

Topic modelling
lda = LatentDirichletAllocation(
    n_components=5,
    random_state=42
)
lda.fit(doc_term_matrix)


The model prioritises interpretability and stability over complexity.

Sentiment scoring
df["sentiment"] = df["clean_text"].apply(get_sentiment)


Sentiment is treated as a complementary signal, not a standalone decision metric.

**5. Results / Outputs**
Topic assignments

File: outputs/topic_assignments.csv

Each record includes:

raw text

cleaned text

assigned topic

sentiment polarity score

This makes the model auditable and easy to validate.

Topic frequency chart

File: outputs/topic_frequency.png

The chart shows how frequently each topic appears across the dataset.
This helps identify dominant themes as well as less frequent but potentially important topics.

(Insert screenshot of topic_frequency.png here)

**6. Why This Matters**

This project demonstrates how NLP can be applied in a way that is useful, explainable, and operational, rather than experimental.

Key impacts:

Converts qualitative text into structured analytical signals

Reduces manual review and subjective interpretation

Enables prioritisation based on both topic prevalence and sentiment

Produces reusable outputs suitable for dashboards or further modelling

Embeds privacy considerations directly into the pipeline design

The modular architecture allows this system to scale to larger datasets or integrate into broader analytics workflows.

Although implementations vary across organisations, these principles apply broadly to most data analytics environments.

**7. Reflection & Future Enhancements
**
Building this pipeline reinforced that the value of NLP lies less in model sophistication and more in design choices, interpretability, and trust.

Key learnings:

Text preprocessing has a greater impact on topic quality than algorithm choice

Simple, explainable models are easier to operationalise

Combining topic modelling with sentiment adds meaningful decision context

Future enhancements could include:

tracking topic trends over time

comparing multiple topic models

aggregating sentiment at topic level

integrating outputs into BI dashboards

**8. How to Reproduce**

From the project root:

pip install -r requirements.txt
python scripts/pipeline.py


This will:

preprocess the data

train the topic model

calculate sentiment

generate outputs and charts

Final Note

This repository reflects an analytics as a system mindset rather than a one off analysis. It demonstrates how applied NLP can be structured, governed, and reused to support real analytical decision making.
