# Email Spam Classifier (TF-IDF + Logistic Regression)

End-to-end spam detection pipeline trained on ~193k deduplicated emails, progressing from a custom bag-of-words `MLPClassifier` prototype to a production `TfidfVectorizer` + `LogisticRegression` pipeline with cost-sensitive threshold tuning.

## Model Comparison

| Stage | Dataset Size | Vectorizer | Classifier | Accuracy | Spam Precision | Spam Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Prototype** | 200 emails (3-fold CV) | Custom `Counter` + `PorterStemmer` | `MLPClassifier (44, 44)` | 82.53% | 81.33% | 81.33% |
| **Full (`t=0.50`)** | 193,070 deduplicated | TF-IDF `(1, 2)`-grams (30k features) | `LogisticRegression (C=2.0)` | 98.25% | 97.66% | 98.67% |
| **Tuned (`t=0.70`)** | 193,070 deduplicated | TF-IDF `(1, 2)`-grams (30k features) | `LogisticRegression (C=2.0)` | — | **98.77%** | **96.56%** |

## Key Engineering Decisions
- **Leakage Prevention:** Deduplicated cleaned email bodies prior to the stratified 80/20 train-test split.
- **Self-Contained Pipeline:** Embedded `clean_text` directly as `TfidfVectorizer(preprocessor=clean_text)` so raw email strings can be scored directly at inference time.
- **Cost-Sensitive Thresholding:** Raised the decision threshold from `0.50` to `0.70`, cutting false positives (legitimate emails flagged as spam) nearly in half (`431` -> `220`).
