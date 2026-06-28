# SMS Spam Classifier

A Naive Bayes text classifier that detects spam SMS messages.

## About

Trains a `MultinomialNB` model to label SMS messages as legitimate ("ham") or spam, using a Bag-of-Words representation of the message text.

## Dataset

- **Source:** [SMS Spam Collection](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv)
- **Size:** 5,574 labeled SMS messages
- The script downloads the dataset automatically on first run and caches it locally as `sms.tsv`.

## Pipeline

1. Download (or reuse) `sms.tsv`, map labels to `0` (ham) / `1` (spam)
2. Inspect class distribution
3. Feature extraction with `CountVectorizer` (Bag-of-Words, English stop words removed)
4. 80/20 train/test split, stratified to preserve class balance
5. Train a `MultinomialNB` classifier
6. Evaluate with accuracy, precision, recall, and a confusion matrix

## Results

Evaluated on a held-out test set of 1,115 messages:

| Metric | Score |
|---|---|
| Accuracy | 98% |
| Precision (Spam) | 0.91 |
| Recall (Spam) | 0.96 |

952 ham and 143 spam messages were classified correctly, with 6 spam messages missed and 14 ham messages misclassified as spam.

## Tech Stack

- Python 3
- pandas
- scikit-learn (`CountVectorizer`, `MultinomialNB`, `train_test_split`)

## Getting Started

```bash
pip install -r requirements.txt
python main.py
```
