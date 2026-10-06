import os
import urllib.request

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

DATASET_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
DATASET_FILE = "sms.tsv"
COLUMN_NAMES = ["label", "message_text"]
LABEL_MAP = {"ham": 0, "spam": 1}


def ensure_dataset(filename=DATASET_FILE, url=DATASET_URL):
    """Download the dataset if it is missing. Returns True if it was downloaded."""
    if os.path.exists(filename):
        return False
    urllib.request.urlretrieve(url, filename)
    return True


def load_messages(filename=DATASET_FILE):
    """Load the TSV file and add a numeric label column (ham=0, spam=1)."""
    df = pd.read_csv(filename, sep="\t", header=None, names=COLUMN_NAMES)
    df["label_numeric"] = df.label.map(LABEL_MAP)
    return df


def extract_features(texts):
    """Bag-of-Words features. Returns (vectorizer, word count matrix)."""
    vectorizer = CountVectorizer(stop_words="english", lowercase=True)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def split_data(features, labels, test_size=0.2, random_state=42):
    """Stratified train/test split."""
    return train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )


def train_model(features_train, labels_train):
    classifier = MultinomialNB()
    classifier.fit(features_train, labels_train)
    return classifier


def evaluate(classifier, features_test, labels_test):
    """Returns accuracy, classification report text, confusion matrix."""
    pred = classifier.predict(features_test)
    return (
        accuracy_score(labels_test, pred),
        classification_report(labels_test, pred, target_names=["Ham (Legit)", "Spam"]),
        confusion_matrix(labels_test, pred),
    )


def predict_message(message, vectorizer, classifier):
    """Returns 'Spam' or 'Ham' for a single message."""
    vector = vectorizer.transform([message])
    return "Spam" if classifier.predict(vector)[0] == 1 else "Ham"
