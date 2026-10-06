import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spam_classifier as sc  # noqa: E402

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sms.tsv")


@pytest.fixture(scope="module")
def df():
    return sc.load_messages(DATA_FILE)


@pytest.fixture(scope="module")
def pipeline(df):
    vectorizer, matrix = sc.extract_features(df["message_text"])
    X_train, X_test, y_train, y_test = sc.split_data(matrix, df["label_numeric"])
    model = sc.train_model(X_train, y_train)
    return vectorizer, model, X_train, X_test, y_train, y_test


# ---- data loading ----
def test_dataset_size_and_class_counts(df):
    assert len(df) == 5572
    assert (df.label == "ham").sum() == 4825
    assert (df.label == "spam").sum() == 747


def test_labels_mapped_to_numbers(df):
    assert set(df.label_numeric.unique()) == {0, 1}
    assert df.label_numeric.isna().sum() == 0
    assert (df[df.label == "spam"].label_numeric == 1).all()


# ---- features ----
def test_stop_words_removed_and_lowercased():
    vec, _ = sc.extract_features(["The FREE prize and the winner"])
    vocab = set(vec.get_feature_names_out())
    assert "the" not in vocab and "and" not in vocab
    assert {"free", "prize", "winner"} <= vocab


def test_feature_matrix_shape(df, pipeline):
    vec, _, *_ = pipeline
    matrix = vec.transform(df["message_text"])
    assert matrix.shape[0] == len(df)
    assert matrix.shape[1] == len(vec.vocabulary_)


# ---- split ----
def test_split_sizes(pipeline):
    _, _, X_train, X_test, y_train, y_test = pipeline
    assert X_test.shape[0] == len(y_test) == 1115
    assert X_train.shape[0] == len(y_train) == 5572 - 1115


def test_split_is_stratified(pipeline, df):
    *_, y_train, y_test = pipeline
    overall = df.label_numeric.mean()
    assert y_train.mean() == pytest.approx(overall, abs=0.002)
    assert y_test.mean() == pytest.approx(overall, abs=0.002)


def test_split_is_reproducible(df):
    _, matrix = sc.extract_features(df["message_text"])
    a = sc.split_data(matrix, df["label_numeric"])[3]
    b = sc.split_data(matrix, df["label_numeric"])[3]
    assert list(a.index) == list(b.index)


# ---- model quality ----
def test_exact_confusion_matrix(pipeline):
    _, model, _, X_test, _, y_test = pipeline
    _, _, cm = sc.evaluate(model, X_test, y_test)
    assert cm.tolist() == [[952, 14], [6, 143]]


def test_accuracy_precision_recall(pipeline):
    _, model, _, X_test, _, y_test = pipeline
    acc, _, cm = sc.evaluate(model, X_test, y_test)
    tn, fp, fn, tp = cm.ravel()
    assert acc >= 0.98
    assert tp / (tp + fp) >= 0.90
    assert tp / (tp + fn) >= 0.95


# ---- predictions ----
@pytest.mark.parametrize(
    "message",
    [
        "URGENT! You have won a $1000 cash prize. Call now!",
        "WINNER!! As a valued network customer you have been selected to receive a 900 prize reward. Call 09061701461 now",
    ],
)
def test_obvious_spam(pipeline, message):
    vec, model, *_ = pipeline
    assert sc.predict_message(message, vec, model) == "Spam"


@pytest.mark.parametrize(
    "message",
    ["Ok, I'll see you at dinner tonight", "Can you pick up some milk on your way home?"],
)
def test_obvious_ham(pipeline, message):
    vec, model, *_ = pipeline
    assert sc.predict_message(message, vec, model) == "Ham"


def test_unknown_words_do_not_crash(pipeline):
    vec, model, *_ = pipeline
    assert sc.predict_message("zzzqqq xxyyzz", vec, model) in ("Spam", "Ham")


# ---- dataset download ----
def test_ensure_dataset_skips_when_file_exists(tmp_path, monkeypatch):
    f = tmp_path / "sms.tsv"
    f.write_text("ham\thi\n")

    def boom(*a, **k):
        raise AssertionError("should not download")

    monkeypatch.setattr(sc.urllib.request, "urlretrieve", boom)
    assert sc.ensure_dataset(str(f)) is False


def test_ensure_dataset_downloads_when_missing(tmp_path, monkeypatch):
    f = tmp_path / "sms.tsv"
    calls = []
    monkeypatch.setattr(sc.urllib.request, "urlretrieve", lambda url, name: calls.append((url, name)))
    assert sc.ensure_dataset(str(f)) is True
    assert calls == [(sc.DATASET_URL, str(f))]
