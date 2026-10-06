import os

from spam_classifier import (
    DATASET_FILE,
    ensure_dataset,
    evaluate,
    extract_features,
    load_messages,
    predict_message,
    split_data,
    train_model,
)


def main():
    if not os.path.exists(DATASET_FILE):
        print(f"File '{DATASET_FILE}' not found. Downloading...")
        try:
            ensure_dataset(DATASET_FILE)
            print("Download complete!")
        except Exception as e:
            print(f"Error downloading file: {e}")
            return
    else:
        print(f"File '{DATASET_FILE}' found locally. Using existing file.")

    messages_df = load_messages(DATASET_FILE)

    print("First 5 rows of data:")
    print(messages_df.head())
    print("\nClass Distribution:")
    print(messages_df["label"].value_counts())

    vectorizer, word_counts_matrix = extract_features(messages_df["message_text"])
    labels = messages_df["label_numeric"]
    features_train, features_test, labels_train, labels_test = split_data(word_counts_matrix, labels)

    classifier = train_model(features_train, labels_train)

    accuracy, report, matrix = evaluate(classifier, features_test, labels_test)
    print(f"\nAccuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(matrix)

    fake_message = "URGENT! You have won a $1000 cash prize. Call now!"
    print(f"\nTest Prediction: {predict_message(fake_message, vectorizer, classifier)}")


if __name__ == "__main__":
    main()
