import pandas as pd
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Data Acquisition
# Define URL and local filename
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
filename = "sms.tsv"
column_names = ['label', 'message_text']

# Check if file exists locally. If not, download it.
if not os.path.exists(filename):
    print(f"File '{filename}' not found. Downloading...")
    try:
        urllib.request.urlretrieve(url, filename)
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading file: {e}")
        exit()
else:
    print(f"File '{filename}' found locally. Using existing file.")

# Read from the LOCAL file (this satisfies the requirement to run offline)
messages_df = pd.read_csv(filename, sep='\t', header=None, names=column_names)

# Convert labels to numbers: 'ham' becomes 0, 'spam' becomes 1
messages_df['label_numeric'] = messages_df.label.map({'ham': 0, 'spam': 1})

# 2. Data Exploration
print("First 5 rows of data:")
print(messages_df.head())

print("\nClass Distribution:")
print(messages_df['label'].value_counts())

# 3. Feature Extraction (Bag-of-Words)
# We convert text to a matrix of token counts
vectorizer = CountVectorizer(stop_words='english', lowercase=True)

# 'X' is now 'word_counts_matrix' - much clearer!
word_counts_matrix = vectorizer.fit_transform(messages_df['message_text'])
labels = messages_df['label_numeric']

# 4. Data Splitting
features_train, features_test, labels_train, labels_test = train_test_split(
    word_counts_matrix, labels, test_size=0.2, random_state=42, stratify=labels
)

# 5. Model Training
# MultinomialNB is chosen because it works well with word counts
classifier = MultinomialNB()
classifier.fit(features_train, labels_train)

# 6. Model Evaluation
labels_pred = classifier.predict(features_test)

print(f"\nAccuracy: {accuracy_score(labels_test, labels_pred):.2f}")

print("\nClassification Report:")
print(classification_report(labels_test, labels_pred, target_names=['Ham (Legit)', 'Spam']))

print("\nConfusion Matrix:")
print(confusion_matrix(labels_test, labels_pred))

# Optional: Test a fake message
fake_message = ["URGENT! You have won a $1000 cash prize. Call now!"]
fake_message_vector = vectorizer.transform(fake_message)
prediction = classifier.predict(fake_message_vector)
print(f"\nTest Prediction: {'Spam' if prediction[0] == 1 else 'Ham'}")