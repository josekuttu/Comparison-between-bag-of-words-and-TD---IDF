import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# STEP 1: Load Your Dataset
# -----------------------
# Replace 'your_dataset.csv' with your actual file path
# and change column names accordingly
df = pd.read_csv("D:\\PG STUDY RELATED\\Semester 3\\NLP\\dataset\\YoutubeCommentsDataSet.csv")

# Clean NaNs in text column
df = df.dropna(subset=['Comment'])        # or: df['text'] = df['text'].fillna('')

# Optional: Also clean label column if needed
df = df.dropna(subset=['Sentiment'])

# Preview data
print(df.head())

# Change 'text' and 'label' below if your column names are different
X = df['Comment']      # Text column
y = df['Sentiment']     # Label column

# -----------------------
# STEP 2: Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------
# STEP 3A: Bag-of-Words
# -----------------------
bow_vectorizer = CountVectorizer(stop_words='english')
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

model_bow = LogisticRegression(max_iter=1000)
model_bow.fit(X_train_bow, y_train)
y_pred_bow = model_bow.predict(X_test_bow)

print("\n🔤 BoW Classification Report:\n")
print(classification_report(y_test, y_pred_bow))
bow_acc = accuracy_score(y_test, y_pred_bow)

# -----------------------
# STEP 3B: TF-IDF
# -----------------------
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model_tfidf = LogisticRegression(max_iter=1000)
model_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)

print("\n📘 TF-IDF Classification Report:\n")
print(classification_report(y_test, y_pred_tfidf))
tfidf_acc = accuracy_score(y_test, y_pred_tfidf)

# -----------------------
# STEP 4: Compare Accuracy
# -----------------------
results_df = pd.DataFrame({
    'Vectorizer': ['Bag-of-Words', 'TF-IDF'],
    'Accuracy': [bow_acc, tfidf_acc]
})

sns.barplot(x='Vectorizer', y='Accuracy', data=results_df)
plt.title("Vectorizer Accuracy Comparison")
plt.ylim(0, 1)
plt.show()
