#CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

docs = ["Mayur is a nice boy.", "Mayur rock! wohooo!", "My name is Mayur, and I am a Pythonista!"]
cv = CountVectorizer()
X = cv.fit_transform(docs)
print(X.todense())
print(cv.vocabulary_)

from sklearn.feature_extraction import DictVectorizer

docs = [{"Mayur": 1, "is": 1, "awesome": 2}, {"No": 1, "I": 1, "dont": 2, "wanna": 3, "fall": 1, "in": 2, "love": 3}]
dv = DictVectorizer()
X = dv.fit_transform(docs)
print(X.todense())

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tfidf_vectorizer = TfidfVectorizer()
cv_vectorizer = CountVectorizer()
docs = ["Mayur is a Guitarist", "Mayur is Musician", "Mayur is also a programmer"]
X_idf = tfidf_vectorizer.fit_transform(docs)
X_cv = cv_vectorizer.fit_transform(docs)
print(X_idf.todense())
print(tfidf_vectorizer.vocabulary_)
print(X_cv.todense())
