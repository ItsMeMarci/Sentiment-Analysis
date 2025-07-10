import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from typing import Final
import argparse

train_data = pd.read_csv('train.tsv', sep='\t')
test_data = pd.read_csv('test.tsv', sep='\t')

print(train_data["label"].value_counts())
print(test_data["label"].value_counts())

vectorizer = CountVectorizer()
vectorizer.fit(train_data.text)
features = vectorizer.transform(train_data.text)
#features = vectorizer.fit_transform(train_data.text)
cls = SGDClassifier()
model = cls.fit(features, train_data.label)

#egyes osztályok legerősebb jellemzői, amelyek alapján a leginkább az adott osztályba sorolódik (negatív, neutral, positive):
#a 2-es index a pozitívakat jelenti, az 1-es a neutrálisakat, a 0-s a negatívakat
print(sorted(zip(model.coef_[2], vectorizer.get_feature_names_out()),reverse=True)[:10])

#kiértékelés train data-n: arra jó, hogy ha nem lenne kellően magas, akkor rossz modellt választottam volna
#erre írhatok egy assert majd, hogy kellően magas-e az accuracy(99% feletti)
prediction = model.predict(features)
accuracy_score(y_true=train_data.label, y_pred=prediction)
print(classification_report(y_true=train_data.label, y_pred=prediction))

#kiértékelés test data-n:
test_features = vectorizer.transform(test_data.text)
prediction = model.predict(test_features)
accuracy_score(y_true=test_data.label, y_pred=prediction)
#baseline: leggyakoribb osztály
dummy_clf = DummyClassifier(strategy="most_frequent") # tanító adatbázis leggyakoribb osztálya lesz mindig a predikció
dummy_clf.fit(features, train_data.label) # ugyanazon a tanító adatbázison "tanítjuk"
baseline_prediction = dummy_clf.predict(test_features) # predikció a kiértékelő adatbázison
accuracy_score(baseline_prediction, test_data.label)

#lineáris gép:
cls = SGDClassifier(alpha=0.001)
model = cls.fit(features, train_data.label)
prediction = model.predict(test_features)
accuracy_score(y_true=test_data.label, y_pred=prediction)

#döntési fa:
cls = DecisionTreeClassifier()
model = cls.fit(features, train_data.label)
prediction = model.predict(test_features)
accuracy_score(y_true=test_data.label, y_pred=prediction)

#egy másik lineáris gép:
cls = LogisticRegression()
model = cls.fit(features, train_data.label)
prediction = model.predict(test_features)
accuracy_score(y_true=test_data.label, y_pred=prediction)

#confusion matrix:
print(classification_report(y_true=test_data.label, y_pred=prediction))
confusion_matrix(y_true=test_data.label, y_pred=prediction)

#bigram tanítás:
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=2)
features_bigram = bigram_vectorizer.fit_transform(train_data.text)

cls = SGDClassifier(alpha=0.001) # új, üres osztályozó
model_bigram = cls.fit(features_bigram, train_data.label)

prediction_bigram = model_bigram.predict(bigram_vectorizer.transform(test_data.text))
#bigram kiértékelés:
print(accuracy_score(y_true=test_data.label, y_pred=prediction_bigram))
print(classification_report(y_true=test_data.label, y_pred=prediction_bigram))

#tf-idf:
vectorizer = CountVectorizer()
cv_counts = vectorizer.fit_transform(train_data.text)
idf_transformer = TfidfTransformer(use_idf=True).fit(cv_counts)
features_idf = idf_transformer.transform(cv_counts)

cls = SGDClassifier()
model_idf = cls.fit(features_idf, train_data.label)

prediction_idf = model_idf.predict(idf_transformer.transform(vectorizer.transform(test_data.text)))

print(accuracy_score(y_true=test_data.label, y_pred=prediction_idf))
print(classification_report(y_true=test_data.label, y_pred=prediction_idf))


#beolvasások
def data_reading():
    return train_data, test_data

def data_label_distribution():
    return data_reading()[0]["label"].value_counts(), data_reading()[1]["label"].value_counts()

def vectorizing():
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(train_data.text)
    return features

def bigram_vectorizing():
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=2)
    features = bigram_vectorizer.fit_transform(train_data.text)
    return features

def tfidf_vectorizing():
    idf_transformer = TfidfTransformer(use_idf=True).fit(vectorizing())
    features_idf = idf_transformer.transform(cv_counts)

vectorizer = CountVectorizer()
cv_counts = vectorizer.fit_transform(train_data.text)
idf_transformer = TfidfTransformer(use_idf=True).fit(cv_counts)
features_idf = idf_transformer.transform(cv_counts)

cls = SGDClassifier(alpha=0.001) # új, üres osztályozó
model_bigram = cls.fit(features_bigram, train_data.label)
#legerősebb jellemzőkhöz: külön függvény a negatívra(0-ás index), neutrálisra(1-es), és a pozitívra(2-es)
print(sorted(zip(model_bigram.coef_[1], bigram_vectorizer.get_feature_names_out()),reverse=True)[:10])


