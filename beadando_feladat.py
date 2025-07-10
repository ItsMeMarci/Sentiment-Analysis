import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import Final
import argparse

TRAIN: Final[pd.DataFrame] = pd.read_csv('train.tsv', sep='\t')
TEST: Final[pd.DataFrame] = pd.read_csv('test.tsv', sep='\t')

class Sentiment:
    def __init__(self, model: str = "bag-of-words", vectorizer: str = "tfidf"):
        self._train_data = TRAIN
        self._test_data = TEST
        self._model = model
        self._vectorizer = vectorizer

    @property
    def train_data(self):
        return self._train_data

    @property
    def test_data(self):
        return self._test_data

    def details(self):
        return self._train_data.describe(), self.train_data.columns, self.train_data.shape, self.train_data.dtypes, \
            self.train_data.head(10)

    def data_value_counts(self):
        return self._train_data["label"].value_counts()

    def strongest_features(self, pred):
        predikcio = 0
        if pred == "pozitiv":
            predikcio = 2
        elif pred == "neutralis":
            predikcio = 1
        else:
            pred = "negativ"
        vectorizer = CountVectorizer()
        cv_counts = vectorizer.fit_transform(self.train_data.text)
        idf_transformer = TfidfTransformer(use_idf=True).fit(cv_counts)
        features_idf = idf_transformer.transform(cv_counts)
        cls = SGDClassifier(alpha=0.001)
        model_idf = cls.fit(features_idf, self.train_data.label)
        strongest = sorted(zip(model_idf.coef_[predikcio], vectorizer.get_feature_names_out()), reverse=True)[:5]
        szoveg = f"Egy {pred} predikciot ezek a szavak befolyasolnak legerosebben, minel nagyobb a szam, annal inkabb: \n{strongest}"
        return szoveg

    def acc_score(self):
        vectorizer = CountVectorizer()
        cv_counts = vectorizer.fit_transform(self.train_data.text)
        idf_transformer = TfidfTransformer(use_idf=True).fit(cv_counts)
        features_idf = idf_transformer.transform(cv_counts)
        cls = SGDClassifier(alpha=0.001)
        model_idf = cls.fit(features_idf, self.train_data.label)
        prediction_idf = model_idf.predict(idf_transformer.transform(vectorizer.transform(self.test_data.text)))
        return accuracy_score(y_true=self.test_data.label, y_pred=prediction_idf)

    def classification_report(self):
        vectorizer = CountVectorizer()
        cv_counts = vectorizer.fit_transform(self.train_data.text)
        idf_transformer = TfidfTransformer(use_idf=True).fit(cv_counts)
        features_idf = idf_transformer.transform(cv_counts)
        cls = SGDClassifier(alpha=0.001)
        model_idf = cls.fit(features_idf, self.train_data.label)
        prediction_idf = model_idf.predict(idf_transformer.transform(vectorizer.transform(self.test_data.text)))
        return classification_report(y_true=self.test_data.label, y_pred=prediction_idf)

    def custom_text_prediction(self, custom_text):
        if len(custom_text.split()) < 6:
            return "legalabb 6 szo hosszu kommentet irj"
        vectorizer = CountVectorizer()
        cv_counts = vectorizer.fit_transform(self.train_data.text)
        idf_transformer = TfidfTransformer(use_idf=True).fit(cv_counts)
        features_idf = idf_transformer.transform(cv_counts)
        cls = SGDClassifier()
        model_idf = cls.fit(features_idf, self.train_data.label)
        custom_features = idf_transformer.transform(vectorizer.transform([custom_text]))
        custom_prediction = model_idf.predict(custom_features)
        return custom_prediction

    def test_label_column(self):
        valid_labels = {"POSITIVE", "NEUTRAL", "NEGATIVE"}
        assert set(self.train_data["label"]).issubset(valid_labels), "Label oszlop helytelen erteket tartalmaz"

    def test_acc_score(self):
        too_high_accuracy = 0.96
        too_low_accuracy = 0.33
        assert self.acc_score() < too_high_accuracy
        assert self.acc_score() > too_low_accuracy

    def test_strongest_features(self):
        which = "neutralis"
        assert self.strongest_features(which).split()[1] == which

    def test_custom_text_prediction(self):
        text = "it is too short"
        assert self.custom_text_prediction(text) == "legalabb 6 szo hosszu kommentet irj"

def main():
        parser = argparse.ArgumentParser(description="Sentiment analysis app")

        parser.add_argument('-data', '-d', action="store_true", default=False)
        parser.add_argument('-acc', '-a', action="store_true", default=False)
        parser.add_argument('-classification', '-c', action="store_true", default=False)
        parser.add_argument('-strongest', '-s', choices=["pozitiv", "neutralis", "negativ"])

        args = parser.parse_args()

        sentiment = Sentiment()

        if args.data:
            print(sentiment.train_data)
        if args.strongest == "pozitiv":
            print(sentiment.strongest_features("pozitiv"))
        elif args.strongest == "neutralis":
            print(sentiment.strongest_features("neutralis"))
        elif args.strongest == "negativ":
            print(sentiment.strongest_features("negativ"))
        if args.acc:
            print(sentiment.acc_score())
        if args.classification:
            print(sentiment.classification_report())

        sentiment.test_label_column()
        sentiment.test_acc_score()
        sentiment.test_strongest_features()
        sentiment.test_custom_text_prediction()

if __name__ == '__main__':
    main()



