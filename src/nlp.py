from dataloader import DataLoader
from datacleaner import DataCleaner
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from tqdm import tqdm
import time
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

class TfidfLogisticRegression:
    def __init__(self, data):
        self.data = data
        self.tfidf = TfidfVectorizer()
        self.model = LogisticRegression(n_jobs=-1)

    def run(self):
        X, y = DataLoader.split(self.data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # We create tf*idf BoW matrix of dimension |D| x |V|
        X_train_tfidf = self.tfidf.fit_transform(X_train)
        X_test_tfidf = self.tfidf.transform(X_test)

        start_time = time.time()
        with tqdm(total=1, desc="Fitting model") as pbar:
            self.model.fit(X_train_tfidf, y_train)
            pbar.update(1)
        end_time = time.time()

        print(f"Time taken for fitting: {end_time - start_time:.2f} seconds")

        y_train_pred = self.model.predict(X_train_tfidf)
        y_test_pred = self.model.predict(X_test_tfidf)

        print(precision_score(y_train, y_train_pred, average='binary'),
              recall_score(y_train, y_train_pred, average='binary'))
        print(precision_score(y_test, y_test_pred, average='binary'),
              recall_score(y_test, y_test_pred, average='binary'))

        # Calculate AUC
        y_train_proba = self.model.predict_proba(X_train_tfidf)[:, 1]
        y_test_proba = self.model.predict_proba(X_test_tfidf)[:, 1]

        train_auc = roc_auc_score(y_train, y_train_proba)
        test_auc = roc_auc_score(y_test, y_test_proba)

        print(f"Train AUC: {train_auc:.2f}")
        print(f"Test AUC: {test_auc:.2f}")

        # Plot AUC curve
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

        plt.figure()
        plt.plot(fpr_train, tpr_train, label='Train AUC')
        plt.plot(fpr_test, tpr_test, label='Test AUC')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('AUC Curve')
        plt.legend()
        plt.show()

        # Example usage of explain_instance for the first test instance
        X_test_instance = X_test.iloc[0]  # Change index as needed
        filename = "lime_explanation.html" 
        c = make_pipeline(self.tfidf, self.model)
        explainer = LimeTextExplainer(class_names=['male', 'female'])
        exp = explainer.explain_instance(X_test_instance, c.predict_proba, num_features=5)
        exp.save_to_file(filename)


class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model  

    def fit(self, X, y=None):
        return self
    
    def get_word2vec_embeddings(self, text):
        words = text.split()
        embeddings = [self.model.wv[word] for word in words if word in self.model.wv]  
        if len(embeddings) == 0:
            return np.zeros(self.model.vector_size) 
        return np.mean(embeddings, axis=0)

    def transform(self, X):
        return np.array([self.get_word2vec_embeddings(post) for post in X])
    

class Word2VecLogisticRegression:
    def __init__(self, data):
        self.data = data
        self.word2vec_model = None
        self.lr_word2vec = LogisticRegression(n_jobs=-1)
        self.X_test_word2vec = None
        self.y_train = None
        self.y_test = None

    def train_word2vec(self, X_train):
        posts = [post.split() for post in X_train]
        self.word2vec_model = Word2Vec(posts)

    def get_word2vec_embeddings(self, text):
        words = text.split()
        embeddings = [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
        if len(embeddings) == 0:
            return np.zeros(self.word2vec_model.vector_size)
        return np.mean(embeddings, axis=0)

    def run(self):
        X, y = DataLoader.split(self.data)
        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.train_word2vec(X_train)

        # We create a word2vec embedding of |V| x d=100 (default)
        X_train_word2vec = np.array([self.get_word2vec_embeddings(post) for post in X_train])
        self.X_test_word2vec = np.array([self.get_word2vec_embeddings(post) for post in X_test])

        start_time = time.time()
        with tqdm(total=1, desc="Fitting model with Word2Vec") as pbar:
            self.lr_word2vec.fit(X_train_word2vec, self.y_train)
            pbar.update(1)
        end_time = time.time()

        print(f"Time taken for fitting with Word2Vec: {end_time - start_time:.2f} seconds")
        y_train_pred = self.lr_word2vec.predict(X_train_word2vec)
        y_test_pred = self.lr_word2vec.predict(self.X_test_word2vec)

        print(precision_score(self.y_train, y_train_pred, average='binary'),
              recall_score(self.y_train, y_train_pred, average='binary'))
        print(precision_score(self.y_test, y_test_pred, average='binary'),
              recall_score(self.y_test, y_test_pred, average='binary'))

        # Calculate and plot AUC curve
        y_train_proba = self.lr_word2vec.predict_proba(X_train_word2vec)[:, 1]
        y_test_proba = self.lr_word2vec.predict_proba(self.X_test_word2vec)[:, 1]

        fpr_train, tpr_train, _ = roc_curve(self.y_train, y_train_proba)
        fpr_test, tpr_test, _ = roc_curve(self.y_test, y_test_proba)

        plt.figure()
        plt.plot(fpr_train, tpr_train, label='Train AUC')
        plt.plot(fpr_test, tpr_test, label='Test AUC')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('AUC Curve')
        plt.legend(loc='best')
        plt.show()

        # Merged explain_instance logic
        # Example usage of explain_instance for the first test instance
        X_test_instance = X_test.iloc[0]  # Change index as needed
        filename = "lime_explanation_word2vec.html"  
        c_word2vec = make_pipeline(Word2VecTransformer(self.word2vec_model), self.lr_word2vec)
        explainer_word2vec = LimeTextExplainer(class_names=['male', 'female'])
        exp_word2vec = explainer_word2vec.explain_instance(X_test_instance, c_word2vec.predict_proba, num_features=5)
        exp_word2vec.save_to_file(filename)