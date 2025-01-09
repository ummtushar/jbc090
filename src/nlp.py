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
from gensim.utils import simple_preprocess


class TfidfLogisticRegression:
    """A classifier that combines TF-IDF vectorization with Logistic Regression.
    
    This class implements a pipeline that converts text data into TF-IDF features
    and uses logistic regression for binary classification of gender-based text.
    
    Attributes:
        data (pd.DataFrame): Input dataset containing text and labels
        tfidf (TfidfVectorizer): TF-IDF vectorizer for text feature extraction
        model (LogisticRegression): Logistic regression classifier
    """

    def __init__(self, data):
        self.data = data
        self.tfidf = TfidfVectorizer()
        self.model = LogisticRegression(n_jobs=-1)

    def run(self):
        """Executes the complete training and evaluation pipeline.
        
        Performs the following steps:
        1. Splits data into training and test sets
        2. Converts text to TF-IDF features
        3. Trains the logistic regression model
        4. Evaluates performance using precision, recall, and AUC
        5. Generates ROC curve visualization
        6. Creates LIME explanation for a sample prediction
        """
        X, y = DataLoader.split(self.data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # We create tf*idf BoW matrix of dimension |D| x |V|
        X_train_tfidf = self.tfidf.fit_transform(X_train)
        X_test_tfidf = self.tfidf.transform(X_test)

        # fitting the tfidf model on training data
        start_time = time.time()
        with tqdm(total=1, desc="Fitting model") as pbar:
            self.model.fit(X_train_tfidf, y_train)
            pbar.update(1)
        end_time = time.time()

        print(f"Time taken for fitting: {end_time - start_time:.2f} seconds")

        # predicting the labels using the trained model
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
        filename = "lime_explanation_orignal.html" 
        c = make_pipeline(self.tfidf, self.model)
        explainer = LimeTextExplainer(class_names=['male', 'female'])
        exp = explainer.explain_instance(X_test_instance, c.predict_proba, num_features=5)
        exp.save_to_file(filename)

    def test(self):
        """Evaluates model performance on a predefined test set.
        
        Tests the model on a curated set of quotes from various speakers
        and calculates the accuracy ratio of predictions.
        """
        test_cases_data = {
            "input": [
                "I stand before you today, humbled by the task before us, grateful for the trust you have bestowed, mindful of the sacrifices borne by our ancestors.",
                "We realize the importance of our voices only when we are silenced. We must raise our voices and speak out against injustice.",
                "There is no limit to what we, as women, can accomplish. We must believe in ourselves and our capabilities.",
                "The future belongs to those who believe in the beauty of their dreams. We must not only dream but also act.",
                "I have learned that as long as I hold fast to my beliefs and values, I will be able to overcome any obstacle.",
                "In the face of adversity, we must not lose hope. Together, we can create a world where equality and justice prevail.",
                "We cannot all succeed when half of us are held back. It is time to lift each other up and strive for equality.",
                "I am not afraid to stand up for what I believe in. My voice is my power, and I will use it to inspire change.",
                "The most common way people give up their power is by thinking they don’t have any. We must reclaim our power.",
                "Let us not seek to satisfy our thirst for freedom by drinking from the cup of bitterness and hatred.",
                "Ask not what your country can do for you – ask what you can do for your country.",
                "The only thing we have to fear is fear itself.",
                "Injustice anywhere is a threat to justice everywhere.",
                "I have a dream that one day this nation will rise up and live out the true meaning of its creed.",
                "Success is not final, failure is not fatal: It is the courage to continue that counts."
            ],
            "expected_output": [
                1,  # Michelle Obama
                1,  # Malala Yousafzai
                1,  # Michelle Obama
                1,  # Eleanor Roosevelt
                1,  # Oprah Winfrey
                1,  # Kamala Harris
                1,  # Malala Yousafzai
                1,  # Greta Thunberg
                1,  # Alice Walker
                1,  # Martin Luther King Jr. (included for contrast)
                0,    # John F. Kennedy
                0,    # Franklin D. Roosevelt
                0,    # Martin Luther King Jr.
                0,    # Martin Luther King Jr.
                0     # Winston Churchill
            ]
        }
        test_cases_df = pd.DataFrame(test_cases_data)
        X = test_cases_df['input']
        X = self.tfidf.transform(X)
        y = test_cases_df['expected_output']

        # calculating prediction ratio
        results = self.model.predict(X)
        correct_predictions = sum(results == y)
        total_predictions = len(y)
        accuracy_ratio = correct_predictions / total_predictions
        print(f"Accuracy Ratio: {accuracy_ratio:.2f}")



class Word2VecTransformer(BaseEstimator, TransformerMixin): # taken from lab session 4 to average out tokens of the document into one vector of 'd' dimensions
    def __init__(self):
        self.w2v = None

    def fit(self, X, y=None):
        """Trains the Word2Vec model on the input documents.
        
        Args:
            X (array-like): List of text documents
            y: Ignored (included for scikit-learn API compatibility)
            
        Returns:
            self: Returns the instance itself
        """
        self.w2v = Word2Vec([simple_preprocess(doc) for doc in X])
        return self

    def transform(self, X):
        """Converts documents into averaged Word2Vec vectors.
        
        Args:
            X (array-like): List of text documents
            
        Returns:
            np.array: Document vectors where each row represents a document
        """
        vec_X = []
        for doc in X:
            vec = []
            for token in simple_preprocess(doc):
                try:
                    vec.append(self.w2v.wv[token])
                except KeyError:
                    pass
            if not vec:
                vec.append(self.w2v.wv['the'])  
            vec_X.append(np.mean(vec, axis=0))  # return the document as one vector
        return np.array(vec_X)

class Word2VecLogisticRegression:
    """A classifier that combines Word2Vec embeddings with Logistic Regression.
    
    This class implements a pipeline that converts text data into Word2Vec features
    and uses logistic regression for binary classification of gender-based text.
    
    Attributes:
        data (pd.DataFrame): Input dataset containing text and labels
        word2vec_model (Word2VecTransformer): Word2Vec feature transformer
        lr_word2vec (LogisticRegression): Logistic regression classifier
        X_test_word2vec (np.array): Transformed test data
        y_train (np.array): Training labels
        y_test (np.array): Test labels
    """

    def __init__(self, data):
        self.data = data
        self.word2vec_model = Word2VecTransformer() 
        self.lr_word2vec = LogisticRegression(n_jobs=-1)
        self.X_test_word2vec = None
        self.y_train = None
        self.y_test = None

    def run(self):
        """Executes the complete training and evaluation pipeline.
        
        Performs the following steps:
        1. Splits data into training and test sets
        2. Converts text to Word2Vec features
        3. Trains the logistic regression model
        4. Evaluates performance using precision, recall, and AUC
        5. Generates ROC curve visualization
        6. Creates LIME explanation for a sample prediction
        """
        X, y = DataLoader.split(self.data)
        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.word2vec_model.fit(X_train)  
        X_train_word2vec = self.word2vec_model.transform(X_train)  # Transform training data
        self.X_test_word2vec = self.word2vec_model.transform(X_test)  # Transform test data
   
        #fitting the word2vec model on the training data
        start_time = time.time()
        with tqdm(total=1, desc="Fitting model with Word2Vec") as pbar:
            self.lr_word2vec.fit(X_train_word2vec, self.y_train)
            pbar.update(1)
        end_time = time.time()

        print(f"Time taken for fitting with Word2Vec: {end_time - start_time:.2f} seconds")

        #predicting the training and testing data using fitted model
        y_train_pred = self.lr_word2vec.predict(X_train_word2vec)
        y_test_pred = self.lr_word2vec.predict(self.X_test_word2vec)

        #calculating precision and recall to further calculate F1
        print(precision_score(self.y_train, y_train_pred, average='binary'),
              recall_score(self.y_train, y_train_pred, average='binary'))
        print(precision_score(self.y_test, y_test_pred, average='binary'),
              recall_score(self.y_test, y_test_pred, average='binary'))

        # Calculate and plot AUC curve
        y_train_proba = self.lr_word2vec.predict_proba(X_train_word2vec)[:, 1]
        y_test_proba = self.lr_word2vec.predict_proba(self.X_test_word2vec)[:, 1]

        train_auc = roc_auc_score(self.y_train, y_train_proba)
        test_auc = roc_auc_score(self.y_test, y_test_proba)

        print(f"Train AUC: {train_auc:.2f}")
        print(f"Test AUC: {test_auc:.2f}")

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

        X_test_instance = X_test.iloc[0]  # Change index as needed
        filename = "lime_explanation_word2vec_orignal.html"  
        c_word2vec = make_pipeline(self.word2vec_model, self.lr_word2vec)
        explainer_word2vec = LimeTextExplainer(class_names=['male', 'female'])
        exp_word2vec = explainer_word2vec.explain_instance(X_test_instance, c_word2vec.predict_proba, num_features=5)
        exp_word2vec.save_to_file(filename)

    def test(self):
        """Evaluates model performance on a predefined test set.
        
        Tests the model on a curated set of quotes from various speakers
        and calculates the accuracy ratio of predictions.
        """
        test_cases_data = {
                        "input": [
                            "I stand before you today, humbled by the task before us, grateful for the trust you have bestowed, mindful of the sacrifices borne by our ancestors.",
                            "We realize the importance of our voices only when we are silenced. We must raise our voices and speak out against injustice.",
                            "There is no limit to what we, as women, can accomplish. We must believe in ourselves and our capabilities.",
                            "The future belongs to those who believe in the beauty of their dreams. We must not only dream but also act.",
                            "I have learned that as long as I hold fast to my beliefs and values, I will be able to overcome any obstacle.",
                            "In the face of adversity, we must not lose hope. Together, we can create a world where equality and justice prevail.",
                            "We cannot all succeed when half of us are held back. It is time to lift each other up and strive for equality.",
                            "I am not afraid to stand up for what I believe in. My voice is my power, and I will use it to inspire change.",
                            "The most common way people give up their power is by thinking they don’t have any. We must reclaim our power.",
                            "Let us not seek to satisfy our thirst for freedom by drinking from the cup of bitterness and hatred.",
                            "Ask not what your country can do for you – ask what you can do for your country.",
                            "The only thing we have to fear is fear itself.",
                            "Injustice anywhere is a threat to justice everywhere.",
                            "I have a dream that one day this nation will rise up and live out the true meaning of its creed.",
                            "Success is not final, failure is not fatal: It is the courage to continue that counts."
                        ],
                        "expected_output": [
                            1,  # Michelle Obama
                            1,  # Malala Yousafzai
                            1,  # Michelle Obama
                            1,  # Eleanor Roosevelt
                            1,  # Oprah Winfrey
                            1,  # Kamala Harris
                            1,  # Malala Yousafzai
                            1,  # Greta Thunberg
                            1,  # Alice Walker
                            1,  # Martin Luther King Jr. (included for contrast)
                            0,    # John F. Kennedy
                            0,    # Franklin D. Roosevelt
                            0,    # Martin Luther King Jr.
                            0,    # Martin Luther King Jr.
                            0     # Winston Churchill
                        ]
                    }
        test_cases_df = pd.DataFrame(test_cases_data)
        X = test_cases_df['input']
        X = self.word2vec_model.transform(X)
        y = test_cases_df['expected_output']
        
        # calculating prediction ratio 
        results = self.lr_word2vec.predict(X)
        correct_predictions = sum(results == y)
        total_predictions = len(y)
        accuracy_ratio = correct_predictions / total_predictions
        print(f"Accuracy Ratio: {accuracy_ratio:.2f}")

        
