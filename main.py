from dataloader import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from nlp import TfidfLogisticRegression, Word2VecLogisticRegression


# df_gender = DataLoader("data/gender.csv").load_data()
# data_cleaner = DataCleaner(df_gender)
# df_gender = data_cleaner.scrubber(df_gender)
# df_gender = data_cleaner.gender_swap(df_gender)
df_gender = pd.read_csv("data/df_gender_augmented.csv") # REMOVE THIS LINE, ONLY THE LINES ABOVE SHOULD FIT

X, y = DataLoader.split(df_gender)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


df_gender = pd.read_csv("data/df_gender_augmented.csv")
tfidf_model = TfidfLogisticRegression(df_gender)
tfidf_model.run()
tfidf_model.explain_instance(X_test.iloc[0], 'lime_explanation.html')
word2vec_model = Word2VecLogisticRegression(df_gender)
word2vec_model.run()
word2vec_model.explain_instance(X_test.iloc[0], 'lime_explanation_word2vec.html')
