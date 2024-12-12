from dataloader import DataLoader
import pandas as pd
from nlp import TfidfLogisticRegression, Word2VecLogisticRegression
from datacleaner import DataCleaner


# df_gender = DataLoader("./data/gender.csv").load_data() 
# data_cleaner = DataCleaner(df_gender)
# df_gender = data_cleaner.scrubber(df_gender) #please note that this can take 10+ hours, for ease, I have uplodaded a file called df_geneder_augmented.csv that can be run directly
# df_gender = data_cleaner.gender_swap(df_gender)
df_gender = pd.read_csv("./data/df_gender_augmented.csv") # REMOVE THIS LINE, ONLY THE LINES ABOVE SHOULD FIT

tfidf_model = TfidfLogisticRegression(df_gender)
tfidf_model.run()
word2vec_model = Word2VecLogisticRegression(df_gender)
word2vec_model.run()

