from dataloader import DataLoader
import pandas as pd
from nlp import TfidfLogisticRegression, Word2VecLogisticRegression
from datacleaner import DataCleaner


df_gender = DataLoader("./data/gender.csv").load_data() 
data_cleaner = DataCleaner(df_gender)
df_gender = data_cleaner.scrubber(df_gender) #please note that this can take 10+ hours, for ease, I have uplodaded a file called df_geneder_augmented.csv that can be run directly under the link - https://tuenl-my.sharepoint.com/:x:/g/personal/t_gupta_student_tue_nl/EXvzzx2mKPBIo1j_x_BCMeMBGdDdSoZeo2vpGlZtbeODPg?e=MwdNws 
df_gender = data_cleaner.gender_swap(df_gender) 

tfidf_model = TfidfLogisticRegression(df_gender)
tfidf_model.run()
word2vec_model = Word2VecLogisticRegression(df_gender)
word2vec_model.run()

