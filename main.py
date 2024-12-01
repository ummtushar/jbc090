from dataloader import DataLoader
from datacleaner import DataCleaner

df_gender = DataLoader("data/gender.csv").load_data()

df_gender = DataCleaner(df_gender).scrubber(df_gender).gender_swap(df_gender)

# df_gender = df_gender.gender_swap(df_gender)


