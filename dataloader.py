import pandas as pd


class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.filepath)
        return self.df
    
    @staticmethod
    def split(df) -> tuple:
        X, y = df['post'], df['female']
        return X, y

