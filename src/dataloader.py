import pandas as pd


class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads the CSV data and returns dataframe.
        """
        self.df = pd.read_csv(self.filepath)
        return self.df
    
    @staticmethod
    def split(df) -> tuple:
        """
        Splits the data from the posts (X) and gender (y) for classification.
        """
        X, y = df['post'], df['female']
        return X, y

