import pandas as pd
import scrubadub_spacy, scrubadub
import spacy

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def scrubber(self, df):
        nlp = spacy.load("en_core_web_sm")
        scrubber = scrubadub.Scrubber()
        
        # Add the SpacyEntityDetector with the loaded model
        scrubber.add_detector(scrubadub_spacy.detectors.SpacyEntityDetector(model="en_core_web_sm"))

        for index, row in self.df.iterrows():
            text = row['post']
            result = scrubber.clean(text)
            self.df.at[index, 'post'] = result

        return self.df
    
    def gender_swap(self, df):
        def change_gender(string):
            # A Dictionary to store the mapping of genders
            # The user can add his words too.
            dictionary = {
                "batman": "batwoman", "batwoman": "batman",
                "boy": "girl", "girl": "boy",
                "boyfriend": "girlfriend", "girlfriend": "boyfriend",
                "father": "mother", "mother": "father",
                "husband": "wife", "wife": "husband",
                "he": "she", "she": "he",
                "his": "her", "her": "his",
                "male": "female", "female": "male",
                "man": "woman", "woman": "man",
                "Mr": "Ms", "Ms": "Mr",
                "sir": "madam", "madam": "sir",
                "son": "daughter", "daughter": "son",
                "uncle": "aunt", "aunt": "uncle",
            }
        
            string += ' '  # Append a space at the end
        
            n = len(string)
        
            # 'temp' string will hold the intermediate words
            # and 'ans' string will be our result
            temp = ""
            ans = ""
        
            for i in range(n):
                if string[i] != ' ':
                    temp += string[i]
                else:
                    # If this is a 'male' or a 'female' word then
                    # swap this with its counterpart
                    if temp in dictionary:
                        temp = dictionary[temp]
        
                    ans += temp + ' '
                    temp = ""
        
            return ans

        for index, row in self.df.iterrows():
            self.df.at[index, 'post'] = change_gender(row['post'])

        return self.df



