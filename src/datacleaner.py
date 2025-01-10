import pandas as pd
import scrubadub_spacy, scrubadub
import spacy

class DataCleaner:
    """A class for cleaning and processing text data in a DataFrame.
    
    This class provides methods to scrub sensitive information from text
    and to swap gendered terms within the text.
    
    Attributes:
        df (pd.DataFrame): Input DataFrame containing text data to be cleaned.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def scrubber(self, df):
        """Cleans the text data in the DataFrame by removing sensitive information.
        
        This method uses the scrubadub library along with a spaCy model to detect
        and remove sensitive entities from the 'post' column of the DataFrame.
        
        Returns:
            pd.DataFrame: The cleaned DataFrame with sensitive information scrubbed.
        """
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
        """Swaps gendered terms in the text data within the DataFrame.
        
        This method replaces male terms with female counterparts and vice versa
        in the 'post' column of the DataFrame.
        
        Returns:
            pd.DataFrame: The DataFrame with gendered terms swapped.
        """
        def change_gender(string):
            """Replaces gendered words in the input string based on a predefined dictionary.
            
            Args:
                string (str): The input string to process.
                
            Returns:
                str: The modified string with gendered terms swapped.
            """
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

        new_rows = []
        for index, row in self.df.iterrows():
            modified_post = change_gender(row['post'])
            modified_label = row['female']
            new_rows.append({'post': modified_post, 'female': modified_label})
        
        new_df = pd.DataFrame(new_rows)
        self.df = pd.concat([new_df, self.df], ignore_index=True)

        return self.df



