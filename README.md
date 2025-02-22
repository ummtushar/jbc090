# 🚀 Codebase for **"Impact of Scrubber and Gender Swapping on predictions of NLP models"**
### *JBC090: Language and AI*

This repository is for the experiments described in ["Impact of Scrubber and Gender Swapping on predictions of NLP models"](https://plum-roxie-42.tiiny.site). If you use anything related to the repository or paper, please cite the following work:


```bibtex
@techreport{jbc090-2024,
  author       = {Jasmijn Verhaegh and Tushar Gupta and Polina Stepanova and Noa Verrijt},
  title        = {Impact of Scrubber and Gender Swapping on Predictions of NLP Models},
  year         = {2024},
  institution  = {Eindhoven University of Technology},
  url          = {https://github.com/ummtushar/jbc090.git},
  note         = {Technical report}
}
```

## 📆 Overview

This project is designed to analyze and process text data, focusing on gender-related content. It includes functionalities for data loading, cleaning, and applying machine learning models to classify text based on gender. The project utilizes various libraries such as Pandas, Scikit-learn, gensim and SpaCy for data manipulation and natural language processing.

## Table of Contents

- [✅ Tl;dr](https://github.com/ummtushar/jbc090#-tl-dr)
- [🧭 Installation](https://github.com/ummtushar/jbc090#-installation)
- [🔃 Reproduction](https://github.com/ummtushar/jbc090#-reproduction)
- [🔋 Resources](https://github.com/ummtushar/jbc090#-resources)
- [📈 Data](https://github.com/ummtushar/jbc090#-data)
- [🤖 Models](https://github.com/ummtushar/jbc090#-models)
- [🧪 Experimental Manipulation](https://github.com/ummtushar/jbc090#-experimental-manipulation)
- [🪪 License](https://github.com/ummtushar/jbc090#-license)
- [📞 Contact](https://github.com/ummtushar/jbc090#-contact)

## ✅ Tl;dr

- Load and preprocess text data from CSV files.
- Clean text data using SpaCy and scrubadub for sensitive information.
- Gender swapping functionality to analyze gender bias in text.
- Train and evaluate machine learning models (Logistic Regression) for gender classification.
- Use of LIME for model interpretability.
- Support for Word2Vec embeddings for enhanced text representation.

## 🧭 Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/ummtushar/jbc090.git
   cd jbc090
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file consist of the following (main and important) libraries:
   ```
   gensim==4.3.3
   lime==0.2.0.1
   matplotlib==3.9.2
   numpy==1.26.4
   pandas==2.2.3
   scikit-learn==1.5.2
   tqdm==4.67.0
   ```

4. Download the necessary SpaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 🔃 Reproduction

To run the project, you can execute the main script. Here’s how to do it:

1. Ensure your data files are in the `data` directory.
2. Open a terminal and run:
   ```bash
   python src/main.py
   ```

3. The script will load the data, clean it, and train the models. You can modify the script to change parameters or add new functionalities.


## 🔋 Resources

We used a MacBook with an Apple Silicon M2 Pro chip to run the logistic regression models with TF-IDF and Word2Vec embeddings. Loading and cleaning the data takes approximately 30 minutes. However, the Scrubber took a little over 12 hours to run. The TfidfLogisticRegression model trains in a few minutes, while the Word2VecLogisticRegression model takes about 30 minutes to an hour, depending on dataset size or split size (In our case, it was a 80/20 split). The entire process, including data handling and model training, completes in 14-15 hours. Approximately 16GB of RAM is recommended to handle data processing and embeddings efficiently.

On using [Deloitte's Emission Calculator](https://www.deloitte.com/uk/en/services/consulting/content/ai-carbon-footprint-calculator.html), the CO2 emissions were calculated to be a score of 5 between a scale of 1 to 10 indicating that there was a relatively medium emissions compared to other ML tasks requiring similar compute. The results can found [here](https://i.postimg.cc/Qtb841yQ/Screenshot-2025-01-09-at-13-06-40.png).

## 📈 Data

The project expects the following CSV file in the `data` directory:

- `gender.csv`: Contains text data with gender labels.

Make sure the data files are formatted correctly as expected by the `DataLoader` class.

## 🤖 Models

The project implements the following models:

- **TfidfLogisticRegression**: A logistic regression model using TF-IDF features.
- **Word2VecLogisticRegression**: A logistic regression model using Word2Vec embeddings.

Each model is trained and evaluated on the provided datasets, and results are printed to the console.

## 🧪 Experimental Manipulation

This section provides guidance on how to modify various elements of the experiment to explore different outcomes or configurations. Below are the key components that can be adjusted:

### Data Loading and Cleaning

- **Data Source**: You can change the data source by modifying the file path in the `DataLoader` class in `src/main.py`:
  ```python:src/main.py
  # Change the input file path here
  df_gender = DataLoader("./data/gender.csv").load_data()
  ```

- **Data Cleaning**: Adjust the cleaning process by modifying the `scrubber` and `gender_swap` methods:
  - **Scrubber**: Modify the cleaning model in `src/datacleaner.py`:
  ```python:src/dataclearner.py
  def scrubber(self, df):
        nlp = spacy.load("en_core_web_sm")
        scrubber = scrubadub.Scrubber()
        
        # Add the SpacyEntityDetector with the changed model
        scrubber.add_detector(scrubadub_spacy.detectors.SpacyEntityDetector(model="en_core_web_sm"))

        for index, row in self.df.iterrows():
            text = row['post']
            result = scrubber.clean(text)
            self.df.at[index, 'post'] = result
  ```

  - **Gender Swapping**: Change the gender swapping logic in the dictionary in  `src/dataclearner.py`:
  ```python:src/dataclearner.py
  def gender_swap(self, df):
        def change_gender(string):
            # Change the logic in the dictionary below
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
  ```

### Model Training and Evaluation

- **Model Parameters**: You can adjust the parameters of the logistic regression models:
  - **TF-IDF Model**: Modify parameters in `src/nlp.py`:
  ```python:src/nlp.py
  class TfidfLogisticRegression:
      def __init__(self):
          # Modify TF-IDF parameters
          self.vectorizer = TfidfVectorizer(
              max_features=5000,
              min_df=5,
              max_df=0.7
          )
          # Modify LogisticRegression parameters
          self.model = LogisticRegression(
              C=1.0,
              max_iter=100,
              random_state=42
          )
  ```

  - **Word2Vec Model**: Adjust the Word2Vec training settings in `src/nlp.py`:
  ```python:src/nlp.py
  class Word2VecLogisticRegression:
      def __init__(self):
          # Modify Word2Vec parameters
          self.w2v_model = Word2Vec(
              vector_size=100,
              window=5,
              min_count=1,
              workers=4
          )
          # Modify LogisticRegression parameters
          self.model = LogisticRegression(
              C=1.0,
              max_iter=100,
              random_state=42
          )
  ```

### Testing and Validation

- **Test Cases**: You can modify the test method to evaluate different inputs in `src/nlp.py`:
  ```python:src/nlp.py
  def test(self):
      # Add or modify test cases
      test_cases_data = {
          "input": [
          "Example text 1",
          "Example text 2"
            ],
      expected_output = {
        "input": [
        "1",
        "0"
          ],
      }
      test_cases_df = pd.DataFrame(test_cases_data)
      X = test_cases_df['input']
      X = self.tfidf.transform(X)
      y = test_cases_df['expected_output']
      
      results = self.model.predict(X)
      correct_predictions = sum(results == y)
      total_predictions = len(y)
      accuracy_ratio = correct_predictions / total_predictions
      print(f"Accuracy Ratio: {accuracy_ratio:.2f}")
  ```

### Additional Configurations

- **Scrubber and SpaCy Model**: Ensure the correct SpaCy model is downloaded:
  ```python:src/nlp.py
  # Download and configure SpaCy model
  !python -m spacy download en_core_web_sm
  nlp = spacy.load("en_core_web_sm")
  ```

By following these guidelines and modifying the code snippets shown above, you can effectively manipulate the experimental setup to explore different scenarios and outcomes. Adjust the parameters and logic as needed to suit your research or project goals.


## 🪪 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## 📞 Contact

For any inquiries, please contact:

- Group - [Tushar Gupta](mailto:t.gupta@student.tue.nl), [Polina Stepanova](mailto:p.stepanova@student.tue.nl), [Noa Verrijt](mailto:n.f.verrijt@student.tue.nl), [Jasmijn Verhaegh](mailto:j.m.verhaegh@student.tue.nl)

---

Thank you for visiting this project! We hope you find it useful for your text analysis needs.
