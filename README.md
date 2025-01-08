# 🚀 Codebase for the course: JBC090- Language and AI
### *Impact of Scrubber and Gender Swapping on predictions of NLP models*

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
- [♻️ Reproduction](https://github.com/ummtushar/jbc090#-reproduction)
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

   Alternatively, you can install only the packages that are directly imported in your `src/nlp.py` file:
   ```bash
   pip install gensim==4.3.3 lime==0.2.0.1 matplotlib==3.9.2 numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2 tqdm==4.67.0
   ```

4. Download the necessary SpaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## ♻️ Reproduction

To run the project, you can execute the main script. Here’s how to do it:

1. Ensure your data files are in the `data` directory.
2. Open a terminal and run:
   ```bash
   python src/main.py
   ```

3. The script will load the data, clean it, and train the models. You can modify the script to change parameters or add new functionalities.


## 🔋 Resources

We used a MacBook with an Apple Silicon M2 Pro chip to run the logistic regression models with TF-IDF and Word2Vec embeddings. Loading and cleaning the data takes approximately 30 minutes. However, the Scrubber took a little over 12 hours to run. The TfidfLogisticRegression model trains in a few minutes, while the Word2VecLogisticRegression model takes about 30 minutes to an hour, depending on dataset size or split size (In our case, it was a 80/20 split). The entire process, including data handling and model training, completes in 14-15 hours. Approximately 16GB of RAM is recommended to handle data processing and embeddings efficiently.

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

- **Data Source**: You can change the data source by modifying the file path in the `DataLoader` class.
  - **File Path**: Update the path in `src/main.py` to load a different dataset.
   

- **Data Cleaning**: Adjust the cleaning process by modifying the `scrubber` and `gender_swap` methods in the `DataCleaner` class.
  - **Scrubber**: Modify the cleaning logic in `notebooks/eda.ipynb`.


  - **Gender Swapping**: Change the gender swapping logic in `notebooks/eda.ipynb`.


### Model Training and Evaluation

- **Model Parameters**: You can adjust the parameters of the logistic regression models in the `TfidfLogisticRegression` and `Word2VecLogisticRegression` classes.
  - **TF-IDF Model**: Modify parameters in `src/nlp.py`.
  

  - **Word2Vec Model**: Adjust the Word2Vec training settings in `src/nlp.py`.
 

### Testing and Validation

- **Test Cases**: You can change the test cases to evaluate different inputs by modifying the `test` method in the `TfidfLogisticRegression` and `Word2VecLogisticRegression` classes.
  - **Test Inputs**: Update the test cases in `src/nlp.py`.


### Additional Configurations

- **Scrubber and SpaCy Model**: Ensure the correct SpaCy model is downloaded and used for text processing.
  - **SpaCy Model**: Download the necessary model in `notebooks/eda.ipynb`.


By following these guidelines, you can effectively manipulate the experimental setup to explore different scenarios and outcomes. Adjust the parameters and logic as needed to suit your research or project goals.


## 🪪 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## 📞 Contact

For any inquiries, please contact:

- Group - [Tushar Gupta](mailto:t.gupta@student.tue.nl), [Polina Stepanova](mailto:p.stepanova@student.tue.nl), [Noa Verrijt](mailto:n.f.verrijt@student.tue.nl), [Jasmijn Verhaegh](mailto:j.m.verhaegh@student.tue.nl)

---

Thank you for visiting this project! We hope you find it useful for your text analysis needs.
