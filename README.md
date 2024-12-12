# Project Title

## Overview

This project is designed to analyze and process text data, focusing on gender-related content. It includes functionalities for data loading, cleaning, and applying machine learning models to classify text based on gender. The project utilizes various libraries such as Pandas, Scikit-learn, and SpaCy for data manipulation and natural language processing.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
- [License](#license)
- [Contact](#contact)

## Features

- Load and preprocess text data from CSV files.
- Clean text data using SpaCy and scrubadub for sensitive information.
- Gender swapping functionality to analyze gender bias in text.
- Train and evaluate machine learning models (Logistic Regression, SVM, KNN) for gender classification.
- Use of LIME for model interpretability.
- Support for Word2Vec embeddings for enhanced text representation.

## Installation

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

4. Download the necessary SpaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

To run the project, you can execute the main script. Here’s how to do it:

1. Ensure your data files are in the `data` directory.
2. Open a terminal and run:
   ```bash
   python src/main.py
   ```

3. The script will load the data, clean it, and train the models. You can modify the script to change parameters or add new functionalities.

## Data

The project expects the following CSV file in the `data` directory:

- `gender.csv`: Contains text data with gender labels.

Make sure the data files are formatted correctly as expected by the `DataLoader` class.

## Models

The project implements the following models:

- **TfidfLogisticRegression**: A logistic regression model using TF-IDF features.
- **Word2VecLogisticRegression**: A logistic regression model using Word2Vec embeddings.

Each model is trained and evaluated on the provided datasets, and results are printed to the console.


## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries, please contact:

- Your Name - [Tushar Gupta](mailto:t.gupta@student.tue.nl), [Polina Stepanova](mailto:p.stepanova@student.tue.nl), [Noa Verrijt](mailto:n.f.verrijt@student.tue.nl), [Jasmijn Verhaegh](mailto:j.m.verhaegh@student.tue.nl)

---

Thank you for visiting this project! We hope you find it useful for your text analysis needs.
