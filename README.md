# Machine Learning Project: Text Article Classification with Reuter Dataset using Support Vector Machine

This project focuses on building a text classification system using the Reuter Dataset, a widely used benchmark dataset in the field of Natural Language Processing (NLP). The goal is to classify multiple labeled text articles into predefined categories using Support Vector Machine (SVM) algorithm.

## Overview

In this project, we leverage the power of Support Vector Machine, a popular supervised learning algorithm, to classify text articles into different categories based on the content. The Reuter Dataset provides a diverse collection of news articles with multiple labels, making it suitable for multi-class classification tasks.

## Dataset

The Reuter Dataset consists of news articles categorized into various topics such as politics, business, sports, etc. Each article can belong to multiple categories, making it a multi-labeled classification problem. The dataset provides a rich source of text data for training and testing our classification model.

## Methodology

1. **Data Preprocessing**: Clean and preprocess the text data by removing stopwords, punctuation, and performing tokenization.

2. **Feature Extraction**: Convert the text data into numerical features using techniques like TF-IDF (Term Frequency-Inverse Document Frequency).

3. **Model Training**: Train a Support Vector Machine classifier using the preprocessed text features.

4. **Model Evaluation**: Evaluate the performance of the trained model using metrics like accuracy, precision, recall, and F1-score.

## Requirements

- Python 3.x
- Scikit-learn
- NumPy
- Pandas

## Usage

1. Clone this repository:

```bash
git clone https://github.com/yourusername/multi-label-text-classification.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook `text_classification_with_svm.ipynb` to train and evaluate the SVM model.

## Results

After training and evaluating the model, we achieve competitive performance in classifying text articles into multiple categories.

## References

- [Reuter Dataset](https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Support Vector Machine (SVM)](https://scikit-learn.org/stable/modules/svm.html)

## Author

[JAIVIK VAGHANI ](https://github.com/jaivikvaghani)

## License

This project is licensed under the MIT License - see the [GITHUB](Gti) file for details.

--- 

Feel free to customize this README according to your project specifics. Good luck with your machine learning project!
