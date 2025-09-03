# ⚖️ Cross-Lingual Telugu Legal Text Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange)](https://scikit-learn.org/)

This project presents a novel machine learning method for the cross-lingual classification of Telugu legal documents. By leveraging **FastText word embeddings** trained on summarized English judgments, the system can accurately categorize Telugu legal text without requiring a large labeled Telugu dataset. This research contributes to advancing cross-lingual Natural Language Processing (NLP) and provides a practical tool for legal professionals managing linguistic diversity.

***

### 📌 Project Overview

The primary goal of this study is to bridge the language gap in legal text analysis. We apply six different machine learning algorithms to the task of classifying Telugu legal documents into relevant legal domains. The models are trained using knowledge derived from English-language legal summaries, demonstrating an effective cross-lingual transfer learning approach. The study culminates in identifying the most accurate model, providing a swift and reliable tool for the analysis of Telugu legal content.

***

### ✨ Key Features

* **Cross-Lingual Approach:** Classifies Telugu text by leveraging word embeddings and patterns learned from English legal summaries.
* **FastText Embeddings:** Utilizes advanced FastText word embeddings to capture semantic relationships that transcend language barriers.
* **Comprehensive Model Comparison:** Implements and rigorously evaluates six classical machine learning models:
    * K-Nearest Neighbors (KNN)
    * Random Forest
    * Decision Tree
    * Multi-layer Perceptron (MLP)
    * Support Vector Machine (SVM)
    * Logistic Regression
* **High Accuracy:** The **Random Forest** classifier was identified as the top-performing model, demonstrating high accuracy in classifying legal domains.
* **Practical Application:** Offers a valuable tool for scholars and legal professionals, accelerating the process of sorting and analyzing Telugu legal documents.

***

### ⚙️ Methodology & Workflow

The workflow of the project is structured as follows:

1.  **Data Collection:** Gathering a corpus of Telugu legal documents for classification and a parallel or comparable corpus of summarized English judgments for training the embeddings.
2.  **Word Embeddings:** Generating cross-lingual word embeddings using the **FastText** model. This allows the system to understand the semantic context of Telugu words based on the English legal corpus.
3.  **Model Training:** Training the six different machine learning classifiers on the vector representations of the text data.
4.  **Cross-Lingual Classification:** Applying the trained models to classify the unseen Telugu legal text into predefined categories.
5.  **Evaluation and Comparison:** Systematically evaluating each model's performance using standard metrics (Accuracy, Precision, Recall, F1-Score) to identify the most effective algorithm.

***

### 📊 Results & Key Findings

Through extensive comparative studies, the following key result was established:

* The **Random Forest** model achieved the highest classification accuracy among the six tested algorithms, proving to be the most effective for this cross-lingual task.
* The overall system demonstrated a high degree of reliability in correctly categorizing documents into their respective legal domains.

***

### 🛠️ Tech Stack

* **Programming Language:** Python 3
* **Machine Learning:** Scikit-learn
* **NLP & Embeddings:** Gensim / `fasttext`
* **Data Manipulation:** Pandas, NumPy
* **Development Environment:** Jupyter Notebook

***

### 🚀 Getting Started

To get a local copy up and running, follow these steps.

#### 1. Prerequisites

Make sure you have Python 3.8+ and pip installed on your system.

#### 2. Clone the Repository

```
git clone [https://github.com/Sravan94-git/Domain-classification-of-legal-documents.git](https://github.com/Sravan94-git/Domain-classification-of-legal-documents.git)
cd Domain-classification-of-legal-documents
```

#### 3.Set Up Data and Embeddings
```
This project requires a specific dataset and pre-trained models.

Dataset: Place your Telugu and English legal text files (.txt, .csv) inside a /data directory in the project root.

FastText Model: Download a pre-trained FastText model or train your own. Place the model file (e.g., model.bin) inside a /models directory.
```

#### 4. Install Dependencies
```
It is recommended to use a virtual environment.

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate    # On Windows

# Install the required packages
pip install -r requirements.txt
```
