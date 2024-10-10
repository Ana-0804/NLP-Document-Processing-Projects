
# **Sentiment Analysis on Product Reviews**

## **Overview**
This project implements a basic sentiment analysis system to classify product reviews as either positive or negative. It utilizes natural language processing (NLP) techniques for text preprocessing and a machine learning model (Naive Bayes) for classification.

## **Objective**

The goal is to classify product reviews based on their text content, determining whether they express a positive or negative sentiment.

## **Tools and Libraries Used**

- **NLTK**: For natural language preprocessing
- **scikit-learn**: For implementing and evaluating the machine learning model
- **Pandas**: For data handling and manipulation.

## **Dataset**

The dataset consists of a small collection of manually labeled product reviews. Each review is labeled as "positive" (1) or "negative" (0).


## **Project Workflow**

### **1. Text Preprocessing**

- **Tokenization**: Breaking down text into individual words
- **Stop-word Removal**:  Eliminating common words that do not contribute much meaning.
- **Vectorization**: Converting cleaned text into numerical features using CountVectorizer from scikit-learn.


### **2. Model Training**

- A **Naive Bayes Classifier**  is used for classification, with the dataset split into training (75%) and testing (25%) sets.

### **3. Model Evaluation**

- The model's accuracy is measured by comparing predicted sentiment labels with true labels in the test set. The model achieved an accuracy of 85% on the small dataset.

## **Challenges**

- **Limited Dataset**: The small dataset may affect the model's ability
- **Feature Selection**: A simple bag-of-words representation was used; exploring more advanced techniques like TF-IDF could improve feature extraction.

## **Installation**

To run this project, make sure you have the following libraries installed.

`pip install nltk scikit-learn pandas`


## **Usage**

1. **Download the NLTK stopwords** 

`nltk.download('stopwords')`
  
2. **Run the sentiment analysis script**

3. **View Results**: The accuracy of the model will be printed.


