# FB-Post-Classification-with-NLP-BERT

## Facebook Post Classification Using BERT
Project Overview
In this project, we aim to build a text classification model using BERT (Bidirectional Encoder Representations from Transformers) to classify Facebook posts into three categories based on their content: Appreciation, Complaint, and Feedback. Businesses leverage these insights to better understand customer needs and preferences, drawing from a rich dataset of user-generated content on Facebook business pages.

## Dataset
The dataset used, "FB_posts_labeled.txt", consists of 7961 user-generated posts on various business pages, each labeled under one of three categories: Appreciation (1), Complaint (2), and Feedback (3). An additional set of 2039 unlabeled posts, "FB_posts_unlabeled.txt", is used for testing the model's performance.

## Model
We use TensorFlow and the BERT model from TensorFlow Hub, fine-tuning it on our labeled dataset. BERT, which stands for Bidirectional Encoder Representations from Transformers, processes words in relation to all the other words in a sentence, rather than one-by-one in order. This allows it to understand the context of a word based on all of its surroundings (left and right).

## Dependencies
- TensorFlow 2.13+
- TensorFlow Text 2.13+
- TensorFlow Hub
- TensorFlow Models Official
Install the necessary libraries using:

```python
pip install tensorflow tensorflow-text tensorflow-hub tf-models-official
```

## Setup
1. Data Preprocessing: Split the data into training and test sets.
2. Model Selection: Load a BERT model and its corresponding preprocessing model from TensorFlow Hub.
3. Model Training: Train the classifier using the selected BERT model with the AdamW optimizer.
4. Evaluation: Evaluate the model on the test dataset using the F1-score as the performance metric. The highest submission on the test dataset has an F-score of 0.8403721.

## Usage
```python
# Load the model
classifier_model = build_classifier_model()

# Predict on new data
predictions = classifier_model.predict(["Your customer service is excellent!"])
```

## Results
The fine-tuned BERT model achieved an F1 score of 0.834 on the test set, indicating a high level of accuracy in classifying posts into the specified categories. These results underscore the potential of advanced NLP models like BERT to understand and categorize customer feedback effectively.

