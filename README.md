# Fake News Detection using Natural Language Processing (NLP)

## Project Overview

This project focuses on detecting fake news articles using Natural Language Processing (NLP), Machine Learning, and Deep Learning techniques. The system classifies news articles as either **Fake** or **Real** by analyzing their textual content.

The project follows the complete NLP pipeline, including data collection, preprocessing, exploratory data analysis (EDA), feature engineering, model development, evaluation, and comparison.

---

## Objectives

- Detect fake news articles automatically.
- Apply NLP preprocessing techniques.
- Develop Machine Learning and Deep Learning models.
- Compare model performance using evaluation metrics.
- Select the best-performing model for future deployment.

---

## Dataset

Dataset Used:
- Fake.csv
- True.csv

Source:
- Kaggle Fake and Real News Dataset

The dataset contains:
- News Title
- News Text
- Subject
- Date
- Label (Fake or Real)

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- TensorFlow / Keras
- WordCloud
- VS Code

---

## NLP Pipeline

### 1. Data Collection
- Load Fake.csv and True.csv
- Merge datasets
- Assign labels
- Save merged dataset

### 2. Data Preprocessing
- Lowercasing
- Remove punctuation
- Remove numbers
- Tokenization
- Stop-word removal
- Stemming
- Clean text

### 3. Exploratory Data Analysis (EDA)
- Dataset statistics
- Class distribution
- Word frequency analysis
- Word Clouds
- Visualizations

### 4. Feature Engineering
- TF-IDF Vectorization (for Naive Bayes)

### 5. Model Development

#### Machine Learning Model
- Multinomial Naive Bayes

#### Deep Learning Model
- Convolutional Neural Network (CNN)

### 6. Model Evaluation
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC

---

## Project Structure

```
Fake-News-Detection/
│
├── data/
│   ├── Fake.csv
│   ├── True.csv
│   └── merged_dataset.csv
│
├── graphs/
│   ├── class_distribution.png
│   ├── wordcloud_fake.png
│   ├── wordcloud_real.png
│   └── model_comparison.png
│
├── models/
│   ├── naive_bayes_model.pkl
│   └── cnn_model.keras
│
├── 1_data_collection.py
├── 2_preprocessing.py
├── 3_eda.py
├── 4_feature_engineering.py
├── 5_model_naive_bayes.py
├── 6_model_cnn.py
├── 7_evaluation.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/fake-news-detection.git
```

Navigate to the project:

```bash
cd fake-news-detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the files in the following order:

```bash
python 1_data_collection.py
```

```bash
python 2_preprocessing.py
```

```bash
python 3_eda.py
```

```bash
python 4_feature_engineering.py
```

```bash
python 5_model_naive_bayes.py
```

```bash
python 6_model_cnn.py
```

```bash
python 7_evaluation.py
```

---

## Evaluation Metrics

The following metrics are used to compare model performance:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score

---

## Results

The Naive Bayes and CNN models are trained and evaluated on the Fake News dataset. Their performances are compared using standard evaluation metrics, and the best-performing model is selected for future deployment.

---

## Future Improvements

- Deploy the best model using Flask.
- Improve CNN accuracy using pre-trained word embeddings.
- Add BERT for performance comparison.
- Develop a user-friendly web interface.

---

## Author

**Denuri Vilara**

BSc (Hons) Data Science

Sri Lanka Technology Campus (SLTC)

---

## License

This project is developed for academic purposes.
