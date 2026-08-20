# 📰 Fake News Detection Using BiLSTM and GloVe

A Natural Language Processing (NLP) and Deep Learning project that classifies news articles as **Real** or **Fake** using a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network with **pre-trained GloVe word embeddings**.

The project includes data preprocessing, text cleaning, tokenization, sequence padding, model training, evaluation using classification metrics and a confusion matrix, and prediction on new/unseen news text.

---

## 🚀 Project Overview

The rapid growth of online news and social media has made the spread of misleading and fabricated information a significant challenge.

This project explores how NLP and deep learning can be used to automatically classify news articles into two categories:

* ✅ **Real News**
* ❌ **Fake News**

The model learns patterns from the textual content of news articles and uses a Bidirectional LSTM architecture to perform binary classification.

---

## ✨ Features

* 📰 Real and fake news dataset processing
* 🧹 Text preprocessing and cleaning
* 🔤 Stopword removal
* 🔡 Lowercase conversion
* ✂️ Removal of non-alphabetic characters
* 🔢 Text tokenization
* 📏 Sequence padding
* 🧠 Bidirectional LSTM model
* 📚 Pre-trained GloVe word embeddings
* 📊 Training and validation monitoring
* 📈 Model accuracy and loss evaluation
* 🔲 Confusion matrix visualization
* 🔮 Prediction on new news articles
* 🏷️ Classification into Real or Fake

---

## 🛠️ Technologies Used

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Programming language           |
| Pandas       | Data manipulation              |
| NumPy        | Numerical operations           |
| NLTK         | Natural Language Processing    |
| TensorFlow   | Deep Learning framework        |
| Keras        | Neural network implementation  |
| Scikit-learn | Data splitting and evaluation  |
| Matplotlib   | Visualization                  |
| Seaborn      | Confusion matrix visualization |
| WordCloud    | Text visualization             |
| GloVe        | Pre-trained word embeddings    |
| BiLSTM       | Text classification model      |

---

## 🧠 Model Architecture

The project uses a **Bidirectional LSTM** neural network.

```text
                News Article
                     │
                     ▼
             Text Preprocessing
                     │
                     ▼
                Tokenization
                     │
                     ▼
               Sequence Padding
                     │
                     ▼
            GloVe Embedding Layer
                     │
                     ▼
          Bidirectional LSTM (15)
                     │
                     ▼
          Global Max Pooling 1D
                     │
                     ▼
             Dense Layer (1)
                     │
                     ▼
             Sigmoid Activation
                     │
                     ▼
          ┌────────────────────┐
          │                    │
          ▼                    ▼
       Real News           Fake News
```

### Model Configuration

* Maximum sequence length: **100**
* Maximum vocabulary size: **20,000**
* Embedding dimension: **50**
* LSTM units: **15**
* LSTM type: **Bidirectional**
* Pooling: **GlobalMaxPool1D**
* Output activation: **Sigmoid**
* Loss function: **Binary Crossentropy**
* Optimizer: **Adam**
* Batch size: **32**
* Epochs: **10**

---

## 📂 Dataset

The project uses two CSV files:

```text
Fake.csv
True.csv
```

The datasets contain news articles with information such as:

* Title
* Text
* Subject
* Date

The `subject` and `date` columns are removed during preprocessing.

The two datasets are then combined into a single dataset:

```text
fake_news.csv
```

### Dataset Statistics

The combined dataset contains:

```text
44,898 news articles
```

The labels are converted into numerical values:

```text
TRUE → 0
FAKE → 1
```

---

## 🔄 Data Preprocessing

The following preprocessing steps are applied to the news text.

### 1. Removing News Source Information

Reuters source information is removed from the article text when present.

### 2. Combining Title and Article Text

The title and processed article text are combined into a single feature:

```python
df['final news'] = df['title'] + " " + df['text processed']
```

### 3. Lowercasing

All text is converted to lowercase.

### 4. Stopword Removal

Common English stopwords are removed using NLTK.

### 5. Removing Non-Alphabetic Characters

Characters other than English alphabetic characters are removed.

### 6. Tokenization

The cleaned text is converted into numerical sequences using the Keras tokenizer.

### 7. Padding

All sequences are padded to a fixed length of:

```text
100 tokens
```

This produces the final input tensor used by the neural network.

---

## 📚 GloVe Word Embeddings

The project uses **pre-trained GloVe word embeddings** to represent words as numerical vectors.

The configured embedding dimension is:

```text
50
```

The embedding layer is initialized using the pre-trained GloVe vectors and kept non-trainable during model training.

This allows the model to use previously learned semantic relationships between words.

---

## 🧪 Train-Test Split

The dataset is divided into training and testing data using:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=0
)
```

### Split

* **80%** → Training data
* **20%** → Testing data

A validation split of **20%** is also used during training.

---

## 🏋️ Model Training

The BiLSTM model is trained using:

```text
Batch Size: 32
Epochs: 10
Optimizer: Adam
Loss: Binary Crossentropy
```

The model learns to distinguish between real and fake news based on patterns in the article text.

---

## 📊 Model Evaluation

The trained model is evaluated using:

* Training accuracy
* Testing accuracy
* Training loss
* Validation accuracy
* Validation loss
* Confusion matrix
* Prediction probabilities

### Current Model Result

The current experiment achieved approximately:

| Metric              |  Result |
| ------------------- | ------: |
| Training Accuracy   | ~52.30% |
| Testing Accuracy    | ~52.29% |
| Validation Accuracy | ~51.31% |

These results indicate that the current model configuration has **limited predictive performance** and is close to random classification.

The project therefore serves as an implementation and experimentation of an NLP + BiLSTM pipeline rather than a production-ready fake-news classifier.

---

## 📉 Training Performance

The training history tracks:

```python
accuracy
val_accuracy
loss
val_loss
```

These values can be plotted to analyze whether the model is learning effectively and to identify potential underfitting or other training issues.

---

## 🔲 Confusion Matrix

A confusion matrix is generated to compare:

```text
Actual Class
     vs
Predicted Class
```

The matrix contains predictions for:

* Real News
* Fake News

This helps analyze the types of classification errors made by the model.

---

## 🔮 Making Predictions

The project includes a function that accepts new news text and predicts whether it is Real or Fake.

Example:

```python
testsent = [
    "Example news article one...",
    "Example news article two..."
]

df_testsent = predict_text(testsent)
```

The prediction is converted using a probability threshold:

```python
prediction >= 0.5 → Fake
prediction < 0.5 → Real
```

The output contains the input text and its predicted category.

---

## 📁 Project Structure

```text
fake-news-detection-bilstm/
│
├── Fake News detection using lstm.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    ├── Fake.csv
    └── True.csv
```

> **Note:** Large datasets and GloVe embedding files may be better referenced through their original source rather than committed directly to the repository.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/fake-news-detection-bilstm.git
```

Move into the project directory:

```bash
cd fake-news-detection-bilstm
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Stopwords

Run:

```python
import nltk

nltk.download('stopwords')
```

### 4. Prepare the Dataset

Place the following files in the appropriate project directory:

```text
Fake.csv
True.csv
```

### 5. Prepare GloVe Embeddings

The model expects a 50-dimensional GloVe embedding file:

```text
glove.68.50d.txt
```

Update the `EMBEDDING_FILE` path in the notebook according to where the file is stored on your system.

---

## ▶️ How to Run

The project was developed using **Google Colab / Jupyter Notebook**.

### Step 1

Open:

```text
Fake News detection using lstm.ipynb
```

### Step 2

Upload or provide access to:

```text
Fake.csv
True.csv
glove.68.50d.txt
```

### Step 3

Run the notebook cells sequentially.

The notebook will:

```text
Load Dataset
      ↓
Clean Dataset
      ↓
Preprocess Text
      ↓
Tokenize Text
      ↓
Pad Sequences
      ↓
Load GloVe Embeddings
      ↓
Build BiLSTM Model
      ↓
Train Model
      ↓
Evaluate Model
      ↓
Generate Predictions
```

---

## 🧪 Example Prediction

Example input:

```text
Trey Gowdy discusses due process and the terror watch list.
```

The trained model produces a prediction probability, which is then converted into:

```text
Real
```

or

```text
Fake
```

depending on the model's output.

---

## 🔮 Future Improvements

The current model provides a foundation for experimentation. Future improvements could include:

* Increase BiLSTM capacity
* Tune the number of LSTM units
* Use trainable embeddings
* Use larger GloVe embeddings
* Implement pretrained transformer models such as BERT
* Experiment with DistilBERT or RoBERTa
* Improve text preprocessing
* Preserve word boundaries during cleaning
* Add attention mechanisms
* Perform hyperparameter tuning
* Use cross-validation
* Address dataset bias
* Add precision, recall and F1-score analysis
* Build a Streamlit web application
* Create a REST API for predictions
* Deploy the model as a web application
* Add real-time news classification

---

## ⚠️ Limitations

This project should not be treated as a reliable fact-checking system.

The current model achieved approximately **52.3% test accuracy**, indicating that substantial improvement is required before using the model for real-world classification.

Important limitations include:

* Model performance is dependent on the training dataset.
* News language can change over time.
* The model may learn dataset-specific patterns.
* Classification does not establish whether a news claim is factually true.
* The current model has relatively low predictive performance.
* The dataset may contain biases or artifacts.
* A news article's writing style alone cannot reliably determine factual accuracy.

---

## 🎯 Learning Outcomes

Through this project, the following concepts were explored:

* Natural Language Processing
* Text preprocessing
* Tokenization
* Sequence padding
* Word embeddings
* GloVe embeddings
* LSTM networks
* Bidirectional LSTM
* Binary classification
* TensorFlow/Keras
* Model evaluation
* Confusion matrices
* Prediction pipelines
* Deep learning for text classification

---

## 👩‍💻 Author

**NagaVenkataLakshmi HamsaVarshitha Divvela**

B.Tech – Computer Science & Engineering (AI & ML)

RVR & JC College of Engineering

**2025 Graduate**

---

## ⭐ Project

If you find this project useful for learning NLP and deep learning, consider giving the repository a ⭐ on GitHub.

---

## 📌 Disclaimer

This project is developed for **educational and research purposes** to demonstrate the application of NLP and deep learning techniques to news-text classification.

The predictions generated by the model should not be considered a substitute for professional fact-checking or verification from reliable sources.
