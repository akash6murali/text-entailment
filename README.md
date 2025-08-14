# Text Entailment Project

A comprehensive machine learning project for text entailment classification, implementing multiple approaches from baseline models to advanced neural architectures.

## 🎯 Project Overview

This project addresses the Natural Language Inference (NLI) task, also known as Text Entailment. Given a premise (fact) and a hypothesis, the system predicts whether the hypothesis is:

- **Entailment**: The premise logically supports the hypothesis
- **Neutral**: The premise neither supports nor contradicts the hypothesis
- **Contradiction**: The premise logically contradicts the hypothesis

## 🏗️ Project Architecture

```
text_entailment_cursor/
├── code/                           # Main source code
│   ├── main.py                     # Entry point and orchestration
│   ├── text_entailment__util/      # Core utility modules
│   ├── baseline_model.py           # Simple baseline implementation
│   ├── bert_pretrained.py          # BERT-based models
│   ├── sentence_bert_similarity.py # Sentence-BERT implementation
│   ├── bilstm_encoder.py          # BiLSTM with attention
│   └── snli_data_explorer.py      # Data exploration and analysis
├── data/                           # Data storage
│   ├── input_data/                 # Raw input datasets
│   └── process_data/               # Preprocessed data
├── docs/                           # Documentation and visualizations
├── log/                            # Execution logs
└── requirements.txt                 # Python dependencies
```

## 🚀 Features

### 1. **Baseline Models** (`text_entailment__util/baseline_models.py`)

- **Logistic Regression**: Linear classification with TF-IDF features
- **Support Vector Machine (SVM)**: Kernel-based classification
- **Random Forest**: Ensemble tree-based classification
- Automatic confusion matrix visualization and ROC-AUC scoring

### 2. **Neural Network Models** (`text_entailment__util/attention_models.py`)

- **LSTM**: Long Short-Term Memory networks
- **BiLSTM**: Bidirectional LSTM with attention mechanisms
- Sequence padding and vocabulary management
- Memory usage monitoring and performance tracking

### 3. **Transformer Models** (`bert_pretrained.py`)

- **BERT**: Bidirectional Encoder Representations from Transformers
- **Sentence-BERT**: Sentence-level BERT embeddings
- Custom dataset classes for SNLI format
- Training with early stopping and learning rate scheduling

### 4. **Data Processing Pipeline** (`text_entailment__util/data_preprocessing.py`)

- Text tokenization and cleaning
- Stop word removal and lemmatization
- Sentence boundary detection
- Automated data preprocessing and storage

### 5. **Utility Functions** (`text_entailment__util/`)

- **Project Logging**: Structured logging with timestamps
- **Data Modeling**: Feature engineering and dataset preparation
- **Model Evaluation**: Comprehensive metrics and visualizations

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)
- 8GB+ RAM (16GB+ recommended for large datasets)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd text_entailment_cursor
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 📊 Data Setup

### SNLI Dataset

The project is designed to work with the Stanford Natural Language Inference (SNLI) dataset:

1. Download SNLI dataset from [Stanford NLP](https://nlp.stanford.edu/projects/snli/)
2. Place `snli_1.0_train.jsonl` in `data/input_data/` directory
3. The preprocessing pipeline will automatically handle the data format

### Data Format

Each line in the JSONL file should contain:

```json
{
  "sentence1": "premise text",
  "sentence2": "hypothesis text",
  "gold_label": "entailment|neutral|contradiction"
}
```

## 🚀 Usage

### Quick Start

```bash
# Run the complete pipeline
python code/main.py
```

### Individual Components

#### 1. Data Preprocessing

```python
from text_entailment__util.data_preprocessing import preprocessing

# Preprocess SNLI data
df = preprocessing('path/to/snli_1.0_train.jsonl')
```

#### 2. Baseline Models

```python
from text_entailment__util.baseline_models import logistic_regression, random_forest

# Train and evaluate models
logistic_regression(X_train, X_test, y_train, y_test)
random_forest(X_train, X_test, y_train, y_test)
```

#### 3. Neural Models

```python
from text_entailment__util.attention_models import lstm_model, blstm_model

# Train LSTM models
lstm_model(X_train, X_test, y_train, y_test, vocab_size, max_seq_len, label_encoder)
```

#### 4. BERT Models

```python
from code.bert_pretrained import BertNLITrainer

# Initialize BERT trainer
trainer = BertNLITrainer('bert-base-uncased')
# Load and train data
trainer.load_data('path/to/data.jsonl')
trainer.train()
```

## 📈 Model Performance

The project includes comprehensive evaluation metrics:

- **Classification Report**: Precision, Recall, F1-Score
- **Confusion Matrix**: Visual representation of predictions
- **ROC-AUC Score**: Model discrimination ability
- **Memory Usage**: Resource consumption monitoring
- **Training Time**: Performance benchmarking

## 🔧 Configuration

### Model Parameters

- **Word2Vec**: `vector_size=100`, `window=5`, `min_count=1`
- **LSTM**: `units=128/64`, `dropout=0.2`, `epochs=5`
- **BERT**: `max_length=128`, `batch_size=16`, `learning_rate=2e-5`

### Data Parameters

- **Test Split**: 20% for evaluation
- **Sequence Length**: Configurable maximum length
- **Vocabulary Size**: Dynamic based on dataset

## 📝 Logging

The project uses structured logging with:

- Timestamp information
- Log levels (INFO, ERROR, DEBUG)
- Performance metrics
- Error tracking and debugging

Logs are stored in the `log/` directory with timestamped filenames.

## 🧪 Experimentation

### Jupyter Notebooks

- `EDA.ipynb`: Exploratory data analysis
- `bert_pretrained.ipynb`: BERT model experimentation
- `Baseline_model.ipynb`: Baseline model development

### Hyperparameter Tuning

- Model architectures can be modified in respective files
- Learning rates, batch sizes, and model dimensions are configurable
- Early stopping and validation splits are implemented

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or sequence length
2. **CUDA Issues**: Ensure PyTorch/TensorFlow CUDA compatibility
3. **NLTK Data**: Download required NLTK datasets
4. **File Paths**: Ensure data files are in correct directories

### Performance Tips

- Use GPU acceleration for transformer models
- Reduce dataset size for faster experimentation
- Monitor memory usage during training
- Use early stopping to prevent overfitting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Stanford NLP for the SNLI dataset
- Hugging Face for transformer implementations
- TensorFlow and PyTorch communities
- NLTK and scikit-learn developers

## 📞 Support

For questions or issues:

1. Check the existing documentation
2. Review the code comments
3. Open an issue on the repository

---

**Happy Text Entailment Modeling! 🚀**
