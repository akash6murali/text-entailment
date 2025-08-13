"""
Sentence-BERT Similarity Model for SNLI
Using pre-trained sentence embeddings and cosine similarity
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import json
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

class SentenceBERTSimilarity:
    """
    SNLI classifier using Sentence-BERT embeddings and cosine similarity
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the model
        
        Args:
            model_name: Sentence transformer model name
        """
        print(f"Loading Sentence-BERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.classifier = None
        self.label_to_id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
    def load_data(self, file_path: str, max_examples: int = None) -> Tuple[List[str], List[str], List[int]]:
        """
        Load SNLI data from JSONL file
        
        Returns:
            premises, hypotheses, labels
        """
        premises = []
        hypotheses = []
        labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                    
                example = json.loads(line.strip())
                
                # Skip invalid examples
                if example.get('gold_label') == '-':
                    continue
                
                premises.append(example['sentence1'])
                hypotheses.append(example['sentence2'])
                labels.append(self.label_to_id[example['gold_label']])
        
        print(f"Loaded {len(premises)} examples from {file_path}")
        return premises, hypotheses, labels
    
    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Encode sentences using Sentence-BERT
        
        Args:
            sentences: List of sentences to encode
            
        Returns:
            Sentence embeddings array of shape (n_sentences, embedding_dim)
        """
        print(f"Encoding {len(sentences)} sentences...")
        embeddings = self.model.encode(sentences, show_progress_bar=True)
        return embeddings
    
    def compute_similarities(self, premise_embeddings: np.ndarray, 
                           hypothesis_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities between premise and hypothesis embeddings
        
        Args:
            premise_embeddings: Embeddings for premises
            hypothesis_embeddings: Embeddings for hypotheses
            
        Returns:
            Array of cosine similarities
        """
        similarities = []
        
        for i in range(len(premise_embeddings)):
            # Compute cosine similarity between premise[i] and hypothesis[i]
            similarity = cosine_similarity(
                premise_embeddings[i].reshape(1, -1),
                hypothesis_embeddings[i].reshape(1, -1)
            )[0][0]
            similarities.append(similarity)
        
        return np.array(similarities)
    
    def create_features(self, premise_embeddings: np.ndarray, 
                       hypothesis_embeddings: np.ndarray) -> np.ndarray:
        """
        Create feature vectors from embeddings
        
        Options:
        1. Just cosine similarity
        2. Multiple similarity metrics
        3. Concatenated embeddings
        4. Element-wise operations
        """
        # Compute cosine similarities
        cosine_sims = self.compute_similarities(premise_embeddings, hypothesis_embeddings)
        
        # Option 1: Just cosine similarity
        features = cosine_sims.reshape(-1, 1)
        
        # Option 2: Add more features (uncomment to use)
        # # Element-wise operations
        # element_multiply = premise_embeddings * hypothesis_embeddings
        # element_subtract = np.abs(premise_embeddings - hypothesis_embeddings)
        # 
        # # Combine features
        # features = np.column_stack([
        #     cosine_sims,
        #     np.mean(element_multiply, axis=1),  # Mean of element-wise product
        #     np.mean(element_subtract, axis=1),  # Mean of element-wise difference
        # ])
        
        return features
    
    def simple_threshold_classify(self, similarities: np.ndarray) -> np.ndarray:
        """
        Simple threshold-based classification using cosine similarity
        
        High similarity -> Entailment
        Low similarity -> Contradiction
        Medium similarity -> Neutral
        """
        predictions = []
        
        for sim in similarities:
            if sim > 0.8:  # High similarity
                predictions.append(0)  # entailment
            elif sim < 0.3:  # Low similarity
                predictions.append(2)  # contradiction
            else:  # Medium similarity
                predictions.append(1)  # neutral
        
        return np.array(predictions)
    
    def train_classifier(self, features: np.ndarray, labels: np.ndarray):
        """
        Train a logistic regression classifier on similarity features
        """
        print("Training classifier...")
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.classifier.fit(features, labels)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.classifier, features, labels, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained classifier
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train_classifier first.")
        
        return self.classifier.predict(features)
    
    def evaluate(self, true_labels: np.ndarray, predictions: np.ndarray, 
                method_name: str = "Model") -> Dict:
        """
        Evaluate predictions and print metrics
        """
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, 
                                     target_names=list(self.label_to_id.keys()),
                                     output_dict=True)
        
        print(f"\n{method_name} Results:")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, 
                                  target_names=list(self.label_to_id.keys())))
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }
    
    def plot_similarity_distribution(self, similarities: np.ndarray, labels: np.ndarray):
        """
        Plot distribution of cosine similarities by label
        """
        plt.figure(figsize=(12, 8))
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'similarity': similarities,
            'label': [self.id_to_label[label] for label in labels]
        })
        
        # Plot distributions
        for i, label in enumerate(['entailment', 'neutral', 'contradiction']):
            data = df[df['label'] == label]['similarity']
            plt.subplot(2, 2, i+1)
            plt.hist(data, bins=50, alpha=0.7, label=label)
            plt.title(f'{label.title()} - Similarity Distribution')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Frequency')
            plt.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.3f}')
            plt.legend()
        
        # Combined plot
        plt.subplot(2, 2, 4)
        for label in ['entailment', 'neutral', 'contradiction']:
            data = df[df['label'] == label]['similarity']
            plt.hist(data, bins=30, alpha=0.5, label=label)
        
        plt.title('All Labels - Similarity Distributions')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        plt.savefig("similarity_sistribution.png")
    
    def analyze_results(self, similarities: np.ndarray, true_labels: np.ndarray, 
                       predictions: np.ndarray):
        """
        Analyze model results and provide insights
        """
        print("\nAnalysis:")
        
        # Similarity statistics by true label
        for label_id, label_name in self.id_to_label.items():
            label_sims = similarities[true_labels == label_id]
            print(f"{label_name.title()} similarities - Mean: {label_sims.mean():.3f}, "
                  f"Std: {label_sims.std():.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=list(self.label_to_id.keys()),
                    yticklabels=list(self.label_to_id.keys()))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        plt.savefig("confusion_matrix.png")

def main():
    """
    Main function to run the complete pipeline
    """
    print("SNLI Sentence-BERT Similarity Classification")
    print("=" * 50)
    
    # Initialize model
    model = SentenceBERTSimilarity()
    
    # Load data (using small subset for testing)
    print("\n1. Loading data...")
    train_premises, train_hypotheses, train_labels = model.load_data(
        '/Users/akashmurali/Documents/NLP/project/snli_1.0/snli_1.0_train.jsonl', max_examples=5000
    )
    dev_premises, dev_hypotheses, dev_labels = model.load_data(
        '/Users/akashmurali/Documents/NLP/project/snli_1.0/snli_1.0_dev.jsonl', max_examples=1000
    )
    
    # Encode sentences
    print("\n2. Encoding sentences...")
    train_premise_emb = model.encode_sentences(train_premises)
    train_hypothesis_emb = model.encode_sentences(train_hypotheses)
    dev_premise_emb = model.encode_sentences(dev_premises)
    dev_hypothesis_emb = model.encode_sentences(dev_hypotheses)
    
    # Create features
    print("\n3. Creating features...")
    train_features = model.create_features(train_premise_emb, train_hypothesis_emb)
    dev_features = model.create_features(dev_premise_emb, dev_hypothesis_emb)
    
    # Extract similarities for analysis
    train_similarities = model.compute_similarities(train_premise_emb, train_hypothesis_emb)
    dev_similarities = model.compute_similarities(dev_premise_emb, dev_hypothesis_emb)
    
    # Method 1: Simple threshold classification
    print("\n4. Method 1: Threshold Classification")
    threshold_predictions = model.simple_threshold_classify(dev_similarities)
    model.evaluate(np.array(dev_labels), threshold_predictions, "Threshold Method")
    
    # Method 2: Trained classifier
    print("\n5. Method 2: Trained Classifier")
    model.train_classifier(train_features, np.array(train_labels))
    classifier_predictions = model.predict(dev_features)
    model.evaluate(np.array(dev_labels), classifier_predictions, "Trained Classifier")
    
    # Analysis and visualization
    print("\n6. Analysis and Visualization")
    model.plot_similarity_distribution(dev_similarities, np.array(dev_labels))
    model.analyze_results(dev_similarities, np.array(dev_labels), classifier_predictions)

if __name__ == "__main__":
    main()