"""
BiLSTM Encoder Model for SNLI
Encode premise and hypothesis separately, then classify
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import json
from typing import List, Tuple, Dict
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

class Vocabulary:
    """
    Vocabulary class to handle word-to-index mapping
    """
    
    def __init__(self):
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = Counter()
        
    def build_vocabulary(self, sentences: List[str], min_freq: int = 2):
        """
        Build vocabulary from sentences
        
        Args:
            sentences: List of sentences
            min_freq: Minimum frequency for word inclusion
        """
        print("Building vocabulary...")
        
        # Count all words
        for sentence in sentences:
            words = sentence.lower().split()
            self.word_counts.update(words)
        
        # Add words that meet minimum frequency
        for word, count in self.word_counts.items():
            if count >= min_freq and word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        print(f"Vocabulary size: {len(self.word_to_idx)}")
        print(f"Most common words: {self.word_counts.most_common(10)}")
    
    def sentence_to_indices(self, sentence: str, max_length: int = 50) -> List[int]:
        """
        Convert sentence to list of word indices
        
        Args:
            sentence: Input sentence
            max_length: Maximum sequence length (pad or truncate)
            
        Returns:
            List of word indices
        """
        words = sentence.lower().split()
        indices = []
        
        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx['<UNK>'])
        
        # Pad or truncate to max_length
        if len(indices) < max_length:
            indices.extend([self.word_to_idx['<PAD>']] * (max_length - len(indices)))
        else:
            indices = indices[:max_length]
        
        return indices
    
    def __len__(self):
        return len(self.word_to_idx)

class SNLIDataset(Dataset):
    """
    PyTorch Dataset for SNLI data
    """
    
    def __init__(self, premises: List[str], hypotheses: List[str], labels: List[int],
                 vocab: Vocabulary, max_length: int = 50):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.premises)
    
    def __getitem__(self, idx):
        premise_indices = self.vocab.sentence_to_indices(self.premises[idx], self.max_length)
        hypothesis_indices = self.vocab.sentence_to_indices(self.hypotheses[idx], self.max_length)
        
        return {
            'premise': torch.tensor(premise_indices, dtype=torch.long),
            'hypothesis': torch.tensor(hypothesis_indices, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class BiLSTMEncoder(nn.Module):
    """
    BiLSTM Encoder for sentence encoding
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 300, 
                 hidden_dim: int = 512, num_layers: int = 1, dropout: float = 0.3):
        super(BiLSTMEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Encoded sentence representation of shape (batch_size, hidden_dim * 2)
        """
        # Embedding: (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # BiLSTM: (batch_size, seq_length, hidden_dim * 2)
        lstm_out, (hidden, cell) = self.bilstm(embedded)
        
        # Use the last hidden state (concatenated forward and backward)
        # hidden: (num_layers * 2, batch_size, hidden_dim)
        # Take the last layer's hidden states
        forward_hidden = hidden[-2]  # Last layer, forward direction
        backward_hidden = hidden[-1]  # Last layer, backward direction
        
        # Concatenate forward and backward hidden states
        sentence_repr = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return sentence_repr

class SNLIClassifier(nn.Module):
    """
    SNLI Classifier using BiLSTM encoders
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 300,
                 hidden_dim: int = 512, num_layers: int = 1, 
                 dropout: float = 0.3, num_classes: int = 3):
        super(SNLIClassifier, self).__init__()
        
        # Shared BiLSTM encoder for both premise and hypothesis
        self.encoder = BiLSTMEncoder(vocab_size, embedding_dim, hidden_dim, 
                                   num_layers, dropout)
        
        # Classifier head
        # Input: concatenated premise and hypothesis representations
        classifier_input_dim = hidden_dim * 2 * 2  # 2 for bidirectional, 2 for premise+hypothesis
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, premise, hypothesis):
        """
        Forward pass
        
        Args:
            premise: Premise tensor of shape (batch_size, seq_length)
            hypothesis: Hypothesis tensor of shape (batch_size, seq_length)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Encode premise and hypothesis separately
        premise_repr = self.encoder(premise)
        hypothesis_repr = self.encoder(hypothesis)
        
        # Concatenate representations
        combined_repr = torch.cat([premise_repr, hypothesis_repr], dim=1)
        
        # Classify
        logits = self.classifier(combined_repr)
        
        return logits

class BiLSTMTrainer:
    """
    Trainer class for BiLSTM model
    """
    
    def __init__(self, device: str = 'cpu'):
        self.model = None
        self.device = device
        self.label_to_id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
    
    def set_model(self, model: SNLIClassifier):
        """Set the model after initialization"""
        self.model = model.to(self.device)
        
    def load_data(self, file_path: str, max_examples: int = None) -> Tuple[List[str], List[str], List[int]]:
        """
        Load SNLI data from JSONL file
        """
        premises = []
        hypotheses = []
        labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                    
                example = json.loads(line.strip())
                
                if example.get('gold_label') == '-':
                    continue
                
                premises.append(example['sentence1'])
                hypotheses.append(example['sentence2'])
                labels.append(self.label_to_id[example['gold_label']])
        
        print(f"Loaded {len(premises)} examples from {file_path}")
        return premises, hypotheses, labels
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 10, learning_rate: float = 0.001):
        """
        Train the model
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                premise = batch['premise'].to(self.device)
                hypothesis = batch['hypothesis'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = self.model(premise, hypothesis)
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Validation phase
            val_accuracy = self.evaluate(val_loader)
            val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        return train_losses, val_accuracies
    
    def evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluate the model
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                premise = batch['premise'].to(self.device)
                hypothesis = batch['hypothesis'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(premise, hypothesis)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        return accuracy
    
    def detailed_evaluation(self, data_loader: DataLoader):
        """
        Detailed evaluation with metrics and confusion matrix
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                premise = batch['premise'].to(self.device)
                hypothesis = batch['hypothesis'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(premise, hypothesis)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, 
                                  target_names=list(self.label_to_id.keys())))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=list(self.label_to_id.keys()),
                    yticklabels=list(self.label_to_id.keys()))
        plt.title('BiLSTM Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        plt.savefig("bilstm_confusion_matrix.png")
    
    def plot_training_curves(self, train_losses: List[float], val_accuracies: List[float]):
        """
        Plot training curves
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Training loss
        ax1.plot(train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        
        # Validation accuracy
        ax2.plot(val_accuracies)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        
        plt.tight_layout()
        plt.show()
        plt.savefig("bilstm_training_curves.png")

def main():
    """
    Main function to run BiLSTM training
    """
    print("SNLI BiLSTM Classification")
    print("=" * 40)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize trainer
    trainer = BiLSTMTrainer(device)
    
    # Load data
    print("\n1. Loading data...")
    train_premises, train_hypotheses, train_labels = trainer.load_data(
        '../snli_1.0/snli_1.0_train.jsonl', max_examples=50000
    )
    dev_premises, dev_hypotheses, dev_labels = trainer.load_data(
        '../snli_1.0/snli_1.0_dev.jsonl', max_examples=10000
    )
    
    # Build vocabulary
    print("\n2. Building vocabulary...")
    vocab = Vocabulary()
    all_sentences = train_premises + train_hypotheses + dev_premises + dev_hypotheses
    vocab.build_vocabulary(all_sentences, min_freq=2)
    
    # Create datasets
    print("\n3. Creating datasets...")
    train_dataset = SNLIDataset(train_premises, train_hypotheses, train_labels, vocab)
    dev_dataset = SNLIDataset(dev_premises, dev_hypotheses, dev_labels, vocab)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    print("\n4. Initializing model...")
    model = SNLIClassifier(
        vocab_size=len(vocab),
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )
    
    trainer.set_model(model)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n5. Training model...")
    train_losses, val_accuracies = trainer.train(
        train_loader, dev_loader, 
        num_epochs=15, 
        learning_rate=0.001
    )
    
    # Final evaluation
    print("\n6. Final evaluation...")
    trainer.detailed_evaluation(dev_loader)
    
    # Plot training curves
    print("\n7. Plotting results...")
    trainer.plot_training_curves(train_losses, val_accuracies)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()