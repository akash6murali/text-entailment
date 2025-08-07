"""
SNLI Dataset Explorer
statistics and analysis of the SNLI dataset
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import os
from pathlib import Path

class SNLIExplorer:
    """
    SNLI dataset explorer and analyzer
    """
    
    def __init__(self, data_dir='./snli_1.0'):
        self.data_dir = Path(data_dir)
        self.splits = ['train', 'dev', 'test']
        self.file_paths = {
            'train': self.data_dir / 'snli_1.0_train.jsonl',
            'dev': self.data_dir / 'snli_1.0_dev.jsonl',
            'test': self.data_dir / 'snli_1.0_test.jsonl'
        }
        
        # Check if files exist
        for split, path in self.file_paths.items():
            if not path.exists():
                print(f"Warning: {split} file not found at {path}")
    
    def load_split_data(self, split='train', max_examples=None):
        """Load data from a specific split"""
        if split not in self.file_paths:
            raise ValueError(f"Split must be one of {list(self.file_paths.keys())}")
        
        data = []
        file_path = self.file_paths[split]
        
        print(f"Loading {split} data from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                    
                try:
                    example = json.loads(line.strip())
                    data.append(example)
                except json.JSONDecodeError:
                    print(f"Skipping malformed line {i+1}")
                    continue
        
        print(f"Loaded {len(data)} examples from {split} split")
        return data
    
    def basic_statistics(self):
        """Get basic statistics for all splits"""
        print("=" * 60)
        print("BASIC DATASET STATISTICS")
        print("=" * 60)
        
        stats = {}
        for split in self.splits:
            if not self.file_paths[split].exists():
                continue
                
            data = self.load_split_data(split)
            
            # Basic counts
            total_examples = len(data)
            valid_examples = len([ex for ex in data if ex.get('gold_label') != '-'])
            invalid_examples = total_examples - valid_examples
            
            # Label distribution
            labels = [ex['gold_label'] for ex in data if ex.get('gold_label') != '-']
            label_counts = Counter(labels)
            
            stats[split] = {
                'total': total_examples,
                'valid': valid_examples,
                'invalid': invalid_examples,
                'labels': label_counts
            }
            
            print(f"\n{split.upper()} SET:")
            print(f"  Total examples: {total_examples:,}")
            print(f"  Valid examples: {valid_examples:,}")
            print(f"  Invalid examples: {invalid_examples:,}")
            print(f"  Label distribution:")
            for label, count in label_counts.items():
                percentage = (count / valid_examples) * 100
                print(f"    {label:>12}: {count:>6,} ({percentage:>5.1f}%)")
        
        return stats
    
    def text_length_analysis(self, split='train', max_examples=5000):
        """Analyze text lengths in the dataset"""
        print(f"\n TEXT LENGTH ANALYSIS ({split.upper()})")
        print("=" * 50)
        
        data = self.load_split_data(split, max_examples)
        valid_data = [ex for ex in data if ex.get('gold_label') != '-']
        
        # Calculate lengths
        premise_lengths = []
        hypothesis_lengths = []
        
        for ex in valid_data:
            premise_lengths.append(len(ex['sentence1'].split()))
            hypothesis_lengths.append(len(ex['sentence2'].split()))
        
        # Statistics
        stats = {
            'premise': {
                'mean': np.mean(premise_lengths),
                'std': np.std(premise_lengths),
                'min': np.min(premise_lengths),
                'max': np.max(premise_lengths),
                'median': np.median(premise_lengths),
                'q25': np.percentile(premise_lengths, 25),
                'q75': np.percentile(premise_lengths, 75)
            },
            'hypothesis': {
                'mean': np.mean(hypothesis_lengths),
                'std': np.std(hypothesis_lengths),
                'min': np.min(hypothesis_lengths),
                'max': np.max(hypothesis_lengths),
                'median': np.median(hypothesis_lengths),
                'q25': np.percentile(hypothesis_lengths, 25),
                'q75': np.percentile(hypothesis_lengths, 75)
            }
        }
        
        print(f"Premise lengths (words):")
        print(f"  Mean: {stats['premise']['mean']:.1f} ± {stats['premise']['std']:.1f}")
        print(f"  Median: {stats['premise']['median']:.1f}")
        print(f"  Range: {stats['premise']['min']:.0f} - {stats['premise']['max']:.0f}")
        print(f"  Q25-Q75: {stats['premise']['q25']:.1f} - {stats['premise']['q75']:.1f}")
        
        print(f"\nHypothesis lengths (words):")
        print(f"  Mean: {stats['hypothesis']['mean']:.1f} ± {stats['hypothesis']['std']:.1f}")
        print(f"  Median: {stats['hypothesis']['median']:.1f}")
        print(f"  Range: {stats['hypothesis']['min']:.0f} - {stats['hypothesis']['max']:.0f}")
        print(f"  Q25-Q75: {stats['hypothesis']['q25']:.1f} - {stats['hypothesis']['q75']:.1f}")
        
        return premise_lengths, hypothesis_lengths
    
    def annotator_analysis(self, split='train', max_examples=5000):
        """Analyze annotator label patterns"""
        print(f"\n ANNOTATOR ANALYSIS ({split.upper()})")
        print("=" * 50)
        
        data = self.load_split_data(split, max_examples)
        
        # Count annotator patterns
        annotator_counts = Counter()
        agreement_patterns = defaultdict(int)
        
        for ex in data:
            if ex.get('gold_label') == '-':
                continue
                
            ann_labels = ex.get('annotator_labels', [])
            num_annotators = len(ann_labels)
            annotator_counts[num_annotators] += 1
            
            # Check agreement
            if num_annotators > 1:
                unique_labels = set(ann_labels)
                if len(unique_labels) == 1:
                    agreement_patterns['full_agreement'] += 1
                elif len(unique_labels) == 2:
                    agreement_patterns['partial_agreement'] += 1
                else:
                    agreement_patterns['no_agreement'] += 1
        
        print("Annotator count distribution:")
        for count, freq in sorted(annotator_counts.items()):
            percentage = (freq / sum(annotator_counts.values())) * 100
            print(f"  {count} annotators: {freq:>5,} examples ({percentage:>5.1f}%)")
        
        if agreement_patterns:
            print("\nAnnotator agreement patterns:")
            total = sum(agreement_patterns.values())
            for pattern, count in agreement_patterns.items():
                percentage = (count / total) * 100
                print(f"  {pattern.replace('_', ' ').title()}: {count:>5,} ({percentage:>5.1f}%)")
    
    def sample_examples(self, split='dev', num_examples=5):
        """Show sample examples from the dataset"""
        print(f"\n SAMPLE EXAMPLES ({split.upper()})")
        print("=" * 50)
        
        data = self.load_split_data(split, num_examples * 2)
        valid_data = [ex for ex in data if ex.get('gold_label') != '-']
        
        # Get examples from each label
        examples_by_label = defaultdict(list)
        for ex in valid_data:
            label = ex['gold_label']
            if len(examples_by_label[label]) < 2:  # 2 examples per label
                examples_by_label[label].append(ex)
        
        for label in ['entailment', 'neutral', 'contradiction']:
            if label in examples_by_label:
                print(f"\n{label.upper()} Examples:")
                for i, ex in enumerate(examples_by_label[label], 1):
                    print(f"  {i}. Premise: {ex['sentence1']}")
                    print(f"     Hypothesis: {ex['sentence2']}")
                    if len(ex.get('annotator_labels', [])) > 1:
                        print(f"     Annotators: {ex['annotator_labels']}")
                    print()
    
    def vocabulary_analysis(self, split='train', max_examples=10000):
        """Analyze vocabulary in the dataset"""
        print(f"\n VOCABULARY ANALYSIS ({split.upper()})")
        print("=" * 50)
        
        data = self.load_split_data(split, max_examples)
        valid_data = [ex for ex in data if ex.get('gold_label') != '-']
        
        # Collect all words
        all_words = []
        premise_words = []
        hypothesis_words = []
        
        for ex in valid_data:
            p_words = ex['sentence1'].lower().split()
            h_words = ex['sentence2'].lower().split()
            
            premise_words.extend(p_words)
            hypothesis_words.extend(h_words)
            all_words.extend(p_words + h_words)
        
        # Vocabulary statistics
        vocab_size = len(set(all_words))
        premise_vocab = len(set(premise_words))
        hypothesis_vocab = len(set(hypothesis_words))
        
        # Most common words
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(20)
        
        print(f"Total vocabulary size: {vocab_size:,}")
        print(f"Premise vocabulary: {premise_vocab:,}")
        print(f"Hypothesis vocabulary: {hypothesis_vocab:,}")
        print(f"Total words: {len(all_words):,}")
        
        print(f"\nMost common words:")
        for word, count in most_common:
            print(f"  {word:>10}: {count:>6,}")
        
        return word_counts
    
    def plot_distributions(self, split='train', max_examples=5000):
        """Create visualizations of data distributions"""
        print(f"\n CREATING VISUALIZATIONS ({split.upper()})")
        print("=" * 50)
        
        try:
            # Get data
            premise_lengths, hypothesis_lengths = self.text_length_analysis(split, max_examples)
            data = self.load_split_data(split, max_examples)
            valid_data = [ex for ex in data if ex.get('gold_label') != '-']
            labels = [ex['gold_label'] for ex in valid_data]
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'SNLI Dataset Analysis - {split.upper()} Split', fontsize=16)
            
            # 1. Label distribution
            label_counts = Counter(labels)
            axes[0, 0].bar(label_counts.keys(), label_counts.values())
            axes[0, 0].set_title('Label Distribution')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Text length distributions
            axes[0, 1].hist(premise_lengths, bins=50, alpha=0.7, label='Premise', color='blue')
            axes[0, 1].hist(hypothesis_lengths, bins=50, alpha=0.7, label='Hypothesis', color='red')
            axes[0, 1].set_title('Text Length Distribution')
            axes[0, 1].set_xlabel('Length (words)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            
            # 3. Length by label
            length_by_label = defaultdict(list)
            for ex in valid_data:
                label = ex['gold_label']
                length_by_label[label].append(len(ex['sentence1'].split()) + len(ex['sentence2'].split()))
            
            labels_for_box = list(length_by_label.keys())
            lengths_for_box = [length_by_label[label] for label in labels_for_box]
            axes[1, 0].boxplot(lengths_for_box, labels=labels_for_box)
            axes[1, 0].set_title('Combined Text Length by Label')
            axes[1, 0].set_ylabel('Total Length (words)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. Premise vs Hypothesis length scatter
            sample_indices = np.random.choice(len(premise_lengths), 
                                            min(1000, len(premise_lengths)), 
                                            replace=False)
            sample_premise = [premise_lengths[i] for i in sample_indices]
            sample_hypothesis = [hypothesis_lengths[i] for i in sample_indices]
            
            axes[1, 1].scatter(sample_premise, sample_hypothesis, alpha=0.5)
            axes[1, 1].set_title('Premise vs Hypothesis Length')
            axes[1, 1].set_xlabel('Premise Length (words)')
            axes[1, 1].set_ylabel('Hypothesis Length (words)')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Could not create visualizations: {e}")
            print("Make sure matplotlib is installed: pip install matplotlib")
    
    def comprehensive_analysis(self):
        """Run all analyses"""
        print("COMPREHENSIVE SNLI DATASET ANALYSIS")
        print("=" * 60)
        
        # Basic statistics
        self.basic_statistics()
        
        # Text length analysis
        self.text_length_analysis('train', 5000)
        
        # Annotator analysis
        self.annotator_analysis('train', 5000)
        self.annotator_analysis('dev', 2000)
        
        # Sample examples
        self.sample_examples('dev', 3)
        
        # Vocabulary analysis
        self.vocabulary_analysis('train', 10000)
        
        # Visualizations
        self.plot_distributions('train', 5000)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)

def main():
    """Main function to run the analysis"""
    # Initialize explorer
    explorer = SNLIExplorer('./snli_1.0')
    
    # Run comprehensive analysis
    explorer.comprehensive_analysis()

if __name__ == "__main__":
    main()