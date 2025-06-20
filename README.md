# Disentangled Biomedical Text Classification with Hypergraph Contrastive Learning

This repository implements a novel approach for zero-shot multi-label biomedical text classification using feature disentanglement and contrastive learning, as described in our research paper.

## Overview

The model addresses the challenge of classifying biomedical abstracts into previously unseen MeSH (Medical Subject Headings) categories. Our approach combines:

- **Feature Disentanglement**: Separates biomedical content from linguistic style variations
- **Contrastive Learning**: Enhances representation quality through multi-view learning
- **Zero-shot Classification**: Enables prediction on unseen medical concepts
- **PubMedBERT Integration**: Leverages domain-specific pre-trained embeddings

## Key Features

- **Disentangled Representations**: Splits text into content-discriminative (z_I) and stylistic variance (z_V) components
- **Multi-objective Training**: Combines ELBO, classification, contrastive, and similarity losses
- **MeSH Anchor Embeddings**: Uses medical concept definitions for semantic alignment
- **Scalable Inference**: Employs FAISS for efficient similarity search
- **Comprehensive Evaluation**: Supports both seen and zero-shot evaluation metrics

```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.15+
- FAISS
- Other dependencies listed in `requirements.txt`

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/biomedical-disentanglement.git
cd biomedical-disentanglement

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### Required Data Files

1. **Training Data** (`data/train.csv`):
   ```csv
   abstract,mesh_D000001,mesh_D000002,...
   "BACKGROUND: Type 2 diabetes...",1,0,...
   ```

2. **MeSH Definitions** (`data/mesh_definitions.csv`):
   ```csv
   mesh_code,definition
   D000001,"A disease characterized by..."
   ```

3. **Class Lists**:
   - `data/seen_classes.txt`: List of seen MeSH codes (one per line)
   - `data/unseen_classes.txt`: List of unseen MeSH codes (one per line)

### Data Format

- **Abstracts**: Raw PubMed abstract text
- **Labels**: Multi-hot encoded vectors for MeSH codes
- **MeSH Definitions**: Scope notes from MeSH vocabulary

## Usage

### Training

```bash
python train.py
```



### Inference

```python
from main_model import DisentangledBiomedicalClassifier

# Load trained model
model = DisentangledBiomedicalClassifier.load_model('outputs/best_model.pt')

# Zero-shot prediction
predictions = model.predict_zero_shot(
    input_ids=tokenized_text['input_ids'],
    attention_mask=tokenized_text['attention_mask'],
    top_k=10
)

# Get predicted MeSH codes
mesh_codes = predictions['predicted_mesh_codes']
```




## Directory Structure

```
biomedical-disentanglement/
├── data_preprocessing.py      # Data loading and preprocessing
├── feature_extraction.py     # Feature extraction modules
├── contrastive_learning.py   # Contrastive learning components
├── loss_functions.py         # Loss function implementations
├── zero_shot_classifier.py   # Zero-shot classification
├── main_model.py             # Main model architecture
├── train.py                  # Training script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/                     # Data directory
│   ├── train.csv
│   ├── mesh_definitions.csv
│   ├── seen_classes.txt
│   └── unseen_classes.txt
└── outputs/                  # Output directory
    ├── models/
    ├── logs/
    └── results/
```

### Feature Analysis

```python
# Extract content features for analysis
content_features = model.get_content_features(input_ids, attention_mask)

# Analyze feature disentanglement
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
embedded = tsne.fit_transform(content_features.numpy())
```

### Custom Evaluation

```python
from zero_shot_classifier import ZeroShotEvaluator

evaluator = ZeroShotEvaluator(mesh_codes)
metrics = evaluator.compute_metrics(similarities, true_labels)
ranking_metrics = evaluator.compute_ranking_metrics(similarities, true_labels)
```



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
