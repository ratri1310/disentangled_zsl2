import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

class BiomedicalTextProcessor:
    """
    Preprocesses PubMed abstracts for biomedical text classification.
    """
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.section_headers = [
            "BACKGROUND:", "METHODS:", "RESULTS:", "CONCLUSIONS:", 
            "OBJECTIVE:", "DESIGN:", "SETTING:", "PARTICIPANTS:",
            "INTERVENTIONS:", "MEASUREMENTS:", "MAIN OUTCOME MEASURES:"
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess biomedical text.
        
        Args:
            text: Raw biomedical text
            
        Returns:
            Cleaned text
        """
        # Remove section headers while preserving content
        for header in self.section_headers:
            text = text.replace(header, "")
        
        # Normalize numerical digits to 'n'
        text = re.sub(r'\d+', 'n', text)
        
        # Normalize special symbols while preserving medical abbreviations
        text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\[\]\/\%\+\=\<\>]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize_text(self, text: str, max_length: int = 512) -> Dict:
        """
        Tokenize text using PubMedBERT tokenizer.
        
        Args:
            text: Cleaned text
            max_length: Maximum sequence length
            
        Returns:
            Tokenized inputs
        """
        return self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
    
    def process_mesh_definitions(self, mesh_data: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        Process MeSH definitions to create class anchor embeddings.
        
        Args:
            mesh_data: Dictionary mapping MeSH codes to definitions
            
        Returns:
            Dictionary mapping MeSH codes to tokenized definitions
        """
        processed_mesh = {}
        for mesh_code, definition in mesh_data.items():
            cleaned_def = self.clean_text(definition)
            tokenized = self.tokenize_text(cleaned_def)
            processed_mesh[mesh_code] = tokenized
        
        return processed_mesh

class BiomedicalDataset(Dataset):
    """
    Dataset class for biomedical abstracts and MeSH labels.
    """
    
    def __init__(self, abstracts: List[str], labels: np.ndarray, processor: BiomedicalTextProcessor):
        self.abstracts = abstracts
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.abstracts)
    
    def __getitem__(self, idx):
        abstract = self.abstracts[idx]
        label = self.labels[idx]
        
        # Clean and tokenize abstract
        cleaned_abstract = self.processor.clean_text(abstract)
        tokenized = self.processor.tokenize_text(cleaned_abstract)
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def load_and_preprocess_data(data_path: str, mesh_path: str) -> Tuple[BiomedicalDataset, Dict]:
    """
    Load and preprocess the biomedical dataset.
    
    Args:
        data_path: Path to the abstracts and labels data
        mesh_path: Path to MeSH definitions
        
    Returns:
        Processed dataset and MeSH data
    """
    # Load data (assuming CSV format)
    data = pd.read_csv(data_path)
    mesh_data = pd.read_csv(mesh_path)
    
    # Create processor
    processor = BiomedicalTextProcessor()
    
    # Process abstracts
    abstracts = data['abstract'].tolist()
    
    # Convert multi-hot labels (assuming they're in separate columns or encoded)
    label_columns = [col for col in data.columns if col.startswith('mesh_')]
    labels = data[label_columns].values
    
    # Create dataset
    dataset = BiomedicalDataset(abstracts, labels, processor)
    
    # Process MeSH definitions
    mesh_dict = dict(zip(mesh_data['mesh_code'], mesh_data['definition']))
    processed_mesh = processor.process_mesh_definitions(mesh_dict)
    
    return dataset, processed_mesh

if __name__ == "__main__":
    # Example usage
    processor = BiomedicalTextProcessor()
    
    sample_text = "BACKGROUND: Type 2 diabetes mellitus is a chronic condition. METHODS: We analyzed n patients."
    cleaned = processor.clean_text(sample_text)
    tokenized = processor.tokenize_text(cleaned)
    
    print("Original:", sample_text)
    print("Cleaned:", cleaned)
    print("Tokenized shape:", tokenized['input_ids'].shape)da
