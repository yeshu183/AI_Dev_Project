import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json
from tqdm import tqdm
import re
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# Configuration
class Config:
    def __init__(self):
        # Dataset paths
        self.data_root = '/kaggle/input/crohme-data'
        
        # Model parameters
        self.embed_dim = 256
        self.hidden_dim = 512
        self.num_layers = 1
        self.dropout = 0.3
        self.max_seq_len = 150
        
        # Training parameters
        self.batch_size = 32
        self.num_epochs = 10
        self.learning_rate = 0.001
        self.teacher_forcing_ratio = 0.9
        self.teacher_forcing_decay = 0.9
        self.grad_clip = 5.0
        
        # Tokenizer parameters
        self.special_tokens = {
            'PAD': '<PAD>',
            'START': '<START>',
            'END': '<END>',
            'UNK': '<UNK>'
        }
        
        # Image preprocessing
        self.img_height = 128
        self.img_width = 512
        
        # Checkpoint parameters
        self.checkpoint_dir = 'checkpoints'
        self.log_dir = 'logs'
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Tokenizer for LaTeX expressions
class LaTeXTokenizer:
    def __init__(self, config):
        self.config = config
        self.token2idx = {}
        self.idx2token = {}
        self.build_vocab([])  # Initialize with special tokens
        
    def build_vocab(self, latex_expressions):
        # Add special tokens
        vocab = [self.config.special_tokens[token] for token in ['PAD', 'START', 'END', 'UNK']]
        
        # Add all unique tokens from latex expressions
        all_tokens = []
        for expr in latex_expressions:
            tokens = self._tokenize(expr)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counter = Counter(all_tokens)
        tokens = [token for token, _ in token_counter.most_common()]
        
        # Add tokens to vocabulary that aren't already special tokens
        for token in tokens:
            if token not in vocab:
                vocab.append(token)
        
        # Create mappings
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx2token = {idx: token for idx, token in enumerate(vocab)}
        
        return self
    
    def _tokenize(self, latex_str):
        """
        Tokenize a LaTeX string.
        This is a simplified approach - in a production system, you might need 
        more sophisticated tokenization based on LaTeX syntax.
        """
        # Remove extra whitespace
        latex_str = latex_str.strip()
        
        # Handle special LaTeX commands
        pattern = r'(\\[a-zA-Z]+|[^a-zA-Z0-9\s])'
        
        # Split by the pattern but keep the delimiters
        parts = re.split(f'({pattern})', latex_str)
        
        # Filter out empty strings and strip whitespace
        tokens = [part.strip() for part in parts if part.strip()]
        
        return tokens
    
    def encode(self, latex_str):
        """Convert LaTeX string to token IDs"""
        tokens = self._tokenize(latex_str)
        
        # Add START and END tokens
        tokens = [self.config.special_tokens['START']] + tokens + [self.config.special_tokens['END']]
        
        # Convert to indices, using UNK for unknown tokens
        unk_idx = self.token2idx[self.config.special_tokens['UNK']]
        indices = [self.token2idx.get(token, unk_idx) for token in tokens]
        
        return indices
    
    def decode(self, indices):
        """Convert token IDs back to LaTeX string"""
        # Convert indices to tokens
        start_idx = self.token2idx[self.config.special_tokens['START']]
        end_idx = self.token2idx[self.config.special_tokens['END']]
        pad_idx = self.token2idx[self.config.special_tokens['PAD']]
        
        # Filter out special tokens
        tokens = []
        for idx in indices:
            if idx == end_idx:  # Stop at END token
                break
            if idx not in [start_idx, pad_idx]:  # Skip START and PAD tokens
                tokens.append(self.idx2token[idx])
        
        # Join tokens (with space between symbols for readability)
        latex = ' '.join(tokens)
        
        # Clean up spaces around certain symbols
        latex = re.sub(r'\s+', ' ', latex)  # Replace multiple spaces with single space
        for symbol in ['+', '-', '=', '>', '<', '\\leq', '\\geq']:
            latex = latex.replace(f' {symbol} ', f' {symbol} ')
        
        return latex.strip()
    
    @property
    def vocab_size(self):
        return len(self.token2idx)
# Dataset class for CROHME
class CROHMEDataset(Dataset):
    def __init__(self, data_dir, tokenizer, config, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.config = config
        self.transform = transform if transform else self._get_default_transform()
        
        # Get all image paths
        split_dir = os.path.join(data_dir, split)
        self.image_paths = sorted(glob.glob(os.path.join(split_dir, 'images', '*.png')))
        
        # Load all LaTeX expressions
        self.latex_expressions = []
        for img_path in self.image_paths:
            # Get corresponding label path
            file_id = os.path.basename(img_path).split('.')[0]
            label_path = os.path.join(split_dir, 'labels', f"{file_id}.txt")
            
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    latex = f.read().strip()
                self.latex_expressions.append(latex)
            else:
                print(f"Warning: Label not found for {img_path}")
                self.latex_expressions.append("")  # Empty placeholder
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and transform image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        # Get and encode LaTeX
        latex = self.latex_expressions[idx]
        encoded_latex = self.tokenizer.encode(latex)
        
        # Pad sequence if needed
        if len(encoded_latex) > self.config.max_seq_len:
            encoded_latex = encoded_latex[:self.config.max_seq_len]
        else:
            pad_idx = self.tokenizer.token2idx[self.config.special_tokens['PAD']]
            encoded_latex = encoded_latex + [pad_idx] * (self.config.max_seq_len - len(encoded_latex))
        
        return {
            'image': image,
            'latex_tokens': torch.tensor(encoded_latex, dtype=torch.long),
            'latex_str': latex,
            'image_path': img_path
        }
    
    def _get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((self.config.img_height, self.config.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])