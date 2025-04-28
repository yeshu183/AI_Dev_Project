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

# Attention module
class AttentionModule(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        # Attention layers
        self.attn = nn.Linear(encoder_dim + decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)
        
    def forward(self, encoder_features, decoder_hidden):
        """
        encoder_features: (batch_size, feature_size, height, width)
        decoder_hidden: (batch_size, decoder_dim)
        """
        batch_size = encoder_features.size(0)
        
        # Reshape encoder features
        feature_size = encoder_features.size(1)
        num_pixels = encoder_features.size(2) * encoder_features.size(3)
        
        encoder_features = encoder_features.permute(0, 2, 3, 1).contiguous()
        encoder_features = encoder_features.view(batch_size, num_pixels, feature_size)
        
        # Repeat decoder hidden state
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, num_pixels, 1)
        
        # Calculate attention weights
        attention_input = torch.cat((decoder_hidden, encoder_features), dim=2)
        attention = torch.tanh(self.attn(attention_input))
        attention = self.v(attention).squeeze(2)
        
        # Apply softmax to get attention weights
        alpha = F.softmax(attention, dim=1)
        alpha = alpha.unsqueeze(2)
        
        # Apply attention weights to encoder features
        context_vector = (encoder_features * alpha).sum(dim=1)
        
        return context_vector, alpha
class Encoder(nn.Module):
    def __init__(self, encoded_dim):
        super().__init__()
        
        # Use ResNet18 as backbone
        resnet = models.resnet18(pretrained=True)
        
        # Remove final fully connected layer and pooling
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # Adjust first conv layer to accept grayscale images
        self.resnet[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Add a projection layer to get the desired dimension
        self.projection = nn.Conv2d(512, encoded_dim, kernel_size=1)
        
    def forward(self, images):
        features = self.resnet(images)
        features = self.projection(features)
        return features
# Decoder - LSTM with attention
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, encoder_dim, num_layers=1, dropout=0.5):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Attention module
        self.attention = AttentionModule(encoder_dim, hidden_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_dim + encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward_step(self, encoder_features, prev_token, hidden=None):
        """Single step forward"""
        # Get embeddings
        embed = self.embedding(prev_token)  # (batch_size, 1, embed_dim)
        
        batch_size = prev_token.size(0)
        
        # Initialize hidden state if None
        if hidden is None:
            # Initialize hidden state and cell state as zeros
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(encoder_features.device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(encoder_features.device)
            hidden = (h_0, c_0)
        
        # Extract hidden state (ignore cell state for attention)
        hidden_state = hidden[0][-1]  # Last layer's hidden state
        
        # Rest of the method remains the same...
        # Apply attention
        context, _ = self.attention(encoder_features, hidden_state)
        context = context.unsqueeze(1)  # (batch_size, 1, encoder_dim)
        
        # Concatenate embedding and context vector
        lstm_input = torch.cat([embed, context], dim=2)
        
        # LSTM step
        output, hidden = self.lstm(lstm_input, hidden)
    
        # Project to vocabulary space
        output = self.output(self.dropout(output))
        
        return output, hidden
    
    def forward(self, encoder_features, targets=None, teacher_forcing_ratio=0.5, max_len=None):
        """
        Forward pass with optional teacher forcing
        encoder_features: (batch_size, encoder_dim, height, width)
        targets: (batch_size, max_len) - token indices
        """
        batch_size = encoder_features.size(0)
        
        # Initialize hidden state
        # Create hidden state and cell state with appropriate dimensions
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(encoder_features.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(encoder_features.device)
        hidden = (h_0, c_0)
        
        # Determine sequence length
        if targets is not None:
            max_len = targets.size(1)
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(encoder_features.device)
        
        # First input is always <START> token
        start_idx = targets[:, 0] if targets is not None else torch.ones(batch_size).long().to(encoder_features.device)
        input_token = start_idx.unsqueeze(1)  # (batch_size, 1)
        
        # Generate sequence
        for t in range(max_len):
            # Forward step
            output, hidden = self.forward_step(encoder_features, input_token, hidden)
            
            # Store output
            outputs[:, t:t+1, :] = output
            
            # Determine next input token
            use_teacher_forcing = (random.random() < teacher_forcing_ratio) and targets is not None
            
            if use_teacher_forcing and t < max_len - 1:
                # Use ground truth as next input
                input_token = targets[:, t+1:t+2]
            else:
                # Use model's prediction as next input
                _, top_indices = output.topk(1, dim=2)
                input_token = top_indices.squeeze(2)
        
        return outputs
# Complete model combining encoder and decoder
class HandwrittenMathRecognizer(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        
        # Encoder and decoder
        self.encoder = Encoder(encoded_dim=config.hidden_dim)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            encoder_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
        
    def forward(self, images, targets=None, teacher_forcing_ratio=0.5):
        # Encode images
        encoder_features = self.encoder(images)
        
        # Decode with or without teacher forcing
        outputs = self.decoder(
            encoder_features, 
            targets, 
            teacher_forcing_ratio, 
            max_len=self.config.max_seq_len if targets is None else None
        )
        
        return outputs
    
    def generate(self, images):
        """Generate LaTeX expressions without teacher forcing"""
        with torch.no_grad():
            encoder_features = self.encoder(images)
            outputs = self.decoder(
                encoder_features, 
                targets=None, 
                teacher_forcing_ratio=0.0, 
                max_len=self.config.max_seq_len
            )
            
            # Get predicted tokens
            _, predicted = outputs.max(2)
            
            return predicted