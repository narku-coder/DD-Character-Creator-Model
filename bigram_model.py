import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # The Embedding Table: A simple lookup table where every character 
        # gets its own row of numbers representing its "meaning" to the model.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (Batch, Time) tensors of integers
        
        # We pass the input through the embedding table to get our predictions (logits)
        logits = self.token_embedding_table(idx) # Outputs shape: (Batch, Time, Channel)
        
        if targets is None:
            loss = None
        else:
            # PyTorch's cross_entropy function expects a 2D tensor, so we have to reshape
            # our batches to stretch them out into a single line.
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            
            # Calculate the Loss (how wrong the model's predictions are)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

# Instantiate the model
model = BigramLanguageModel(vocab_size)

# Test the forward pass with our batch
logits, loss = model(xb, yb)
print(f"\nModel initialized.")
print(f"Initial untrained loss: {loss.item():.4f}")