import torch
import os
import torch.nn as nn
from torch.nn import functional as F
import json
import re
import psycopg2

# --- 0. HARDWARE ACCELERATION ---
# Automatically use your GPU if available, otherwise fallback to CPU
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# --- 1. HYPERPARAMETERS (SCALED FOR D&D) ---
batch_size = 16      # Number of sequences processed in parallel
block_size = 512     # Context window: Increased to fit an entire character sheet
n_embd = 128         # Brain power: Increased dimensions for embeddings
n_head = 4           # Number of attention heads
n_layer = 4          # Number of Transformer blocks
learning_rate = 3e-4 # Adjusted learning rate for a larger model
max_iters = 5000     # Number of training steps

# --- 2. DATA LOADING & TOKENIZATION ---
file_path = "dnd_json_dataset.txt"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find {file_path}. Please run the generation script first.")

print(f"Loading dataset from {file_path}...")
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    # Uses .get() to prevent KeyErrors if a user types an unknown character
    return [stoi.get(c, stoi.get(' ', 0)) for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

print("Encoding dataset to PyTorch Tensor... (This may take a moment)")
data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Vocabulary size: {vocab_size} unique characters")
print(f"Training data size:   {len(train_data):,} tokens (90%)")
print(f"Validation data size: {len(val_data):,} tokens (10%)")
print("-" * 50)

# --- 3. BATCH GENERATION ---
torch.manual_seed(1337) 

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    # Move the batch to the GPU/CPU
    x, y = x.to(device), y.to(device)
    return x, y

# --- 4. TRANSFORMER ARCHITECTURE ---
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        # Move position tensor to the correct device
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# Initialize model and move it to the correct device
model = TransformerLanguageModel(vocab_size)
model = model.to(device)

# --- 5. THE TRAINING LOOP ---
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\nBeginning training phase...")
print("This will take a while depending on your hardware. Feel free to grab a coffee!")

for steps in range(max_iters): 
    xb, yb = get_batch('train')
    
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if steps % 500 == 0:
        print(f"Step {steps}/{max_iters} | Loss: {loss.item():.4f}")

print(f"Final training loss: {loss.item():.4f}")
print("Training complete!\n")

# --- 6. INFERENCE / CHAT SCRIPT ---
def generate_response(model, user_query, max_new_tokens=400):
    model.eval()
    
    # We explicitly prompt the model to generate JSON
    formatted_prompt = f"<USER> {user_query} Output strictly in JSON format. <ASSISTANT>\n"
    idx = torch.tensor([encode(formatted_prompt)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] if idx.size(1) > block_size else idx
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
            
            current_text = decode(idx[0].tolist())
            if "<END>" in current_text:
                break
                
    full_text = decode(idx[0].tolist())
    raw_output = full_text.split("<ASSISTANT>\n")[-1].replace("<END>", "").strip()
    
    model.train()
    
    # Attempt to cleanly extract the JSON block using regex in case of trailing tokens
    json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return {"error": "Model output invalid JSON structure.", "raw_text": raw_output}
    else:
        return {"error": "No JSON block detected in output.", "raw_text": raw_output}
    
# --- 6.5 POSTGRESQL RAG ENRICHMENT ---
def enrich_character_data(char_json):
    # Check if the model failed to output a valid sheet
    if "error" in char_json:
        return char_json
        
    features_list = char_json.get("features", [])
    if not features_list:
        char_json["enriched_features"] = []
        return char_json 

    try:
        # Update these credentials to match your local or Render-hosted PostgreSQL database
        conn = psycopg2.connect(
            dbname="neondb", 
            user="Narku_Model", 
            password="V0113yba11ru135", 
            host="ep-empty-butterfly-aq64mw79-pooler.c-8.us-east-1.aws.neon.tech", 
            port="5432"
        )
        cur = conn.cursor()

        # Query using the exact casing from your SQL schema image
        query = """
            SELECT "Name", "Description"
            FROM "ClassFeatures"
            WHERE "Name" = ANY(%s)
        """
        # Pass the list of feature names to the ANY() operator
        cur.execute(query, (features_list,))
        rows = cur.fetchall()

        # Map the results back into a structured array
        enriched_features = []
        for row in rows:
            enriched_features.append({
                "name": row[0],
                "description": row[1]
            })

        char_json["enriched_features"] = enriched_features

        cur.close()
        conn.close()
    except Exception as e:
        char_json["db_error"] = f"Database connection/query failed: {str(e)}"

    return char_json

# --- 7. MOBILE EXPORT (ONNX & JSON) ---
print("\n" + "="*50)
print("  Beginning Mobile Export Process...")
print("="*50)

# 1. Export the Tokenizer to JSON
# Android cannot run your Python encode/decode functions. 
# We save the raw dictionaries so your mobile app can rebuild the logic.
print("Exporting Tokenizer to JSON...")
tokenizer_data = {
    "stoi": stoi,
    "itos": itos
}

with open("dnd_tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer_data, f, ensure_ascii=False, indent=4)
print("-> Saved 'dnd_tokenizer.json'")

# 2. Export the PyTorch Model to ONNX
print("Preparing PyTorch model for ONNX trace...")

# Put the model in evaluation mode (critical for freezing weights)
model.eval()

# Move the model to CPU for the export process to avoid device conflicts
model.to('cpu')

# ONNX requires a "dummy input" to trace the mathematical operations of your network.
# We give it a fake tensor with a batch size of 1 and an arbitrary length of 10.
dummy_input = torch.randint(0, vocab_size, (1, 10), dtype=torch.long)

onnx_file_path = "dnd_transformer.onnx"

torch.onnx.export(
    model,                      # The trained model in memory
    dummy_input,                # The dummy trace tensor
    onnx_file_path,             # The output filename
    export_params=True,         # Embed the trained weights inside the file
    opset_version=14,           # ONNX version 14 is highly compatible with mobile runtimes
    do_constant_folding=True,   # Optimizes the math for faster mobile inference
    input_names=['input_ids'],  # What the Android app will call the input array
    output_names=['logits'],    # What the Android app will call the output array
    
    # IMPORTANT: Dynamic Axes
    # If we don't include this, the Android app will crash unless every prompt is exactly 10 characters long.
    # This tells ONNX that the 'sequence_length' can be any size up to your block_size.
    dynamic_axes={              
        'input_ids': {1: 'sequence_length'},
        'logits': {1: 'sequence_length'}
    }
)

print(f"-> Saved '{onnx_file_path}'")
print("\nExport Complete! You are ready to build the Android app.")
print("="*50 + "\n")

# Put the model back on the original device/training mode if you plan to keep using the terminal
model.to(device)
model.train()

# --- 8. INTERACTIVE TERMINAL ---
print("="*60)
print("  D&D JSON Character Generator & RAG Initialized.")
print("  Example Prompt: 'Generate a Level 5 Battle Master Fighter.'")
print("  Type 'quit' or 'exit' to end the session.")
print("="*60 + "\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ['quit', 'exit']:
        print("Ending session. Goodbye!")
        break
        
    # 1. Generate the structured JSON shell
    base_json = generate_response(model, user_input)
    
    # 2. Enrich the shell with descriptions from the PostgreSQL database
    final_payload = enrich_character_data(base_json)
    
    # 3. Print the final payload nicely
    print(f"\nAssistant (Enriched Payload):\n")
    print(json.dumps(final_payload, indent=4))
    print("\n" + "-" * 40)