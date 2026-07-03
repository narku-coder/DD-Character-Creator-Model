import os
import json
import re
import uuid
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.quantization # Added for CPU speed optimization
import psycopg2
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_NAME = os.environ.get("DB_NAME", "neondb")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASS = os.environ.get("DB_PASSWORD", "password")
DB_PORT = os.environ.get("DB_PORT", "5432")

def save_message_to_db(session_id: str, role: str, content: str, is_json: bool):
    try:
        conn = psycopg2.connect(
            host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT
        )
        cur = conn.cursor()
        query = """
            INSERT INTO "ChatHistory" (session_id, role, content, is_json)
            VALUES (%s, %s, %s, %s)
        """
        cur.execute(query, (session_id, role, content, is_json))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Database logging failed: {e}")

# ==========================================
# 2. LOAD TOKENIZER DICTIONARIES
# ==========================================
tokenizer_path = "dnd_tokenizer.json"
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Missing {tokenizer_path}. Cannot start API.")

with open(tokenizer_path, "r", encoding="utf-8") as f:
    tok_data = json.load(f)

stoi = tok_data["stoi"]
itos = {int(k): v for k, v in tok_data["itos"].items()}
vocab_size = len(stoi)

def encode(s):
    return [stoi.get(c, stoi.get(' ', 0)) for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# ==========================================
# 3. TRANSFORMER ARCHITECTURE (KV CACHE ENABLED)
# ==========================================
block_size = 512
n_embd = 128
n_head = 4
n_layer = 4
device = 'cpu'

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        v = self.value(x) 

        # If we have a cache from previous tokens, concatenate it to the current ones
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
            
        new_kv_cache = (k, v)

        wei = q @ k.transpose(-2, -1) * (C ** -0.5) 
        
        # Causal mask logic adjusted for KV caching
        if T > 1:
            T_full = k.shape[1]
            T_cache = T_full - T
            mask = torch.ones(T, T_full, device=self.tril.device)
            mask[:, T_cache:] = self.tril[:T, :T]
            wei = wei.masked_fill(mask == 0, float('-inf'))
            
        wei = F.softmax(wei, dim=-1)
        return wei @ v, new_kv_cache

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, kv_cache=None):
        if kv_cache is None:
            kv_cache = [None] * len(self.heads)
            
        out_list = []
        new_kv_cache = []
        for i, h in enumerate(self.heads):
            out, cache = h(x, kv_cache[i])
            out_list.append(out)
            new_kv_cache.append(cache)
            
        out = torch.cat(out_list, dim=-1)
        return self.proj(out), new_kv_cache

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

    def forward(self, x, kv_cache=None):
        sa_out, new_kv_cache = self.sa(self.ln1(x), kv_cache)
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, new_kv_cache

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Switched to ModuleList so we can pass the cache sequentially
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, kv_cache=None):
        B, T = idx.shape
        
        # Determine current positional indices based on cache length
        if kv_cache is None:
            pos_idx = torch.arange(T, device=device)
        else:
            past_length = kv_cache[0][0][0].shape[1]
            pos_idx = torch.arange(past_length, past_length + T, device=device)
            pos_idx = torch.clamp(pos_idx, max=block_size - 1)
            
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(pos_idx) 
        x = tok_emb + pos_emb 
        
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)
            
        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, kv_cache[i])
            new_kv_cache.append(cache)
            
        x = self.ln_f(x) 
        logits = self.lm_head(x) 
        return logits, new_kv_cache

# ==========================================
# 4. INITIALIZE & LOAD PRE-TRAINED WEIGHTS
# ==========================================
print("Initializing model architecture...")
model = TransformerLanguageModel(vocab_size)

model_path = "dnd_transformer.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Missing {model_path}. Please train and export the model first.")

print("Loading pre-trained weights into memory...")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval() 

# Optimize CPU Inference by quantizing Linear layers down to 8-bit
print("Quantizing model to INT8 for faster CPU inference...")
model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

model.to(device)
print("Model ready!")

# ==========================================
# 5. INFERENCE & RAG PIPELINE
# ==========================================
def generate_response(user_query, max_new_tokens=400):
    formatted_prompt = f"<USER> {user_query} Output strictly in JSON format. <ASSISTANT>\n"
    idx = torch.tensor([encode(formatted_prompt)], dtype=torch.long).to(device)
    
    # Deterministic Settings
    temperature = 0.2  
    top_k = 10         
    
    if idx.size(1) > block_size:
        idx = idx[:, -block_size:]
        
    full_sequence = idx.tolist()[0]
    
    with torch.no_grad():
        # Step 1: Feed the entire prompt ONCE to fill the KV Cache
        logits, kv_cache = model(idx, kv_cache=None)
        next_logits = logits[:, -1, :]
        
        for _ in range(max_new_tokens):
            # Apply Temperature Scaling and Top-K Filtering
            next_logits = next_logits / temperature
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = -float('Inf')
            
            # Predict the next token
            probs = F.softmax(next_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) 
            
            full_sequence.append(idx_next.item())
            if "<END>" in decode([idx_next.item()]):
                break
                
            # Sliding Window fallback if the output gets too long for the buffer
            current_length = kv_cache[0][0][0].shape[1]
            if current_length >= block_size - 1:
                context_idx = torch.tensor([full_sequence[-(block_size-1):]], dtype=torch.long).to(device)
                logits, kv_cache = model(context_idx, kv_cache=None)
                next_logits = logits[:, -1, :]
                continue
                
            # Step 2: Pass ONLY the new token forward, utilizing the KV Cache
            logits, kv_cache = model(idx_next, kv_cache=kv_cache)
            next_logits = logits[:, -1, :]
            
    full_text = decode(full_sequence)
    raw_output = full_text.split("<ASSISTANT>\n")[-1].replace("<END>", "").strip()
    
    json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return {"error": "Invalid JSON structure.", "raw_text": raw_output}
    return {"error": "No JSON detected.", "raw_text": raw_output}

def enrich_character_data(char_json):
    if "error" in char_json: return char_json
        
    features_list = char_json.get("features", [])
    if not features_list:
        char_json["enriched_features"] = []
        return char_json 

    try:
        conn = psycopg2.connect(
            host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT
        )
        cur = conn.cursor()
        query = 'SELECT "Name", "Description" FROM "ClassFeatures" WHERE "Name" = ANY(%s)'
        cur.execute(query, (features_list,))
        rows = cur.fetchall()

        char_json["enriched_features"] = [{"name": r[0], "description": r[1]} for r in rows]
        cur.close()
        conn.close()
    except Exception as e:
        char_json["db_error"] = str(e)

    return char_json

# ==========================================
# 6. FASTAPI ENDPOINTS & BACKGROUND TASKS
# ==========================================
app = FastAPI(title="D&D Generator API")

tasks_db = {}

class ChatRequest(BaseModel):
    session_id: str
    prompt: str

@app.get("/")
def read_root():
    return {"status": "D&D Generator API is live"}

def process_character_task(task_id: str, session_id: str, prompt: str):
    print(f"[{task_id}] Task started.")
    try:
        base_json = generate_response(prompt)
        final_payload = enrich_character_data(base_json)
        
        if "error" in final_payload:
            tasks_db[task_id] = {"status": "failed", "error": final_payload["error"]}
            print(f"[{task_id}] Task failed during generation.")
            return
            
        save_message_to_db(session_id, "assistant", json.dumps(final_payload), is_json=True)
        
        tasks_db[task_id] = {
            "status": "completed",
            "data": final_payload
        }
        print(f"[{task_id}] Task successfully completed.")
        
    except Exception as e:
        tasks_db[task_id] = {"status": "failed", "error": str(e)}
        print(f"[{task_id}] Task encountered an exception: {e}")

@app.post("/generate")
async def generate_character(request: ChatRequest, background_tasks: BackgroundTasks):
    save_message_to_db(request.session_id, "user", request.prompt, is_json=False)
    
    task_id = str(uuid.uuid4())
    tasks_db[task_id] = {"status": "processing"}
    
    background_tasks.add_task(process_character_task, task_id, request.session_id, request.prompt)
    
    return {"task_id": task_id, "status": "processing"}

@app.get("/status/{task_id}")
async def check_status(task_id: str):
    task_info = tasks_db.get(task_id)
    
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")
        
    return task_info