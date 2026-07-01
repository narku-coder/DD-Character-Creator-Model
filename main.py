import os
import json
import re
import uuid
import torch
import torch.nn as nn
from torch.nn import functional as F
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
# 3. TRANSFORMER ARCHITECTURE
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

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x) 
        return wei @ v 

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

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
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 
        return logits, None

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
model.to(device)
print("Model ready!")

# ==========================================
# 5. INFERENCE & RAG PIPELINE
# ==========================================
def generate_response(user_query, max_new_tokens=400):
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
            
            if "<END>" in decode(idx[0].tolist()):
                break
                
    full_text = decode(idx[0].tolist())
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

# Temporary in-memory dictionary to hold task status
tasks_db = {}

class ChatRequest(BaseModel):
    session_id: str
    prompt: str

@app.get("/")
def read_root():
    return {"status": "D&D Generator API is live"}

# This function runs in the background, keeping the HTTP route free
def process_character_task(task_id: str, session_id: str, prompt: str):
    print(f"[{task_id}] Task started.")
    try:
        # 1. Run Inference
        base_json = generate_response(prompt)
        
        # 2. Apply RAG
        final_payload = enrich_character_data(base_json)
        
        if "error" in final_payload:
            tasks_db[task_id] = {"status": "failed", "error": final_payload["error"]}
            print(f"[{task_id}] Task failed during generation.")
            return
            
        # 3. Log the Assistant response to the database
        save_message_to_db(session_id, "assistant", json.dumps(final_payload), is_json=True)
        
        # 4. Mark task as completed so Flutter can fetch the data
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
    # 1. Log the User message immediately
    save_message_to_db(request.session_id, "user", request.prompt, is_json=False)
    
    # 2. Create a unique task ID
    task_id = str(uuid.uuid4())
    
    # 3. Initialize the task status
    tasks_db[task_id] = {"status": "processing"}
    
    # 4. Kick off the heavy PyTorch and Postgres processes in the background
    background_tasks.add_task(process_character_task, task_id, request.session_id, request.prompt)
    
    # 5. Instantly return to Flutter so Render doesn't throw a 502 Timeout
    return {"task_id": task_id, "status": "processing"}

@app.get("/status/{task_id}")
async def check_status(task_id: str):
    # Flutter calls this route to check if the transformer is done
    task_info = tasks_db.get(task_id)
    
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")
        
    return task_info