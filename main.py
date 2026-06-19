from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import data_tokenizer # Your updated script containing the model and RAG functions
import psycopg2

app = FastAPI(title="D&D Generator API")

DB_PARAMS = {
    "dbname": "neondb",
    "user": "Narku_Model",
    "password": "V0113yba11ru135",
    "host": "ep-empty-butterfly-aq64mw79-pooler.c-8.us-east-1.aws.neon.tech", # or your Render DB external host
    "port": "5432"
}

class ChatRequest(BaseModel):
    session_id: str  # Send a unique ID from the phone to group messages
    prompt: str

def save_message_to_db(session_id: str, role: str, content: str, is_json: bool):
    try:
        conn = psycopg2.connect(**DB_PARAMS)
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
        print(f"Failed to log message to PostgreSQL: {e}")

@app.post("/generate")
async def generate_character(request: ChatRequest):
    try:
        # 1. Generate JSON shell using the loaded model
        base_json = data_tokenizer.generate_response(data_tokenizer.model, request.prompt)
        
        # 2. Enrich with PostgreSQL RAG
        final_payload = data_tokenizer.enrich_character_data(base_json)
        
        if "error" in final_payload:
            raise HTTPException(status_code=400, detail=final_payload["error"])
            
        return final_payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run locally using: uvicorn main:app --reload