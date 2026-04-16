import sqlite3
import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS so your Vercel frontend can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD AI FILES ---
# Ensure mood_model.joblib and vectorizer.joblib are in the same folder
model = joblib.load('mood_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('goodwill.db')
    conn.execute('CREATE TABLE IF NOT EXISTS posts (text TEXT, mood TEXT)')
    conn.close()

init_db()

# --- DATA MODELS ---
class Message(BaseModel):
    msg: str

# --- ROUTES ---

@app.get("/")
def read_root():
    return {"status": "API is running", "docs": "/docs"}

@app.get("/messages")
def get_messages():
    conn = sqlite3.connect('goodwill.db')
    cursor = conn.execute('SELECT text, mood FROM posts ORDER BY ROWID DESC')
    messages = [{"text": row[0], "mood": row[1]} for row in cursor.fetchall()]
    conn.close()
    return messages
@app.post("/add")
async def add_message(message: Message):
    user_text = message.msg
    
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # AI PREDICTION
    vectorized_text = vectorizer.transform([user_text])
    prediction = model.predict(vectorized_text)[0]
    mood = "😊 Positive" if prediction == 1 else "😟 Negative"
    
    # SAVE TO DATABASE
    conn = sqlite3.connect('goodwill.db')
    conn.execute('INSERT INTO posts VALUES (?, ?)', (user_text, mood))
    conn.commit()
    conn.close()
    
    return {"status": "success", "mood": mood}

if __name__ == "__main__":
    import uvicorn
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

