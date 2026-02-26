import os
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google.generativeai import types

# =============================
# تحميل ملف البيئة (Environment Variables)
# =============================
# افتراضًا أن لديك ملف .env يحتوي على GEMINI_API_KEY
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

# =============================
# إنشاء تطبيق FastAPI
# =============================
app = FastAPI(title="Data Science Chatbot",
              description="🤖 بوت متخصص في مجال علم البيانات فقط باستخدام Gemini",
              version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# اختيار الموديل وتحديد شخصية النظام
# =============================
# استخدام system_instruction لتعريف قواعد البوت
system_instruction = """
أنت مساعد ذكي متخصص فقط في **علم البيانات** (Data Science).
- إذا سُئلت عن أي شيء خارج علم البيانات: اعتذر بأدب وقل أنك مختص فقط في هذا المجال.
"""
model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_instruction)

# =============================
# نموذج البيانات المستقبلة من المستخدم (Request Body)
# =============================
class ChatRequest(BaseModel):
    user_id: str
    message: str

# =============================
# تخزين المحادثات (Conversation History)
# =============================
conversations = {}


# =============================
# نقطة المحادثة الرئيسية
# =============================
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        user_id = req.user_id
        user_message = req.message

        if user_id not in conversations:
            conversations[user_id] = []

        # إضافة رسالة المستخدم الجديدة إلى السجل
        conversations[user_id].append({"role": "user", "parts": [user_message]})
    
        # إرسال الطلب إلى Gemini مع سجل المحادثة المحدث
        response = model.generate_content(
            conversations[user_id]
        )

        # إضافة رد المساعد إلى السجل
        conversations[user_id].append({"role": "model", "parts": [response.text]})

        # إعادة الرد والسجل المحدث
        readable_history = []
        for msg in conversations[user_id]:
            role = "👤 المستخدم" if msg['role'] == 'user' else "🤖 المساعد"
            readable_history.append(f"{role}: {msg['parts'][0]}")

        return {
            "reply": response.text,
            "history": readable_history
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def get_status():
    return {"status": "ok", "message": "API is running successfully."}