import streamlit as st
import requests
import uuid
import edge_tts
import asyncio
import os
from faster_whisper import WhisperModel
from audio_recorder_streamlit import audio_recorder
import tempfile



async def generate_tts(text, filename="question_audio.mp3"):
    communicate = edge_tts.Communicate(text, voice="en-US-GuyNeural")
    await communicate.save(filename)

API_URL = "http://backend:8000"


st.set_page_config(page_title="AI Interview Coach", layout="centered")

# ----------------- Session Setup ------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "round" not in st.session_state:
    st.session_state.round = 0
if "done" not in st.session_state:
    st.session_state.done = False

# ----------------- App UI ------------------

st.title("🧠 AI Interview Prep Coach")

if st.session_state.round == 0:
    st.subheader("Start a new interview")
    difficulty = st.selectbox("Select difficulty", ["easy", "medium", "hard"])
    category = st.selectbox("Select category", ["behavioral", "technical"])
    rounds = st.slider("Number of questions", 1, 5, 3)

    if st.button("🎬 Start Interview"):
        res = requests.post(f"{API_URL}/start", json={
            "session_id": st.session_state.session_id,
            "rounds": rounds,
            "difficulty": difficulty,
            "category": category
        })
        if res.status_code == 200:
            st.session_state.question = res.json()["question"]
            st.session_state.round = 1
            st.session_state.user_answer = ""  # Clear answer when starting
            if "question_audio_generated" not in st.session_state:
                st.session_state.question_audio_generated = False
            if "evaluation_audio_generated" not in st.session_state:
                st.session_state.evaluation_audio_generated = False
            if "feedback_audio_generated" not in st.session_state:
                st.session_state.feedback_audio_generated = False
            if "summary_audio_generated" not in st.session_state:
                st.session_state.summary_audio_generated = False

            if not st.session_state.question_audio_generated:
                asyncio.run(generate_tts(st.session_state["question"]))
                st.session_state.question_audio_generated = True

            st.rerun()
        else:
            st.error("Could not start interview. Try again.")

elif st.session_state.done:
    st.markdown("---")
    st.header("📊 Final Summary")
    res = requests.get(f"{API_URL}/summary", params={"session_id": st.session_state.session_id})
    if res.status_code == 200:
        st.session_state.summary=res.json().get("summary", "")
        if not st.session_state.summary_audio_generated:
                asyncio.run(generate_tts(st.session_state["summary"],filename="summary_audio.mp3"))
                st.session_state.summary_audio_generated = True
        # Add Play Audio
        if os.path.exists("summary_audio.mp3"):
            st.audio("summary_audio.mp3", format="audio/mp3")
            st.session_state.summary_audio_generated = False
        st.markdown(st.session_state.summary)
    else:
        st.error("Could not fetch summary.")

    if st.button("🔁 Restart"):
        st.session_state.clear()
        st.rerun()

else:
    st.subheader(f"Question {st.session_state.round}")
    st.markdown(f"**{st.session_state.get('question', '')}**")
    if not st.session_state.question_audio_generated:
                asyncio.run(generate_tts(st.session_state["question"]))
                st.session_state.question_audio_generated = True
    # Add Play Audio
    if os.path.exists("question_audio.mp3"):
        st.audio("question_audio.mp3", format="audio/mp3")

    st.title("🎙️ Record Your Answer")

    audio_bytes = audio_recorder()
    user_answer=""
    

    if st.button("✅ Submit Answer") and audio_bytes:
        if audio_bytes:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                audio_path = f.name

            st.audio(audio_bytes, format="audio/wav")

            model = WhisperModel("tiny", compute_type="int8")  # or "base"
            segments, _ = model.transcribe(audio_path)
            user_answer = " ".join([seg.text for seg in segments])
            st.success(f"📝 Transcription:\n\n{user_answer}")
            
        res = requests.post(f"{API_URL}/answer", json={
            "session_id": st.session_state.session_id,
            "user_message": user_answer
        })
        if res.status_code == 200:
            data = res.json()
            st.session_state.evaluations = data["evaluations"]
            st.session_state.feedback = data["feedback"]
            st.session_state.done = data["done"]
            st.session_state.flag=0
            st.markdown("### 🧾 Evaluation")
            if not st.session_state.evaluation_audio_generated:
                asyncio.run(generate_tts(st.session_state["evaluations"],filename="evaluation_audio.mp3"))
                st.session_state.evaluation_audio_generated = True
            # Add Play Audio
            if os.path.exists("evaluation_audio.mp3"):
                st.audio("evaluation_audio.mp3", format="audio/mp3")
                st.session_state.evaluation_audio_generated = False
            st.info(st.session_state.get("evaluations", ""))
            st.session_state.question = data.get("question", "")
            st.session_state.question_audio_generated = False 
            st.session_state.round += 1
            st.markdown("### 💡 Feedback")
            if not st.session_state.question_audio_generated:
                asyncio.run(generate_tts(st.session_state["feedback"],filename="feedback_audio.mp3"))
                st.session_state.feedback_audio_generated = True
            # Add Play Audio
            if os.path.exists("feedback_audio.mp3"):
                st.audio("feedback_audio.mp3", format="audio/mp3")
                st.session_state.feedback_audio_generated = False
            st.success(st.session_state.get("feedback", ""))
        if not st.session_state.done:      
                if st.button("Next Question"): 
                    st.rerun()           
        
                    
        else:
                if st.button("Get Final Summary"):
                    st.rerun()
            

        