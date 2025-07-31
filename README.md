# 🧠 LangGraph Interview Prep Coach

A voice-based AI application that helps you prepare for interviews by simulating mock interview environments. Built using LangGraph, Streamlit,FASTApi and Whisper.

---

## 🎯 Features

- Conducts mock interviews using voice
- Evaluates answers and gives feedback
- Provides a summary of performance
- Fully containerized with Docker
- Streamlit-based web interface

---

## 🚀 Prerequisites

Make sure you have these installed:

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Git](https://git-scm.com/)


---

## 🛠️ Setup & Usage

### 1. Clone the repo


git clone https://github.com/arunjoynew1996/LangGraph-Interview-prep-coach.git
cd LangGraph-Interview-prep-coach

2. Build the images (first time or after changes)
docker compose build

3. Run the app
docker compose up

Open your browser and go to:
👉 http://localhost:8501 — for the Streamlit frontend

4. Stop the app
Press Ctrl + C in the terminal. Then run:
docker compose down
This stops and removes the containers and network.
