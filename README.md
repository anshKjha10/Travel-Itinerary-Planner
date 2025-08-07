# 🧳 Travel Itinerary Planner Chatbot

An intelligent, AI-powered travel planning chatbot built with **Streamlit**, **LangChain**, **Groq**, and **Geoapify API**. This app generates personalized day-wise itineraries based on your selected destination, trip duration, and budget — and even lets you chat about your travel plans!

---

## 🚀 Features

- 🗺️ **Top Tourist Places**: Get the most popular attractions in any city using OpenStreetMap + Geoapify.
- 🧠 **LLM-Powered Itinerary**: Creates smart, budget-aware daily plans using Groq's **Gemma 2 9B** model.
- 💬 **Conversational Chatbot**: Ask follow-up questions about the places or itinerary — the bot remembers your context!
- 🧠 **RAG Pipeline**: Uses **LangChain’s Retrieval-Augmented Generation** for context-aware answers.
- 🧭 **Local Vector Store**: Embeds place info using HuggingFace and stores in **FAISS** for fast semantic search.

---

## 🧩 Tech Stack

| Layer       | Tool / Library |
|-------------|----------------|
| Frontend    | Streamlit      |
| LLM         | Groq (Gemma 2 9B) |
| Embeddings  | HuggingFace (`all-MiniLM-L6-v2`) |
| Vector DB   | FAISS          |
| Retrieval   | LangChain RAG  |
| Data Source | Geoapify + OpenStreetMap |

---


pip install -r requirements.txt
streamlit run app.py
