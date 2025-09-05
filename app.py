import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
import requests

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
api_key = os.getenv("GEOAPIFY_API_KEY")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

llm = ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it")
st.set_page_config(page_title="Travel Itinerary Planner Chatbot", page_icon=":airplane:")
st.title("Travel Itinerary Planner Chatbot")

session_id = "default_session"
if 'store' not in st.session_state:
    st.session_state.store = {}

if "places" not in st.session_state:
    st.session_state["places"] = []
if "places_history" not in st.session_state:
    st.session_state["places_history"] = []
if "itinerary_history" not in st.session_state:
    st.session_state["itinerary_history"] = []

country = st.text_input("Enter the Country you want to visit : ")
city = st.text_input("Enter the City you want to visit in that country : ")
duration = st.number_input("Enter the duration of your trip in days : ")
budget = st.number_input("Enter your budget in INR : ", min_value=100)

def get_top_popular_places(city, country, limit=10):
    api_key = os.getenv("GEOAPIFY_API_KEY")
    geo = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": f"{city}, {country}", "format": "json", "limit": 1},
        headers={"User-Agent": "Mozilla/5.0"}
    ).json()
    if not geo:
        return f"No location found for {city}, {country}"
    lat, lon = geo[0]["lat"], geo[0]["lon"]

    url = "https://api.geoapify.com/v2/places"
    params = {
        "categories": "tourism",
        "filter": f"circle:{lon},{lat},10000",  
        "limit": limit,
        "apiKey": api_key
    }
    resp = requests.get(url, params=params)
    data = resp.json()
    features = data.get("features", [])
    if not features:
        return f"No popular places found for {city}, {country}"

    top_places = []
    for f in features:
        prop = f["properties"]
        name = prop.get("name", "Unknown")
        address = prop.get("formatted", "")
        top_places.append(f"🏙️ **{name}**\n📍 {address}")
    return top_places

def embded_places_and_create_retriever(places_list):
    embeddings = get_embeddings()
    docs = [Document(page_content=place) for place in places_list]
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store.as_retriever()

if st.button("Show Top Places"):
    places = get_top_popular_places(city, country)
    st.session_state["places"] = places
    if isinstance(places, list) and places:
        st.session_state["places_history"].append((city, country, places))
        st.write("## Top Popular Places: ")
        for p in places:
            st.write(p)
        retriever = embded_places_and_create_retriever(places)
        
        contextualize_q_system_prompt = (
            """
            Given the chat history and a follow-up question, rewrite the follow-up question to be a standalone question.
            Assume the context is a list of travel places fetched for the user.
            Do not answer the question — only rewrite it if needed.
            """
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )
        
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        st.session_state["retriever"] = history_aware_retriever
    else:
        st.write(places)

if st.session_state["places"]:
    itinerary_prompt = f"""
        You are a travel planner. Suggest a {duration}-day itinerary for {city}, {country}.
        Consider a total budget of ₹{budget}. Use the following popular places:

        {chr(10).join(st.session_state["places"])}

        Create a daily plan that balances sightseeing, travel time, and cost.
        Mention affordable or free options where possible.
    """

    if st.button("Suggest Itinerary"):
        st.session_state.response = llm.invoke(itinerary_prompt)
        st.session_state["itinerary_history"].append((city, country, duration, budget, st.session_state.response.content))

    if st.session_state["itinerary_history"]:
        last_city, last_country, last_duration, last_budget, last_itinerary = st.session_state["itinerary_history"][-1]
        st.write("### Best Itinerary:")
        st.markdown(f"**{last_city}, {last_country} ({last_duration} days, ₹{last_budget}):**\n\n{last_itinerary}")    
        
    user_query = st.text_input("Ask about these places or your itinerary:")
    if user_query:
        travel_prompt = (
            f"""
    You are a travel assistant. Answer the user's question based on the following places and their travel plan.
    <context>
    {{context}}
    <context>
    Question: {{input}}
    Make sure your answer considers the trip duration and budget, and suggests affordable or free options where possible.
    """
        )
        travel_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", travel_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = st.session_state.get("retriever", None)
        if history_aware_retriever is not None:
            document_chain = create_stuff_documents_chain(llm=llm, prompt=travel_prompt_template)
            rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if "store" not in st.session_state:
                    st.session_state.store = {}
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                return st.session_state.store[session_id]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            response = conversational_rag_chain.invoke(
                {"input": user_query},
                config={"configurable": {"session_id": session_id}}
            )
            st.markdown(f"**Answer:** {response['answer']}")
        else:
            st.warning("Please click 'Show Top Places' first to initialize the retriever.")
 
