import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
import os
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.pinecone import PineconeVectorStore

from llama_index.core import StorageContext
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
api_key = os.environ["PINECONE_API_KEY"]
model = os.environ["OPENAI_GPT_MODEL"]


st.set_page_config(page_title="Chat with the Huberman about sleep, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = os.environ["OPENAI_API_KEY"]
st.title("Chat with the knowledge from Huberman lab podcast about sleep, powered by LlamaIndex 💬🦙")
st.info("Check out the Huberman Lab podacst here [link](https://www.hubermanlab.com/)", icon="🎙️")
         
# Initialize the chat messages history
st.session_state.messages = st.session_state.get('messages', [
    {"role": "assistant", "content": "Ask me a question about sleep!"}
])

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):       
        pc = PineconeGRPC(api_key=api_key)
        try:
            index_name = "llama-integration-example"
            # Initialize your index
            pinecone_index = pc.Index(index_name)

            # Initialize VectorStore
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            loaded_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        except Exception as e:
            print("Error accessing Pinecone index:", str(e))
            loaded_index = None

        

        return loaded_index
llm = OpenAI(model=model, temperature=0)
index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True, llm=llm )

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history