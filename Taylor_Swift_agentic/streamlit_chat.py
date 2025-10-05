import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.analysis_assistant import AnalysisAssistant
from src.agents.openai_client import OpenAIClient
from src.agents.ollama_client import OllamaClient
import src.config as config

st.set_page_config(
    page_title="Taylor Swift Analysis Assistant",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .main {
        padding: 2rem;
    }
    h1 {
        color: #1DB954;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎵 Taylor Swift Analysis Assistant")
st.markdown("Ask questions about Taylor Swift's discography, eras, topics, and audio features")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    use_openai = st.toggle("Use OpenAI (Fast, Demo Mode)", value=config.USE_OPENAI)
    
    if use_openai:
        st.info("Using GPT-4o-mini: ~2-5s responses")
        if not config.OPENAI_API_KEY:
            st.error("⚠️ OpenAI API key not set! Add to config.py or environment variable.")
    else:
        st.info(f"Using {config.MODEL}: ~30-90s responses")
    
    st.divider()
    
    # Suggested questions
    st.subheader("Suggested Questions")
    suggestions = [
        "What are Taylor Swift's different eras?",
        "Generate insights from the analysis",
        "What makes Reputation era distinctive?",
        "Which songs are most similar to my favorites?",
        "What topics appear most in her lyrics?"
    ]
    
    for suggestion in suggestions:
        if st.button(suggestion, key=f"suggest_{suggestion[:20]}"):
            st.session_state.suggested_question = suggestion
    
    st.divider()
    
    # Clear conversation
    if st.button("Clear Conversation", type="secondary"):
        st.session_state.messages = []
        if 'agent' in st.session_state:
            st.session_state.agent.reset()
        st.rerun()

# Initialize agent
@st.cache_resource
def get_agent(use_openai):
    if use_openai:
        from src.agents.openai_client import OpenAIClient
        client = OpenAIClient(model=config.MODEL)
    else:
        from src.agents.ollama_client import OllamaClient
        client = OllamaClient(model=config.MODEL)
    
    # Create assistant with the client
    assistant = AnalysisAssistant()
    assistant.client = client
    return assistant

# Load agent
try:
    agent = get_agent(use_openai)
    st.session_state.agent = agent
except Exception as e:
    st.error(f"Error loading agent: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle suggested question
if "suggested_question" in st.session_state:
    prompt = st.session_state.suggested_question
    del st.session_state.suggested_question
else:
    prompt = st.chat_input("Ask about Taylor Swift's music...")

# Process user input
if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if prompt.lower() == "insights":
                    response = agent.suggest_insights()
                else:
                    response = agent.ask(prompt)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
st.caption("Powered by BERTopic, Spotify API, and LLMs | Data analysis of Taylor Swift's discography")
