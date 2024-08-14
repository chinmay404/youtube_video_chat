import streamlit as st
from yt_response import YTResponseGenerator  # Ensure this import matches your project structure

st.set_page_config(page_title="ChatWithYouTube")
st.title('Chat With Youtube Video')
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    groq_api_key = None

# Prompt user to enter GROQ API Key manually if not found in file
if groq_api_key is None:
    with st.sidebar:
        st.title('Configuration')
        groq_api_key = st.text_input(label="Enter your GROQ API Key:", type="password")
        if groq_api_key:
            st.success('Groq API Key saved!', icon='✅')
        else:
            st.error('Please enter a GROQ API Key.', icon='⚠️')
else:
    with st.sidebar:
        st.title('Configuration')
        st.success('Groq API Key loaded from config.toml', icon='✅')
        

# Inside the 'todo' container, add checkboxes


# Initialize YTResponseGenerator if not already initialized
if "yt_response_generator" not in st.session_state:
    if groq_api_key:
        st.session_state.yt_response_generator = YTResponseGenerator(groq_api_key)
        st.success('Groq API Key loaded successfully!', icon='✅')
    else:
        st.error('Groq API Key not found. Please enter manually.', icon='⚠️')
        st.stop()

with st.sidebar:
    st.title('ChatWithYouTube')
    youtube_url = st.text_input('Enter YouTube Video URL:')
    
    st.markdown('')
    todo = st.container(height=300)
    
with todo:
    st.header('Todo List')
    todos = ["Conversation History", "Summury", "Add Chat Memory"]
    for index, task in enumerate(todos):
        st.checkbox(task, key=f"checkbox_{index}")



# Check if a YouTube URL has been entered
if youtube_url:
    # Extract transcript and initialize embeddings and chain
    st.session_state.yt_response_generator.get_transcript(youtube_url)
    st.session_state.yt_response_generator.gen_embeddings()
    st.session_state.yt_response_generator.load_qa_chain(st.session_state.yt_response_generator.llm_instance)
    st.success('Preprocessing complete!', icon='✅')
else:
    st.error('Please enter a YouTube Video URL.', icon='⚠️')



# Store chatbot generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you with the YouTube video?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input(disabled=not hasattr(st.session_state, "yt_response_generator")):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Process the user's query based on the YouTube video URL
    if youtube_url:
        st.session_state.yt_response_generator.handle_user_input(youtube_url + " " + prompt)
    else:
        st.error('Please enter a YouTube Video URL.', icon='⚠️')


# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Generate response using the existing YTResponseGenerator instance
            response = st.session_state.yt_response_generator.generate_response(prompt)
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
