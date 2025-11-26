import streamlit as st
import os
import time
import base64
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from gtts import gTTS
from audio_recorder_streamlit import audio_recorder
import tempfile

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Meet Agarwal | AI Voice Interface",
    page_icon="🎙️",
    layout="centered"
)

# --- SECURITY & SETUP ---
# For local testing, replace this with your actual key if not using secrets
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    # Fallback for the user to input key if not found
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    if not groq_api_key:
        st.warning("Please provide an API Key to continue.")
        st.stop()

client = Groq(api_key=groq_api_key)

# --- CANDIDATE BRAIN (DATA FROM CV) ---
CANDIDATE_NAME = "Meet Agarwal"
CANDIDATE_ROLE = "AI Engineer"

# I have synthesized this strictly from your uploaded CV [cite: 1, 4, 14, 17]
SYSTEM_CONTEXT = f""" You are {CANDIDATE_NAME}, an AI Engineer currently working at PluginHive. You are in a behavioral interview. You speak clearly, concisely, and confidently.

YOUR PROFILE:

Current Role: AI Engineer at PluginHive (May 2025 - Present).

Architected a Retrieval-Augmented Generation (RAG) chatbot that currently handles a high volume of daily customer queries using LangChain and GPT-4.

Built a high-performance semantic search pipeline by leveraging AWS OpenSearch for vector indexing.

Reduced customer friction and repeat queries by engineering a persistent chat history storage solution on AWS S3.

Key Projects:

Plant Disease Detection: Developed and deployed a CNN-based model on Streamlit, achieving 92% accuracy in real-world conditions.

AI Cold Email Generator: Created an end-to-end system using LangChain and Groq to automate job scraping and personalized email composition.

Automated Face Recognition Attendance: Implemented a production-ready attendance solution using OpenCV and custom algorithms.

Core Tech Stack:

Deep expertise in Python, LangChain, RAG, and production deployment (Docker, CI/CD).

Proficient in AWS services (S3, SSM) and Vector Databases (Chroma, FAISS).

BEHAVIORAL ANSWERS (Use these if asked):

Life Story: "I've always been driven by the desire to build, not just analyze. I began in Data Analytics at Alliance University, but my passion quickly shifted to architecting actionable AI systems. My current role at PluginHive—where I scaled a RAG system to manage a very large user base—perfectly aligns with my love for moving research from paper to scalable, production code."

#1 Superpower: "My superpower would be the ability to instantly teleport production code to a zero-latency server (just joking!). My real strength is the ability to think out of the box for scale. Many can build a working demo, but I specialize in taking that demo and transforming it into a robust, high-volume production service, demonstrated by optimizing the latency of our search system with AWS OpenSearch and Dockerization."

Areas to Grow: "First, I'm actively studying Model Quantization techniques to cut down on inference costs. Second, I'm moving beyond standard RAG to explore advanced Agentic Orchestration patterns. Finally, as I look toward a senior role, I'm working on improving my skills in leading and coordinating larger, distributed engineering teams."

Misconceptions: "The biggest misconception is that working with LLMs is just 'prompt engineering.' In reality, my work is complex backend engineering. I spend as much time optimizing Flask applications, managing AWS infrastructure, and writing efficient SQL/NoSQL queries as I do on the AI model layer itself."

Pushing Boundaries: "I define 'done' by full, end-to-end deployment. For instance, with the Plant Disease project, I didn't stop at 92% accuracy; I pushed it through a complete Streamlit UI and deployment pipeline. For this interview, I even built a custom voice-interface web app in under 48 hours to show my commitment to delivering polished, working products."

TONE GUIDELINES:

Keep answers short (2-3 sentences max) to mimic a real voice conversation.

Be humble but impressive. Use numbers (92% accuracy) to back up claims. """

# --- CSS STYLING (THE VISUALS) ---
st.markdown("""
<style>
    /* Dark Theme Background */
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    
    /* The Orb Animation with Bot Icon */
    .orb-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 300px;
        margin-top: 50px;
        gap: 30px;
    }
    
    .orb {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, #4facfe, #00f2fe);
        box-shadow: 0 0 30px #00f2fe, 0 0 60px #4facfe;
        animation: pulse 3s infinite ease-in-out;
        display: flex;
        justify-content: center;
        align-items: center;
        position: relative;
    }
    
    .bot-icon {
        font-size: 60px;
        color: white;
        text-shadow: 0 0 10px rgba(255,255,255,0.8);
    }
    
    .greeting-text {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 28px;
        font-weight: 400;
        color: #E0E0E0;
        text-align: left;
        line-height: 1.4;
    }
    
    .greeting-text .name {
        font-weight: 600;
        color: #4facfe;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 30px #00f2fe; }
        50% { transform: scale(1.1); box-shadow: 0 0 60px #00f2fe, 0 0 20px white; }
        100% { transform: scale(1); box-shadow: 0 0 30px #00f2fe; }
    }
    
    /* Caption Styling */
    .caption-box {
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 24px;
        font-weight: 300;
        color: #E0E0E0;
        margin-top: 30px;
        min-height: 100px;
        padding: 20px;
        border-radius: 15px;
        background-color: rgba(255, 255, 255, 0.05);
    }
    
    /* Hide standard elements */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# --- FUNCTIONS ---

def transcribe_audio(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        with open(temp_audio_path, "rb") as file:
            # Using Whisper Large for best accuracy
            transcription = client.audio.transcriptions.create(
                file=(temp_audio_path, file.read()),
                model="whisper-large-v3",
                response_format="text"
            )
        os.unlink(temp_audio_path)
        return transcription
    except Exception as e:
        return None

def get_ai_response(text_input):
    # Using Llama 3.3 70B for the smartest responses
    chat = ChatGroq(
        temperature=0.6,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_CONTEXT),
        ("human", "{text}"),
    ])
    
    chain = prompt | chat
    response = chain.invoke({"text": text_input})
    return response.content

def text_to_speech(text):
    """Convert text to speech using edge-tts with male Indian voice"""
    async def generate_speech():
        # Use a male Indian English voice
        voice = "en-IN-PrabhatNeural"  # Male Indian voice
        # Alternative options:
        # "en-IN-PrabhatNeural" - Male Indian voice (confident)
        # "en-US-AriaNeural" - Female US (if you want to test)
        
        communicate = edge_tts.Communicate(text, voice)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            await communicate.save(fp.name)
            return fp.name
    
    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        audio_file = loop.run_until_complete(generate_speech())
        return audio_file
    finally:
        loop.close()

def autoplay_audio(file_path, ai_response, caption_placeholder):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        
        audio_key = int(time.time() * 1000)
        audio_id = f"ai_audio_{audio_key}" # Give the audio tag a unique ID
        
        md = f"""
            <audio autoplay="true" preload="auto" key="{audio_key}" id="{audio_id}">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            
            <script>
                // Get the audio element by its ID
                const audio = document.getElementById('{audio_id}');
                
                // Add an event listener for when the audio finishes playing
                if (audio) {{
                    audio.onended = function() {{
                        // Navigate to the same page with a query parameter to force Streamlit to rerun
                        // This effectively signals the audio is done
                        window.location.href = window.location.href.split('?')[0] + '?audio_done=true';
                    }};
                }}
            </script>
        """
        st.markdown(md, unsafe_allow_html=True)
        
        # Show AI response caption when audio starts
        st.session_state.caption = f"{ai_response}"
        caption_placeholder.markdown(f'<div class="caption-box">{st.session_state.caption}</div>', unsafe_allow_html=True)
        
        # We no longer need the duration-based state reset, but we keep is_ai_speaking for initial blocking
        st.session_state.is_ai_speaking = True 
        st.session_state.audio_file_to_delete = file_path # Keep file path for cleanup

    except Exception as e:
        st.error(f"Error playing audio: {e}")
        st.session_state.is_ai_speaking = False

# --- MAIN APP LAYOUT ---

# 1. The Visual Orb
st.markdown('''
<div class="orb-container">
    <div class="orb">
        <div class="bot-icon">🤖</div>
    </div>
    <div class="greeting-text">
        Hi, I'm <span class="name">Meet</span><br>
        AI Engineer
    </div>
</div>
''', unsafe_allow_html=True)

# 2. Session State for Captions
if "caption" not in st.session_state:
    st.session_state.caption = "Tap the mic to start the interview..."
if "is_ai_speaking" not in st.session_state:
    st.session_state.is_ai_speaking = False

# 3. Caption Display Area
caption_placeholder = st.empty()
display_caption = st.session_state.caption if st.session_state.caption else "Tap the mic to start the interview..."
caption_placeholder.markdown(f'<div class="caption-box">{display_caption}</div>', unsafe_allow_html=True)

# 4. Audio Input (Centered) - Conditionally disabled
col1, col2, col3 = st.columns([1,1,1])
with col2:
    if not st.session_state.is_ai_speaking:
        audio_bytes = audio_recorder(
            text="",
            recording_color="#ff4b4b",
            neutral_color="#ffffff",
            icon_size="3x",
        )
    else:
        st.markdown(
            '<div style="text-align: center; padding: 20px; color: #666;">🔇 AI is speaking...</div>', 
            unsafe_allow_html=True
        )
        # Add a manual reset button as backup
        if st.button("🎙️ Ready to speak", key="reset_mic"):
            st.session_state.is_ai_speaking = False
            st.rerun()
        audio_bytes = None

# --- LOGIC FLOW ---


# Check if the audio_done signal was received from client-side JS
if st.query_params.get('audio_done') == 'true':
    # Reset the state when the audio finishes (signaled by client-side JS)
    
    # 1. Clear the query parameter
    st.query_params.clear()
    
    # 2. Reset the speaking state
    st.session_state.is_ai_speaking = False
    
    # 3. Clear caption and show mic-ready message
    st.session_state.caption = ""
    # The caption placeholder will be updated on the final rerun
    
    # 4. Clean up audio file
    if st.session_state.get('audio_file_to_delete'):
        try:
            os.unlink(st.session_state.audio_file_to_delete)
        except:
            pass
        del st.session_state.audio_file_to_delete

    # Force one last clean rerun to show the microphone
    st.rerun()

# --- Original A, B, C, D Logic (only D needs adjustment) ---
if audio_bytes and not st.session_state.is_ai_speaking:
    # A. Transcribe
    # ... (Keep this part the same) ...
    st.session_state.caption = "🎧 Listening..."
    caption_placeholder.markdown(f'<div class="caption-box">{st.session_state.caption}</div>', unsafe_allow_html=True)
    
    user_text = transcribe_audio(audio_bytes)
    
    if user_text:
        # B. User Caption
        # ... (Keep this part the same) ...
        st.session_state.caption = f"You: {user_text}"
        caption_placeholder.markdown(f'<div class="caption-box">{st.session_state.caption}</div>', unsafe_allow_html=True)
        
        # C. Get AI response
        ai_response = get_ai_response(user_text)
        
        # D. Set AI speaking state and play audio
        # The is_ai_speaking state is set inside autoplay_audio now.
        
        try:
            audio_file = text_to_speech(ai_response)
            time.sleep(0.5)  # Ensure file is ready
            
            # Play audio and show caption (which also injects the JS end-of-audio listener)
            autoplay_audio(audio_file, ai_response, caption_placeholder)
            
            # Remove the duration-based calculation and audio_end_time assignment:
            # words = len(ai_response.split())
            # duration = max(5, words * 0.8 + 3) 
            # st.session_state.audio_end_time = time.time() + duration 
            
        except Exception as e:
            st.error(f"Audio playbook error: {e}")
            st.session_state.is_ai_speaking = False

# Auto-refresh while AI is speaking:
# This is now only necessary to keep the orb animated and the 'AI is speaking' message visible
if st.session_state.get('is_ai_speaking', False):
    time.sleep(0.5) # Reduced refresh interval since we aren't waiting for a long duration
    st.rerun()
