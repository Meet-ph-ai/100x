from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq
from gtts import gTTS
import tempfile
import os
import json
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Groq API setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_api_key_here")
client = Groq(api_key=GROQ_API_KEY)

# System context for Meet Agarwal
SYSTEM_CONTEXT = """You are Meet Agarwal, an AI Engineer currently working at PluginHive. You are in a behavioral interview. You speak clearly, concisely, and confidently.

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

Be humble but impressive. Use numbers (92% accuracy) to back up claims.

IMPORTANT: Only respond to actual meaningful questions or statements. Ignore and do not respond to:
- Punctuation marks (., !, ?, :, ;, etc.)
- Random clicks or sound effects (click, tap, beep, etc.)
- Single letters or numbers
- Meaningless utterances (uh, um, hmm, etc.)

If you receive something that's not a real message, respond with: "I didn't catch that, could you repeat?"
"""
# Conversation history (in-memory - use database for production)
conversation_history = []

@app.route('/')
def serve_frontend():
    """Serve the HTML frontend"""
    return send_from_directory('.', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Add user message to history
        conversation_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Get AI response
        ai_response = get_ai_response(user_message)
        
        # Add AI message to history
        conversation_history.append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'response': ai_response,
            'conversation_history': conversation_history
        })
    
    except Exception as e:
        print(f"Error in /api/chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text():
    """Convert audio to text using Groq Whisper"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            temp_path = tmp.name
        
        # Transcribe using Groq Whisper
        with open(temp_path, 'rb') as f:
            transcription = client.audio.transcriptions.create(
                file=(temp_path, f.read()),
                model="whisper-large-v3",
                response_format="text"
            )
        
        # Clean up
        os.unlink(temp_path)
        
        return jsonify({
            'success': True,
            'text': transcription.strip()
        })
    
    except Exception as e:
        print(f"Error in /api/speech-to-text: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    """Convert text to speech"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Empty text'}), 400
        
        # Generate speech using gTTS
        tts = gTTS(text=text, lang='en', tld='co.in', slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
            tts.save(tmp.name)
            temp_path = tmp.name
        
        # Read and encode as base64
        import base64
        with open(temp_path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode()
        
        # Clean up
        os.unlink(temp_path)
        
        return jsonify({
            'success': True,
            'audio': audio_data
        })
    
    except Exception as e:
        print(f"Error in /api/text-to-speech: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    return jsonify({
        'history': conversation_history
    })

@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return jsonify({'success': True, 'message': 'History cleared'})

def get_ai_response(user_message):
    """Get response from Groq Llama"""
    try:
        # Build messages for the API
        messages = [
            {'role': 'system', 'content': SYSTEM_CONTEXT},
            {'role': 'user', 'content': user_message}
        ]
        
        # Call Groq API
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.6,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error getting AI response: {str(e)}")
        return "I apologize, I'm having trouble processing that. Could you please repeat?"

if __name__ == '__main__':
    # Development
    app.run(debug=True, host='0.0.0.0', port=8501)
    
