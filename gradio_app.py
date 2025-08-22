# if you dont use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()

#VoiceBot UI with Gradio - Real-time Conversation Version
import os
import gradio as gr
import time
import json
import tempfile
import uuid
from datetime import datetime

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

# System prompts
initial_consultation_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            What's in this image?. Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Donot say 'In the image I see' but say 'With what I see, I think you have ....'
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
            Keep your answer concise (max 3 sentences). No preamble, start your answer right away please"""

follow_up_prompt = """You are continuing a medical consultation as a professional doctor. 
            The patient is asking follow-up questions. Respond naturally and professionally as a doctor would. 
            Keep responses concise and helpful. Don't use markdown formatting. 
            Answer as if speaking to a real patient. Maximum 3 sentences."""

greeting_prompt = """You are a professional doctor greeting a new patient. 
            The patient just said something but hasn't provided an image yet. 
            Greet them warmly and ask them to describe their concern or upload an image if they have one. 
            Keep it brief and professional. Maximum 2 sentences."""

class DoctorConversation:
    def __init__(self):
        self.conversation_history = []
        self.session_start = datetime.now()
        self.has_initial_image = False
        
    def add_to_history(self, user_input, doctor_response):
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user': user_input,
            'doctor': doctor_response
        })
    
    def get_context_for_llm(self, max_exchanges=3):
        """Get recent conversation context for the LLM"""
        if not self.conversation_history:
            return ""
        
        context = "Previous conversation:\n"
        recent_history = self.conversation_history[-max_exchanges:]
        
        for exchange in recent_history:
            context += f"Patient: {exchange['user']}\n"
            context += f"Doctor: {exchange['doctor']}\n"
        
        return context + "\nCurrent question: "
    
    def reset(self):
        self.conversation_history = []
        self.session_start = datetime.now()
        self.has_initial_image = False

# Global conversation manager
doctor_session = DoctorConversation()

def text_to_speech_with_elevenlabs_fixed(input_text):
    """Generate speech and return the file path"""
    try:
        from elevenlabs.client import ElevenLabs
        import elevenlabs
        
        ELEVENLABS_API_KEY = os.environ.get("ELEVEN_API_KEY")
        if not ELEVENLABS_API_KEY:
            print("ElevenLabs API key not found")
            return None
            
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        
        # Create a unique temporary file
        temp_dir = tempfile.gettempdir()
        unique_id = str(uuid.uuid4())[:8]
        output_filepath = os.path.join(temp_dir, f"doctor_response_{unique_id}.mp3")
        
        audio = client.generate(
            text=input_text,
            voice="Aria",
            output_format="mp3_22050_32",
            model="eleven_turbo_v2"
        )
        
        elevenlabs.save(audio, output_filepath)
        return output_filepath
        
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

def process_conversation(audio_filepath, image_filepath, chat_history):
    """Main function to process user input and generate doctor response"""
    
    if not audio_filepath:
        return chat_history, None, "Please record your voice message first.", gr.Audio(value=None)
    
    try:
        # Transcribe the audio
        user_text = transcribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3"
        )
        
        if not user_text or user_text.strip() == "":
            return chat_history, None, "Sorry, I couldn't understand what you said. Please try again.", gr.Audio(value=None)
        
        # Determine the type of response needed
        doctor_response = ""
        
        if image_filepath and not doctor_session.has_initial_image:
            # First consultation with image
            query = initial_consultation_prompt + " " + user_text
            doctor_response = analyze_image_with_query(
                query=query,
                encoded_image=encode_image(image_filepath),
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
            doctor_session.has_initial_image = True
            
        elif doctor_session.conversation_history:
            # Follow-up conversation
            context = doctor_session.get_context_for_llm()
            full_query = follow_up_prompt + "\n\n" + context + user_text
            
            # Use Groq for text-only follow-up
            from groq import Groq
            client = Groq()
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": full_query}],
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
            doctor_response = chat_completion.choices[0].message.content
            
        else:
            # First interaction without image
            query = greeting_prompt + " Patient said: " + user_text
            from groq import Groq
            client = Groq()
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": query}],
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
            doctor_response = chat_completion.choices[0].message.content
        
        # Clean up the response
        doctor_response = doctor_response.strip()
        
        # Add to conversation history
        doctor_session.add_to_history(user_text, doctor_response)
        
        # Update chat interface
        chat_history.append([user_text, doctor_response])
        
        # Generate speech response
        voice_file = text_to_speech_with_elevenlabs_fixed(doctor_response)
        
        # Return updated chat, audio response, status, and clear audio input
        return chat_history, voice_file, f"✅ Response generated. Ready for next question!", gr.Audio(value=None)
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Error in process_conversation: {e}")
        return chat_history, None, error_msg, gr.Audio(value=None)

def clear_conversation():
    """Reset the conversation"""
    global doctor_session
    doctor_session.reset()
    return [], None, "Conversation cleared. Ready for new consultation!", gr.Audio(value=None)

def save_conversation():
    """Save current conversation to file"""
    if not doctor_session.conversation_history:
        return "No conversation to save."
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save to desktop or user's documents folder
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        if os.path.exists(desktop):
            filename = os.path.join(desktop, f"doctor_consultation_{timestamp}.json")
        else:
            filename = f"doctor_consultation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'session_start': doctor_session.session_start.isoformat(),
                'conversation_history': doctor_session.conversation_history
            }, f, indent=2)
        return f"Conversation saved as {filename}"
    except Exception as e:
        return f"Error saving conversation: {str(e)}"

def update_status_for_recording():
    """Update status when recording starts"""
    return "🎤 Recording... Speak now!"

def update_status_for_stop():
    """Update status when recording stops"""
    return "⏹️ Recording stopped. Click 'Send' to process your question."

# Create the Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(), 
    title="AI Doctor - Real-time Consultation"
) as app:
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>🩺 AI Doctor - Real-time Consultation</h1>
        <p>Have a continuous conversation with your AI doctor!</p>
        <div style="background: #f0f8ff; padding: 10px; border-radius: 5px; margin: 10px 0;">
            <strong>💡 How it works:</strong> Record → Send → Get Response → Record Again → Send → Continue...
        </div>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # Main chat interface
            chatbot = gr.Chatbot(
                label="💬 Conversation with Doctor",
                height=400,
                show_label=True,
                type="tuples",
                bubble_full_width=False,
                show_copy_button=True
            )
            
            # Status display
            status_display = gr.Textbox(
                label="📊 Status",
                value="🟢 Ready! Record your voice and click Send to start consultation.",
                interactive=False,
                max_lines=2
            )
            
        with gr.Column(scale=2):
            # Input section
            with gr.Group():
                gr.Markdown("### 🎤 Record Your Question")
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Click to record your voice",
                    show_label=True
                )
                
                gr.Markdown("### 📷 Upload Image (Optional)")
                gr.Markdown("*Only needed for initial consultation with visual analysis*")
                image_input = gr.Image(
                    type="filepath",
                    label="Medical image",
                    show_label=False,
                    height=150
                )
                
                # Send button - prominently displayed
                send_btn = gr.Button(
                    "🚀 Send Message", 
                    variant="primary", 
                    size="lg",
                    scale=1
                )
            
            # Doctor's response section
            with gr.Group():
                gr.Markdown("### 🔊 Doctor's Voice Response")
                audio_output = gr.Audio(
                    label="Listen to doctor's response",
                    show_label=True,
                    interactive=False
                )
            
            # Control buttons
            with gr.Row():
                clear_btn = gr.Button("🗑️ New Session", variant="secondary")
                save_btn = gr.Button("💾 Save Chat", variant="secondary")
    
    # Instructions panel
    with gr.Accordion("📋 Complete Instructions", open=False):
        gr.Markdown("""
        ## 🚀 Quick Start Guide:
        
        **For First Question (with image):**
        1. 🎤 Click microphone and record your question
        2. 📷 Upload a medical image (optional)
        3. 🚀 Click "Send Message"
        4. 🔊 Listen to doctor's response
        
        **For Follow-up Questions:**
        1. 🎤 Record another question (no need to upload image again)
        2. 🚀 Click "Send Message" 
        3. 🔊 Get response
        4. 🔄 Repeat for continuous conversation
        
        ## 💡 Tips:
        - **Speak clearly** into your microphone
        - **Wait for response** before asking next question
        - **Use "New Session"** to start fresh consultation
        - **Save important conversations** for later reference
        
        ## 🩺 Conversation Flow:
        ```
        You: Record + Send → Doctor: Responds → You: Record + Send → Doctor: Responds...
        ```
        """)
    
    # Event handlers with proper clearing
    send_btn.click(
        fn=process_conversation,
        inputs=[audio_input, image_input, chatbot],
        outputs=[chatbot, audio_output, status_display, audio_input],
        show_progress=True
    )
    
    clear_btn.click(
        fn=clear_conversation,
        outputs=[chatbot, audio_output, status_display, audio_input]
    )
    
    save_btn.click(
        fn=save_conversation,
        outputs=[status_display]
    )
    
    # Update status based on recording state
    audio_input.start_recording(
        fn=update_status_for_recording,
        outputs=[status_display]
    )
    
    audio_input.stop_recording(
        fn=update_status_for_stop,
        outputs=[status_display]
    )

if __name__ == "__main__":
    print("🩺 Starting AI Doctor Application...")
    print("📍 Access at: http://127.0.0.1:7860")
    print("💡 Usage: Record → Send → Get Response → Repeat")
    
    app.launch(
        debug=True,
        share=False,
        server_port=7860,
        server_name="127.0.0.1",
        show_error=True
    )