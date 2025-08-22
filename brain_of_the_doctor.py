# if you dont use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()

#Step1: Setup GROQ API key
import os

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")

#Step2: Convert image to required format
import base64


#image_path="acne.jpg"

def encode_image(image_path):   
    image_file=open(image_path, "rb")
    return base64.b64encode(image_file.read()).decode('utf-8')

#Step3: Setup Multimodal LLM 
from groq import Groq

query="Is there something wrong with my face?"
#model = "meta-llama/llama-4-maverick-17b-128e-instruct"
model="meta-llama/llama-4-scout-17b-16e-instruct"
#model = "meta-llama/llama-4-scout-17b-16e-instruct"
#model="llama-3.2-90b-vision-preview" #Deprecated

def analyze_image_with_query(query, model, encoded_image):
    client=Groq()  
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }]
    chat_completion=client.chat.completions.create(
        messages=messages,
        model=model
    )

    return chat_completion.choices[0].message.content

def analyze_follow_up_query(query, model, conversation_context=""):
    """
    Handle follow-up queries without images
    """
    client = Groq()
    
    # Build the full context
    full_query = query
    if conversation_context:
        full_query = conversation_context + "\n\nCurrent question: " + query
    
    messages = [
        {
            "role": "user",
            "content": full_query
        }
    ]
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )
    
    return chat_completion.choices[0].message.content

def get_conversational_response(query, model, chat_history=None):
    """
    Enhanced function to handle both image-based and text-based conversations
    """
    client = Groq()
    
    # Build conversation context from history
    messages = []
    
    if chat_history:
        # Add conversation history as context
        for user_msg, doctor_msg in chat_history[-5:]:  # Keep last 5 exchanges
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": doctor_msg})
    
    # Add current query
    messages.append({"role": "user", "content": query})
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )
    
    return chat_completion.choices[0].message.content