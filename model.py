import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv()
CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)

from huggingface_hub import login
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
#login(token = "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")
login(token=ACCESS_TOKEN, add_to_git_credential=True)

class ChatModel:
    
    def __init__(self, model_id: str = "google/gemma-7b-it", device="cuda"):

        ACCESS_TOKEN = os.getenv(
            'ACCESS_TOKEN' )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=CACHE_DIR, token=ACCESS_TOKEN
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
            cache_dir=CACHE_DIR,
            token=ACCESS_TOKEN,
            #force_download=True,
        )
        self.model.eval()
        self.chat = []
        self.device = device

    def generate(self, question: str, context: str = None, max_new_tokens: int = 250):
        
    # Placeholder check for query clarity - implement specific criteria as needed
        if len(question.split()) < 3:  # Example criterion for considering a query to be unclear
            # Respond with an introduction and prompt for more information
            
            response = "Hello! I am an AI educational assistant here to help you prepare your course material. It seems your question might be a bit too broad or not fully detailed. Could you please provide more context or specify your question further? This will help me assist you more effectively."
        else:
            if context is None or context == "":
                prompt = f"Give a detailed answer to the following question. Question: {question}"
            else:
                prompt = f"""Imagine you are an integral part of a cutting-edge educational support system designed for Machine Learning education. This system dynamically incorporates uploaded class notes into a searchable database. Your role is to provide precise, insightful responses to inquiries that directly relate to the content of these uploaded materials, creating a personalized and contextually aware learning experience.
    
    Your tasks include:
    
    Analysing and correlating the inquiry with specific sections of the uploaded class notes where relevant information can be        found. 
    Also Produce responses based on the class notes content. Connect concepts and suggest further readings for comprehensive          understanding.
    
    Please don't sound robotic 
    
    Given this context, your immediate task is to respond to the following query based on the information in the class notes:
    
    Context: {context}
    Question: {question}"""
    
            chat = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            print(formatted_prompt)
            inputs = self.tokenizer.encode(
                formatted_prompt, add_special_tokens=False, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(formatted_prompt):].strip()  # Remove input prompt from response
        
        return response
