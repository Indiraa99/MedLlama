from llama_cpp import Llama

# Path to your `.gguf` file
model_path = r"C:\Users\kuppa\.cache\lm-studio\models\lmstudio-community\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Load the Llama model
print("Loading the Llama model...")
llama = Llama(model_path=model_path)

# Define a sample prompt
prompt = "I have headache, what can i do?"

# Generate a response from the model
print("Generating response...")
output = llama(prompt, max_tokens=100, temperature=0.7, top_k=40)
print(f"Model Response: {output['choices'][0]['text']}")

