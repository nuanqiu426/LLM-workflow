from transformers import AutoModelForCausalLM, AutoTokenizer
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model_path = "/workspace/LLM-finetune/codeLLM/huggingface/unsloth/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
)
print(model)