import torch
import transformers
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipeline = transformers.pipeline(
    "text-generation",
    model = model_id,
    torch_dtype= torch.float16,
    device_map = "auto"
    )
sequences = pipeline(
    'What is nutritional value of eggs per 100gm in table format',
    do_sample = True,
    top_k = 4,
    num_return_sequences = 1,
    eos_token_id = tokenizer.eos_token_id,
    truncation = True,
    max_length = 300,
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")