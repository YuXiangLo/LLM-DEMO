import json
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
import torch

# Load the JSON file
with open('test.json', 'r') as file:
    data = json.load(file)

# Configuration and model loading as before
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoPeftModelForCausalLM.from_pretrained("./fine-tuned-llama", quantization_config=nf4_config)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = model.to("cuda")
model.eval()

# Instruction remains the same
instruction = "Find the input keyword in the following questions."

# Process each input
for item in data:
    input_text = f"Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n\n### Instruction:\n{instruction}\n\n### Input:\n{item['input']}\n\n### Response:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
    response = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    # print(response)
    response = response.split("### Response:")[-1].strip()
    print(response)
    # Here you can further process or store 'response'

