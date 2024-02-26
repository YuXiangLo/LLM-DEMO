import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer

# Load the fine-tuned model and tokenizer
model_name = "fine_tuned_bart"  # Adjust the directory name as needed
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

df = pd.read_csv('data/input_leetcode_test_v3.csv')
X_test = df['Sentence(Processed)'].tolist()
X_test = ['None' if isinstance(item, float) and math.isnan(item) else item for item in X_test]
Index = df['ID']

input_texts = X_test  # Replace with your test data
max_length = 128  # Adjust this as needed for your task

# Generate predictions for each input text
generated_texts = []
for (input_text, ID) in zip(input_texts, Index):
    input_ids = tokenizer.encode(input_text, max_length=max_length, return_tensors="pt", truncation="longest_first")
    # Generate the output sequence using the model
    output_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_texts.append(generated_text)
    
    print(ID, '"' + generated_text + '"',sep=',')

