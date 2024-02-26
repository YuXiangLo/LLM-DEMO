import torch
import math
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, random_split

# 1. Load the pretrained BART model and its tokenizer.
model_name = 'facebook/bart-large'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset
class CustomDataset(Dataset):
    def __init__(self, source_texts, target_texts, tokenizer, max_length=32):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, index):
        source_tokenized = self.tokenizer(self.source_texts[index], return_tensors='pt', max_length=self.max_length, truncation=True)
        target_tokenized = self.tokenizer(self.target_texts[index], return_tensors='pt', max_length=self.max_length, truncation=True)
        return {"source_ids": source_tokenized['input_ids'].squeeze(), 
                "source_mask": source_tokenized['attention_mask'].squeeze(),
                "target_ids": target_tokenized['input_ids'].squeeze()}

df = pd.read_csv('data/input_leetcode_train_v3.csv')
X = df['Sentence(Processed)'].tolist()
y = df['Label(Main)'].tolist()
X = ['None' if isinstance(item, float) and math.isnan(item) else item for item in X]
y = ['None' if isinstance(item, float) and math.isnan(item) else item for item in y]

X_test = X[:-20]
y_test = y[:-20]
X = X[-20:]
y = y[-20:]

# TODO remove test from X and y

# Initialize dataset and dataloader
dataset = CustomDataset(X, y, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Define the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 3) # 3 epochs

# Define a function to calculate evaluation loss
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["source_ids"].to(device)
            masks = batch["source_mask"].to(device)
            targets = batch["target_ids"].to(device)

            outputs = model(input_ids=inputs, attention_mask=masks, labels=targets)
            total_loss += outputs.loss.item()

    return total_loss / len(dataloader)

model.to(device)

def generate_predictions(model, input_texts, actual_texts, device):
    model.to(device)  # Ensure the model is on the correct device
    max_length = 512
    acc = 0
    for input_text, actual_text in zip(input_texts, actual_texts):
        input_ids = tokenizer.encode(input_text, max_length=max_length, return_tensors="pt", truncation="longest_first").to(device)
        output_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
        generated_text = tokenizer.decode(output_ids[0].cpu(), skip_special_tokens=True)
        acc += generated_text == actual_text
    
    acc /= len(actual_texts)
    return acc

# Training loop with loss print at every 0.33 epoch
print_every = len(train_dataloader) // 3  # Calculate the number of steps for 0.33 epoch
for epoch in range(5):  # 3 epochs
    model.train()
    for i, batch in enumerate(train_dataloader):
        inputs = batch["source_ids"].to(device)
        masks = batch["source_mask"].to(device)
        targets = batch["target_ids"].to(device)

        outputs = model(input_ids=inputs, attention_mask=masks, labels=targets)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if (i + 1) % print_every == 0 or i == len(train_dataloader) - 1:
            eval_loss = evaluate(model, val_dataloader)
            acc = generate_predictions(model, X_test, y_test, device)
            print(f"Epoch: {epoch + (i + 1) / len(train_dataloader):.2f}, Training Loss: {loss.item()}, Evaluation Loss: {eval_loss}, acc: {acc}")

# Save the model
model.save_pretrained("./fine_tuned_bart/")
tokenizer.save_pretrained("./fine_tuned_bart/")
