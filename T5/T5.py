import torch
import pandas as pd
import math
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from transformers import Trainer
import csv

model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Custom Loss
def custom_loss(outputs, labels, lambda_value=5.0):
    ce_loss = F.cross_entropy(outputs.view(-1, outputs.shape[-1]), labels.view(-1), ignore_index=tokenizer.pad_token_id)
    empty_penalty = ((outputs.argmax(dim=-1).sum(dim=1) == tokenizer.pad_token_id).float() * lambda_value).mean()
    return ce_loss + empty_penalty

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        loss = custom_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

torch.cuda.empty_cache()

df = pd.read_csv('data/input_leetcode_train_v3.csv')
X = df['Sentence(Processed)'].tolist()
y = df['Label(Main)'].tolist()

df2 = pd.read_csv('data/input_leetcode_test_v3.csv')
X_test = df2['Sentence(Processed)'].tolist()
Index = df2['ID']

X = ['None' if isinstance(item, float) and math.isnan(item) else item for item in X]
X_test = ['None' if isinstance(item, float) and math.isnan(item) else item for item in X_test]
y = ['None' if isinstance(item, float) and math.isnan(item) else item for item in y]

X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=1126)

# period = int(100) # 5-fold (total 500)
# eval_start = 400
# eval_end = eval_start + period

# X_train = X[:eval_start] + X[eval_end:]
# X_eval = X[eval_start:eval_end]

# y_train = y[:eval_start] + y[eval_end:]
# y_eval = y[eval_start:eval_end]


MAX_LENGTH = 512
X_train_tokenized = tokenizer(X_train, max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors="pt")
y_train_tokenized = tokenizer(y_train, max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors="pt")
X_eval_tokenized = tokenizer(X_eval, max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors="pt")
y_eval_tokenized = tokenizer(y_eval, max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors="pt")

X_test_tokenized = tokenizer(X_test, max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors="pt")

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.X["input_ids"][idx],
            "attention_mask": self.X["attention_mask"][idx],
            "labels": self.y["input_ids"][idx]
            }

train_dataset = MyDataset(X_train_tokenized, y_train_tokenized)
eval_dataset = MyDataset(X_eval_tokenized, y_eval_tokenized)
test_dataset = MyDataset(X_test_tokenized, X_test_tokenized)

training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=2,
    num_train_epochs=4.5,
    logging_dir='./logs',
    logging_steps=50,
    do_train=True,
    evaluation_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    push_to_hub=False,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

def predict_one_by_one(model, dataset):
    decoded_predictions = []
    model.eval()  # Set model to evaluation mode
    for item in dataset:
        input_ids = item["input_ids"].unsqueeze(0).to(device)  # Add a batch dimension
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)  # Add a batch dimension
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoded_predictions.append(decoded)
    return decoded_predictions

decoded_predictions = predict_one_by_one(model, eval_dataset)
for pred, actual in zip(decoded_predictions, y_eval):
    print("pred:  ", pred)
    print("actual:", actual)

test_decoded_predictions = predict_one_by_one(model, test_dataset)
with open('predictions.csv', 'w', newline='') as csvfile:
    fieldnames = ['ID', 'Prediction']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    
    for idx, pred in zip(Index, test_decoded_predictions):
        writer.writerow({'ID': idx, 'Prediction': pred})

