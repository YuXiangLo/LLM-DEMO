import pandas as pd
import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast
from torch.utils.data import DataLoader

def read_data(file_path):
    df = pd.read_csv(file_path)
    df.fillna('None', inplace=True)
    return df

def preprocess_data(df):
    questions = ["find the input in this problem" for _ in df.index]
    contexts = df['Sentence(Processed)'].tolist()
    IDs = df['ID'].tolist()
    return contexts, questions, IDs

def predict(model, tokenizer, contexts, questions):
    predictions = []
    for context, question in zip(contexts, questions):
        inputs = tokenizer(context, question, truncation=True, padding=True, max_length=256, return_tensors="pt")

        # Move inputs to GPU
        for key in inputs:
            inputs[key] = inputs[key].to(device)

        outputs = model(**inputs)
        start_positions = outputs.start_logits.argmax(dim=-1).item()  # .item() extracts the scalar value from the tensor
        end_positions = outputs.end_logits.argmax(dim=-1).item()

        pred_answer = tokenizer.decode(inputs['input_ids'][0][start_positions:end_positions+1])  # [0] to take the first item since we're processing one-by-one
        predictions.append(pred_answer)

    return predictions


def main():
    # Checking for GPU availability
    global device
    device = torch.device("cuda")

    df = read_data('data/input_leetcode_test_v2.csv')
    contexts, questions, IDs = preprocess_data(df)

    tokenizer = BertTokenizerFast.from_pretrained('./my_qa_model')
    model = BertForQuestionAnswering.from_pretrained('./my_qa_model')
    model.eval()  # Set the model to evaluation mode
    
    # Load model onto the GPU
    model.to(device)

    predictions = predict(model, tokenizer, contexts, questions)

    print('ID,Prediction')
    for context, question, prediction, ID in zip(contexts, questions, predictions, IDs):
        print(f"{ID},\"{prediction}\"")

if __name__ == "__main__":
    main()

