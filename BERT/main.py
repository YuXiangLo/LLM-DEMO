import pandas as pd
import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QADataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'start_positions': self.encodings['start_positions'][idx],
            'end_positions': self.encodings['end_positions'][idx]
        }

    def __len__(self):
        return len(self.encodings.input_ids)

def read_data(file_path):
    df = pd.read_csv(file_path)
    df.fillna('None', inplace=True)
    return df

def preprocess_data(df):
    questions = ["find the input in this problem" for _ in df.index]
    contexts = df['Sentence(Processed)'].tolist()
    answers = [{"text": ans, "answer_start": context.find(ans)} for ans, context in zip(df['Label(Main)'], contexts)]
    return contexts, questions, answers

def add_token_positions(encodings, answers, tokenizer):
    start_positions, end_positions = [], []
    valid_answers = []
    
    for i, answer in enumerate(answers):
        if answer['answer_start'] == -1:  # Answer not found in the context
            continue

        start_pos = encodings.char_to_token(i, answer['answer_start'])
        end_pos = encodings.char_to_token(i, answer['answer_start'] + len(answer['text']) - 1)

        # In case the answer is truncated or goes beyond the context
        if start_pos is None:
            start_pos = tokenizer.model_max_length
        if end_pos is None:
            end_pos = tokenizer.model_max_length

        start_positions.append(start_pos)
        end_positions.append(end_pos)
        valid_answers.append(i)

    # Filter out the invalid answers
    encodings['input_ids'] = [encodings['input_ids'][i] for i in valid_answers]
    encodings['attention_mask'] = [encodings['attention_mask'][i] for i in valid_answers]
    if 'offset_mapping' in encodings:
        encodings['offset_mapping'] = [encodings['offset_mapping'][i] for i in valid_answers]

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


def main():
    df = read_data('data/input_leetcode_train_v2.csv')
    contexts, questions, answers = preprocess_data(df)
    train_contexts, eval_contexts, train_questions, eval_questions, train_answers, eval_answers =\
        train_test_split(contexts, questions, answers, test_size=0.1, random_state=42)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    train_encodings =\
        tokenizer(train_contexts, train_questions, truncation=True, padding=True, max_length=256, return_tensors="pt", return_offsets_mapping=True)
    eval_encodings =\
        tokenizer(eval_contexts, eval_questions, truncation=True, padding=True, max_length=256, return_tensors="pt", return_offsets_mapping=True)

    add_token_positions(train_encodings, train_answers, tokenizer)
    add_token_positions(eval_encodings, eval_answers, tokenizer)

    model = BertForQuestionAnswering.from_pretrained("bert-large-uncased")
    model.to(device)

    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        num_train_epochs=5,
        do_train=True,
        evaluation_strategy="steps",
        push_to_hub=False,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        output_dir='./results',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=QADataset(train_encodings),
        eval_dataset=QADataset(eval_encodings),
    )

    trainer.train()

    model.save_pretrained('./my_qa_model')
    tokenizer.save_pretrained('./my_qa_model')

if __name__ == "__main__":
    main()

