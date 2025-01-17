
import pandas as pd
from datasets import Dataset
from transformers import CamembertTokenizer, CamembertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

train_df = pd.read_csv('D:/pythonProject/sii/train.csv')
test_df = pd.read_csv('D:/pythonProject/sii/test.csv')

def clean_text(text):
    text = ' '.join(text.split())
    text = text.lower()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

train_df['Text'] = train_df['Text'].apply(clean_text)
test_df['Text'] = test_df['Text'].apply(clean_text)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['Text'], train_df['Label'], test_size=0.2, random_state=42, stratify=train_df['Label']
)

label_map = {'fake': 0, 'biased': 1, 'true': 2}
train_labels = train_labels.map(label_map)
val_labels = val_labels.map(label_map)

train_dataset = Dataset.from_pandas(pd.DataFrame({'text': train_texts, 'label': train_labels}))
val_dataset = Dataset.from_pandas(pd.DataFrame({'text': val_texts, 'label': val_labels}))

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=3)
model = model.to(device)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    learning_rate=3e-5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)

    predictions = torch.argmax(logits, dim=-1)

    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)

    predictions = predictions.cpu()
    labels = labels.cpu()

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

eval_results = trainer.evaluate(val_dataset)
print(f"Final evaluation results: {eval_results}")

test_dataset = Dataset.from_pandas(pd.DataFrame({'text': test_df['Text']}))
test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

predictions = trainer.predict(test_dataset)
predicted_labels = torch.argmax(torch.tensor(predictions.predictions), dim=-1).numpy()

reverse_label_map = {v: k for k, v in label_map.items()}
test_df['Label'] = [reverse_label_map[label] for label in predicted_labels]

test_df.to_csv('D:/pythonProject/sii/predictions.csv', index=False)