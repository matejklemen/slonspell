import os
import time
import random
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from transformers import CamembertTokenizer, CamembertForMaskedLM, AdamW, get_linear_schedule_with_warmup

# Configuration
device = torch.device("cpu")
batch_size = 8
epochs = 2
output_dir = 'fine_tuned_models/sloberta_masks/'
seed_val = 42


def load_dataset(filepath):
    """Loads the dataset from a given filepath and returns sentences and labels."""
    df = pd.read_csv(filepath, delimiter='\t', header=None, names=['sentence', 'label'])
    print(f'Number of training sentences: {df.shape[0]:,}')
    return df.sentence.values, df.label.values


def initialize_tokenizer(model_path):
    """Initializes the Camembert tokenizer."""
    return CamembertTokenizer.from_pretrained(model_path)


def isNaN(x):
    """Checks if a value is NaN."""
    return str(x) == str(1e400 * 0)


def tokenize_sentences(sentences, labels, tokenizer):
    """Tokenizes sentences and processes the labels."""
    input_ids, attention_masks, label_list = [], [], []
    skipped_sentences = 0

    for idx_sentence, sent in enumerate(sentences):
        if idx_sentence % 10000 == 0:
            print(f"Progress: {idx_sentence / len(sentences):.2%}")

        if isNaN(sent):
            skipped_sentences += 1
            continue

        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=False,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        try:
            cur_labels = labels[idx_sentence].split(' ')
            labels_in = []
            masked_idx_pointer = 0

            for token in encoded_dict['input_ids'][0].numpy():
                if token == 32004:  # <mask> token
                    labels_in.append(int(cur_labels[masked_idx_pointer]))
                    masked_idx_pointer += 1
                else:
                    labels_in.append(-100)

            assert len(encoded_dict['input_ids'][0]) == len(labels_in)

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            label_list.append(torch.tensor([labels_in]))

        except Exception as e:
            print(f"Error processing sentence {idx_sentence}: {e}")
            skipped_sentences += 1

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_list,
                                                                                     dim=0), skipped_sentences


def create_dataloaders(dataset, train_size, batch_size):
    """Splits the dataset into training, validation, and test sets and returns corresponding dataloaders."""
    val_size = test_size = (len(dataset) - train_size) // 2

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

    return train_dataloader, validation_dataloader, test_dataloader


# Model Initialization
def initialize_model(model_path, device):
    """Initializes the Camembert model."""
    model = CamembertForMaskedLM.from_pretrained(model_path)
    return model.to(device)


def initialize_optimizer(model, learning_rate=2e-5, eps=1e-8):
    """Initializes the optimizer."""
    return AdamW(model.parameters(), lr=learning_rate, eps=eps)


def initialize_scheduler(optimizer, train_dataloader, epochs):
    """Initializes the learning rate scheduler."""
    total_steps = len(train_dataloader) * epochs
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# Utility Functions
def flat_accuracy(preds, labels):
    """Calculates the accuracy of predictions."""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """Formats elapsed time into a string hh:mm:ss."""
    return str(datetime.timedelta(seconds=int(round(elapsed))))


def set_seed(seed_val):
    """Sets the seed for reproducibility."""
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)


def save_model(model, tokenizer, output_dir):
    """Saves the model and tokenizer."""
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# Training Loop
def train(model, train_dataloader, optimizer, scheduler):
    """Performs training over the dataset."""
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and step != 0:
            elapsed = format_time(time.time() - t0)
            print(f"  Batch {step:>5,} of {len(train_dataloader):>5,}. Elapsed: {elapsed}.")

        b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]

        model.zero_grad()
        loss, logits = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)[:2]

        total_train_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return total_train_loss / len(train_dataloader)


def validate(model, validation_dataloader):
    """Performs validation and calculates accuracy and loss."""
    model.eval()
    total_eval_loss = 0
    total_eval_accuracy = 0

    for batch in validation_dataloader:
        b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]

        with torch.no_grad():
            logits = model(b_input_ids, attention_mask=b_input_mask).logits

        loss = torch.nn.CrossEntropyLoss()(logits.view(-1, model.config.vocab_size), b_labels.view(-1))
        total_eval_loss += loss.item()

        total_eval_accuracy += flat_accuracy(logits.detach().cpu().numpy(), b_labels.to('cpu').numpy())

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    return avg_val_accuracy, avg_val_loss


def train_and_evaluate(model, train_dataloader, validation_dataloader, optimizer, scheduler, epochs, output_dir):
    """Trains and evaluates the model, saving the best version."""
    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(epochs):
        print(f"\n======== Epoch {epoch_i + 1} / {epochs} ========")
        print('Training...')

        t0 = time.time()
        avg_train_loss = train(model, train_dataloader, optimizer, scheduler)
        training_time = format_time(time.time() - t0)

        print(f"\n  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training epoch took: {training_time}")

        save_model(model, tokenizer, output_dir)

        print("\nRunning Validation...")

        t0 = time.time()
        avg_val_accuracy, avg_val_loss = validate(model, validation_dataloader)
        validation_time = format_time(time.time() - t0)

        print(f"  Accuracy: {avg_val_accuracy:.2f}")
        print(f"  Validation Loss: {avg_val_loss:.2f}")
        print(f"  Validation took: {validation_time}")

        training_stats.append({
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        })

    print("\nTraining complete!")
    print(f"Total training took {format_time(time.time() - total_t0)} (h:mm:ss)")
    save_model(model, tokenizer, output_dir)


if __name__ == "__main__":
    set_seed(seed_val)

    sentences, labels = load_dataset('train_data/data_SloBERTa_masks_model.txt')
    tokenizer = initialize_tokenizer('sloBERTaModel')

    input_ids, attention_masks, labels_torch, skipped_sentences = tokenize_sentences(sentences, labels, tokenizer)
    print(f"Number of sentences skipped: {skipped_sentences}")

    dataset = TensorDataset(input_ids, attention_masks, labels_torch)
    # This should be adjusted accordingly
    train_size = len(dataset) - 2
    train_dataloader, validation_dataloader, test_dataloader = create_dataloaders(dataset, train_size, batch_size)

    model = initialize_model('sloBERTaModel', device)
    optimizer = initialize_optimizer(model)
    scheduler = initialize_scheduler(optimizer, train_dataloader, epochs)

    train_and_evaluate(model, train_dataloader, validation_dataloader, optimizer, scheduler, epochs, output_dir)
