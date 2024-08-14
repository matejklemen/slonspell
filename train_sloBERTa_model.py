import pandas as pd
from transformers import CamembertTokenizer
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertForMaskedLM, AdamW, BertConfig, CamembertForMaskedLM
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import random
import numpy as np
import os, time
import random
import numpy as np
import os, time

from torch.utils.data import TensorDataset, random_split

device = torch.device("mps")  # if torch.cuda.is_available() else "cpu"

df = []

# Load the dataset into a pandas dataframe.
li = pd.read_csv('../fine_tune_chatgpt/prepare_bert_data/train_SloBERTa_o_b_z.tsv', delimiter='\t', header=None,
                 names=['sentence', 'label'])
df.append(li)

df = pd.concat(df, axis=0, ignore_index=True)
#
# df.drop(df.tail(2).index,inplace=True)
# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))

# Display 10 random rows from the data.
df.sample(100)

# SLO
# tokenizer = BertTokenizer.from_pretrained('drive/My Drive/ucenje_BERT/pretrained_slo_bert/', do_lower_case=True)

# just SLO
tokenizer = CamembertTokenizer.from_pretrained('../sloBERTaModel')
# tokenizer = CamembertTokenizer.from_pretrained('drive/My Drive/ucenje_BERT/bert_just_slo_maj_2021_model/')

# drive/My Drive/ucenje_BERT/bert_just_slo_maj_2021_model/

# MULTILINGUAL
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)

print(tokenizer)

sentences = df.sentence.values
labels = df.label.values


# Tokenize all of the sentences and map the tokens to thier word IDs.
def isNaN(x):
    return str(x) == str(1e400 * 0)


input_ids = []
attention_masks = []
label_list = []
stevec = 0
stevec_izpuscenih = 0
print(len(sentences))
# For every sentence...
for idx_sentence in range(len(sentences)):
    # if stevec > 1000:
    # break

    sent = sentences[idx_sentence]

    if (stevec % 10000 == 0):
        print("progress: ", stevec / len(sentences))

    if isNaN(sent):
        stevec_izpuscenih += 1
        continue

    stevec += 1

    encoded_dict = tokenizer.encode_plus(
        sent,  # Sentence to encode.
        add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
        max_length=512,  # Pad & truncate all sentences. #200
        # pad_to_max_length = True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    try:

        labels_in = []
        cur_labels = labels[idx_sentence].split(' ')
        # print(cur_labels)

        # if idx_sentence == 400:
        #  print(sent)
        #  print(labels[idx_sentence])

        maskedindexesIndexPointer = 0

        # print(encoded_dict['input_ids'][0].numpy())

        for token in encoded_dict['input_ids'][0].numpy():
            if token == 32004:  # <mask> token -> 32004
                labels_in.append(int((cur_labels[maskedindexesIndexPointer])))
                maskedindexesIndexPointer += 1
            else:
                labels_in.append(-100)

        # print(len(encoded_dict['input_ids'][0].numpy()))
        # print(len(labels_in))

        assert len(encoded_dict['input_ids'][0].numpy()) == len(labels_in)

        # assert maskedindexesIndexPointer == len(cur_labels) - 1
        # if maskedindexesIndexPointer != len(cur_labels) - 1:
        #  print(idx_sentence)
        # print(encoded_dict['input_ids'])

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

        label_list.append(torch.tensor([labels_in]))

    except:
        print(sent)
        print(cur_labels)
        print(maskedindexesIndexPointer)
        print(len(cur_labels) - 1)
        stevec_izpuscenih += 1

    # break

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
# labels = torch.tensor(labels)
labels_torch = torch.cat(label_list, dim=0)

# Print sentence 0, now as a list of IDs.
print('Original: ', tokenizer.tokenize(sentences[0]))
print('Token IDs:', input_ids[0])
print('Labels:', labels_torch[0])
print('Original: ', tokenizer.tokenize(sentences[4]))
print('Token IDs:', input_ids[4])
print('Labels:', labels_torch[4])
print('Original: ', tokenizer.tokenize(sentences[152]))
print('Token IDs:', input_ids[152])
print('Labels:', labels_torch[152])
print('Original: ', tokenizer.tokenize(sentences[400]))
print('Token IDs:', input_ids[400])
print('Labels:', labels_torch[400])

print("st izpuscenih")
print(stevec_izpuscenih)

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels_torch)

# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = len(dataset) - 100  # int(0.864 * len(dataset))
val_and_test_size = len(dataset) - train_size

val_size = int(0.5 * val_and_test_size)
test_size = int(0.5 * val_and_test_size)

print(train_size)
print(val_size)
print(test_size)

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 8

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
    train_dataset,  # The training samples.
    sampler=RandomSampler(train_dataset),  # Select batches randomly
    batch_size=batch_size  # Trains with this batch size.
)

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
    val_dataset,  # The validation samples.
    sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
    batch_size=batch_size  # Evaluate with this batch size.
)

# For test the order doesn't matter, so we'll just read them sequentially.
test_dataloader = DataLoader(
    test_dataset,  # The test samples.
    sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially.
    batch_size=batch_size  # Evaluate with this batch size.
)

# slo
model = CamembertForMaskedLM.from_pretrained(
    '../sloBERTaModel'
)

model.to(device)

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # args.learning_rate - default is 5e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                  )

# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
# epochs = 4
epochs = 2

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)

print(scheduler)
print(scheduler.get_last_lr())


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

output_dir = '../fine_tuned_models/sloberta_masks_jan_29/'

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

total_stevec = 0
stevci = []
learningRates = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    # model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # print(1)
        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # print(len(b_input_ids))
        # print(len(b_labels))
        # print(len(b_input_mask))
        # print(2)
        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        # model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        # print(b_input_ids.size())
        # print(b_input_mask.size())
        # print(b_labels.size())
        # break
        """loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)"""

        returned_values_from_model = model(b_input_ids,
                                           token_type_ids=None,
                                           attention_mask=b_input_mask,
                                           labels=b_labels)

        loss = returned_values_from_model.loss
        logits = returned_values_from_model.logits

        # print(3)
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()
        # print(4)
        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # print(5)

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()
        total_stevec += 1

        if total_stevec % 1000 == 0:
            learningRates.append(scheduler.get_last_lr())
            stevci.append(total_stevec)

            print(scheduler.get_last_lr())

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    stevec = 0

    num_right_predictions = 0
    num_all_predictions = 0

    num_right_mask_predictions = 0
    num_all_masks = 0
    num_right_split_predictions = 0
    num_all_split = 0
    num_right_concat_predictions = 0
    num_all_concats = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        batch_numpy = tuple(t.numpy() for t in batch)
        b_input_numpy_ids, b_input_numpy_mask, b_numpy_labels = batch_numpy

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.

            # print(b_input_ids)
            # print(b_input_mask)

            returned_values_from_model2 = model(b_input_ids,
                                                token_type_ids=None,
                                                attention_mask=b_input_mask)

            logits = returned_values_from_model2.logits

            logits2 = logits[0]
            for batch_index in range(0, len(b_input_numpy_ids)):
                # print(batch_index)
                for i in range(0, len(b_input_numpy_ids[batch_index])):
                    # print(torch.argmax(logits2[0, i]).item())
                    if b_input_numpy_ids[batch_index][i] == 32004:
                        if b_numpy_labels[batch_index][i] == torch.argmax(logits2[batch_index, i]).item():
                            num_right_predictions += 1
                        num_all_predictions += 1

                        if b_numpy_labels[batch_index][i] == 1:
                            if torch.argmax(logits2[batch_index, i]).item() == 1:
                                num_right_mask_predictions += 1
                            num_all_masks += 1

                        if b_numpy_labels[batch_index][i] == 2:
                            if torch.argmax(logits2[batch_index, i]).item() == 2:
                                num_right_split_predictions += 1
                            num_all_split += 1

                        if b_numpy_labels[batch_index][i] == 3:
                            if torch.argmax(logits2[batch_index, i]).item() == 3:
                                num_right_concat_predictions += 1
                            num_all_concats += 1

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits[0].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

    print('    DONE.')
    print('Accuracy: ' + str((num_right_predictions / num_all_predictions) * 100) + '%')
    print('Accuracy, masks: ' + str((num_right_mask_predictions / num_all_masks) * 100) + '%')
    print('Accuracy, splits: ' + str((num_right_split_predictions / num_all_split) * 100) + '%')
    print('Accuracy, concats: ' + str((num_right_concat_predictions / num_all_concats) * 100) + '%')
    print(num_right_predictions)
    print(num_all_predictions)
    print(num_right_mask_predictions)
    print(num_right_split_predictions)
    print(num_right_concat_predictions)
    print(num_all_masks)

print("")
print("Training complete!")

print(stevci)
print(learningRates)

print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))
