import os
import csv
import re
import random
from transformers import CamembertTokenizer
import make_text_incorrect_functions

# Constants
FINAL_PUNCTUATION = ['!', '.', '?', ':', ';']

# Global counters
global_counter_masks = 0
global_counter_concats = 0
global_counter_splits = 0
write_row_counter = 0

# Configuration flags
print_counters = False

# Initialize tokenizer
tokenizer = CamembertTokenizer.from_pretrained("../sloBERTaModel")

# Output file handler
out_file = None


def insert_spaces_before_quote(match):
    """
    Inserts a space before a quote if it's not preceded by punctuation or a space.
    """
    if match.group(0)[0] not in FINAL_PUNCTUATION and match.group(0)[0] != " ":
        return match.group(0)[0] + " " + match.group(0)[-1]
    return match.group(0)


def preprocess_line(line):
    """
    Cleans and preprocesses the line by removing unwanted characters and formatting it properly.
    """
    line = line.strip()
    line = re.sub(r"^(\.* *)* *", "", line)
    line = re.sub(r"(\. *)+ *$", ".", line)
    line = re.sub(r'(,* *\.+ *)+', ". ", line)
    line = re.sub(r'\s+', " ", line)
    line = re.sub(r'.["\'»«”“`]', insert_spaces_before_quote, line)

    line = make_text_incorrect_functions.add_spaces_before_and_after_punctuation(line)
    line = make_text_incorrect_functions.remove_double_spaces(line)

    return line


def process_line_for_model(line, use_masks=True, use_split=True, use_concat=True, mask_token="<mask>",
                           split_token="<splited_word>", concat_token="<concated_word>", start_token="<s>",
                           end_token="</s>"):
    """
    Processes the line for the SloBERTa model, applying optional transformations for masks, splits, and concatenations.
    """
    global write_row_counter, global_counter_splits, global_counter_concats, global_counter_masks

    line = preprocess_line(line)
    original_line = line

    if use_split:
        line = make_text_incorrect_functions.split_random_words(line, 0.08)
        if print_counters:
            global_counter_splits += line.count(split_token)

    if use_concat:
        line = make_text_incorrect_functions.concat_random_words(line, 0.38)
        if print_counters:
            global_counter_concats += line.count(concat_token)

    if use_masks:
        line = make_text_incorrect_functions.zamenjaj_besede_nagajivke(line, 0.01)
        line = make_text_incorrect_functions.replace_some_predefined_characters(line, 0.6, 8)
        line = make_text_incorrect_functions.replace_some_ž_š_č_characters(line, 0.05)
        line = make_text_incorrect_functions.replace_random_chars_in_text_2(line)

    incorrect_line, original_line = make_text_incorrect_functions.mark_false_words(line, original_line, mask_token,
                                                                                   split_token, concat_token)
    if print_counters:
        global_counter_masks += original_line.count(mask_token)

    original_line = make_text_incorrect_functions.make_T5_row(original_line, mask_token)
    bert_model_line = make_text_incorrect_functions.make_false_words_mark_sloBERTa_model(
        incorrect_line, original_line, mask_token, split_token, concat_token, start_token=start_token,
        end_token=end_token
    )

    if incorrect_line:
        write_row_counter += 1
        out_file.write(bert_model_line + '\n')


def process_text_files():
    """
    Processes all text files in the data folder and writes processed lines to the output file.
    """
    global out_file, write_row_counter

    with open("../train_data/data_SloBERTa_masks_model.txt", "w", newline="") as out_file:
        os.chdir("../data_folders/ucne_mnozice")

        text_files = [f for f in os.listdir() if f.endswith(".txt")]

        for file_counter, file_name in enumerate(text_files, start=1):
            with open(file_name, "r") as file:
                print(f"Processing file {file_name}, file number {file_counter}/{len(text_files)}")
                if print_counters:
                    print(
                        f"Splits: {global_counter_splits}, Concats: {global_counter_concats}, Masks: {global_counter_masks}")

                cur_sentence_group = ""
                cur_sentence_group_length = random.randint(30, 128)
                line_counter = 0

                for line in file:
                    line_counter += 1
                    for sentence in line.split("."):
                        sentence += "."

                        if len(sentence.split()) > 3:
                            if len(tokenizer.encode(cur_sentence_group + sentence)) < cur_sentence_group_length:
                                cur_sentence_group += " " + sentence
                                cur_sentence_group_length = random.randint(10, 128)
                            else:
                                process_line_for_model(cur_sentence_group, use_masks=True, use_split=False,
                                                       use_concat=False)
                                cur_sentence_group = sentence
                                break


if __name__ == "__main__":
    process_text_files()
