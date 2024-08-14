import os
import csv
import re
import support_functions
import make_text_incorrect_functions
import random

from transformers import T5ForConditionalGeneration, T5Tokenizer

# tokenizer = T5Tokenizer.from_pretrained("./masks_model_nov_3/")
write_row_counter = 0
final_punct = ['!', '.', '?', ':', ';']
num_all_files = 93610420
num_to_sample_out = 25000000

global_counter_masks = 0
global_counter_concats = 0
global_counter_splits = 0
print_counters = False

out_file = None

def insert_spaces_before_quote(line):
    if not line.group(0)[0] in final_punct and line.group(0)[0] != " ":
        return line.group(0)[0] + " " + line.group(0)[-1]
    return line.group(0)

def preprocess_line(line):
    line = line.strip()
    
    line = re.sub("^(\.* *)* *", "", line)
    line = re.sub("(\. *)+ *$", ".", line)
    
    line = re.sub('(,* *\.+ *)+', ". ", line)
    
    line = re.sub(r'\s+', " ", line)
    
    line = re.sub('.["\'»«”“`]', insert_spaces_before_quote, line)
    
    # line = re.sub(r'(\d,+\d)+', "909873828123", line)
    # line = re.sub(r'(\d\.+\d)+', "908873928124", line)
    
    line = support_functions.add_spaces_before_and_after_punctuation(line)
    line = support_functions.remove_double_spaces(line)
    
    return line


def write_down_line_masks_model(line, use_masks=True, use_split=True, use_concat=True, mask_token="<mask>", split_token="<splited_word>", concat_token="<concated_word>", start_token="<s>", end_token="</s>"):
    global write_row_counter, global_counter_splits, global_counter_concats, global_counter_masks, print_counters, out_file
    
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
    
    #line = make_text_incorrect_functions.replace_word_with_contextual_wrong_word(line, 0.02)
    
    incorrect_line, original_line = make_text_incorrect_functions.mark_false_words(line, original_line, mask_token, split_token, concat_token)
    
    if print_counters:
        global_counter_masks += original_line.count(mask_token)
    
    original_line = make_text_incorrect_functions.make_T5_row(original_line, mask_token)
    
    bert_model_line = make_text_incorrect_functions.make_false_words_mark_sloBERTa_model(incorrect_line, original_line, mask_token, split_token, concat_token, start_token=start_token, end_token=end_token)
    
    if not (incorrect_line == None or incorrect_line == ""):
        write_row_counter += 1
    
        out_file.write(bert_model_line + '\n')


def write_down_lines_EngBERT_model(output_file_path, num_lines=-1):
    """
    function prepares dataset for english BERT
    :param num_lines: number of lines to include in dataset, -1 for all lines
    """
    global out_file

    with open(output_file_path, "w", newline="") as out_file:
        with open("english_corpora/download/collected_texts.txt", "r", newline="") as in_file:
            file_in_text = in_file.read()

            for line_inx, line in enumerate(file_in_text.split("\n")):
                write_down_line_masks_model(line, use_masks=True, use_split=False, use_concat=False, mask_token="[MASK]", start_token="[CLS]", end_token="[SEP]")

                if num_lines != -1 and line_inx > num_lines:
                    break


def split_file(input_file, train_file, eval_file, eval_ratio=0.05):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    random.shuffle(lines)

    eval_size = int(len(lines) * eval_ratio)

    eval_lines = lines[:eval_size]
    train_lines = lines[eval_size:]

    with open(train_file, 'w') as trainfile:
        trainfile.writelines(train_lines)

    with open(eval_file, 'w') as evalfile:
        evalfile.writelines(eval_lines)


def write_down_lines_SloBERTa_model():
    with open("data_BERT_masks_model.txt", "w", newline="") as out_file:

        os.chdir("data_folders/oct_30_ucne_mnozice")

        text_files = [f for f in os.listdir() if f.endswith(".txt")]
        file_counter = 0;
        write_row_counter = 0;
        line_counter = 0

        lines_to_pic_dict = {}
        lines_to_pick = random.sample(range(0, num_all_files), num_to_sample_out)
        for line_num in lines_to_pick:
            lines_to_pic_dict[line_num] = 1

        assert len(lines_to_pick) == num_to_sample_out

        for file in text_files:
            file_counter += 1;

            with open(file, "r") as file:
                print("file is: ", file, "write_row_counter is: ", write_row_counter)
                print("working on file num: ", file_counter, " of: ", len(text_files))

                if print_counters:
                    print("global_counter_splits: ", global_counter_splits)
                    print("global_counter_concats: ", global_counter_concats)
                    print("global_counter_masks: ", global_counter_masks)

                cur_sentence_group = ""
                cur_sentence_group_length = random.randint(30, 128)

                # Iterate through each line in the file
                for line in file:
                    line_counter += 1

                    if line_counter in lines_to_pic_dict:
                        #print("line_counter in: ", line_counter)
                        for sentence in line.split("."):
                            sentence = sentence + "."

                            if len(sentence.split(" ")) > 3:

                                if len(tokenizer.encode(cur_sentence_group + sentence)) < cur_sentence_group_length:
                                    cur_sentence_group += " " + sentence
                                    cur_sentence_group_length = random.randint(10, 128)
                                else:
                                    write_down_line_masks_model(cur_sentence_group, use_masks=True, use_split=False, use_concat=False)
                                    #write_down_line_word_shape_model(cur_line_group)
                                    #write_down_line_wo_model(cur_line_group)

                                    cur_sentence_group = sentence
                                    break

    
if __name__ == "__main__":
    write_down_lines_EngBERT_model("data_EngBERT_masks_model.txt", -1)
    split_file("data_EngBERT_masks_model.txt", "./train_eng_bert_model/data_EngBERT_masks_model_train.txt", "./train_eng_bert_model/data_EngBERT_masks_model_eval.txt", eval_ratio=0.05)
    # split_file("./train_eng_bert_model/EngBERT_clang8_dataset_train.txt", "./train_eng_bert_model/EngBERT_clang8_dataset_train.txt", "./train_eng_bert_model/EngBERT_clang8_dataset_eval.txt", 0.005)
    