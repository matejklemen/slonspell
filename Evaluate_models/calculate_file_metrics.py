from evaluate import read_file
from transformers import CamembertTokenizer


def calculate_error_rate(source_file_path, target_file_path):
    """
    function calculates percentage of errors in source file, compared to target file
    """
    source_text = read_file(source_file_path)
    target_text = read_file(target_file_path)

    num_words_with_mistakes = 0
    num_all_words = 0

    for s_word, t_word in zip(source_text.split(" "), target_text.split(" ")):
        if s_word != t_word:
            if t_word != "<mask>" and t_word != "mask":
                print(f"{s_word}\t{t_word}\"")

            num_words_with_mistakes += 1
        num_all_words += 1

    print(f"#mistakes: {num_words_with_mistakes}\n#all_words: {num_all_words}\n%-mistakes: {num_words_with_mistakes/num_all_words*100}")


def get_number_of_tokens(file_path, tokenizer_path):
    """
    function calculates number of tokens, when file (specified in file_path) is tokenized with specified tokenizer (tokenizer path)
    """
    tokenizer = CamembertTokenizer.from_pretrained(tokenizer_path)

    file_text = read_file(file_path)

    tokens = tokenizer.tokenize(file_text)

    print(f"Num of tokens: {len(tokens)}")


if __name__ == "__main__":
    # calculate_error_rate("SloSpell/solar/solar_target_napake_č.txt", "SloSpell/solar/solar_target.txt")

    get_number_of_tokens("../prepared_data_train_models/data_sloBERTa_masks_model.txt", "../sloBERTaModel")

