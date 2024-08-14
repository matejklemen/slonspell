from openai import OpenAI
import re
from classla import Pipeline

# Initialize the pipeline
nlp = Pipeline('sl', processors='tokenize,ner')

api_key = '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'

client = OpenAI(api_key=api_key)
model_to_use = ''  # 'gpt-3.5-turbo-0125' 'gpt-4-0125-preview' 'gpt-3.5-turbo-1106' 'gpt-4-1106-preview'
row_name_to_add = ''


def use_chatgpt_3_5():
    """
    function sets global variables for ChatGPT 3.5 turbo
    :return:
    """
    global model_to_use, row_name_to_add

    model_to_use = 'gpt-3.5-turbo-0125'
    row_name_to_add = 'ChatGPT 3.5 improved razlaga'


def use_chatgpt_4():
    """
    function sets global variables for ChatGPT 4 turbo
    :return:
    """
    global model_to_use, row_name_to_add

    model_to_use = 'gpt-4-0125-preview'
    row_name_to_add = 'ChatGPT 4 improved razlaga'


def query_chatgpt(incorrect_sentence):
    """
    function creates query for chatGPT and executes it
    :param incorrect_sentence
    :return: chatGPT response
    """
    global client

    response = client.chat.completions.create(
        model=model_to_use,
        messages=[
            {"role": "system",
             "content": "You are an expert Slovene proofreader. Do not explain your answers. Mark incorrect words in provided sentence. Do not shorten or lengthen the input sentence."},
            {"role": "user",
             "content": "Je pa tuti res , da je ta razred š vedno pod povprečjem v Evropi in že za radi tega se mi zdi , da je še mogoče nati kupce zate vrste avtomobilov ."},
            {"role": "assistant",
             "content": "Je pa <napaka> res , da je ta razred <napaka> vedno pod povprečjem v Evropi in že <split> <split> tega se mi zdi , da je še mogoče <napaka> kupce <concat> vrste avtomobilov ."},
            {"role": "user",
             "content": f"{incorrect_sentence}"},
        ],

        # mogoče še zmanjšaj temperaturo

        # timeout=10
    )

    return response.choices[0].message.content


def post_process_line(line):
    """
    function converts all spli and concat tokens to mask tokens
    """
    line = re.sub("<split>", "<mask>", line)
    line = re.sub("<concat>", "<mask>", line)
    line = re.sub("<napaka>", "<mask>", line)

    return line


def print_out_result(TP, FP, TN, FN, include_metrics=False):
    """
    function prints TP, FP, TN and FN
    """
    print(f"\nTP: {TP}\nTN: {TN}\nFP: {FP}\nFN: {FN}")

    if include_metrics:
        # Calculations for positive class
        accuracy_pos = TP / (TP + FP)
        recall_pos = TP / (TP + FN)

        # Calculations for negative class
        accuracy_neg = TN / (TN + FN)
        recall_neg = TN / (TN + FP)

        # F1 Score for positive and negative class
        F1_pos = 2 * TP / (2 * TP + FP + FN)
        F1_neg = 2 * TN / (2 * TN + FN + FP)

        # F0.5 Score (weighing precision more than recall)
        F05_pos = (1 + 0.5 ** 2) * TP / ((1 + 0.5 ** 2) * TP + 0.5 ** 2 * FN + FP)
        F05_neg = (1 + 0.5 ** 2) * TN / ((1 + 0.5 ** 2) * TN + 0.5 ** 2 * FP + FN)

        # Combined F1 and F0.5 Score (assuming balanced classes)
        F1_combined = (F1_pos + F1_neg) / 2
        F05_combined = (F05_pos + F05_neg) / 2

        print("accuracy_pos:", accuracy_pos)
        print("recall_pos:", recall_pos)
        print("Accuracy neg: " + str(accuracy_neg))
        print("Recall neg: " + str(recall_neg))
        print("F1 pos: " + str(F1_pos))
        print("F1 neg: " + str(F1_neg))
        print("F1 combined: " + str(F1_combined))
        print("F05_pos: " + str(F05_pos))
        print("F05_neg: " + str(F05_neg))
        print("F05_combined: " + str(F05_combined))


def read_file(file_path):
    """
    function reads file and returns text
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def align_file(file_path, new_file_path):
    """
    function introduces spaces before every punctuation
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    punct_with_no_space = re.compile(r'(?<! )(\.|,|;|:|\?|!|=|/|\'|_|\)|…|\“|\”)')
    aligned_content = punct_with_no_space.sub(r' \1 ', content)

    content_with_single_spaces = re.sub(r' {2,}', ' ', aligned_content)
    content_with_single_spaces = re.sub("NAPAKA / Č / SKUPAJ _ NARAZEN / ", "NAPAKA/Č/SKUPAJ_NARAZEN/", content_with_single_spaces)
    content_with_single_spaces = re.sub("NAPAKA / Č / TUJA _ BESEDA / ", "NAPAKA/Č/TUJA_BESEDA/", content_with_single_spaces)
    content_with_single_spaces = re.sub("NAPAKA / Č / NESTANDARDNO / ", "NAPAKA/Č/NESTANDARDNO/", content_with_single_spaces)
    content_with_single_spaces = re.sub("NAPAKA / Č / FORMAT / ", "NAPAKA/Č/FORMAT/", content_with_single_spaces)
    content_with_single_spaces = re.sub("NAPAKA / Č / NESIGUREN / ", "NAPAKA/Č/NESIGUREN/", content_with_single_spaces)
    content_with_single_spaces = re.sub("NAPAKA / Č / ", "NAPAKA/Č/", content_with_single_spaces)
    content_with_single_spaces = re.sub(r"(\. )+\.", ".", content_with_single_spaces)
    content_with_single_spaces = re.sub(r"(\? )+\?", "?", content_with_single_spaces)
    content_with_single_spaces = re.sub(r"(\, )+\,", ",", content_with_single_spaces)
    content_with_single_spaces = re.sub(r"(\" )+\"", "\"", content_with_single_spaces)
    content_with_single_spaces = re.sub(r"(: )+:", ":", content_with_single_spaces)
    content_with_single_spaces = re.sub("<concated _ word>", "<concated_word>", content_with_single_spaces)
    content_with_single_spaces = re.sub("<split _ word>", "<split_word>", content_with_single_spaces)
    content_with_single_spaces = re.sub(r"“ ", "“", content_with_single_spaces)
    content_with_single_spaces = re.sub(r" ”", "”", content_with_single_spaces)

    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(content_with_single_spaces)


def reformat_chatgpt_annotated_file(source_file_path, corrected_file_path, new_corrected_file_path):
    """
    function reformats file annotated with chatGPT.
    it completes half filled corrected sentences, with sentence from source..
    if corrected word in corrected sentence includes error label it marks the whole word as error
    """
    source_text = read_file(source_file_path)
    corrected_text = read_file(corrected_file_path)

    with open(new_corrected_file_path, "w", encoding="utf-8") as new_corrected_file:
        for s_line, c_line in zip(source_text.split("\n"), corrected_text.split("\n")):
            s_line = s_line.strip()
            c_line = c_line.strip()

            new_c_line_split = []

            for c_word in c_line.split(" "):
                if "<split>" in c_word:
                    new_c_line_split.append("<split>")
                elif "<concat>" in c_word:
                    new_c_line_split.append("<concat>")
                elif "<napaka>" in c_word:
                    new_c_line_split.append("<napaka>")
                else:
                    new_c_line_split.append(c_word)

            c_line_split = new_c_line_split
            s_line_split = s_line.split(" ")

            for c_word_idx, s_word in enumerate(s_line_split):
                if c_word_idx >= len(c_line_split):
                    new_c_line_split.append(s_word)

            new_corrected_file.write(" ".join(new_c_line_split) + "\n")


def calculate_line_score(s_line, t_line, c_line, ner_words=[]):
    """
    function calculates TP, FP... for provided lines
    """
    TN = 0
    FP = 0
    TP = 0
    FN = 0

    for s_token, t_token, c_token in zip(s_line.split(" "), t_line.split(" "), c_line.split(" ")):
        if s_token == "IGNORE" or s_token in ner_words:
            continue
        if s_token == t_token and c_token != "<mask>":
            TN += 1
        elif s_token == t_token and c_token == "<mask>":
            FP += 1
        elif s_token != t_token and c_token == "<mask>":
            TP += 1
        else:
            FN += 1

    return TP, TN, FP, FN


def evaluate_using_chatGPT(source_file_path, target_file_path, corrected_file_path):
    """
    function evaluates chatGPT model based on source file and target file
    """
    source_text = read_file(source_file_path)
    target_text = read_file(target_file_path)

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    num_all_lines = len(source_text.split("\n"))

    with open(corrected_file_path, "w", encoding="utf-8") as corr_file:
        for line_idx, (s_line, t_line) in enumerate(zip(source_text.split("\n"), target_text.split("\n"))):
            c_line = query_chatgpt(s_line)

            corr_file.write(f'{c_line}\n')

            c_line = post_process_line(c_line)

            TP_out, TN_out, FP_out, FN_out = calculate_line_score(s_line, t_line, c_line, [])

            TP += TP_out
            TN += TN_out
            FP += FP_out
            FN += FN_out

            print_out_result(TP, FP, TN, FN, include_metrics=False)
            print(f"Progress: {(line_idx + 1) / num_all_lines}")

        print_out_result(TP, FP, TN, FN, include_metrics=True)


def is_number(value):
    """
    returns true if value is a number
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def namedEntityWords(text_input):
    """
    Function returns indexes of words that represent names
    """
    text = text_input
    doc = nlp(text)
    conll_output = doc.to_conll()

    lines = conll_output.strip().split('\n')
    word_data = [line.split('\t') for line in lines]

    named_entity_words = []

    # Loop through each word and check NER tag
    for idx, word_info in enumerate(word_data):
        if len(word_info) >= 9 and is_number(word_info[0]):
            word = word_info[1]
            ner_tag = word_info[9]

            if ner_tag.startswith('NER=B-PER') or ner_tag.startswith('NER=I-PER'):
                named_entity_words.append(word)

    return named_entity_words


def reformat_evaluated_file(source_file_path, corrected_file_path, corr_file_out_path):
    """
    function reformats results of ChatGPT model, so it places labels like NAPAKA/Č/ and NAPAKA/Č/SKUPAJ_NARAZEN before words
    """
    source_text = read_file(source_file_path)
    corrected_text = read_file(corrected_file_path)
    new_corr_text = ""

    with open(corr_file_out_path, "w", encoding="utf-8") as corr_file_out:
        for s_line, c_line in zip(source_text.split("\n"), corrected_text.split("\n")):
            for index, (s_word, c_word) in enumerate(zip(s_line.split(" "), c_line.split(" "))):
                if index != 0:
                    new_corr_text += " "

                if c_word == "<napaka>" or c_word == "<mask>":
                    new_corr_text += "NAPAKA/Č/" + s_word
                elif c_word == "<split>" or c_word == "<concat>":
                    new_corr_text += "NAPAKA/Č/" + s_word
                else:
                    new_corr_text += s_word

            new_corr_text += "\n"

        corr_file_out.write(new_corr_text)


def evaluate_on_annotated_file(source_file_path, target_file_path, corrected_file_path, include_names=True):
    """
    function evaluates file that has annotations that start with NAPAKA/....
    """
    source_text = read_file(source_file_path)
    target_text = read_file(target_file_path)
    corrected_text = read_file(corrected_file_path)

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for t_line, c_line in zip(target_text.split("\n"), corrected_text.split("\n")):
        for t_word, c_word in zip(t_line.split(" "), c_line.split(" ")):
            if t_word.startswith("NAPAKA/Č/FORMAT") or t_word.startswith("NAPAKA/Č/IGNORE") or t_word.startswith("NAPAKA/Č/NESIGUREN") or t_word.startswith("NAPAKA/Č/NESTANDARDNO") or t_word.startswith("NAPAKA/Č/TUJA_BESEDA") or c_word.startswith("NAPAKA/Č/TUJA_BESEDA"):
                # print("went out")
                continue
            elif t_word.startswith("NAPAKA/Č") and c_word.startswith("NAPAKA/Č"):
                TP += 1
            elif c_word.startswith("NAPAKA/Č"):
                FP += 1
            elif t_word.startswith("NAPAKA/Č"):
                FN += 1
            else:
                TN += 1

        # if TP > 0 and FP > 0 and FN > 0 and TN > 0:
        #    print_out_result(TP, FP, TN, FN)

    if TP > 0 and FP > 0 and FN > 0 and TN > 0:
        print_out_result(TP, FP, TN, FN, include_metrics=True)


def evaluate_on_mask_annotated_file(source_file_path, target_file_path, corrected_file_path, include_names=True):
    """
    function evaluates chatGPT model, using already annotated corrected file (corrected_file_path)
    """
    source_text = read_file(source_file_path)
    target_text = read_file(target_file_path)
    corrected_text = read_file(corrected_file_path)

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for s_line, t_line, c_line in zip(source_text.split("\n"), target_text.split("\n"), corrected_text.split("\n")):
        c_line = post_process_line(c_line)

        if c_line == "Pravilno.":
            c_line = s_line

        if not include_names:
            ner_words = namedEntityWords(c_line)
        else:
            ner_words = []

        TP_out, TN_out, FP_out, FN_out = calculate_line_score(s_line, t_line, c_line, ner_words)

        TP += TP_out
        TN += TN_out
        FP += FP_out
        FN += FN_out

        print_out_result(TP, FP, TN, FN)

    print_out_result(TP, FP, TN, FN, include_metrics=True)


def realign_all_files():
    """
    function just calls align function on all files specified in advance
    """
    # lektor source/target
    align_file("source_target_files/lektor/lektor-combined-source-formatted.txt", "source_target_files/lektor/lektor-combined-source-aligned.txt")
    align_file("source_target_files/lektor/lektor-combined-target-formatted.txt", "source_target_files/lektor/lektor-combined-target-aligned.txt")
    align_file("source_target_files/lektor/lektor-combined-target-2-formatted.txt", "source_target_files/lektor/lektor-combined-target-2-aligned.txt")

    # solar source/target
    align_file("source_target_files/solar/solar_target_napake_č.txt",
               "source_target_files/solar/solar_target_napake_č-aligned.txt")
    align_file("source_target_files/solar/solar_target.txt",
               "source_target_files/solar/solar-target-aligned.txt")

    # synthetic source/target
    align_file("source_target_files/synthetic/source.txt",
               "source_target_files/synthetic/source-synthetic-aligned.txt")
    align_file("source_target_files/synthetic/target.txt",
               "source_target_files/synthetic/target-synthetic-aligned.txt")

    # ChatGPT
    align_file("ChatGPT/lektor/ChatGPT-4-lektor-formatted-2.txt", "ChatGPT/lektor/ChatGPT-4-lektor-aligned-formatted.txt")
    align_file("ChatGPT/šolar/ChatGPT-4-šolar-formatted.txt", "ChatGPT/šolar/ChatGPT-4-šolar-aligned-formatted.txt")
    align_file("ChatGPT/synthetic/ChatGPT-4-synthetic-formatted.txt", "ChatGPT/synthetic/ChatGPT-4-synthetic-aligned-formatted.txt")

    # SloNSpell
    align_file("SloNSpell/lektor/SloNSpell_lektor_corrected_data-formatted.txt", "SloNSpell/lektor/SloNSpell_aligned_lektor_corrected_data.txt")
    align_file("SloNSpell/solar/SloNSpell_šolar_corrected_data.txt", "SloNSpell/solar/SloNSpell_aligned_šolar_corrected_data.txt")
    align_file("SloNSpell/synthetic/SloNSpell_synthetic_corrected_data.txt", "SloNSpell/synthetic/SloNSpell_aligned_synthetic_corrected_data.txt")
    align_file("SloNSpell/lektor/lektor_just_mistakes.txt", "SloNSpell/lektor/lektor_just_mistakes_aligned.txt")

    # SloSpell
    align_file("SloSpell/lektor/slospell_preds_lektor-spelling.txt", "SloSpell/lektor/slospell_preds_lektor-spelling-aligned.txt")
    align_file("SloSpell/solar/slospell_preds_solar-eval.txt", "SloSpell/solar/slospell_preds_solar-eval-aligned.txt")
    align_file("SloSpell/synthetic/slospell_preds_synthetic-eval.txt", "SloSpell/synthetic/slospell_preds_synthetic-eval-aligned.txt")

    # source target files nestandardno
    align_file("source_target_files_nestandardno/lektor/lektor_just_mistakes.txt", "source_target_files_nestandardno/lektor/lektor_just_mistakes_aligned.txt")
    align_file("source_target_files_nestandardno/lektor/lektor_source_pT_pZ-shortened-označeno-combined.txt", "source_target_files_nestandardno/lektor/lektor_source_pT_pZ-shortened-označeno-combined_aligned.txt")


def check_if_lines_same(lines1, lines2):
    """
    function checks if all lines are the same and prints out lines that do not match in number of words
    """
    for line1, line2 in zip(lines1, lines2):
        if len(line1.split()) != len(line2.split()):
            print("line1: ", line1)
            print("line2: ", line2)


def check_num_lines_and_words_in_lines():
    """
    function checks for each aligned line if lengths match
    """
    # Solar
    with open("source_target_files/solar/solar_target_napake_č-aligned.txt", "r", encoding="utf-8") as solar_source:
        solar_source_lines = solar_source.read().split("\n")

    with open("source_target_files/solar/solar-target-aligned.txt", "r", encoding="utf-8") as solar_target:
        solar_target_lines = solar_target.read().split("\n")

    with open("SloNSpell/solar/SloNSpell_aligned_šolar_corrected_data.txt", "r", encoding="utf-8") as solar_corrected:
        slonspell_solar_corrected_lines = solar_corrected.read().split("\n")

    with open("ChatGPT/šolar/ChatGPT-4-šolar-aligned-formatted.txt", "r", encoding="utf-8") as solar_corrected:
        chatgpt_solar_corrected_lines = solar_corrected.read().split("\n")

    with open("SloSpell/solar/slospell_preds_solar-eval-aligned.txt", "r", encoding="utf-8") as solar_corrected:
        slospell_solar_corrected_lines = solar_corrected.read().split("\n")

    check_if_lines_same(solar_source_lines, solar_target_lines)

    check_if_lines_same(solar_source_lines, slonspell_solar_corrected_lines)

    check_if_lines_same(solar_source_lines, chatgpt_solar_corrected_lines)

    # check_if_lines_same(solar_source_lines, slospell_solar_corrected_lines)

    # Lektor
    with open("source_target_files/lektor/lektor-combined-source-aligned.txt", "r", encoding="utf-8") as lektor_source:
        lektor_source_lines = lektor_source.read().split("\n")

    with open("source_target_files/lektor/lektor-combined-target-aligned.txt", "r", encoding="utf-8") as lektor_target:
        lektor_target_lines = lektor_target.read().split("\n")

    with open("source_target_files/lektor/lektor-combined-target-2-aligned.txt", "r", encoding="utf-8") as lektor_target:
        lektor_target_2_lines = lektor_target.read().split("\n")

    with open("source_target_files_nestandardno/lektor/lektor_just_mistakes_aligned.txt", "r", encoding="utf-8") as lektor_mistakes_target:
        lektor_mistakes_target_lines = lektor_mistakes_target.read().split("\n")

    with open("source_target_files_nestandardno/lektor/lektor_source_pT_pZ-shortened-označeno-combined_aligned.txt", "r", encoding="utf-8") as lektor_nestandardno_target:
        lektor_nestandardno_target_lines = lektor_nestandardno_target.read().split("\n")

    with open("SloNSpell/lektor/SloNSpell_aligned_lektor_corrected_data.txt", "r", encoding="utf-8") as lektor_corrected:
        slonspell_lektor_corrected_lines = lektor_corrected.read().split("\n")

    with open("SloNSpell/lektor/lektor_just_mistakes_aligned.txt", "r", encoding="utf-8") as lektor_mistakes_corrected:
        slonspell_lektor_mistakes_corrected = lektor_mistakes_corrected.read().split("\n")

    with open("ChatGPT/lektor/ChatGPT-4-lektor-aligned-formatted.txt", "r", encoding="utf-8") as lektor_corrected:
        chatgpt_lektor_corrected_lines = lektor_corrected.read().split("\n")

    # check_if_lines_same(lektor_source_lines, lektor_nestandardno_target_lines)
    #
    # check_if_lines_same(lektor_source_lines, lektor_mistakes_target_lines)

    check_if_lines_same(lektor_source_lines, lektor_target_2_lines)
    #
    check_if_lines_same(lektor_source_lines, slonspell_lektor_corrected_lines)

    # check_if_lines_same(lektor_nestandardno_target_lines, slonspell_lektor_corrected_lines)
    #
    # check_if_lines_same(lektor_mistakes_target_lines, slonspell_lektor_mistakes_corrected)

    # check_if_lines_same(lektor_source_lines, chatgpt_lektor_corrected_lines)

    # Synthetic

    with open("source_target_files/synthetic/source-synthetic-aligned.txt", "r", encoding="utf-8") as synthetic_source:
        synthetic_source_lines = synthetic_source.read().split("\n")

    with open("source_target_files/synthetic/target-synthetic-aligned.txt", "r", encoding="utf-8") as synthetic_target:
        synthetic_target_lines = synthetic_target.read().split("\n")

    with open("SloNSpell/synthetic/SloNSpell_aligned_synthetic_corrected_data.txt", "r", encoding="utf-8") as synthetic_corrected:
        slonspell_synthetic_corrected_lines = synthetic_corrected.read().split("\n")

    with open("ChatGPT/synthetic/ChatGPT-4-synthetic-aligned-formatted.txt", "r", encoding="utf-8") as synthetic_corrected:
        chatgpt_synthetic_corrected_lines = synthetic_corrected.read().split("\n")

    check_if_lines_same(synthetic_source_lines, synthetic_target_lines)

    check_if_lines_same(synthetic_source_lines, slonspell_synthetic_corrected_lines)

    check_if_lines_same(synthetic_source_lines, chatgpt_synthetic_corrected_lines)


def evaluate_solar_fine_tuned_slollama():
    """
    function evaluates slopt model used to correct grammar mistakes
    """
    align_file("fine_tuned_SloLlama/solar/solar_source.txt", "fine_tuned_SloLlama/solar/solar_source_aligned.txt")

    align_file("fine_tuned_SloLlama/solar/solar_target.txt", "fine_tuned_SloLlama/solar/solar_target_aligned.txt")

    align_file("fine_tuned_SloLlama/solar/solar_corrected.txt", "fine_tuned_SloLlama/solar/solar_corrected_aligned.txt")


def label_spelling_mistakes_in_file(file_path, output_file_path):
    """
    function labels just spelling mistakes in file (mistakes starting with NAPAKA/Č and NAPAKA/Č/SKUPAJ_NARAZEN) - it labels mistakes with label: NAPAKA
    """
    with open(file_path, "r", encoding="utf-8") as in_file:
        file_text = in_file.read()

    # file_text = re.sub("NAPAKA/Č/FORMAT/", "IGNORE", file_text)
    # file_text = re.sub("NAPAKA/Č/NESTANDARDNO/", "IGNORE", file_text)
    # file_text = re.sub("NAPAKA/Č/TUJA_BESEDA/", "IGNORE", file_text)

    new_words = []

    for line in file_text.split("\n"):
        for word in line.split(" "):
            if word.startswith("NAPAKA/Č/FORMAT/") or word.startswith("NAPAKA/Č/NESTANDARDNO/") or word.startswith("NAPAKA/Č/TUJA_BESEDA/"):
                new_words.append("IGNORE")
            elif word.startswith("NAPAKA/Č/"):
                new_words.append("NAPAKA")
            else:
                new_words.append(word)
        new_words.append("\n")

    new_file_text = " ".join(new_words)

    with open(output_file_path, "w", encoding="utf-8") as out_file:
        out_file.write(new_file_text)


if __name__ == "__main__":
    # use_chatgpt_4()
    #
    # print("lektor: SloNSpell")
    # evaluate_on_annotated_file("source_target_files/lektor/lektor-combined-source-aligned.txt", "source_target_files/lektor/lektor-combined-target-2.txt", f"SloNSpell/lektor/SloNSpell_lektor_corrected_data-formatted.txt", include_names=True)
    #
    # print("lektor: ChatGpt-4")
    # evaluate_on_annotated_file("source_target_files/lektor/lektor-combined-source-aligned.txt", "source_target_files/lektor/lektor-combined-target-2.txt", f"ChatGPT/lektor/ChatGPT-4-lektor-aligned-formatted.txt", include_names=True)
    #
    # print("lektor: SloSpell")
    # evaluate_on_annotated_file("source_target_files/lektor/lektor-combined-source-aligned.txt", "source_target_files/lektor/lektor-combined-target-2.txt", f"SloSpell/lektor/slospell_preds_lektor-spelling-aligned.txt", include_names=True)
    #
    # print("lektor: HunSpell")
    # evaluate_on_annotated_file("source_target_files/lektor/lektor-combined-source-aligned.txt",
    #                            "source_target_files/lektor/lektor-combined-target-2.txt",
    #                            f"hunspell/lektor/hunspell_preds_lektor-spelling.txt", include_names=True)
    #
    # print("lektor: LanguageTool")
    # evaluate_on_annotated_file("source_target_files/lektor/lektor-combined-source-aligned.txt",
    #                            "source_target_files/lektor/lektor-combined-target-2.txt",
    #                            f"LanguageTool/lektor/languagetool_preds_lektor-spelling 1.txt", include_names=True)

    evaluate_solar_fine_tuned_slollama()


    # evaluate_using_chatGPT("source_target_files/lektor/lektor-combined-source-aligned.txt", "source_target_files/lektor/lektor-combined-target-2-formatted.txt", "SloNSpell/lektor/SloNSpell_aligned_lektor_corrected_data.txt")

    # reformat_chatgpt_annotated_file("ChatGPT/šolar/solar_target_napake_č.txt", "ChatGPT/šolar/corrected-gpt-4-0125-preview.txt", "ChatGPT/šolar/corrected-gpt-4-reformatted.txt")

    # reformat_chatgpt_annotated_file("source_target_files/lektor/lektor-combined-target-formatted.txt",
    #                                 "ChatGPT/lektor/corrected-gpt-4-0125-preview.txt",
    #                                 "ChatGPT/lektor/corrected-gpt-4-reformatted.txt")

    # reformat_chatgpt_annotated_file("ChatGPT/synthetic/target.txt",
    #                                 "ChatGPT/synthetic/corrected-gpt-4-0125-preview.txt",
    #                                 "ChatGPT/synthetic/corrected-gpt-4-reformatted.txt")

    # evaluate_using_chatGPT("lektor/lektor-combined-source.txt", "source_target_files/lektor/lektor-combined-target-2.txt", f"lektor/corrected-{model_to_use}.txt")

    # reformat_evaluated_file("ChatGPT/šolar/solar_target_napake_č.txt", "ChatGPT/šolar/corrected-gpt-4-reformatted.txt", "ChatGPT/šolar/ChatGPT-4-šolar-formatted.txt")

    # reformat_evaluated_file("ChatGPT/synthetic/source.txt", "ChatGPT/synthetic/corrected-gpt-4-reformatted.txt", "ChatGPT/synthetic/ChatGPT-4-synthetic-formatted.txt")

    # reformat_evaluated_file("ChatGPT/lektor/lektor-combined-source.txt", "ChatGPT/lektor/corrected-gpt-4-reformatted.txt", "ChatGPT/lektor/ChatGPT-4-lektor-formatted.txt")

    # realign_all_files()
    #
    # check_num_lines_and_words_in_lines()

    # label_spelling_mistakes_in_file("source_target_files_nestandardno/lektor/lektor_source_pT_pZ-shortened-označeno-combined.txt", "source_target_files_nestandardno/lektor/lektor_just_mistakes.txt")
    # label_spelling_mistakes_in_file("SloNSpell/lektor/SloNSpell_lektor_corrected_data.txt", "SloNSpell/lektor/lektor_just_mistakes.txt")

    # label_spelling_mistakes_in_file("source_target_files/lektor/lektor-combined-target-2.txt", "source_target_files/lektor/lektor-combined-target-2-formatted.txt")

