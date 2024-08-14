import random
import csv
import os
import re
import string
import pickle
from random import randint, choice
from typing import Tuple

besede_nagajivke_nepravilne = ['golop', 'vrak', 'hrip', 'blaš', 'poglet', 'grat', 'drobiš', 'krok', 'mras', 'zemljevit',
                               'paris', 'matevš', 'predlok', 'nahrptnik', 'vrapci', 'retko', 'nagopčnik', 'obras',
                               'tomaš', 'labot', 'sladolet', 'matjaš', 'moš', 'andraš', 'glazba', 'negdo', 'negdaj',
                               'rizba', 'gdo', 'gdaj', 'negdaj', 'gdor', 'napisav', 'napisau', 'domou', 'domol', 'beu',
                               'rou', 'lou', 'kupiu', 'nou', 'miseu', 'siu', 'žouna', 'oseu', 'učitel', 'kluka',
                               'učitelica', 'kegel', 'lubezniv', 'kniga', 'kon', 'zadni', 'sosedni', 'češna',
                               'bližnica', 'prijatelica', 'škoren', 'čevel', 'fotel', 'uhel', 'hmel', 'savinski',
                               'ravnatel', 'tulnje', 'tjulne', 'tjulni', 'tjulen', 'metul', 'knižničar', 'polje',
                               'ragla', 'reditel', 'zdroblen', 'bradla', 'prejšni', 'žemle', 'razsvetlava', 'zemelski',
                               'mravla', 'obluba', 'cinglati', 'von', 'čebelnak', 'življenski']
besede_nagajivke_pravilne = ['golob', 'vrag', 'hrib', 'blaž', 'pogled', 'grad', 'drobiž', 'krog', 'mraz', 'zemljevid',
                             'pariz', 'matevž', 'predlog', 'nahrbtnik', 'vrabci', 'redko', 'nagobčnik', 'obraz',
                             'tomaž', 'labod', 'sladoled', 'matjaž', 'mož', 'andraž', 'glasba', 'nekdo', 'nekdaj',
                             'risba', 'kdo', 'kdaj', 'nekdaj', 'kdor', 'napisal', 'napisal', 'domov', 'domov', 'bel',
                             'rov', 'lov', 'kupil', 'nov', 'misel', 'siv', 'žolna', 'osel', 'učitelj', 'kljuka',
                             'učiteljica', 'kegelj', 'ljubezniv', 'knjiga', 'konj', 'zadnji', 'sosednji', 'češnja',
                             'bližnjica', 'prijateljica', 'škorenj', 'čevelj', 'fotelj', 'uhelj', 'hmelj', 'savinjski',
                             'ravnatelj', 'tjuljnje', 'tjulnje', 'tjuljni', 'tjulenj', 'metulj', 'knjižnjičar', 'polje',
                             'raglja', 'reditelj', 'zdrobljen', 'bradlja', 'prejšnji', 'žemlje', 'razsvetljava',
                             'zemeljski', 'mravlja', 'obljuba', 'cingljati', 'vonj', 'čebelnjak', 'življenjski']
slovenian_alphabet_uppercase = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'Đ', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                                'NJ', 'O', 'P', 'R', 'S', 'Š', 'T', 'U', 'V', 'Z', 'Ž', '_', '/']
slovenian_alphabet_lowercase = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'lj', 'm', 'n', 'nj', 'o',
                                'p', 'r', 's', 't', 'u', 'v', 'z']
linking_words = ["in", "kot", "ali", "oziroma", "ter", "le", "čeprav", "tudi", "vendar", "vključno", "kjer", "ki",
                 "kateri", "a", "ne", "niti"]
filler_words = ["pa"]
link_sentences_filler = ["in"]

valid_chars = "abcčdefghijklmnoprsštuvzž"

lemma_dict = {}
count_lemma_dict = {}
root_lemma_dict = {}

# all_words = {}

with open('dictionaries/root_lemma_dict.pkl', 'rb') as f:
    root_lemma_dict = pickle.load(f)

with open('dictionaries/sloleks_root_dict.pkl', 'rb') as f:
    sloleks_root_dict = pickle.load(f)

with open('dictionaries/sloleks_word_forms_dict.pkl', 'rb') as f:
    sloleks_word_forms_dict = pickle.load(f)

with open('dictionaries/sloleks_pos_tag_dict.pkl', 'rb') as f:
    sloleks_pos_tag_dict = pickle.load(f)

with open('dictionaries/lemma_dict.pkl', 'rb') as f:
    lemma_dict = pickle.load(f)
    # print(lemma_dict)

with open('dictionaries/count_lemma_dict.pkl', 'rb') as f:
    count_lemma_dict = pickle.load(f)


# with open('ssks_besede.txt', 'r') as file:
#     for line in file:
#         word = line.strip()
#         if word not in all_words:
#             all_words[word] = 1

def take_with_prob(probability=0.5):
    return random.random() < probability


def split_word_random(word):
    if len(word) <= 1:
        return [word]

    split_point = random.randint(1, len(word) - 1)
    return [word[:split_point], word[split_point:]]


def split_random_words(text: str, probability: 0.06) -> str:
    words = text.split()

    new_words = []
    for word in words:
        if random.random() < probability and can_modify(word) and len(word) > 1:
            word1, word2 = split_word_random(word)

            if (random.random() < 0.01) or ((len(word1) > 1 or len(
                    word2) > 1) and word1 in root_lemma_dict and word2 in word2 in root_lemma_dict and word1 in sloleks_root_dict and word2 in word2 in sloleks_root_dict):
                new_words.append(word1 + "<splited_word>" + word2)
                # new_words.append("<splited_word>" + word2)
                # print("razdruzil besedi; word1: ", word1, "word2: ", word2)
            else:
                new_words.append(word)
        else:
            new_words.append(word)

    return " ".join(new_words)


def check_concat_word_calid(target_word, prev_word):
    if target_word == "<word_split>" or prev_word == "<word_split>" or target_word in string.punctuation or prev_word in string.punctuation:
        return False
    return True


def concat_random_words(text: str, probability: 0.04) -> str:
    words = text.split()

    new_words = []

    for word in words:
        if random.random() < probability and len(
                new_words) > 0 and "<splited_word>" not in word and "<splited_word>" not in new_words[
            -1] and word not in string.punctuation and new_words[-1] not in string.punctuation and not new_words[
            -1].startswith("<concated_word>"):
            if new_words[-1].startswith("<concated_word>"):
                new_words[-1] = new_words[-1] + word
            else:
                if (random.random() < 0.01) or (
                        (new_words[-1] + word) in root_lemma_dict and (new_words[-1] + word) in sloleks_root_dict):
                    new_words[-1] = "<concated_word>" + new_words[-1] + word

                    # print("zdruzil v besedo: ", new_words[-1])
                else:
                    new_words.append(word)
        else:
            new_words.append(word)

    return " ".join(new_words)


def replace_random_chars_in_word_2(word, charSubselection, charReplaceProb=0.15) -> str:
    while random.random() < charReplaceProb and len(word) > 1:
        index_to_replace = random.randint(0, len(word) - 1)

        chars = list(word)

        char_to_replace = ''

        if chars[index_to_replace].lower() in charSubselection:
            chars[index_to_replace] = random.choice(charSubselection)

        word = ''.join(chars)

    return word


def insert_random_chars_in_word_2(word, charSubselection, charReplaceProb=0.1) -> str:
    while random.random() < charReplaceProb and len(word) > 1:
        index_to_insert = random.randint(0, len(word))

        chars = list(word)

        char_to_insert = ''

        # if chars[index_to_insert].lower() in charSubselection:
        char_to_insert = random.choice(charSubselection)
        chars.insert(index_to_insert, char_to_insert)

        word = ''.join(chars)

    return word


def remove_random_chars_in_word_2(word, charRemoveProb=0.1) -> str:
    while random.random() < charRemoveProb and len(word) > 1:
        index_to_remove = random.randint(0, len(word) - 1)

        chars = list(word)

        chars[index_to_remove] = ''

        word = ''.join(chars)

    return word


def replace_random_chars_in_word(word, charReplaceProb=0.15, checkLower=False) -> str:
    while random.random() < charReplaceProb and len(word) > 1:
        index_to_replace = random.randint(0, len(word) - 1)

        chars = list(word)

        if chars[index_to_replace].isdigit() or chars[index_to_replace] in string.punctuation:
            continue

        char_to_replace = ''

        if checkLower and not chars[index_to_replace].islower():
            print("not lower: ", chars[index_to_replace])
            continue

        if chars[index_to_replace].isupper():
            char_to_replace = random.choice(slovenian_alphabet_uppercase)
        else:
            char_to_replace = random.choice(slovenian_alphabet_lowercase)

        random_value = random.random()

        if random_value < 0.35:
            chars[index_to_replace] = ''
        elif random_value < 0.55:
            char_to_replace = random.choice(slovenian_alphabet_lowercase)
            if char_to_replace != '/':
                len_chars_prej = len(chars)
                chars.insert(index_to_replace, char_to_replace)
                assert len(chars) == len_chars_prej + 1
        elif random_value < 0.8:
            # switch random two chars that stand together
            index_to_switch = index_to_replace
            if index_to_switch == len(word) - 1:
                index_to_switch -= 1

            # print("before chars: ", chars)
            chars[index_to_switch - 1], chars[index_to_switch] = chars[index_to_switch], chars[index_to_switch - 1]
            # print("after chars: ", chars)
        else:
            assert random_value >= 0.80 and random_value <= 1.0

            chars[index_to_replace] = char_to_replace

        word = ''.join(chars)

    return word


def replace_random_chars_in_text_2(input_string: str, p_vowel=0.05, p_consonant=0.05, p_subst_chr=0.02, p_def_chr=0.04, p_insert_chr=0.03) -> str:
    words = input_string.split()

    modified_words = []

    for itr_index, word in enumerate(words):
        new_word = word

        if not word.isdigit() and not word in string.punctuation and can_modify(word):
            new_word = replace_random_chars_in_word_2(word, 'aeiou', p_vowel)

            new_word = replace_random_chars_in_word_2(new_word, 'bcčdfghjklmnprsštvzž', p_consonant)

            new_word = replace_random_chars_in_word_2(new_word, 'abcčdevghijklmnoprsštuvzž', p_subst_chr)

            new_word = remove_random_chars_in_word_2(new_word, p_def_chr)

            new_word = insert_random_chars_in_word_2(new_word, 'abcčdevghijklmnoprsštuvzž', p_insert_chr)

        modified_words.append(new_word)

    return ' '.join(modified_words)


def can_modify(word):
    return "<splited_word>" not in word and "<concated_word>" not in word


def zamenjaj_besede_nagajivke(input_string: str, probability=0.6) -> str:
    words = input_string.split()

    for i, word in enumerate(words):
        for j, w in enumerate(besede_nagajivke_pravilne):
            if take_with_prob(probability) and can_modify(word):
                if word.startswith(w):
                    words[i] = word.replace(w, besede_nagajivke_nepravilne[j], 1)
                elif word.startswith(w[:-1]):
                    words[i] = word.replace(w[:-1], besede_nagajivke_nepravilne[j][:-1], 1)

    return " ".join(words)


def replace_some_predefined_characters(text, prob=0.7, num_repeats=4):
    words = text.split()
    new_words = []
    for word in words:
        new_word = word
        if take_with_prob(prob) and can_modify(word):
            for num_repeat in range(0, num_repeats):
                random_number = random.randint(1, 38)

                if random_number == 1:
                    new_word = word.replace('nj', 'n')
                elif random_number == 2:
                    new_word = word.replace('n', 'nj')
                elif random_number == 3:
                    new_word = word.replace('l', 'lj')
                elif random_number == 4:
                    new_word = word.replace('lj', 'l')
                elif random_number == 5:
                    new_word = word.replace('t', 'd')
                elif random_number == 6:
                    new_word = word.replace('d', 't')
                elif random_number == 7:
                    new_word = word.replace('v', 'u')
                elif random_number == 8:
                    new_word = word.replace('u', 'v')
                elif random_number == 9:
                    new_word = word.replace('u', 'el')
                elif random_number == 10:
                    new_word = word.replace('el', 'u')
                elif random_number == 11:
                    new_word = word.replace('i', 'j')
                elif random_number == 12:
                    new_word = word.replace('j', 'i')
                elif random_number == 13:
                    new_word = word.replace('k', 'kj')
                elif random_number == 14:
                    new_word = word.replace('kj', 'k')
                elif random_number == 15:
                    new_word = word.replace('k', 'h')
                elif random_number == 16:
                    new_word = word.replace('h', 'k')
                elif random_number == 17:
                    new_word = word.replace('k', 'g')
                elif random_number == 18:
                    new_word = word.replace('g', 'k')
                elif random_number == 19:
                    new_word = word.replace('s', 'z')
                elif random_number == 20:
                    new_word = word.replace('z', 's')
                elif random_number == 21:
                    new_word = word.replace('p', 'b')
                elif random_number == 22:
                    new_word = word.replace('b', 'p')
                elif random_number == 23:
                    new_word = word.replace('š', 'ž')
                elif random_number == 24:
                    new_word = word.replace('ž', 'š')
                elif random_number == 25:
                    new_word = word.replace('v', 'l')
                elif random_number == 26:
                    new_word = word.replace('l', 'v')
                elif random_number == 27:
                    new_word = word.replace('u', 'l')
                elif random_number == 28:
                    new_word = word.replace('l', 'u')
                elif random_number == 29:
                    new_word = word.replace('t', 'tj')
                elif random_number == 30:
                    new_word = word.replace('tj', 't')
                elif random_number == 31:
                    new_word = word.replace('i', 'ij')
                elif random_number == 32:
                    new_word = word.replace('ij', 'i')
                elif random_number == 33:
                    new_word = word.replace('j', 'lj')
                elif random_number == 34:
                    new_word = word.replace('lj', 'j')
                elif random_number == 35:
                    new_word = word.replace('ž', 'č')
                elif random_number == 36:
                    new_word = word.replace('č', 'ž')
                elif random_number == 37:
                    new_word = word.replace('e', 'i')
                elif random_number == 38:
                    new_word = word.replace('i', 'e')

                if new_word != word:
                    # print('NOT EQUAL')
                    break
        new_words.append(new_word)
    return ' '.join(new_words)


def add_spaces_before_and_after_punctuation(text):
    punctuation = set(string.punctuation)

    # Initialize an empty string to store the modified text
    modified_text = ""

    # Loop through each character in the text
    for char in text:
        # If the character is a punctuation mark
        if char in punctuation:
            # Add a space before and after the punctuation mark
            modified_text += " " + char + " "
        else:
            # Otherwise, just add the character to the modified text
            modified_text += char

    # Return the modified text
    return modified_text


def shuffle_n_words(words, n):
    if len(words) != len(set(words)):
        return words

    if len(words) == 0:
        return words

    if n > len(words):
        print("vecja dolzina besed as it should be")
        n = len(words)

    words_to_shuffle = random.sample(words, n)
    random.shuffle(words_to_shuffle)
    shuffled_words = []
    i = 0
    # print("n je: ", n)
    # print("words_to_shuffle: ", words_to_shuffle)
    # print("words: ", words)
    for word in words:
        if word in words_to_shuffle:
            shuffled_words.append(words_to_shuffle[i])
            i += 1
        else:
            shuffled_words.append(word)
    return shuffled_words


def weighted_random(max_number):
    if max_number < 2:
        raise ValueError("max_number should be greater than or equal to 2")

    choices = list(range(2, max_number + 1))
    weights = list(reversed(range(1, len(choices) + 1)))

    return random.choices(choices, weights, k=1)[0]


def replace_some_ž_š_č_characters(input_str, prob=0.1):
    words = input_str.split(' ')
    output_str = ''

    for word in words:
        if output_str != '':
            output_str += ' '

        if can_modify(word):
            for char in word:
                if take_with_prob(prob):
                    char = char.lower()

                    if char == 'ž':
                        output_str += 'z'
                    elif char == 'š':
                        output_str += 's'
                    elif char == 'č':
                        output_str += 'c'
                    elif char == 's':
                        output_str += 'š'
                    elif char == 'z':
                        output_str += 'ž'
                    elif char == 'c':
                        output_str += 'č'
                    else:
                        output_str += char
                else:
                    output_str += char
        else:
            output_str += word

    return output_str


def mark_false_words(incorrect_line, original_line, mask_token="<mask>", split_token="<splited_word>", concat_token="<concated_word>"):
    new_incorrect_line = []
    new_original_line = []

    numConcat = incorrect_line.count(concat_token)

    incorrect_line = incorrect_line.strip()
    original_line = original_line.strip()

    incorrect_line = incorrect_line.split(' ')
    original_line = original_line.split(' ')

    if (len(incorrect_line) + numConcat) != len(original_line):
        print("incorrect_line: ", incorrect_line)
        print("original_line: ", original_line)
        print("incorrect_line length: ", len(incorrect_line))
        print("count: ", numConcat)
        print("len(original_line): ", len(original_line))

    assert (len(incorrect_line) + numConcat) == len(original_line)

    # print("incorrect_line: ", incorrect_line)
    # print("original_line: ", original_line)
    plus_orig_line_counter = 0

    for i in range(0, len(incorrect_line)):
        # new_incorrect_line.append(incorrect_line[i])

        if can_modify(incorrect_line[i]):
            new_incorrect_line.append(incorrect_line[i])

            if incorrect_line[i] == original_line[i + plus_orig_line_counter]:
                new_original_line.append(original_line[i + plus_orig_line_counter])
            else:
                new_original_line.append(original_line[i + plus_orig_line_counter] + mask_token + incorrect_line[i])
                # new_original_line.append("<mask>")
        else:
            if split_token in incorrect_line[i]:
                word1, word2 = incorrect_line[i].split(split_token)
                new_incorrect_line.append(word1)
                new_incorrect_line.append(word2)

                new_original_line.append(split_token)
                new_original_line.append(split_token)
            elif concat_token in incorrect_line[i]:
                plus_orig_line_counter += 1

                word0 = incorrect_line[i].split(concat_token)[1]

                new_incorrect_line.append(word0)
                new_original_line.append(concat_token)

    # print("new_incorrect_line: ", new_incorrect_line)
    # print("new_original_line: ", new_original_line)

    assert len(new_incorrect_line) == len(new_original_line)

    new_incorrect_line = " ".join(new_incorrect_line)
    new_original_line = " ".join(new_original_line)

    return (new_incorrect_line, new_original_line)


def remove_double_spaces(text):
    pattern = r'\s{2,}'
    return re.sub(pattern, ' ', text)


def make_false_words_mark_sloBERTa_model(incorrect_line, original_line, mask_token="<mask>", split_token="<splited_word>", concat_token="<concated_word>", start_token="<s>", end_token="</s>"):
    new_masks_line_vector = []
    labels_vector = []

    new_masks_line_vector.append(start_token)

    # print("incorrect_line: ", incorrect_line)
    # print("original_line: ", original_line)

    for incorrect_word, orig_word in zip(incorrect_line.split(" "), original_line.split(" ")):
        if orig_word == mask_token:
            labels_vector.append("1")
        elif orig_word == split_token:
            labels_vector.append("2")
        elif orig_word == concat_token:
            labels_vector.append("3")
        else:
            if incorrect_word != orig_word:
                print("orig_word: ", orig_word)

            assert incorrect_word == orig_word

            labels_vector.append("0")

        new_masks_line_vector.append(incorrect_word)
        new_masks_line_vector.append(mask_token)

    new_masks_line_vector.append(end_token)

    new_masks_line = " ".join(new_masks_line_vector)
    labels = " ".join(labels_vector)

    return new_masks_line + "\t" + labels


def make_T5_row(line, mask_token="<mask>"):
    new_line = []

    for word in line.split(' '):
        if mask_token in word:
            new_line.append(mask_token)
        else:
            new_line.append(word)

    return " ".join(new_line)


def edit_step(word):
    """
    All edits that are one edit away from `word`.
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)