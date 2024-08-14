import json

if __name__ == "__main__":
    words = []

    with open("../lektor/labeled_unlabeled_data/lektor_source_pT_pZ-shortened-označeno-combined.txt", "r", encoding="utf-8") as lektor_whole:
        lektor_text = lektor_whole.read()

    for word in lektor_text.split(" "):
        if "NAPAKA/Č/TUJA_BESEDA" in word: # if word.startswith("NAPAKA/Č/TUJA_BESEDA"):
            words.append(word)

    with open("ChatGPT/lektor/ChatGPT-4-lektor-formatted.txt", "r", encoding="utf-8") as lektor:
        lektor_text = lektor.read()

    new_text = []
    idx_array = 0

    for idx, word in enumerate(lektor_text.split(" ")):
        if "IGNORE" in word:  # if word.startswith("IGNORE") or word.startswith("\nIGNORE"):  # if "IGNORE" in word:
            if idx_array > len(words):
                print("napaka")
            new_text.append(words[idx_array])
            idx_array += 1
        else:
            new_text.append(word)

    with open("ChatGPT/lektor/ChatGPT-4-lektor-formatted-2.txt", "w", encoding="utf-8") as lektor_source:
        lektor_source.write(" ".join(new_text))