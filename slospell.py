import itertools
import os.path
import re
import string
from time import time
from typing import Set, List, Optional

import datasets
from nltk import word_tokenize
from sklearn.metrics import classification_report
from tqdm import trange, tqdm

CACHE_PATH = "_slospell_cache.txt"
ADDITIONAL_VALID_SPECIAL_CHARACTERS = {"”", "—", "„", "”", "–", "…", "“", "‚", "‘", "»", "«", "−", "’", "\"", "´", "..",
									   "→"}


def prepare_sloleks_wordforms() -> Set[str]:
	if os.path.exists(CACHE_PATH):
		print(f"Loading SloSpell word forms from cache...")
		with open(CACHE_PATH, "r") as f_cache:
			valid_word_forms = set(map(lambda _s: _s.strip(), f_cache.readlines()))

	else:
		sloleks = datasets.load_dataset("cjvt/sloleks", trust_remote_code=True, split="train")

		tmp_valid_lemmas = set()
		valid_word_forms = set()
		for idx in trange(len(sloleks)):
			ex = sloleks[idx]
			tmp_valid_lemmas.add(ex["headword_lemma"])

			for _wfs in ex["word_forms"]:
				for _wf, _is_nonstandard in zip(_wfs["forms"], _wfs["is_nonstandard"]):
					if not _is_nonstandard:
						valid_word_forms.add(_wf)

		with open(CACHE_PATH, "w") as fp:
			print(f"Caching valid word forms...")
			for _wf in valid_word_forms:
				print(_wf, file=fp)
		print(f"{len(tmp_valid_lemmas)} lemmas vs {len(valid_word_forms)} word forms")

	return valid_word_forms


def load_solar_eval(file_path):
	raw_examples: List[str] = []
	tokenized_examples: List[List[str]] = []
	binary_labels: List[List[int]] = []
	error_types: List[List[str]] = []

	with open(file_path, "r") as f:
		for _line in f:
			ex = _line.strip()
			raw_examples.append(ex)
			words = ex.split(" ")
			postprocessed_words = []
			binary_error, error_type = [], []
			for _w in words:
				# Example: NAPAKA/Č/KONZ/izpust/življenski
				if _w.startswith("NAPAKA/"):
					parts = _w.split("/")
					binary_error.append(1)
					error_type.append("/".join(parts[: -1]))
					postprocessed_words.append(parts[-1])

				else:
					binary_error.append(0)
					error_type.append("NO_ERROR")
					postprocessed_words.append(_w)

			tokenized_examples.append(postprocessed_words)
			binary_labels.append(binary_error)
			error_types.append(error_type)

	return raw_examples, tokenized_examples, binary_labels, error_types


def slospell(ex):
	predicted_labels = []
	for in_word_or_words in ex:
		# Decimals (e.g., "3", "3.", "3.5", "3,5") are marked as correct
		if re.fullmatch(r"(\d+\.?\d*)|(\d+\,\d+)", in_word_or_words) is not None:
			predicted_labels.append(0)
			continue

		# URLs are ignored, i.e. marked as correct
		if in_word_or_words.startswith("http") or in_word_or_words.startswith("www"):
			predicted_labels.append(0)
			continue

		in_words = word_tokenize(in_word_or_words, language="slovene")

		is_incorrect = []
		for _w in in_words:
			is_w_correct = _w in valid_word_forms or _w.lower() in valid_word_forms or \
						   _w in string.punctuation or _w.isnumeric() or _w in ADDITIONAL_VALID_SPECIAL_CHARACTERS

			is_incorrect.append(not is_w_correct)

		predicted_labels.append(int(any(is_incorrect)))

	return predicted_labels


def visualize_predictions(input_words: List[List[str]], preds: List[List[int]],
						  correct_classes: Optional[List[List[int]]]=None,
						  additional_markers: Optional[List[List[str]]]=None,
						  visualization_path=None):
	_visualization_path = "visualization.html" if visualization_path is None else visualization_path
	assert len(input_words) == len(preds)
	if correct_classes is not None:
		assert len(preds) == len(correct_classes)
	else:
		correct_classes = [[None for _w_pred in _ex_preds] for _ex_preds in preds]

	if additional_markers is None:
		additional_markers = [[None for _w_pred in _ex_preds] for _ex_preds in preds]

	body_code = []

	for idx_ex, (ex_words, ex_preds, ex_correct, ex_markers) in enumerate(zip(input_words, preds, correct_classes, additional_markers)):
		ex_code = []
		for word, pred, corr, mark in zip(ex_words, ex_preds, ex_correct, ex_markers):
			hl_color = "transparent"
			if corr is not None and (pred or corr):
				hl_color = "#f5ffc4" if pred == corr else "#ffd3c4"

			font_weight = "normal"  # color: red?
			if pred:
				font_weight = "bold"

			ex_code.append(f'<span style="font-weight: {font_weight}; background-color: {hl_color}">{word}</span>')

		ex_code = " ".join(ex_code)
		body_code.append(f'<div style="padding-bottom: 10px"><span style="background-color: yellow">[#{1 + idx_ex}]</span> {ex_code}</div>')

	with open(_visualization_path, "w") as f_viz:
		body_code = "\n".join(body_code)
		print(f"<html><body>{body_code}</body></html>", file=f_viz)


def evaluate_predictions(preds: List[List[int]], correct_classes: List[List[int]]):
	flat_preds = list(itertools.chain(*preds))
	flat_correct = list(itertools.chain(*correct_classes))

	print(classification_report(y_true=flat_correct, y_pred=flat_preds, digits=3))


if __name__ == "__main__":
	valid_word_forms = prepare_sloleks_wordforms()
	print(f"{len(valid_word_forms)} valid word forms")

	path_solar_eval = "Šolar Eval ločene črkovne napake/solar_target_napake_č_label.txt"
	input_texts, input_words, correct_labels, err_types = load_solar_eval(path_solar_eval)
	print(f"Loaded {len(input_texts)} test examples")

	ts = time()
	predictions = []
	with open("slospell_preds_solar-eval.txt", "w") as f:
		for curr_ex in tqdm(input_words):
			curr_preds = slospell(curr_ex)
			predictions.append(curr_preds)

			formatted_ex = []
			for _w, _p in zip(curr_ex, curr_preds):
				if _p:
					formatted_ex.append(f"NAPAKA/Č/{_w}")
				else:
					formatted_ex.append(_w)

			formatted_ex = " ".join(formatted_ex)
			print(formatted_ex, file=f)

	te = time()
	print(f"Time: {te - ts}s")

	visualize_predictions(input_words, predictions, correct_labels, visualization_path="visualization_solar_slospell.html")
	evaluate_predictions(predictions, correct_labels)

	path_lektor = "lektor eval final/lektor_source_pT_pZ-shortened-označeno-combined.txt"
	input_texts, input_words, correct_labels, err_types = load_solar_eval(path_lektor)
	print(f"Loaded {len(input_texts)} test examples")

	ts = time()
	predictions = []
	with open("slospell_preds_lektor-spelling.txt", "w") as f:
		for curr_ex in tqdm(input_words):
			curr_preds = slospell(curr_ex)
			predictions.append(curr_preds)

			formatted_ex = []
			for _w, _p in zip(curr_ex, curr_preds):
				if _p:
					formatted_ex.append(f"NAPAKA/Č/{_w}")
				else:
					formatted_ex.append(_w)

			formatted_ex = " ".join(formatted_ex)
			print(formatted_ex, file=f)

	te = time()
	print(f"Time: {te - ts}s")

	visualize_predictions(input_words, predictions, correct_labels, visualization_path="visualization_lektor_slospell.html")
	evaluate_predictions(predictions, correct_labels)

	path_synthetic_eval = "Gigafida generirane črkovne napake/source.txt"
	input_texts, input_words, correct_labels, err_types = load_solar_eval(path_synthetic_eval)
	print(f"Loaded {len(input_texts)} test examples")

	ts = time()
	predictions = []
	with open("slospell_preds_synthetic-eval.txt", "w") as f:
		for curr_ex in tqdm(input_words):
			curr_preds = slospell(curr_ex)
			predictions.append(curr_preds)

			formatted_ex = []
			for _w, _p in zip(curr_ex, curr_preds):
				if _p:
					formatted_ex.append(f"NAPAKA/Č/{_w}")
				else:
					formatted_ex.append(_w)

			formatted_ex = " ".join(formatted_ex)
			print(formatted_ex, file=f)

	visualize_predictions(input_words, predictions, correct_labels, visualization_path="visualization_synthetic-eval_slospell.html")
	evaluate_predictions(predictions, correct_labels)

	te = time()
	print(f"Time: {te - ts}s")


