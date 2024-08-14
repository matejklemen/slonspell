# SlonSpell: Neural Spell-Checker for Slovenian

This repository contains the code for running experiments in the paper "Neural Spell-Checker: Beyond Words with Synthetic Data Generation," which has been accepted to TSD 2024. The core of SlonSpell is a fine-tuned SloBERTa model, designed for improved spell-checking capabilities in Slovenian text. This README will guide you through the process of setting up the environment, generating synthetic training data, training the model, and evaluating its performance.

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Installation

### Prerequisites

Before you begin, ensure that you have the following software installed:

- Python 3.8 or higher
- [PyTorch](https://pytorch.org/get-started/locally/)
- [Transformers](https://huggingface.co/transformers/installation.html)
- Additional dependencies as listed in `requirements.txt`

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/slonspell.git
   cd slonspell
   ```

2. **Install dependencies:**

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download SloBERTa model:**

   Download the pre-trained SloBERTa model from Hugging Face and store it in a folder named `sloBERTaModel`:

   ```bash
   mkdir sloBERTaModel
   # Download the model from Hugging Face and place it in this directory.
   ```

## Data Preparation

To fine-tune the SloBERTa model, you need to prepare the raw text data:

1. **Prepare raw text data:**

   Place your raw text files in a directory named `data_folders`.

   ```bash
   mkdir data_folders
   # Add your raw text files to this directory.
   ```

## Synthetic Data Generation

The model training relies on a synthetic dataset generated from the raw text data. Follow these steps:

1. **Generate synthetic data:**

   Run the `prepare_train_data_BERT_model.py` script to generate the synthetic dataset:

   ```bash
   python prepare_train_data_BERT_model.py
   ```

   The synthetic dataset will be stored in a directory named `train_data`.

## Model Training

Once the synthetic dataset is ready, you can fine-tune the SloBERTa model:

1. **Train the model:**

   Run the `train_sloBERTa_model.py` script to start the training process:

   ```bash
   python train_sloBERTa_model.py
   ```

   The trained model will be saved in the specified output directory.

## Model Evaluation

After training the model, you can evaluate its performance using the provided scripts:

1. **Align model predictions:**

   First, align the model predictions with the source and target data by running the `align_file` function in `evaluate.py`:


2. **Annotate the aligned file:**

   Annotate the aligned file by marking spelling mistakes with the text `NAPAKA/Č` before each error.


3. **Evaluate the model:**

   Use the `evaluate_on_annotated_file` function to get the final score of the model:


## Acknowledgments

This project is based on the research presented in "Neural Spell-Checker: Beyond Words with Synthetic Data Generation," accepted to TSD 2024. We gratefully acknowledge the support of the research community and the contributors to the SloBERTa model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
