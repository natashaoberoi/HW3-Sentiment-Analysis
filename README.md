# RNN Assignment
Natasha Oberoi, UID: 117150537

DATA641

Description: This assignment implement and evaluate multiple Recurrent Neural Network (RNN) architectures for sentiment classification of the IMDB movie review dataset. The goal is to compare RNN, LSTM, and Bidirectional LSTM models under variations in activation functions, optimizers, sequence lengths, and gradient clipping strategies

# File structure
├── data/ 
│ └── IMDB Dataset.csv
├── results/ 
│ ├── metrics.csv 
│ └── plots/ 
│   ├── acc_f1_vs_seq_length.png
│   └──loss_curves_best_worst.png
├── NLP_HW3.ipynb  # data prep, model, runs experiments end-to-end
├── report.pdf 
├── requirements.txt 
└── README.md

# Models implemented
- Simple RNN
- LSTM (Long Short-Term Memory)
- Bidirectional LSTM

# Setup
- Python version 3.12.12 used
- Dependencies: 
```
pip install torch torchvision torchaudio nltk pandas scikit-learn tqdm tabulate matplotlib
```
- Additionally download NLTK tokenizers
```python
import nltk
nltk.download(['punkt', 'punkt_tab'])
```

# How to run
1. Open the notebook NLP_HW3.ipynb in Google Colab.
2. Upload the IMDb Dataset.csv file to the Colab environment from your local repo (third cell down).
3. Change any path names as needed for your system.
4. Run all cells sequentially

# Output files
- metrics.csv — aggregated results from all experiments with model configuration, accuracy, F1-score, and epoch time
- plots/ — visualizations for
    - Accuracy/F1 vs. Sequence Length
    - Training Loss vs. Epochs (for best and worst models)
- report.pdf

# Note
This project can be run on CPU or GPU. The results in this repository were generated using a single A100 GPU.