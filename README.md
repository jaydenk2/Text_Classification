# Text Classification: Quora Duplicate Question Detection

This project implements deep learning models to identify duplicate questions in the Quora Duplicate Questions dataset. It explores various architectures, including Siamese LSTMs and BERT-based classifiers, to determine semantic similarity between text pairs.

## 🚀 Models Implemented
* **Siamese LSTM:** A twin-network architecture using Long Short-Term Memory units to encode sentences into embedding vectors.
* **BERT Classifier:** Fine-tuned BERT model for sentence pair classification.

## 📂 Project Structure
```text
Text_Classification/
├── data/                  # Dataset files (ignored by Git)
├── best_model_bert.pt     # Saved model weights (ignored by Git)
├── main.py                # Main training/evaluation script
├── models.py              # Model architecture definitions
├── utils.py               # Data loading and preprocessing
└── README.md
```

## 🛠️ Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/jaydenk2/Text_Classification.git](https://github.com/jaydenk2/Text_Classification.git)
cd Text_Classification
```

**2. Install Dependencies**
Ensure you have Python 3.8+ and PyTorch installed.
```bash
pip install torch transformers pandas numpy scikit-learn
```

**3. Data Setup**
* Download the **Quora Duplicate Questions** dataset.
* Place the `.tsv` file inside the `data/` folder.
* *Note: Large data files are excluded from this repository via .gitignore.*

## 🏃‍♂️ Usage

To train the model:
```bash
python main.py --model bert --epochs 5
```

To evaluate the saved model:
```bash
python main.py --evaluate --model_path best_model_bert.pt
```

## 📊 Results
* **BERT Accuracy:** [Insert your accuracy here, e.g., 85%]
* **Siamese LSTM Accuracy:** [Insert your accuracy here]

## 📝 Notes
* The trained `best_model_bert.pt` is approximately 420MB and is not included in this repo due to GitHub size limits.
* To reproduce results, please retrain using the provided scripts.