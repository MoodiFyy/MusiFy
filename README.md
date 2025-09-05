
# MusiFy - Music Genre Classification with CNN (GTZAN Dataset)

This project implements a **Convolutional Neural Network (CNN)** for automatic **music genre classification** using the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

The pipeline covers everything from **data preprocessing** and **feature extraction** to **training, evaluation, and results visualization**.

---

## Project Structure

```
├── data/
│   └── gtzan/           # Dataset (GTZAN)
├── models/              # Saved trained models
├── results/             # Evaluation outputs (metrics, confusion matrix, etc.)
├── notebooks/           # Jupyter notebooks for exploration
├── scripts/             # Helper scripts
├── main.py              # Main training & evaluation script
└── README.md            # Project documentation
```

---

## Features

* **Audio Preprocessing**

  * Silence trimming
  * Noise reduction (optional)
  * Normalization
* **Feature Extraction**

  * Log-mel spectrograms
* **CNN Architecture**

  * Multiple convolutional + pooling layers
  * Batch normalization & dropout
  * Dense layers with softmax for classification
* **Training**

  * Stratified train-test split
  * Early stopping & learning rate scheduling
  * Model checkpointing (`models/best_model.h5`)
* **Evaluation**

  * Accuracy, classification report, confusion matrix

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/music-genre-classification.git
cd music-genre-classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

You can download the GTZAN dataset from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

Place it inside:

```
data/gtzan/
```

### 4. Run Training

```bash
python main.py
```

---

## Results

* Model achieves **\~85–90% accuracy** on the test split (varies slightly depending on random seed and noise reduction).
* Confusion matrix and classification reports are printed after evaluation.
* Best model is saved to `models/best_model.h5`.

---

## Requirements

* Python 3.8+
* [TensorFlow](https://www.tensorflow.org/) / Keras
* [Librosa](https://librosa.org/) for audio processing
* Scikit-learn
* tqdm
* (Optional) [noisereduce](https://pypi.org/project/noisereduce/)

Install them via:

```bash
pip install tensorflow librosa scikit-learn tqdm noisereduce
```

---

## References

* [GTZAN Dataset on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
* [Librosa Documentation](https://librosa.org/doc/latest/index.html)
* [Keras CNN Documentation](https://keras.io/api/layers/convolution_layers/convolution2d/)

