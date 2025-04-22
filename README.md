# Transformer-Based Text Classification

A state-of-the-art text classification system that leverages BERT models with fine-tuning for specific domains.

## Overview

This project implements a text classification system using transformer-based architectures, specifically BERT (Bidirectional Encoder Representations from Transformers). The system is designed to be adaptable to various text classification tasks through simple fine-tuning.

## Features

- Pre-trained BERT model fine-tuning for custom classification tasks
- Support for multiple classification domains
- Modular architecture that allows for easy model swapping
- Extensive evaluation metrics and visualization
- Performance optimization for inference

## Installation

```bash
git clone https://github.com/stevenelliottjr/bert-text-classifier.git
cd bert-text-classifier
pip install -r requirements.txt
```

## Usage

### Fine-tuning a classifier

```python
from text_classifier import TextClassifier

# Initialize the classifier with pre-trained BERT
classifier = TextClassifier(model_name='bert-base-uncased', num_labels=4)

# Fine-tune on your dataset
classifier.train(train_texts, train_labels, batch_size=16, epochs=3)

# Save the model
classifier.save('my_text_classifier')
```

### Making predictions

```python
# Load a trained model
classifier = TextClassifier.load('my_text_classifier')

# Make predictions
predictions = classifier.predict(["Your text to classify"])
```

## Performance

The model achieves the following performance on standard benchmarks:

| Dataset     | Accuracy | F1 Score | Precision | Recall |
|-------------|----------|----------|-----------|--------|
| SST-2       | 93.4%    | 93.2%    | 93.0%     | 93.5%  |
| IMDB        | 94.6%    | 94.5%    | 94.9%     | 94.2%  |
| AG News     | 94.8%    | 94.8%    | 94.7%     | 95.0%  |

## Demo

A live demo of this classifier is available at:
[Text Classification Demo](https://text-classifier-demo.streamlit.app)

### Running the Demo Locally

This repository contains two demo applications:

1. **Full Version (`app.py`)**: Includes the BERT model functionality, requires PyTorch and Transformers
2. **Simple Version (`demo_app.py`)**: A lightweight demo using rule-based classification, no ML dependencies

To run the simple demo:

```bash
streamlit run demo_app.py
```

To run the full BERT-based demo (requires additional dependencies):

```bash
streamlit run app.py
```

### Deploying to Streamlit Cloud

To deploy this application on Streamlit Cloud:

1. Fork or clone this repository to your GitHub account
2. Create a new app on [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect to your GitHub repository
4. Select either `app.py` (full version) or `demo_app.py` (simple version) as the main file
5. Deploy!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@software{elliott2025transformer,
  author = {Elliott, Steven},
  title = {Transformer-Based Text Classification},
  url = {https://github.com/stevenelliottjr/bert-text-classifier},
  year = {2025},
}
```