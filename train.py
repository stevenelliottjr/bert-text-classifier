import argparse
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from text_classifier import TextClassifier
import matplotlib.pyplot as plt

def train_model(dataset_path, model_name, num_labels, output_dir, epochs, batch_size):
    """
    Train a BERT-based text classifier on the specified dataset.
    """
    print(f"Loading dataset from {dataset_path}")
    
    # Load dataset (handle different formats)
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith('.json'):
        df = pd.read_json(dataset_path, lines=True)
    else:
        raise ValueError("Unsupported dataset format. Please use CSV or JSON.")
    
    # Check for required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    
    # Split the dataset
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    
    # Initialize the classifier
    classifier = TextClassifier(model_name=model_name, num_labels=num_labels)
    
    # Train the model
    print(f"Training model for {epochs} epochs with batch size {batch_size}")
    history = classifier.train(
        train_texts=train_df['text'].tolist(),
        train_labels=train_df['label'].tolist(),
        val_texts=val_df['text'].tolist(),
        val_labels=val_df['label'].tolist(),
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Save the model
    print(f"Saving model to {output_dir}")
    classifier.save(output_dir)
    
    # Save training history
    with open(f"{output_dir}/training_history.json", 'w') as f:
        json.dump(history, f)
    
    # Plot training history if validation was used
    if 'val_accuracy' in history:
        plot_history(history, output_dir)
    
    # Final evaluation on validation set
    val_dataset = val_df['text'].tolist()
    val_labels = val_df['label'].tolist()
    
    print("Evaluating model on validation set...")
    probs = classifier.predict(val_dataset)
    preds = np.argmax(probs, axis=1)
    accuracy = np.mean(preds == val_labels)
    
    print(f"Validation accuracy: {accuracy:.4f}")
    
    return classifier

def plot_history(history, output_dir):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BERT-based text classifier")
    
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset file (CSV or JSON)")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="Pre-trained model to use")
    parser.add_argument("--num-labels", type=int, required=True,
                        help="Number of labels/classes")
    parser.add_argument("--output-dir", type=str, default="./trained_model",
                        help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size")
    
    args = parser.parse_args()
    
    train_model(
        dataset_path=args.dataset,
        model_name=args.model,
        num_labels=args.num_labels,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )