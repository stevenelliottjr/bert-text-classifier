"""
A simplified text classifier for demo purposes.
This is used as a fallback if the main model fails to load.
"""

import numpy as np
import re

class SimpleTextClassifier:
    """A very simple rule-based classifier for demo purposes."""
    
    def __init__(self, num_labels=2):
        self.num_labels = num_labels
        
        # Simple sentiment keywords
        self.positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love',
            'happy', 'positive', 'perfect', 'recommend', 'awesome', 'fantastic',
            'enjoyed', 'favorite', 'nice', 'superb', 'outstanding'
        ]
        
        self.negative_words = [
            'bad', 'terrible', 'awful', 'worst', 'poor', 'horrible', 'hate',
            'disappointing', 'negative', 'waste', 'boring', 'disappointed',
            'useless', 'rubbish', 'mediocre', 'avoid', 'disaster'
        ]
        
        # News categories keywords
        self.category_words = {
            0: ['world', 'country', 'nation', 'international', 'global', 'foreign', 'president', 'minister'],
            1: ['sports', 'game', 'team', 'player', 'championship', 'win', 'score', 'match', 'tournament'],
            2: ['business', 'market', 'company', 'stock', 'economic', 'trade', 'financial', 'economy'],
            3: ['technology', 'tech', 'app', 'software', 'digital', 'internet', 'computer', 'device', 'innovation']
        }
        
        # Emotion keywords
        self.emotion_words = {
            0: ['happy', 'joy', 'excited', 'pleased', 'delighted', 'thrilled'],
            1: ['sad', 'unhappy', 'depressed', 'miserable', 'gloomy', 'crying'],
            2: ['angry', 'furious', 'outraged', 'annoyed', 'frustrated'],
            3: ['afraid', 'scared', 'frightened', 'terrified', 'anxious'],
            4: ['surprised', 'amazed', 'astonished', 'shocked'],
            5: ['disgusted', 'revolted', 'appalled', 'nauseated']
        }
    
    def predict(self, texts):
        """
        Make predictions on a list of texts.
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            numpy array of probabilities for each class
        """
        results = []
        
        for text in texts:
            if self.num_labels == 2:  # Sentiment
                result = self._predict_sentiment(text)
            elif self.num_labels == 4:  # News categories
                result = self._predict_category(text)
            elif self.num_labels == 6:  # Emotions
                result = self._predict_emotion(text)
            else:
                # Default to random probabilities
                result = np.random.random(self.num_labels)
                result = result / np.sum(result)  # Normalize
            
            results.append(result)
        
        return np.array(results)
    
    def _predict_sentiment(self, text):
        """Simple rule-based sentiment analysis."""
        text = text.lower()
        
        # Count positive and negative words
        pos_count = sum(1 for word in self.positive_words if re.search(r'\b' + word + r'\b', text))
        neg_count = sum(1 for word in self.negative_words if re.search(r'\b' + word + r'\b', text))
        
        # Calculate probabilities
        total = pos_count + neg_count
        if total == 0:
            # No sentiment words found, default to neutral
            probs = np.array([0.5, 0.5])
        else:
            neg_prob = neg_count / total
            pos_prob = pos_count / total
            probs = np.array([neg_prob, pos_prob])
        
        # Add some randomness
        probs = probs + np.random.uniform(-0.1, 0.1, size=2)
        
        # Ensure probs are valid
        probs = np.clip(probs, 0.1, 0.9)
        probs = probs / np.sum(probs)
        
        return probs
    
    def _predict_category(self, text):
        """Simple rule-based news category classification."""
        text = text.lower()
        
        # Count category words
        counts = []
        for category in range(4):
            count = sum(1 for word in self.category_words[category] if re.search(r'\b' + word + r'\b', text))
            counts.append(count)
        
        # Calculate probabilities
        total = sum(counts)
        if total == 0:
            # No category words found, default to uniform
            probs = np.array([0.25, 0.25, 0.25, 0.25])
        else:
            probs = np.array(counts) / total
        
        # Add some randomness
        probs = probs + np.random.uniform(-0.1, 0.1, size=4)
        
        # Ensure probs are valid
        probs = np.clip(probs, 0.1, 0.9)
        probs = probs / np.sum(probs)
        
        return probs
    
    def _predict_emotion(self, text):
        """Simple rule-based emotion classification."""
        text = text.lower()
        
        # Count emotion words
        counts = []
        for emotion in range(6):
            count = sum(1 for word in self.emotion_words[emotion] if re.search(r'\b' + word + r'\b', text))
            counts.append(count)
        
        # Calculate probabilities
        total = sum(counts)
        if total == 0:
            # No emotion words found, default to uniform
            probs = np.array([1/6] * 6)
        else:
            probs = np.array(counts) / total
        
        # Add some randomness
        probs = probs + np.random.uniform(-0.1, 0.1, size=6)
        
        # Ensure probs are valid
        probs = np.clip(probs, 0.1, 0.9)
        probs = probs / np.sum(probs)
        
        return probs
    
    def predict_classes(self, texts):
        """
        Predict the most likely class for each text.
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            numpy array of class indices
        """
        probs = self.predict(texts)
        return np.argmax(probs, axis=1)