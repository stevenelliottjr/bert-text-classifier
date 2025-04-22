import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Set page configuration
st.set_page_config(
    page_title="Text Classifier Demo",
    page_icon="📝",
    layout="wide"
)

# Predefined label mappings for different tasks
LABEL_MAPPINGS = {
    "sentiment": ["Negative", "Positive"],
    "news_category": ["World", "Sports", "Business", "Technology"],
    "emotion": ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Disgust"]
}

# Simple keyword lists for each category
KEYWORDS = {
    "sentiment": {
        0: ['bad', 'terrible', 'awful', 'worst', 'poor', 'horrible', 'hate',
            'disappointing', 'negative', 'waste', 'boring', 'disappointed',
            'useless', 'rubbish', 'mediocre', 'avoid', 'disaster'],
        1: ['good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love',
            'happy', 'positive', 'perfect', 'recommend', 'awesome', 'fantastic',
            'enjoyed', 'favorite', 'nice', 'superb', 'outstanding']
    },
    "news_category": {
        0: ['world', 'country', 'nation', 'international', 'global', 'foreign', 'president', 'minister'],
        1: ['sports', 'game', 'team', 'player', 'championship', 'win', 'score', 'match', 'tournament'],
        2: ['business', 'market', 'company', 'stock', 'economic', 'trade', 'financial', 'economy'],
        3: ['technology', 'tech', 'app', 'software', 'digital', 'internet', 'computer', 'device', 'innovation']
    },
    "emotion": {
        0: ['happy', 'joy', 'excited', 'pleased', 'delighted', 'thrilled'],
        1: ['sad', 'unhappy', 'depressed', 'miserable', 'gloomy', 'crying'],
        2: ['angry', 'furious', 'outraged', 'annoyed', 'frustrated'],
        3: ['afraid', 'scared', 'frightened', 'terrified', 'anxious'],
        4: ['surprised', 'amazed', 'astonished', 'shocked'],
        5: ['disgusted', 'revolted', 'appalled', 'nauseated']
    }
}

def simple_predict(text, task="sentiment"):
    """Simple rule-based classifier"""
    text = text.lower()
    labels = LABEL_MAPPINGS[task]
    keywords = KEYWORDS[task]
    
    # Count keywords for each category
    counts = []
    for i in range(len(labels)):
        count = sum(1 for word in keywords.get(i, []) if re.search(r'\b' + word + r'\b', text))
        counts.append(count)
    
    # Calculate probabilities
    total = sum(counts)
    if total == 0:
        # No keywords found, return uniform distribution
        probs = np.ones(len(labels)) / len(labels)
    else:
        probs = np.array(counts) / total
        
    # Add some randomness for more realistic probabilities
    probs = probs + np.random.uniform(-0.1, 0.1, size=len(labels))
    probs = np.clip(probs, 0.1, 0.9)  # Ensure values are between 0.1 and 0.9
    probs = probs / np.sum(probs)     # Normalize to sum to 1
        
    return probs

def plot_probabilities(probs, labels):
    """Create a bar chart of prediction probabilities"""
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))
    
    bars = ax.barh(y_pos, probs, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Probability')
    ax.set_title('Prediction Probabilities')
    
    # Add probability values as text
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(
            prob + 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{prob:.4f}',
            va='center'
        )
    
    # Highlight the highest probability bar
    bars[np.argmax(probs)].set_color('orange')
    
    st.pyplot(fig)

def main():
    # App title and description
    st.title("Text Classification Demo")
    
    st.write("""
    This interactive demo showcases a simple text classifier. Type or paste text below to 
    classify it using different models.
    
    **Note:** This is a simplified demonstration using keyword matching rather than a full machine learning model.
    """)
    
    # Sidebar for task selection
    st.sidebar.header("Configuration")
    selected_task = st.sidebar.selectbox(
        "Select Classification Task",
        ["sentiment", "news_category", "emotion"]
    )
    
    # Get the labels for the selected task
    current_labels = LABEL_MAPPINGS[selected_task]
    
    # Create two columns for input and visualization
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Input Text")
        
        # Provide some example texts to help users
        example_texts = {
            "sentiment": [
                "I absolutely loved this movie, the acting was superb!",
                "This restaurant was terrible, I'll never go back again."
            ],
            "news_category": [
                "The stock market fell 5% today due to concerns about inflation.",
                "The team scored in the final minutes to win the championship."
            ],
            "emotion": [
                "I just won the lottery, I can't believe it!",
                "I miss my family so much since I moved away."
            ]
        }
        
        selected_example = st.selectbox(
            "Try an example or enter your own text below:",
            [""] + example_texts[selected_task]
        )
        
        # Text input area
        if selected_example:
            text_input = st.text_area("", selected_example, height=200)
        else:
            text_input = st.text_area("Enter text to classify:", height=200)
            
        # Only show prediction button if there's text
        if text_input:
            predict_button = st.button("Classify Text")
        else:
            predict_button = False
            st.info("Enter some text to get a prediction.")
    
    with col2:
        st.subheader("Prediction Results")
        
        if text_input and predict_button:
            # Display a spinner while classifying
            with st.spinner("Classifying..."):
                # Get prediction probabilities
                probs = simple_predict(text_input, selected_task)
                
                # Display the highest confidence prediction
                highest_prob_idx = np.argmax(probs)
                st.success(f"Predicted Class: **{current_labels[highest_prob_idx]}**")
                
                # Display the confidence
                confidence = probs[highest_prob_idx]
                st.metric("Confidence", f"{confidence:.2%}")
                
                # Plot all probabilities
                st.subheader("Probability Distribution")
                plot_probabilities(probs, current_labels)
    
    # Show additional information and features
    st.markdown("---")
    
    # Add tabs for different sections
    tab1, tab2, tab3 = st.tabs(["How it Works", "About", "GitHub"])
    
    with tab1:
        st.header("How Text Classification Works")
        st.write("""
        Text classification is the process of categorizing text into organized groups. In a real-world 
        application, this would use advanced machine learning techniques like:
        
        1. **Preprocessing**: Clean and normalize text data
        2. **Feature Extraction**: Convert text into numerical features
        3. **Model Training**: Train a classifier on labeled examples
        4. **Prediction**: Apply the model to new text inputs
        
        This demo uses a simplified approach based on keyword matching to demonstrate the concept.
        """)
        
        st.subheader("Keywords Used for Classification")
        task_keywords = KEYWORDS[selected_task]
        
        for i, label in enumerate(current_labels):
            keywords = ", ".join(task_keywords.get(i, []))
            st.write(f"**{label}**: {keywords}")
    
    with tab2:
        st.header("About This Project")
        st.write("""
        This project was developed to demonstrate text classification concepts.
        
        **Author**: [Steven Elliott Jr.](https://www.linkedin.com/in/steven-elliott-jr)
        
        **License**: MIT License
        """)
        
    with tab3:
        st.header("GitHub Repository")
        st.write("""
        The complete source code for this project, including a more advanced version using BERT models, 
        is available on GitHub:
        
        [GitHub Repository](https://github.com/stevenelliottjr/bert-text-classifier)
        
        Feel free to star the repository if you find it useful!
        """)

if __name__ == "__main__":
    main()