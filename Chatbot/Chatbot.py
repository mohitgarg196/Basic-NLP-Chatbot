import nltk     #nlp library for tokenization, lemmatization.
import numpy as np      #for numerical operations
import random       # to generate random responses for greetings
import string  # for text preprocessing
import warnings # to supress unnecessary warnings

from nltk.stem import WordNetLemmatizer  # Converts words to their base forms.
from sklearn.feature_extraction.text import TfidfVectorizer     # Converts text into numerical features for comparison
from sklearn.metrics.pairwise import cosine_similarity      # Measures similarity between TF-IDF vectors.
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS      # ENGLISH_STOP_WORDS is a predefined list in the Scikit-learn library that contains common English words, such as "the," "is," "and," "in," etc., which are typically not useful for analyzing the meaning of a sentence. These words are called "stop words".

# Download required NLTK packages
# nltk.download('punkt')  # Tokenizer
# nltk.download('wordnet')  # Lemmatizer
# nltk.download('averaged_perceptron_tagger')  # POS Tagging
# nltk.download('maxent_ne_chunker')  # NER
# nltk.download('words')  # Word List for NER

with open('Chatbot.txt', 'r', errors='ignore') as f:                # Chatbot.txt containing the chatbot's knowledge base
    corpus = f.read()

# Tokenization
sent_tokens = nltk.sent_tokenize(corpus)  # Convert to list of sentences
word_tokens = nltk.word_tokenize(corpus)  # Convert to list of words

# Lemmatizer
lemmatizer = WordNetLemmatizer()        # Reduces words to their base forms (e.g., "running" → "run").

def LemTokens(tokens):              #Lemmatizing a list of tokens means converting a list of words (tokens) into their base or dictionary form (known as the lemma). Eg.:"running," "runs," and "ran" are converted to their base form, "run". "runner" remains the same since it is already in its base form.
    """Lemmatize tokens."""
    return [lemmatizer.lemmatize(token) for token in tokens]


def LemNormalize(text):
    """Normalize text by removing punctuation, converting to lowercase, and lemmatizing."""
    remove_punct = dict((ord(punct), None) for punct in string.punctuation)         # removing punctuation
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct)))      # converting into lowercase, lemmatizing tokens

# Preprocess the stop words
custom_stop_words = [lemmatizer.lemmatize(word) for word in ENGLISH_STOP_WORDS]     # Stop Words: Common words (e.g., "is", "the") are excluded as they don’t add much meaning.

# Define TfidfVectorizer with custom stop words
TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=custom_stop_words)    # TF-IDF (Term Frequency-Inverse Document Frequency): Weighs words based on their importance in the text.

# Suppress warnings
warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None")
TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')

def response(user_input):           # Purpose: Matches user input with the most similar sentence in the corpus. 
    # Temporarily adds the user input to the corpus, computes similarity, and retrieves the most relevant response.
    all_sentences = sent_tokens + [user_input]  # Include user input temporarily
    tfidf = TfidfVec.fit_transform(all_sentences)   # Create TF-IDF vectors
    similarity_scores = cosine_similarity(tfidf[-1], tfidf[:-1])  # Compare user_input with existing sentences -1 means last sentence i.e. user input because all_sentences = sent_tokens + user_input, we added user_input at last. :-1 means corpus text because :-1 means all before -1(last one).
    idx = similarity_scores.argsort()[0][-1]        # argsort will sort the simiaroty matrix in ascending order and then [0] : first row in ascending order and then [-1] : last element of that forst row. So, that is the index of most similar sentence to the user input.
    flat = similarity_scores.flatten()      # will flat 2d matrix into 1d list
    flat.sort()     # sort that 1d list
    req_tfidf = flat[-1]        # highest similarity score

    if req_tfidf < 0.1:  # Lower threshold to consider a response, if similarity is lower than 0.1 will return this generic response.
        return "I'm sorry, I didn't understand that. Could you please rephrase?"
    else:
        return sent_tokens[idx]         # else return the most similar sentence

    
# Greeting responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hello", "hi", "hey there", "hi! How can I assist you?"]


def greeting(sentence):
    """Check if the user's input is a greeting and return an appropriate response."""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)        # return random response from greetings output list


# Main chatbot loop
print("Chatbot: Hi! I'm your NLP chatbot. Type 'bye' to exit.")

while True:
    user_input = input("You: ").strip()     # .strip(): This removes any leading and trailing whitespace characters (spaces, tabs, or newlines) from the input string.
    if user_input.lower() == 'bye':     # End loop if user says "bye"
        print("Chatbot: Goodbye! Have a great day!")
        break
    elif greeting(user_input) is not None:      # check if it is greeting
        print(f"Chatbot: {greeting(user_input)}")           # f"...": The f before the string tells Python to evaluate any expressions inside {} and insert the result directly into the string.
    else:       # else generate response for non-greeting input
        print(f"Chatbot: {response(user_input)}")
