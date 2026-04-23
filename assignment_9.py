#import libraries
import nltk
from nltk.tokenize import word_tokenize  
from nltk.corpus import stopwords          
from nltk.stem import PorterStemmer, WordNetLemmatizer  
from nltk.probability import FreqDist           
from nltk import pos_tag                        


# These datasets are required for tokenization, tagging, and lemmatization
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4') 

#load text files
def load_text(file_path):
    # Opens a text file and returns its contents as a string
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Load the first three texts
text1 = load_text('Text_1.txt')
text2 = load_text('Text_2.txt')
text3 = load_text('Text_3.txt')

# ====================================================================
#Part 1

#nlp processing function
def process_text(text):
    # Convert text to lowercase and tokenize into words
    tokens = word_tokenize(text.lower())

    # Keep only alphabetic words
    tokens = [word for word in tokens if word.isalpha()]

    # Remove common stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Apply stemming (reduces words to root form)
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in tokens]

    # Apply lemmatization (more accurate base form)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]

    # Count frequency of words and get top 20
    freq_dist = FreqDist(lemmatized)
    top_20 = freq_dist.most_common(20)

    return tokens, stemmed, lemmatized, top_20

#name entities
def count_named_entities(text):
    # Tokenize and tag each word with its part of speech
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    # Extract proper nouns
    entities = [word for word, tag in pos_tags if tag in ['NNP', 'NNPS']]

    return entities, len(entities)

#analysis
texts = {
    "Text 1": text1,
    "Text 2": text2,
    "Text 3": text3
}

for name, text in texts.items():
    print(f"\n================ {name} ================\n")

    # Process text and get top words
    tokens, stemmed, lemmatized, top_20 = process_text(text)

    # Print most common words
    if len(top_20) == 0:
        print("⚠️ No tokens found — check preprocessing")
    else:
        print("Top 20 Tokens:")
        for word, freq in top_20:
            print(f"{word}: {freq}")

    # Get named entities (proper nouns)
    entities, count = count_named_entities(text)

    print("\nNamed Entities Found:")
    print(set(entities))  # show unique entities

    print(f"\nTotal Named Entities: {count}")

# ====================================================================
#part 2



from nltk.util import ngrams       # used to generate n-grams
from collections import Counter   # counts frequency

# Load all four texts (including Text 4)
text1 = load_text('Text_1.txt')
text2 = load_text('Text_2.txt')
text3 = load_text('Text_3.txt')
text4 = load_text('Text_4.txt')

#preprocess for n-grams
def preprocess(text):
    # Tokenize and lowercase
    tokens = word_tokenize(text.lower())
    
    # Keep only words
    tokens = [word for word in tokens if word.isalpha()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

#get top trigrams
def get_top_trigrams(text, n=3, top_k=20):
    # Clean and tokenize text
    tokens = preprocess(text)
    
    # Create list of trigrams
    trigram_list = list(ngrams(tokens, n))
    
    # Count frequency of each trigram
    freq = Counter(trigram_list)
    
    # Return most common trigrams
    return freq.most_common(top_k)


#run analysis on all text
texts = {
    "Text 1": text1,
    "Text 2": text2,
    "Text 3": text3,
    "Text 4": text4
}

trigram_results = {}

for name, text in texts.items():
    print(f"\n================ {name} ================\n")
    
    # Get top trigrams
    top_trigrams = get_top_trigrams(text)
    trigram_results[name] = top_trigrams
    
    # Print results
    for tri, count in top_trigrams:
        print(f"{tri}: {count}")


#compare overlap
def trigram_overlap(trigrams1, trigrams2):
    # Convert trigram lists to sets for comparison
    set1 = set([t[0] for t in trigrams1])
    set2 = set([t[0] for t in trigrams2])
    
    # Return number of shared trigrams
    return len(set1.intersection(set2))

print("\n====== TRIGRAM OVERLAP WITH TEXT 4 ======\n")

# Compare Text 4 with the first three texts
for name in ["Text 1", "Text 2", "Text 3"]:
    overlap = trigram_overlap(trigram_results[name], trigram_results["Text 4"])
    print(f"{name} vs Text 4 overlap: {overlap}")