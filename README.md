# Assignment-9

Purpose: This project uses Natural Language Processing (NLP) in Python to analyze four texts (Text_1–Text_4). The goal is to extract meaning from unstructured text and compare writing style and subject matter using techniques like tokenization, stemming, lemmatization, named entity recognition, and n-gram analysis.

Functions:
load_text(file_path) – Loads text from a file
preprocess(text) – Cleans and tokenizes text
process_text(text) – Performs stemming, lemmatization, and finds top words
count_named_entities(text) – Finds named entities using POS tagging (NNP/NNPS)
get_top_trigrams(text) – Returns most common trigrams
trigram_overlap(t1, t2) – Compares trigram similarity between texts

Results
Texts 1–3 share key tokens and entities, showing they have the same subject.
Trigram analysis shows low overlap overall due to strict matching, but Text_4 is most similar in tone to darker texts like Text_1.

Limitations
Trigrams require exact matches and have low overlap
Stopword removal can break phrases
Text_4 is longer
