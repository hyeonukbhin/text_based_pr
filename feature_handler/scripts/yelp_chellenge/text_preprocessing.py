# -*- coding: utf-8 -*-

import os
import codecs
import json
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('white')
sns.set_context('notebook', font_scale=1.5)

import spacy

# spacy.en.language
# %matplotlib inline
import matplotlib.pyplot as plt

data_directory = os.path.join('/home/kist/yelp_13')
businesses_filepath = os.path.join(data_directory, 'business.json')
review_json_filepath = os.path.join(data_directory, 'review.json')
intermediate_directory = os.path.join(data_directory, 'intermediate')
review_txt_filepath = os.path.join(intermediate_directory, 'review_text_all.txt')

healthcare_ids = []

# open the businesses file
with codecs.open(businesses_filepath, encoding='utf_8') as f:
    # iterate through each line (json record) in the file
    for business_json in f:

        # convert the json record to a Python dict
        business = json.loads(business_json)

        # if this business has no categories or is not a target entity, skip to the next one
        if business[u'categories'] is None or u'Health & Medical' not in business[u'categories']:
            continue
        # Remove businesses in BW, Germany
        if u'BW' in business[u'state']:
            continue
        # Remove businesses that are restaurants, food and pets
        if u'Restaurants' in business[u'categories'] or u'Food' in business[u'categories'] or 'Pets' in business[
            u'categories']:
            continue

        # add the business id to our healthcare_ids set
        healthcare_ids.append(business[u'business_id'])

# Turn the list of ids into a set, which is faster for testing whether an element is in the set
healthcare_ids = set(healthcare_ids)

# print the number of unique ids in the dataset
print('{:,}'.format(len(healthcare_ids)), u'health & medical entities in the dataset.')

# Create a new file that contains only the text from reviews about healthcare entities.
# One review per line in the this new file.

review_count = 0
useful = []

# create & open a new file in write mode
with codecs.open(review_txt_filepath, 'w', encoding='utf_8') as review_txt_file:
    # open the existing review json file
    with codecs.open(review_json_filepath, encoding='utf_8') as review_json_file:

        # loop through all reviews in the existing file and convert to dict
        for review_json in review_json_file:
            review = json.loads(review_json)

            # if this review is not in the target set, skip to the next one
            if review[u'business_id'] not in healthcare_ids:
                continue

            # write each review as a line in the new file
            # escape newline characters in the original review text
            if review[u'text'] is None:
                print(review_count)

            review_txt_file.write(review[u'text'].replace('\n', '\\n').replace('\r', '') + '\n')
            review_count += 1
            useful.append(review[u'useful'])

print(u'Text from {:,} healthcare reviews written to the new txt file.'.format(review_count))

# Create a new file that contains only the text from reviews about healthcare entities.
# One review per line in the this new file.

useful = np.array(useful)
luseful = np.log(useful + 1)
df = pd.DataFrame(useful, columns=['useful'])
df.to_csv(os.path.join(intermediate_directory, 'useful.csv'))
print(df)

unique, counts = np.unique(useful, return_counts=True)
# print(unique)
# print(counts)
print(len(unique), len(counts))
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f.set_size_inches(12, 5)
ax1.bar(unique, counts)
ax1.set_title('Distribution of Healthcare Review Usefulness')
ax1.set_xlabel('Useful Votes')
ax1.set_ylabel('Number of Reviews')
ax2.set_title('Distribution in Natural Log')
ax2.hist(luseful, bins=10, width=0.3, align='mid')
ax2.set_xlabel('Log(Useful Votes + 1)')
f.savefig('useful_votes.png', dpi=100)
# plt.show()

import spacy
import pandas as pd
import itertools as it

nlp = spacy.load('en')

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

unigram_sentences_filepath = os.path.join(intermediate_directory, 'unigram_sentences_all.txt')
bigram_model_filepath = os.path.join(intermediate_directory, 'bigram_model_all')
bigram_sentences_filepath = os.path.join(intermediate_directory, 'bigram_sentences_all.txt')
trigram_model_filepath = os.path.join(intermediate_directory, 'trigram_model_all')
trigram_sentences_filepath = os.path.join(intermediate_directory, 'trigram_sentences_all.txt')
trigram_reviews_filepath = os.path.join(intermediate_directory, 'trigram_transformed_reviews_all.txt')


def punct_space(token):
    """Eliminate tokens that are pure punctuation or white space"""

    return token.is_punct or token.is_space


def person(token):
    """Remove tokens that are PERSON entities"""

    return token.ent_type_ == 'PERSON'


def line_review(filename):
    """Generator function (iterator without storing all texts)
    to read in reviews from file and return the original line breaks"""

    with codecs.open(filename, encoding='utf_8') as f:
        for review in f:
            yield review.replace('\\n', '\n')


def lemmatized_sentence_corpus(filename):
    """Generator function to use spaCy to parse reviews, lemmatize the text and yield sentences"""

    for parsed_review in nlp.pipe(line_review(filename), batch_size=10000, n_threads=4):
        for sent in parsed_review.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not (punct_space(token) or person(token))])


# Segment reviews into sentences and normalize the text
# Save the parsed sentences file on disk to avoid storing the entire corpus in RAM
with codecs.open(unigram_sentences_filepath, 'w', encoding='utf_8') as f:
    for sentence in lemmatized_sentence_corpus(review_txt_filepath):
        f.write(sentence + '\n')

# gensim's LineSentence class takes the format: one sentence = one line
# words are preprocessed and separated by whitespace.
unigram_sentences = LineSentence(unigram_sentences_filepath)

# Run a phrase model to link two-words phrases together
bigram_model = Phrases(unigram_sentences)
bigram_model.save(bigram_model_filepath)
bigram_model = Phrases.load(bigram_model_filepath)

# Apply the bigram model to unigram sentences and create a text with bigram sentences
with codecs.open(bigram_sentences_filepath, 'w', encoding='utf-8') as f:
    for unigram_sentence in unigram_sentences:
        bigram_sentence = u' '.join(bigram_model[unigram_sentence])
        f.write(bigram_sentence + '\n')

bigram_sentences = LineSentence(bigram_sentences_filepath)

trigram_model = Phrases(bigram_sentences)
trigram_model.save(trigram_model_filepath)
trigram_model = Phrases.load(trigram_model_filepath)

trigram_sentences = LineSentence(trigram_sentences_filepath)

with codecs.open(trigram_sentences_filepath, 'w', encoding='utf_8') as f:
    for bigram_sentence in bigram_sentences:
        trigram_sentence = u' '.join(trigram_model[bigram_sentence])
        f.write(trigram_sentence + '\n')

# Write a transformed text into a new file, with one review per line
with codecs.open(trigram_reviews_filepath, 'w', encoding='utf-8') as f:
    for parsed_review in nlp.pipe(line_review(review_txt_filepath), batch_size=10000, n_threads=4):
        # Lemmatize the text, removing punctuation and whitespace
        unigram_review = [token.lemma_ for token in parsed_review
                          if not (punct_space(token) or person(token))]

        # Apply the first-order and second-order phrase models
        bigram_review = bigram_model[unigram_review]
        trigram_review = trigram_model[bigram_review]

        #         print(trigram_review[:50])
        # Remove any remaining stopwords
        #         trigram_review = [term for term in trigram_review if term not in spacy.en.language_data.STOP_WORDS]
        #         trigram_review = [term for term in trigram_review if term not in spacy.en.word_sets.stop_words]
        trigram_review = [term for term in trigram_review if term not in nlp.Defaults.stop_words]

        # Write the transformed review as a new line
        trigram_review = u' '.join(trigram_review)
        f.write(trigram_review + '\n')

# Gensim's Doc2Vec class creates vector representations for entire documents
# The input object is an iterator of LineSentence objects
# The default dm=1 refers to the distributed memory algorithm
# The algorithm runs through sentences twice: (1) build the vocab,
# (2) learn a vector representation for each word and for each label (sentence)
# Better results can be achieved by iterating over the data several times

from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

doc2vec_filepath = os.path.join(intermediate_directory, 'doc2vec_model')

X = []
with codecs.open(trigram_reviews_filepath, encoding='utf-8') as f:
    for review in f:
        X.append(review)

labels = ['Review_' + str(x) for x in range(len(X))]


class DocIterator():
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(words=doc.split(), tags=[self.labels_list[idx]])


labeled_reviews = DocIterator(X, labels)

# To produce better results, iterate over the data 10 times
# and control the learning rate for each iteration
doc2vec = Doc2Vec(size=100, window=5, min_count=50, workers=4, iter=10)
doc2vec.build_vocab(labeled_reviews)

doc2vec.train(labeled_reviews, total_examples=doc2vec.corpus_count, epochs=10)

doc2vec.save(doc2vec_filepath)
doc2vec.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

doc2vec = Doc2Vec.load(doc2vec_filepath)

print(doc2vec.most_similar(positive=["pediatrician"], topn=5))

print(doc2vec.most_similar(positive=['pediatrician', 'tooth'], negative=['baby'], topn=5))

print(doc2vec.docvecs.most_similar('Review_10', topn=5))
