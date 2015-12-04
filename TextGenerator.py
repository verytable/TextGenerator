# -*- coding: utf-8 -*-
from os import walk, path
import re
from collections import Counter
from itertools import tee, izip, groupby, imap, islice
import numpy as np
import pickle
import argparse


def nwise(iterable, n=2):
    iters = tee(iterable, n)
    for i, it in enumerate(iters):
        next(islice(it, i, i), None)
    return izip(*iters)


def count_next_word_frequences(ngram_counts):
    next_word_frequences = {}
    ngram_counts_sorted_by_first = \
        sorted(ngram_counts.items(), key=lambda tup: tup[0][:-1])
    ngram_groups = groupby(
        ngram_counts_sorted_by_first, key=lambda tup: tup[0][:-1])
    for words, group in ngram_groups:
        it1, it2 = tee(group, 2)
        sum_counts = sum(imap(lambda bc: bc[1], it1))
        next_word_frequences[words] = \
            dict(imap(lambda bc: (bc[0][-1], bc[1] * 1.0 / sum_counts), it2))
    return next_word_frequences


def get_counts(corpus_dir):
    unigram_counts = Counter()
    bigram_counts = Counter()
    trigram_counts = Counter()
    for root, dirs, files in walk(corpus_dir, topdown=False):
        for name in files:
            with open(path.join(root, name), 'r') as input_file:
                text = input_file.read().replace("\xe2\x80\x99", "'")
                words_and_punctuation = re.compile(
                    "\w+(?:[-‘’']\w+)*|,|:|\.|\?|!").findall(text)
                unigram_counts.update(Counter(nwise(words_and_punctuation, 1)))
                lower_words_and_punctuation = \
                    [word.lower() for word in words_and_punctuation]
                bigram_counts.update(
                    Counter(nwise(lower_words_and_punctuation, 2)))
                trigram_counts.update(
                    Counter(nwise(lower_words_and_punctuation, 3)))
    return unigram_counts, bigram_counts, trigram_counts


def get_capitalized_words(unigram_counts):
    always_capitalized = []
    for (word,), count in unigram_counts.items():
        if not word.islower():
            if not (word.lower(), ) in unigram_counts or \
               unigram_counts[word.lower(), ] * 9 < count:
                always_capitalized.append(word.lower())
    return set(always_capitalized)


def count_and_save_statistics(corpus_dir):
    unigram_counts, bigram_counts, trigram_counts = get_counts(corpus_dir)

    unigram_frequences = count_next_word_frequences(unigram_counts)
    bigram_frequences = count_next_word_frequences(bigram_counts)
    trigram_frequences = count_next_word_frequences(trigram_counts)

    capitalized_words = get_capitalized_words(unigram_counts)

    with open('unigram_frequences.pickle', 'w+') as output_file:
        pickle.dump(unigram_frequences, output_file)

    with open('bigram_frequences.pickle', 'w+') as output_file:
        pickle.dump(bigram_frequences, output_file)

    with open('trigram_frequences.pickle', 'w+') as output_file:
        pickle.dump(trigram_frequences, output_file)

    with open('capitalized_words.pickle', 'w+') as output_file:
        pickle.dump(capitalized_words, output_file)


def upper_first(word):
    return word[0].upper() + word[1:] if len(word) > 0 else word


def beutify(raw_text, min_sentence_length, paragraph_loc, paragraph_scale):
    with open('capitalized_words.pickle', 'r') as input_file:
        capitalized_words = pickle.load(input_file)

    raw_text = [
        word.capitalize() if word in capitalized_words else word
        for word in raw_text
    ]

    sentence_endings = [
        (index, value) for index, value in enumerate(raw_text)
        if value in ['.', '!', '?']
    ]
    sentences_in_current_paragraph_left = \
        int(np.random.normal(paragraph_loc, paragraph_scale))
    last_sentence_ending_index = -1

    paragraph = ['\t']
    paragraphs = []
    for next_sentence_ending in sentence_endings:
        sentence_length = \
            next_sentence_ending[0] - last_sentence_ending_index - 1
        if sentence_length > min_sentence_length:
            sentence = upper_first(
                ' '.join(raw_text[
                    last_sentence_ending_index+1:next_sentence_ending[0]
                ])
            ) + next_sentence_ending[1]
            sentences_in_current_paragraph_left -= 1
            paragraph.append(sentence)
            if sentences_in_current_paragraph_left == 0:
                sentences_in_current_paragraph_left = \
                    int(np.random.normal(paragraph_loc, paragraph_scale))
                paragraphs.append(" ".join(paragraph))
                paragraph = ['\t']
        last_sentence_ending_index = next_sentence_ending[0]
    return re.sub(" ([.,:])", "\1", "\n".join(paragraphs))


def generate_raw_text(words_number):
    raw_text = []

    with open('unigram_frequences.pickle', 'r') as input_file:
        unigram_frequences = pickle.load(input_file)

    first_words, first_words_frequences = zip(*unigram_frequences[()].items())
    first_word = np.random.choice(a=first_words, p=first_words_frequences)
    raw_text.append(first_word)

    with open('bigram_frequences.pickle', 'r') as input_file:
        bigram_frequences = pickle.load(input_file)

    second_words, second_words_frequences = \
        zip(*bigram_frequences[first_word, ].items())
    second_word = np.random.choice(a=second_words, p=second_words_frequences)
    raw_text.append(second_word)

    with open('trigram_frequences.pickle', 'r') as input_file:
        trigram_frequences = pickle.load(input_file)

    last_but_one_word, last_word = first_word, second_word
    for i in range(2, words_number):
        next_words, next_words_frequences = \
            zip(*trigram_frequences[last_but_one_word, last_word, ].items())
        next_word = np.random.choice(a=next_words, p=next_words_frequences)
        raw_text.append(next_word)
        last_but_one_word = last_word
        last_word = next_word
    return raw_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_dir", type=str)
    parser.add_argument("approximate_text_length", type=int)
    parser.add_argument("output_file_name", type=str)

    args = parser.parse_args()

    count_and_save_statistics(args.corpus_dir)

    with open(args.output_file_name, 'w') as output_file:
        raw_text = generate_raw_text(args.approximate_text_length)
        output_file.write(beutify(raw_text, 3, 40, 8).replace('\x01', ''))


if __name__ == '__main__':
    main()
