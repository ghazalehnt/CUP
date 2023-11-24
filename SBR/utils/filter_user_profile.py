import json
import random
from collections import Counter
from os.path import join

from datasets import Dataset
from sentence_splitter import SentenceSplitter
from torchtext.data import get_tokenizer

from torchtext.data.utils import ngrams_iterator


# todo hard coded
idf_ngram_year = 'all'
df_agg = 'max'
idf_ngram_alpha = True
idf_ngram_path = open('data/paths_vars/GoogleNgram_extracted_IDFs', 'r').read().strip()


def filter_user_profile_idf_tf(dataset_config, user_info):
    phrase_sizes = set()  # e.g. [1], [2], [3], [1,2], ..., [1,2,3]
    do_tfidf = False
    if dataset_config['user_text_filter'].startswith("tf-idf_"):
        sp = dataset_config['user_text_filter'].split('_')
        for n in sp[1].split('-'):
            phrase_sizes.add(int(n))
        do_tfidf = True

    user_info = Dataset.from_pandas(user_info)

    tokenizer = get_tokenizer("spacy")
    user_info = user_info.map(tokenize_function_torchtext, fn_kwargs={
        'tokenizer': tokenizer,
        'case_sensitive': dataset_config['case_sensitive'],
        'normalize_negation': dataset_config['normalize_negation'],
        'include_ngrams': phrase_sizes,
        'do_tf': do_tfidf
    }, batched=True)

    # do it only for the vocab
    vocab = {n: [] for n in phrase_sizes}
    for tokens in user_info['tokenized_text']:
        for token in tokens:
            if len(token.split()) > 0:
                vocab[len(token.split())].append(token)
    vocab = {n: set(v) for n, v in vocab.items()}

    # now load idf weights:
    # load from files
    idf_weights = {}
    for n in phrase_sizes:
        ngram_file = join(idf_ngram_path,
                          f"{n}_gram_casesensitive-{dataset_config['case_sensitive']}-{df_agg}_year-{idf_ngram_year}_alphabetic-{idf_ngram_alpha}.json")
        temp = json.load(open(ngram_file, 'r'))
        # filter for the vocab at hand:
        temp = {k: v for k, v in temp.items() if k in vocab[n]}
        idf_weights.update(temp)

    # now let's apply the weights to user data and sort the phrases.
    if do_tfidf:
        user_info = user_info.map(tokens_tf_idf_weights, fn_kwargs={'idf_weights': idf_weights}, batched=True)
        user_info = user_info.remove_columns(['tf'])
    else:
        user_info = user_info.map(tokens_idf_weights, fn_kwargs={'idf_weights': idf_weights}, batched=True)
    user_info = user_info.remove_columns(['text', 'tokenized_text'])

    user_info = user_info.to_pandas()
    user_info['text'] = user_info['sorted_text'].apply(" ".join)
    user_info = user_info.drop(columns=['sorted_text'])

    return user_info


def tokenize_function_torchtext(samples, tokenizer=None, doc_desc_field="text", include_ngrams=None,
                                case_sensitive=True, normalize_negation=True, do_tf=False):
    if include_ngrams is None:
        raise ValueError("the include_ngrams is not given!")
    n = max(include_ngrams)
    tokenized_batch = []
    tf_ret = []
    for text in samples[doc_desc_field]:
        tokens = tokenizer(text)
        if not case_sensitive:  # although this is done in data_loading, we still need it here because there are some cases that it is not done there
            tokens = [t.lower() for t in tokens]
        if normalize_negation:
            while "n't" in tokens:
                tokens[tokens.index("n't")] = "not"
        phrases = [ng for ng in list(ngrams_iterator(tokens, n)) if len(ng.split()) in include_ngrams]
        if do_tf:
            counter = Counter(phrases)
            phrases = [k for k, v in counter.items()]
            tfs = [v for k, v in counter.items()]
            tf_ret.append(tfs)
        else:
            phrases = list(set(phrases))
        tokenized_batch.append(phrases)

    ret_dict = {f"tokenized_{doc_desc_field}": tokenized_batch}
    if do_tf:
        ret_dict["tf"] = tf_ret
    return ret_dict


def tokens_idf_weights(samples, idf_weights=None):
    ret = []
    for tokens in samples['tokenized_text']:
        ret.append([k for k, v in sorted({token: idf_weights[token] if token in idf_weights else 0 for token in tokens}.items(), reverse=True, key=lambda x: x[1])])
    return {f"sorted_text": ret}


def tokens_tf_idf_weights(samples, idf_weights=None):
    ret = []
    for tokens, tfs in zip(samples['tokenized_text'], samples['tf']):
        ret.append([k for k, v in sorted({token: idf_weights[token]*tf if token in idf_weights else 0 for token, tf in zip(tokens, tfs)}.items(), reverse=True, key=lambda x: x[1])])
    return {f"sorted_text": ret}


def filter_user_profile_idf_sentences(dataset_config, user_info):
    phrase_sizes = set([1])
    unique_phrases = False
    # unique_phrases = True  # todo this should be optional to weigh sentences based on their unique terms or repeated?
    sent_splitter = SentenceSplitter(language='en')

    user_info = Dataset.from_pandas(user_info)

    tokenizer = get_tokenizer("spacy")
    user_info = user_info.map(tokenize_by_sent_function_torchtext, fn_kwargs={
        'tokenizer': tokenizer,
        'case_sensitive': dataset_config['case_sensitive'],
        'normalize_negation': dataset_config['normalize_negation'],
        'unique': unique_phrases,
        'sentencizer': sent_splitter
    },
                              batched=True)
    # do it only for the vocab
    vocab = {n: [] for n in phrase_sizes}
    for sent in user_info['tokenized_sentences_text']:
        for tokens in sent:
            for token in tokens:
                if len(token.split()) > 0:
                    vocab[len(token.split())].append(token.lower())
    vocab = {n: set(v) for n, v in vocab.items()}

    # now load idf weights:
    # load from files
    idf_weights = {}
    for n in phrase_sizes:
        ngram_file = join(idf_ngram_path,
                          f"{n}_gram_casesensitive-{False}-{df_agg}_year-{idf_ngram_year}_alphabetic-{idf_ngram_alpha}.json")
        temp = json.load(open(ngram_file, 'r'))
        # filter for the vocab at hand:
        temp = {k: v for k, v in temp.items() if k in vocab[n]}
        idf_weights.update(temp)

    user_info = user_info.map(sort_sentences, fn_kwargs={'idf_weights': idf_weights}, batched=True)
    user_info = user_info.remove_columns(['text', 'tokenized_sentences_text', 'sentences_text'])

    user_info = user_info.to_pandas()
    user_info['text'] = user_info['sorted_text'].apply(" ".join)
    user_info = user_info.drop(columns=['sorted_text'])
    return user_info


def filter_user_profile_random_sentences(dataset_config, user_info):
    phrase_sizes = set([1])
    sent_splitter = SentenceSplitter(language='en')

    user_info["sentences_text"] = user_info['text'].apply(lambda x:
                                                          sentencize_function_torchtext(x,
                                                                                        case_sensitive=dataset_config['case_sensitive'],
                                                                                        normalize_negation=dataset_config['normalize_negation'],
                                                                                        sentencizer=sent_splitter))

    user_info['text'] = user_info['sentences_text'].apply(lambda x: " ".join(random.sample(list(x), len(x))))
    user_info = user_info.drop(columns=['sentences_text'])
    return user_info


def tokenize_by_sent_function_torchtext(samples, tokenizer=None, sentencizer=None, doc_desc_field="text",
                                        case_sensitive=True, normalize_negation=True, unique=False):
    sent_ret = []
    sent_tokens_ret = []

    for text in samples[doc_desc_field]:
        sents = []
        sent_tokens = []
        for s in sentencizer.split(text=text):
            if not case_sensitive:
                s = s.lower()
            if normalize_negation:
                s = s.replace("n't", " not")
            sents.append(s)
            tokens = tokenizer(s)
            if unique:
                tokens = list(set(tokens))
            sent_tokens.append(tokens)
        sent_ret.append(sents)
        sent_tokens_ret.append(sent_tokens)

    return {f"sentences_{doc_desc_field}": sent_ret, f"tokenized_sentences_{doc_desc_field}": sent_tokens_ret}


def sentencize_function_torchtext(text, sentencizer=None,
                                        case_sensitive=True, normalize_negation=True):
    sents = []
    for s in sentencizer.split(text=text):
        if not case_sensitive:
            s = s.lower()
        if normalize_negation:
            s = s.replace("n't", " not")
        sents.append(s)
    return sents


def sort_sentences(samples, idf_weights=None):
    ret = []
    for sentences, sentences_tokens in zip(samples['sentences_text'], samples['tokenized_sentences_text']):
        s_weights = {}
        for i in range(0, len(sentences)):
            # we make sure we use lower case for idf calc
            # I am giving 0 to oov
            # w = [idf_weights[token.lower()] if token.lower() in idf_weights else 0 for token in sentences_tokens[i]]
            # I am NOT giving 0 to oov: done for experiment until now:
            w = [idf_weights[token.lower()] for token in sentences_tokens[i] if token.lower() in idf_weights]
            if len(w) > 0:
                s_weights[i] = sum(w)/len(w)
            else:
                s_weights[i] = 0
        ret.append([sentences[i] for i, v in sorted(s_weights.items(), key=lambda x:x[1], reverse=True)])
    return {f"sorted_text": ret}

