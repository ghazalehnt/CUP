import argparse
import json
import time
from os import listdir, makedirs
from os.path import join, exists
import gzip

import numpy as np


def idf(N, df, smooth, prob_idf):
    if smooth and prob_idf:
        raise ValueError("Not implemented")
    elif smooth:
        raise ValueError("Not implemented")
    elif prob_idf:
        raise ValueError("Not implemented")
    else:
        return np.log10(N/df)

# total doc num is in the totalcount... year, match_count, page_count, volume_count
def acc_num_docs(per_year_stat, year_const):
    return acc_df_weights(per_year_stat, year_const, field=3)


def acc_df_weights(per_year_stat, year_const, field=2):
    from_year = None
    # if year_const.startswith('last_'):
    #     per_year_stat = per_year_stat[-1 * int(year_const[5:])]
    if year_const.startswith('from_'):
        from_year = int(year_const[5:])

    #year,total_term_count,document_count
    df = 0
    for st in per_year_stat:
        sp = st.split(',')
        if from_year is not None:
            if int(sp[0]) < from_year:
                continue
        df += int(sp[field])
    return df


def get_idf_weights(ngram_dir, n, keys, idf_smooth, idf_prob,
                    case_sensitive=True, year_const='all', alphabetic_only=False, agg='max'):
    if year_const != 'all' and not year_const.startswith('from_'):
        raise ValueError(f"year_const = {year_const} not implemented")

    agg_df = {}
    idf_weights = {}

    # first read total number of documents from totalcount file:
    total_num_docs = 0
    with open(join(ngram_dir, f'{n}-grams', f'totalcounts-{n}'), 'r') as f:
        for line in f:# only one line
            sp = line.strip().split("\t")
            total_num_docs += acc_num_docs(sp, year_const)

    files = listdir(join(ngram_dir, f'{n}-grams'))
    counter = 0
    for f in files:
        if f.startswith('totalcounts') or f == 'download.sh':
            continue
        if counter % 100 == 0:
            print(f"{counter} files parsed. {time.time() - start_time}")
        counter += 1
        with gzip.open(join(ngram_dir, f'{n}-grams', f), 'rt') as f:
            for line in f:
                sp = line.strip().split("\t")  ### TODO maybe space is better than specifying \t? but gor 2,3 grams should concat them again...
                k = sp[0]
                if alphabetic_only:
                    if n == 1:
                        if k.isalpha() is False:
                            continue
                    else:
                        isalpha = True
                        for t in k.split():
                            if t.isalpha() is False:
                                isalpha = False
                                break
                        if isalpha is False:
                            continue
                if case_sensitive is False:
                    k = k.lower()
                if k not in agg_df:
                    agg_df[k] = 0
                if keys is None or k in keys:
                    df = acc_df_weights(sp[1:], year_const)
                    if agg == 'sum':
                        agg_df[k] += df
                    elif agg == 'max':
                        if agg_df[k] < df:
                            agg_df[k] = df
    print(f"{counter} files parsed. {time.time() - start_time}")
    for k, df in agg_df.items():
        if df > 0:
            idf_weights[k] = idf(total_num_docs, df, idf_smooth, idf_prob)
    return idf_weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get idf weights for google ngrams.')
    parser.add_argument('--n', type=int, help='n-gram', default=1)
    parser.add_argument('--cs', type=bool, help='case_sensitive', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--year', type=str, help='year_constraint', default='all')
    parser.add_argument('--alpha', type=bool, help='alphabetic', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--agg', type=str, help='aggregator of df when case insensitive', default='max')
    args = parser.parse_args()

    _n = args.n
    _idf_smooth = False
    _idf_prob = False
    _case_sensitive = args.cs
    _year_const = args.year
    _alpha = args.alpha
    _agg = args.agg
    outpath = open('data/paths_vars/GoogleNgram_extracted_IDFs', 'r').read().strip()
    outfile = f"{_n}_gram_casesensitive-{_case_sensitive}-{_agg}_year-{_year_const}_alphabetic-{_alpha}.json"
    print(outfile)
    if exists(join(outpath, outfile)):
        print(f"File Exists: {join(outpath, outfile)}")
        exit()
    makedirs(outpath, exist_ok=True)
    start_time = time.time()
    idfs = get_idf_weights(open('data/paths_vars/GoogleNgram', 'r').read().strip(), _n, None, _idf_smooth, _idf_prob,
                           year_const=_year_const, case_sensitive=_case_sensitive, alphabetic_only=_alpha,
                           agg=_agg)
    # idfs = {k: v for k, v in sorted(idfs.items())}  # TODO sort?
    json.dump(idfs, open(join(outpath, outfile), 'w'))
    print(f"finish: {time.time() - start_time}")
    # print(idfs)

# implemented all, from_year
# note: I also implemented last_year to get only the last years of a word, however, it is problematic as
# we cannot calculate the total docs to calculate the idf, as the last_x differs for each term.
