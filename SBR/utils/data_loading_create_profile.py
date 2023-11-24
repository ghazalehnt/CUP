import json
from builtins import NotImplementedError
from os.path import join

import pandas as pd
from sentence_splitter import SentenceSplitter
from sentence_transformers import SentenceTransformer, util

from SBR.utils.filter_user_profile import filter_user_profile_idf_sentences, filter_user_profile_idf_tf, filter_user_profile_random_sentences

goodreads_rating_mapping = {
    'did not like it': 1,
    'it was ok': 2,
    'liked it': 3,
    'really liked it': 4,
    'it was amazing': 5
}


def sentencize(text, sentencizer, case_sensitive, normalize_negation):
    sents = []
    for s in sentencizer.split(text=text):
        if not case_sensitive:
            s = s.lower()
        if normalize_negation:
            s = s.replace("n't", " not")
        sents.append(s)
    return sents


def filter_user_profile(dataset_config, user_info):
    # filtering the user profile
    # filter-type1.1 idf_sentence
    # TODO having "idf_sentence_unique" and "idf_sentence_repeating" how about random_sentence? SBERT? For SBERT as it is a round robin, it is a bit weird! we choose genres from book1, then choose another sentence from book2 since genre is covered?
    # TODO I think more important than the uniqueness is the split of genres into sentences each not all together
    if dataset_config['user_text_filter'] == "idf_sentence":
        user_info = filter_user_profile_idf_sentences(dataset_config, user_info)
    # filter-type1 idf: we can have idf_1_all, idf_2_all, idf_3_all, idf_1-2_all, ..., idf_1-2-3_all, idf_1_unique, ...
    # filter-type2 tf-idf: tf-idf_1, ..., tf-idf_1-2-3
    elif dataset_config['user_text_filter'].startswith("idf_") or \
            dataset_config['user_text_filter'].startswith("tf-idf_"):
        if dataset_config['case_sensitive'] == True:
            raise ValueError("we don't want to do case_sensitive ngrams")
        if dataset_config['normalize_negation'] == False:
            raise ValueError("we don't want to do not normalize_negation ngrams")
        user_info = filter_user_profile_idf_tf(dataset_config, user_info)
    elif dataset_config['user_text_filter'] == "random_sentence":
        user_info = filter_user_profile_random_sentences(dataset_config, user_info)
    else:
        raise ValueError(
            f"filtering method not implemented, or belong to another script! {dataset_config['user_text_filter']}")
    return user_info


def load_split_dataset(config):
    if 'user_text_filter' in config and config['user_text_filter'] in ["idf_sentence", "random_sentence"]:
        temp_cs = config['case_sensitive']
        config['case_sensitive'] = True

    user_text_fields = config['user_text'].copy()
    item_text_fields = config['item_text'].copy()

    # check:
    for field in user_text_fields:
        if not field.startswith("user.") and not field.startswith("item.") and not field.startswith("interaction.") \
            and not field.startswith("userprofile."):
            raise ValueError(f"{field} in user text field is wrong")
    for field in item_text_fields:
        if not field.startswith("user.") and not field.startswith("item.") and not field.startswith("interaction."):
            raise ValueError(f"{field} in item text field is wrong")
    # read users and items, create internal ids for them to be used
    keep_fields = ["user_id"]
    for field in user_text_fields:
        topkgenres = None
        if field.startswith("user.sorted_genres_"):
            keep_fields.append("sorted_genres")
            topkgenres = int(field[field.rindex("_")+1:])
            break
    keep_fields.extend([field[field.index("user.")+len("user."):] for field in user_text_fields if field.startswith("user.") and not field.startswith("user.sorted_genres_")])
    keep_fields.extend([field[field.index("user.")+len("user."):] for field in item_text_fields if field.startswith("user.") and not field.startswith("user.sorted_genres_")])
    keep_fields = list(set(keep_fields))
    user_info = pd.read_csv(join(config['dataset_path'], "users.csv"), usecols=keep_fields, dtype=str)
    user_info = user_info.fillna('')
    if 'sorted_genres' in user_info.columns:
        if topkgenres is not None:
            user_info[f'sorted_genres_{topkgenres}'] = user_info['sorted_genres'].apply(
                lambda x: ", ".join(
                    [g.replace("'", "").replace('"', "").replace("[", "").replace("]", "").replace("  ", " ").strip()
                     for
                     g in x.split(",")][:topkgenres]))
            user_info = user_info.drop(columns=["sorted_genres"])
        else:
            user_info['sorted_genres'] = user_info['sorted_genres'].apply(
            lambda x: ", ".join([g.replace("'", "").replace('"', "").replace("[", "").replace("]", "").replace("  ", " ").strip() for
                                 g in x.split(",")]))
    user_info = user_info.rename(
        columns={field[field.index("user.") + len("user."):]: field for field in user_text_fields if
                 field.startswith("user.")})

    keep_fields = [field[field.index("userprofile.")+len("userprofile."):] for field in user_text_fields if field.startswith("userprofile.")]
    if len(keep_fields) > 1:
        raise ValueError("more than 1 userprofile fields ?")
    elif len(keep_fields) == 1:
        keep_fields = keep_fields[0]
        up = pd.read_csv(join(config['dataset_path'], f"users_profile_{keep_fields}.csv"), dtype=str)
        up = up.fillna('')
        up = up.rename(columns={"text": f"userprofile.{keep_fields}"})
        user_info = pd.merge(user_info, up, on="user_id")

    keep_fields = ["item_id"]
    keep_fields.extend([field[field.index("item.")+len("item."):] for field in item_text_fields if field.startswith("item.")])
    keep_fields.extend([field[field.index("item.") + len("item."):] for field in user_text_fields if field.startswith("item.")])
    keep_fields = list(set(keep_fields))
    tie_breaker = None
    if len(user_text_fields) > 0 and config['user_item_text_tie_breaker'] != "":
        if config['user_item_text_tie_breaker'].startswith("item."):
            tie_breaker = config['user_item_text_tie_breaker']
            tie_breaker = tie_breaker[tie_breaker.index("item.") + len("item."):]
            keep_fields.extend([tie_breaker])
        else:
            raise ValueError(f"tie-breaker value: {config['user_item_text_tie_breaker']}")

    item_info = pd.read_csv(join(config['dataset_path'], "items.csv"), usecols=keep_fields, low_memory=False, dtype=str)
    if tie_breaker is not None:
        if tie_breaker in ["avg_rating", "average_rating"]:
            item_info[tie_breaker] = item_info[tie_breaker].astype(float)
        else:
            raise NotImplementedError(f"tie-break {tie_breaker} not implemented")
        item_info[tie_breaker] = item_info[tie_breaker].fillna(0)
    item_info = item_info.fillna('')
    item_info = item_info.rename(
        columns={field[field.index("item.") + len("item."):]: field for field in item_text_fields if
                 field.startswith("item.")})
    item_info = item_info.rename(
        columns={field[field.index("item.") + len("item."):]: field for field in user_text_fields if
                 field.startswith("item.")})
    if 'item.genres' in item_info.columns:
        item_info['item.genres'] = item_info['item.genres'].apply(
            lambda x: ", ".join([g.replace("'", "").replace('"', "").replace("[", "").replace("]", "").replace("  ", " ").strip() for
                                 g in x.split(",")]))
    if config["name"] == "Amazon":
        if 'item.category' in item_info.columns:
            item_info['item.category'] = item_info['item.category'].apply(
                lambda x: ", ".join(x[1:-1].split(",")).replace("'", "").replace('"', "").replace("  ", " ")
                .replace("[", "").replace("]", "").strip())
        if 'item.description' in item_info.columns:
            item_info['item.description'] = item_info['item.description'].apply(
                lambda x: ", ".join(x[1:-1].split(",")).replace("'", "").replace('"', "").replace("  ", " ")
                .replace("[", "").replace("]", "").strip())

    train_file = join(config['dataset_path'], "train.csv")
    train_df = pd.read_csv(train_file, dtype=str)  # rating:float64

    # TODO if dataset is cleaned beforehand this could change slightly
    train_df['rating'] = train_df['rating'].fillna(-1)
    if config["name"] == "CGR":
        for k, v in goodreads_rating_mapping.items():
            train_df['rating'] = train_df['rating'].replace(k, v)
    elif config["name"] == "GR_UCSD":
        train_df['rating'] = train_df['rating'].astype(int)
    elif config["name"] == "Amazon":
        train_df['rating'] = train_df['rating'].astype(float).astype(int)
    else:
        raise NotImplementedError(f"dataset {config['name']} not implemented!")

    train_df = train_df.rename(
        columns={field[field.index("interaction.") + len("interaction."):]: field for field in user_text_fields if
                 field.startswith("interaction.")})
    train_df = train_df.rename(
        columns={field[field.index("interaction.") + len("interaction."):]: field for field in item_text_fields if
                 field.startswith("interaction.")})

    # text profile:
    # concat and move the user/item text fields to user and item info:
    sort_reviews = ""
    if len(user_text_fields) > 0:
        sort_reviews = config['user_item_text_choice']

    train_df = train_df.fillna('')
    ## USER:
    # This code works for user text fields from interaction and item file
    user_item_text_fields = [field for field in user_text_fields if field.startswith("item.")]
    user_inter_text_fields = [field for field in user_text_fields if field.startswith("interaction.")]
    user_item_inter_text_fields = user_item_text_fields.copy()
    user_item_inter_text_fields.extend(user_inter_text_fields)

    if len(user_item_inter_text_fields) > 0:
        user_item_merge_fields = ["item_id"]
        user_item_merge_fields.extend(user_item_text_fields)
        if tie_breaker in ["avg_rating", "average_rating"]:
            user_item_merge_fields.append(tie_breaker)

        user_inter_merge_fields = ["user_id", "item_id", 'rating']
        user_inter_merge_fields.extend(user_inter_text_fields)

        temp = train_df[user_inter_merge_fields].\
            merge(item_info[user_item_merge_fields], on="item_id")
        if sort_reviews.startswith("pos_rating_sorted_"):
            pos_threshold = int(sort_reviews[sort_reviews.rindex("_") + 1:])
            temp = temp[temp['rating'] >= pos_threshold]
        # before sorting them based on rating, etc., let's append each row's field together (e.g. title. genres. review.)
        if config['user_text_filter'] == "item_per_chunk":
            temp['text'] = temp[user_item_inter_text_fields].agg('. '.join, axis=1) + "<ENDOFITEM>"
        else:
            temp['text'] = temp[user_item_inter_text_fields].agg('. '.join, axis=1)

        if config['user_text_filter'] in ["item_sentence_SBERT"]:
            # first we sort the items based on the ratings, tie-breaker
            if tie_breaker is None:
                temp = temp.sort_values(['rating'], ascending=[False])
            else:
                temp = temp.sort_values(['rating', tie_breaker], ascending=[False, False])

            # sentencize the user text (r, tgr, ...)
            sent_splitter = SentenceSplitter(language='en')
            temp['sentences_text'] = temp.apply(lambda row: sentencize(row['text'], sent_splitter,
                                                                       config['case_sensitive'],
                                                                       config['normalize_negation']), axis=1)
            temp = temp.drop(columns=['text'])
            temp = temp.drop(columns=[field for field in user_text_fields if not field.startswith("user.") and not field.startswith("userprofile.")])

            # load SBERT
            sbert = SentenceTransformer("all-mpnet-base-v2")  # TODO hard coded
            # "all-MiniLM-L12-v2"
            # "all-MiniLM-L6-v2"
            print("sentence transformer loaded!")

            user_texts = []
            for user_id in list(user_info["user_id"]):
                user_items = []
                user_item_temp = temp[temp["user_id"] == user_id]
                for item_id, sents in zip(user_item_temp["item_id"], user_item_temp['sentences_text']):
                    if len(sents) == 0:
                        continue
                    item = item_info[item_info["item_id"] == item_id].reset_index().loc[0]
                    item_text = '. '.join(list(item[item_text_fields]))
                    scores = util.dot_score(sbert.encode(item_text), sbert.encode(sents))
                    user_items.append([sent for score, sent in sorted(zip(scores[0], sents), reverse=True)])
                user_text = []
                cnts = {i: 0 for i in range(len(user_items))}
                while True:
                    remaining = False
                    for i in range(len(user_items)):
                        if cnts[i] == len(user_items[i]):
                            continue
                        remaining = True
                        user_text.append(user_items[i][cnts[i]])
                        cnts[i] += 1
                    if not remaining:
                        break
                user_texts.append(' '.join(user_text))  # sentences
            user_info['text'] = user_texts
            print(f"user text matching with item done!")
        else:
            if sort_reviews == "":
                temp = temp.groupby("user_id")['text']
            else:
                if sort_reviews == "rating_sorted" or sort_reviews.startswith("pos_rating_sorted_"):

                    if tie_breaker is None:
                        temp = temp.sort_values('rating', ascending=False).groupby(
                            "user_id")['text']
                    elif tie_breaker in ["avg_rating", "average_rating"]:
                        temp = temp.sort_values(['rating', tie_breaker], ascending=[False, False]).groupby(
                            "user_id")['text']
                    else:
                        raise ValueError("Not implemented!")
                else:
                    raise ValueError("Not implemented!")
            temp = temp.apply('. '.join).reset_index()
            user_info = user_info.merge(temp, "left", on="user_id")
            user_info['text'] = user_info['text'].fillna('')

    ## ITEM:
    # This code works for item text fields from interaction and user file
    item_user_text_fields = [field for field in item_text_fields if field.startswith("user.")]
    item_inter_text_fields = [field for field in item_text_fields if field.startswith("interaction.")]
    item_user_inter_text_fields = item_user_text_fields.copy()
    item_user_inter_text_fields.extend(item_inter_text_fields)

    if len(item_user_inter_text_fields) > 0:
        item_user_merge_fields = ["user_id"]
        item_user_merge_fields.extend(item_user_text_fields)

        item_inter_merge_fields = ["user_id", "item_id", 'rating']
        item_inter_merge_fields.extend(item_inter_text_fields)

        temp = train_df[item_inter_merge_fields]. \
            merge(user_info[item_user_merge_fields], on="user_id")
        if sort_reviews.startswith("pos_rating_sorted_"):  # Todo sort_review field in config new?
            pos_threshold = int(sort_reviews[sort_reviews.rindex("_") + 1:])
            temp = temp[temp['rating'] >= pos_threshold]
        # before sorting them based on rating, etc., let's append each row's field together
        temp['text'] = temp[item_user_inter_text_fields].agg('. '.join, axis=1)

        if sort_reviews == "":
            temp = temp.groupby("item_id")['text'].apply('. '.join).reset_index()
        else:
            if sort_reviews == "rating_sorted" or sort_reviews.startswith("pos_rating_sorted_"):
                temp = temp.sort_values('rating', ascending=False).groupby(
                    "item_id")['text'].apply('. '.join).reset_index()
            else:
                raise ValueError("Not implemented!")

        item_info = item_info.merge(temp, "left", on="item_id")
        item_info['text'] = item_info['text'].fillna('')

    # apply filter:  maybe put this beofre the user fields like genre is read so those are always w/o filter
    if 'text' in user_info.columns and config['user_text_filter'] != "":
        if config['user_text_filter'] not in ["item_sentence_SBERT", "item_per_chunk"]:
            user_info = filter_user_profile(config, user_info)

    # text sorting for user is already applied, so next would just be on top:  TODO what if it is from a userprofile file ??? IDK check what we want
    # after moving text fields to user/item info, now concatenate them all and create a single 'text' field:
    user_remaining_text_fields = [field for field in user_text_fields if (field.startswith("user.") or field.startswith("userprofile."))]
    if 'text' in user_info.columns:
        user_remaining_text_fields.append('text')
    if len(user_remaining_text_fields) > 0:
        user_info['text'] = user_info[user_remaining_text_fields].agg('. '.join, axis=1)
        user_info['text'] = user_info['text'].apply(lambda x: x.replace("<end of review>", ""))
        if not config['case_sensitive']:
            user_info['text'] = user_info['text'].apply(str.lower)
        if config['normalize_negation']:
            user_info['text'] = user_info['text'].replace("n\'t", " not", regex=True)
        user_info = user_info.drop(columns=[field for field in user_text_fields if field.startswith("user.")])

    item_remaining_text_fields = [field for field in item_text_fields if field.startswith("item.")]
    if 'text' in item_info.columns:
        item_remaining_text_fields.append('text')
    if len(item_remaining_text_fields) > 0:
        item_info['text'] = item_info[item_remaining_text_fields].agg('. '.join, axis=1)
        if not config['case_sensitive']:
            item_info['text'] = item_info['text'].apply(str.lower)
        if config['normalize_negation']:
            item_info['text'] = item_info['text'].replace("n\'t", " not", regex=True)
    item_info = item_info.drop(columns=[field for field in item_text_fields if field.startswith("item.")])

    if 'user_text_filter' in config and config['user_text_filter'] in ["idf_sentence", "random_sentence"]:
        config['case_sensitive'] = temp_cs
        # since cs was turned off while reading the dataset for user-text to remain intact. user text will change in the followup filder_user_profile function.
        # but item text should be change now latest.
        if temp_cs is False and 'text' in item_info.columns:
            item_info['text'] = item_info['text'].apply(lambda x: {x.lower()})

    return user_info, item_info
