import argparse
import json
import os
import random
from os.path import join, exists

import torch
import numpy as np
from SBR.utils.data_loading_create_profile import load_split_dataset


def main(config_file=None, given_user_text_filter=None, given_limit_training_data=None,
         given_user_text=None, given_item_text=None, given_cs=None):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    config = json.load(open(config_file, 'r'))
    if given_user_text_filter is not None:
        config['dataset']['user_text_filter'] = given_user_text_filter
    if given_limit_training_data is not None:
        config['dataset']['limit_training_data'] = given_limit_training_data
    if given_user_text is not None:
        config['dataset']['user_text'] = given_user_text.split(",")
    if given_item_text is not None:
        config['dataset']['item_text'] = given_item_text.split(",")
    if given_cs is not None:
        config['dataset']['case_sensitive'] = given_cs
    if "<DATA_ROOT_PATH" in config["dataset"]["dataset_path"]:
        DATA_ROOT_PATH = config["dataset"]["dataset_path"][config["dataset"]["dataset_path"].index("<"):
                         config["dataset"]["dataset_path"].index(">")+1]
        config["dataset"]["dataset_path"] = config["dataset"]["dataset_path"]\
            .replace(DATA_ROOT_PATH, open(f"data/paths_vars/{DATA_ROOT_PATH[1:-1]}").read().strip())
    print(config)
    dataset_config = config['dataset']

    user_outfile = join(dataset_config['dataset_path'],
                        f"users_profile_{'-'.join(dataset_config['user_text'])}_{dataset_config['user_text_filter']}{'_i' + '-'.join(dataset_config['item_text']) if dataset_config['user_text_filter'] in ['item_sentence_SBERT'] else ''}_cs{dataset_config['case_sensitive']}_nn{dataset_config['normalize_negation']}.csv")
    item_outfile = join(dataset_config['dataset_path'], f"item_profile_{'-'.join(dataset_config['item_text'])}_cs{dataset_config['case_sensitive']}_nn{dataset_config['normalize_negation']}.csv")

    # check if both profile exists, there is nothing to do:
    if exists(user_outfile) and exists(item_outfile) :
        print(f"user and item profiles exist:\n{user_outfile}\n{item_outfile}")
        exit(1)

    user_info, item_info = load_split_dataset(dataset_config)
    if not exists(user_outfile):
        user_info[["user_id", "text"]].to_csv(user_outfile, index=False)
        print(f"user profile done:\n{user_outfile}")
    if not exists(item_outfile):
        item_info[["item_id", "text"]].to_csv(item_outfile, index=False)
        print(f"item profile done:\n{item_outfile}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-c', type=str, default=None, help='config file, to train')
    parser.add_argument('--user_text_filter', type=str, default=None, help='user_text_filter used only if given, otherwise read from the config')
    parser.add_argument('--limit_training_data', '-l', type=str, default=None, help='the file name containing the limited training data')
    parser.add_argument('--user_text', default=None, help='user_text (tg,tgr,tc,tcsr)')
    parser.add_argument('--item_text', default=None, help='item_text (tg,tgd,tc,tcd)')
    parser.add_argument('--not_case_sensitive', type=bool, default=None)
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.config_file):
        raise ValueError(f"Config file does not exist: {args.config_file}")
    main(config_file=args.config_file,
         given_user_text_filter=args.user_text_filter,
         given_limit_training_data=args.limit_training_data,
         given_user_text=args.user_text,
         given_item_text=args.item_text,
         given_cs=not args.not_case_sensitive if args.not_case_sensitive is not None else None)


