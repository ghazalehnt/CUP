import argparse
import json
import os
import random
from os.path import join

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from SBR.trainer.supervised import SupervisedTrainer
from SBR.utils.data_loading import load_data
from SBR.utils.others import get_model
from SBR.utils.statics import shorten_names, shorten_strategies


def main(op, config_file=None, result_folder=None, given_eval_model=None, given_eval_pos_file=None,
         given_eval_neg_file=None, given_lr=None, given_tbs=None, given_user_text_file_name=None,
         given_item_text_file_name=None, given_k1k2=None):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_only = False
    if op in ["train", "trainonly"]:
        config = json.load(open(config_file, 'r'))
        if given_lr is not None:
            config['trainer']['lr'] = given_lr
        if given_tbs is not None:
            config['dataset']['train_batch_size'] = given_tbs
        if given_user_text_file_name is not None:
            config['dataset']['user_text_file_name'] = given_user_text_file_name
        if given_item_text_file_name is not None:
            config['dataset']['item_text_file_name'] = given_item_text_file_name
        if given_k1k2 is not None:
            config["model"]["k1"] = given_k1k2
            config["model"]["k2"] = given_k1k2
        if "<DATA_ROOT_PATH" in config["dataset"]["dataset_path"]:
            DATA_ROOT_PATH = config["dataset"]["dataset_path"][config["dataset"]["dataset_path"].index("<"):
                             config["dataset"]["dataset_path"].index(">")+1]
            config["dataset"]["dataset_path"] = config["dataset"]["dataset_path"]\
                .replace(DATA_ROOT_PATH, open(f"data/paths_vars/{DATA_ROOT_PATH[1:-1]}").read().strip())
        if "<EXP_ROOT_PATH>" in config["experiment_root"]:
            config["experiment_root"] = config["experiment_root"]\
                .replace("<EXP_ROOT_PATH>", open("data/paths_vars/EXP_ROOT_PATH").read().strip())
        print(config)
        exp_dir_params = []
        for param in config['params_in_exp_dir']:
            p1 = param[:param.index(".")]
            p2 = param[param.index(".")+1:]
            if param == "dataset.validation_neg_sampling_strategy" and config[p1][p2].startswith("f:"):
                temp = config[p1][p2]
                temp = temp[temp.index("f:validation_neg_")+len("f:validation_neg_"):]
                exp_dir_params.append(f"f-{temp}")
            elif param == "dataset.test_neg_sampling_strategy" and config[p1][p2].startswith("f:"):
                temp = config[p1][p2]
                temp = temp[temp.index("f:test_neg_")+len("f:test_neg_"):]
                exp_dir_params.append(f"f-{temp}")
            elif param == "dataset.user_text_file_name" or param == "dataset.item_text_file_name":
                temp = config[p1][p2]
                for s, v in shorten_strategies.items():
                    temp = temp.replace(s, v)
                for s, v in shorten_names.items():
                    temp = temp.replace(s, v)
                exp_dir_params.append(temp)
            elif isinstance(config[p1][p2], list):
                exp_dir_params.append('-'.join([str(v) for v in config[p1][p2]]))
            else:
                exp_dir_params.append(str(config[p1][p2]))
        exp_dir = join(config['experiment_root'], "_".join(exp_dir_params))
        
        config["experiment_dir"] = exp_dir
        # check if the exp dir exists, the config file is the same as given.
        if os.path.exists(join(exp_dir, "config.json")):
            config2 = json.load(open(join(exp_dir, "config.json"), 'r'))
            if config != config2:
                print(f"GivenConfig: {config}")
                raise ValueError(f"{exp_dir} exists with different config != {config_file}")
        os.makedirs(exp_dir, exist_ok=True)
        json.dump(config, open(join(exp_dir, "config.json"), 'w'), indent=4)
    elif op in ["test", "log"]:
        config = json.load(open(join(result_folder, "config.json"), 'r'))
        test_only = True
        exp_dir = config["experiment_dir"]
        if given_eval_pos_file is not None:
            config["dataset"]["alternative_pos_test_file"] = given_eval_pos_file
        if given_eval_neg_file is not None:
            config["dataset"]["test_neg_sampling_strategy"] = given_eval_neg_file
    else:
        raise ValueError("op not defined!")

    logger = SummaryWriter(exp_dir)
    for k, v in config["dataset"].items():
        logger.add_text(f"dataset/{k}", str(v))
    for k, v in config["trainer"].items():
        logger.add_text(f"trainer/{k}", str(v))
    for k, v in config["model"].items():
        logger.add_text(f"model/{k}", str(v))
    logger.add_text("exp_dir", exp_dir)
    print("experiment_dir:")
    print(exp_dir)

    train_dataloader, valid_dataloader, test_dataloader, users, items, relevance_level, padding_token = \
        load_data(config['dataset'],
                  pretrained_model=config['model']['pretrained_model'] if 'pretrained_model' in config['model'] else None,
                  word_vector_model=config['model']['word2vec_file'] if 'word2vec_file' in config['model'] else None,
                  exp_dir=exp_dir if 'word2vec_file' in config['model'] else None)
    print("Data load done!")
    # needed for item-item relatedness
    temp = {ex: internal for ex, internal in zip(items['item_id'], items['internal_item_id'])}
    json.dump(temp, open(join(exp_dir, "item_internal_ids.json"), 'w'))

    model = get_model(config['model'], users, items, device, config['dataset'], exp_dir, test_only=test_only)
    print("Get model Done!")

    trainer = SupervisedTrainer(config=config['trainer'], model=model, device=device, logger=logger, exp_dir=exp_dir,
                                test_only=test_only, relevance_level=relevance_level,
                                users=users, items=items,
                                dataset_eval_neg_sampling=
                                {"validation": config["dataset"]["validation_neg_sampling_strategy"],
                                "test": config["dataset"]["test_neg_sampling_strategy"]},
                                to_load_model_name=given_eval_model,
                                padding_token=padding_token,
                                unique_user_item=config['dataset']['load_unique_user_item'] if 'load_unique_user_item' in config['dataset'] else False)
    if op == "train":
        trainer.fit(train_dataloader, valid_dataloader)
        trainer.evaluate(test_dataloader, valid_dataloader)
    elif op == "trainonly":
        trainer.fit(train_dataloader, valid_dataloader)
    elif op == "test":
        trainer.evaluate(test_dataloader, valid_dataloader)
    elif op == "log":
        trainer.user_prec_file_out = join(exp_dir, "user_prec_output")
        trainer.item_prec_file_out = join(exp_dir, "item_prec_output")
        trainer.log()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-c', type=str, default=None, help='config file, to train')
    parser.add_argument('--result_folder', '-r', type=str, default=None, help='result forler, to evaluate')
    parser.add_argument('--eval_model_name', default=None, help='test time model to load, only for op == test')
    parser.add_argument('--eval_pos_file', default=None, help='test positive file, only for op == test')
    parser.add_argument('--eval_neg_file', default=None, help='test neg file, only for op == test')
    parser.add_argument('--trainer_lr', default=None, help='trainer learning rate')
    parser.add_argument('--train_batch_size', default=None, help='train_batch_size')
    parser.add_argument('--user_text_file_name', default=None, help='user_text_file_name')
    parser.add_argument('--item_text_file_name', default=None, help='item_text_file_name')
    parser.add_argument('--model_k1k2', type=int, default=None, help='model.k1k2')
    parser.add_argument('--op', type=str, help='operation train/test/trainonly')
    args, _ = parser.parse_known_args()

    if args.op in ["train", "trainonly"]:
        if not os.path.exists(args.config_file):
            raise ValueError(f"Config file does not exist: {args.config_file}")
        if args.result_folder:
            raise ValueError(f"OP==train does not accept result_folder")
        if args.eval_model_name or args.eval_pos_file or args.eval_neg_file:
            raise ValueError(f"OP==train does not accept test-time eval pos/neg/model.")
        main(op=args.op, config_file=args.config_file,
             given_lr=float(args.trainer_lr) if args.trainer_lr is not None else args.trainer_lr,
             given_tbs=int(args.train_batch_size) if args.train_batch_size is not None else args.train_batch_size,
             given_user_text_file_name=args.user_text_file_name, given_item_text_file_name=args.item_text_file_name,
             given_k1k2=args.model_k1k2)
    elif args.op == "test":
        if not os.path.exists(join(args.result_folder, "config.json")):
            raise ValueError(f"Result folder does not exist: {args.config_file}")
        if args.config_file:
            raise ValueError(f"OP==test does not accept config_file")
        main(op=args.op, result_folder=args.result_folder,
             given_eval_model=args.eval_model_name,
             given_eval_pos_file=args.eval_pos_file,
             given_eval_neg_file=args.eval_neg_file)
    elif args.op == "log":
        if not os.path.exists(join(args.result_folder, "config.json")):
            raise ValueError(f"Result folder does not exist: {args.config_file}")
        if args.config_file:
            raise ValueError(f"OP==test does not accept config_file")
        main(op=args.op, result_folder=args.result_folder)
    else:
        raise ValueError("--op not given")
