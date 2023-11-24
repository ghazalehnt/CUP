import argparse
import json
import time
from collections import Counter, defaultdict
import os
import pandas as pd
import numpy as np

from SBR.utils.metrics import calculate_ranking_metrics, get_significance

relevance_level = 1


def get_metrics(ground_truth, prediction_scores, ranking_metrics=None, calc_pytrec=True, exp_dir=None, anchor_path=None):
    if len(ground_truth) == 0:
        return {}
    start = time.time()
    results = calculate_ranking_metrics(gt=ground_truth, pd=prediction_scores,
                                        relevance_level=relevance_level,
                                        given_ranking_metrics=ranking_metrics,
                                        calc_pytrec=calc_pytrec,
                                        avg=False,
                                        exp_dir=exp_dir,
                                        anchor_path=anchor_path)
    # prep dict
    results_dict = dict()
    for m in results:
        if m.startswith("significance"):
            continue
        results_dict[m] = dict(zip(ground_truth.keys(), results[m]))
    # micro avg over user-item pairs:
    micro_res = defaultdict()
    for m in results:
        assert len(results[m]) == len(ground_truth)
        micro_res[m] = np.array(results[m]).mean().tolist()

    if exp_dir:
        json.dump(results_dict, open(exp_dir + "per_useritem.js", "w"))

    if anchor_path:
        anchor_useritem_dict = json.load(open(anchor_path + "per_useritem.js"))
        for m in results_dict.keys():
            micro_res["significance_" + m] = get_significance(anchor_useritem_dict[m], results_dict[m])
    else:
        for m in results_dict.keys():
            micro_res["significance_" + m] = 1.0
    print(f"ranking metrics in {time.time() - start}")
    return micro_res


def group_item_threshold(train_item_count, thresholds):
    groups = {thr: set() for thr in sorted(thresholds)}
    if len(thresholds) > 0:
        groups['rest'] = set()
        for item in train_item_count:
            added = False
            for thr in sorted(thresholds):
                if train_item_count[item] <= thr:
                    groups[thr].add(str(item))
                    added = True
                    break
            if not added:
                groups['rest'].add(str(item))

    ret_group = {}
    for gr in groups:
        if gr == 0:
            new_gr = "unseen"
        elif gr == 'rest':
            new_gr = "seen"
        else:
            new_gr = f"niche{gr}"
        ret_group[new_gr] = groups[gr]
    return ret_group


def group_users_ratios(train_user_count, ratios):
    sorted_users = [k for k, v in sorted(train_user_count.items(), key=lambda x: x[1])]  # sporadic to bibliophilic
    n_users = len(sorted_users)
    if len(ratios) != 3:
        raise ValueError("3 ratios must be given")
    cnts = [int(r*n_users) for r in ratios]
    if sum(cnts) < n_users:
        cnts[0] += 1
        if sum(cnts) < n_users:
            cnts[1] +=1
        if sum(cnts) < n_users:
            raise ValueError("check here1")
    elif sum(cnts) > n_users:
        raise ValueError("check here2")
    groups = {"sporadic": set(sorted_users[:cnts[0]]),
              "regular": set(sorted_users[cnts[0]:-cnts[2]]),
              "bibliophilic": set(sorted_users[-cnts[2]:])}
    return groups


def get_results(prediction_qid, ground_truth_qid, ranking_metrics, item_thresholds, user_ratios,
                train_file=None, item_file=None, exp_dir=None, anchor_path=None, neg_strategy=None, outfile_name=None):
    ret = []
    # we may have some users who only exist in training set (not in all datasets)
    train_ds = pd.read_csv(train_file, dtype=str)
    all_items = pd.read_csv(item_file, dtype=str)["item_id"]

    train_item_count = Counter(train_ds['item_id'])
    for itemid in all_items:
        if itemid not in train_item_count:
            train_item_count[itemid] = 0
    item_groups = group_item_threshold(train_item_count, item_thresholds)
    train_user_count = Counter(train_ds['user_id'])
    user_groups = group_users_ratios(train_user_count, user_ratios)

    sorted_grps = []
    sorted_grps.extend(user_groups.keys())
    sorted_grps.extend(item_groups.keys())
    for gu in user_groups:
        for gi in item_groups:
            sorted_grps.append(f"{gu}-{gi}")
    qid_groups = defaultdict(set)
    # user groups:
    for qid in ground_truth_qid.keys():
        tuid = qid[:qid.index("_")]
        for g in user_groups:
            if tuid in user_groups[g]:
                qid_groups[g].add(qid)
                break
    # item groups:
    for qid in ground_truth_qid.keys():
        tiid = qid[qid.index("_")+1:]
        for g in item_groups:
            if tiid in item_groups[g]:
                qid_groups[g].add(qid)
                break
    # user-item groups:
    for qid in ground_truth_qid.keys():
        tuid = qid[:qid.index("_")]
        tiid = qid[qid.index("_") + 1:]
        for gu in user_groups:
            if tuid in user_groups[gu]:
                for gi in item_groups:
                    if tiid in item_groups[gi]:
                        qid_groups[f"{gu}-{gi}"].add(qid)
                        break
                break

    # stats:
    eval_pos_items = []
    for qid in ground_truth_qid.keys():
        eval_pos_items.append(qid[qid.index("_")+1:])
    eval_pos_items = set(eval_pos_items)

    ret.append(f"#total Eval pos user-item pairs: {len(ground_truth_qid.keys())}")
    for g in sorted_grps:
        ret.append(f"#pos user-item pairs in {g}: {len(qid_groups[g])}")

    # calc metrics:
    # total:
    total_micro_results = get_metrics(ground_truth=ground_truth_qid,
                                      prediction_scores=prediction_qid,
                                      ranking_metrics=ranking_metrics,
                                      exp_dir=exp_dir + f"/{outfile_name}_ALL_{neg_strategy}_",
                                      anchor_path=anchor_path + f"/{outfile_name}_ALL_{neg_strategy}_" if anchor_path else None)

    metric_header = sorted(total_micro_results.keys())
    m_dict = {"ALL":  {f"{h}_micro": total_micro_results[h] for h in metric_header}}
    ret.append(m_dict)
    
    # groups:
    for g in qid_groups:
        group_micro_results = get_metrics(
            ground_truth={k: v for k, v in ground_truth_qid.items() if k in qid_groups[g]},
            prediction_scores={k: v for k, v in prediction_qid.items() if k in qid_groups[g]},
            ranking_metrics=ranking_metrics,
            exp_dir=exp_dir + "/"+outfile_name+g+f"_{neg_strategy}_",
            anchor_path=anchor_path + "/"+outfile_name+g+f"_{neg_strategy}_" if anchor_path else None)
        metric_header = sorted(group_micro_results.keys())
        m_dict = {g:  {f"{h}_micro": group_micro_results[h] for h in metric_header}}
        ret.append(m_dict)

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # required: path to gt and pd to be evaluated:
    parser.add_argument('--pos_gt_path', type=str, default=None, help='path to gt file')
    parser.add_argument('--neg_gt_path', type=str, default=None, help='path to gt file')
    parser.add_argument('--pred_path', type=str, default=None, help='path to pd file')
    parser.add_argument('--out_path', type=str, default=None, help='path to output file')
    parser.add_argument('--train_file_path', type=str, default=None, help='path to similar runs config file')
    parser.add_argument('--item_file_path', type=str, default=None, help='path to similar runs config file')

    parser.add_argument('--anchor_path', type=str, default=None,
                        help='path to baseline to compute significance. if left none, significance is not computed')

    # optional: threshold to group items:
    parser.add_argument('--item_thresholds', type=int, nargs='+', default=None, help='item thresholds')

    # optional: ratio of user groups:
    parser.add_argument('--user_ratios', type=int, nargs='+', default=None, help='user thresholds')

    args, _ = parser.parse_known_args()

    ranking_metrics_ = ["ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20", "P_1", "P_5", "recip_rank", "map"]

    if "test_predicted_test_neg_SB_BM25_100_" in args.pred_path:
        ng = "SB_BM25"
    elif "test_predicted_test_neg_standard_100_" in args.pred_path:
        ng = "standard"
    else:
        raise ValueError("ng not specified")

    item_thrs = args.item_thresholds
    user_rates = args.user_ratios
    if user_rates is not None:
        if sum(user_rates) != 100:
            raise ValueError(f"rations must sum up to 100: {user_rates}")
        user_rates = [r/100 for r in user_rates]

    if item_thrs is None or user_rates is None:
        raise ValueError("use this script only with both item threshold and user rations!")

    # over user-item pairs
    prediction_raw = json.load(open(args.pred_path))
    if len(prediction_raw.keys()) == 1 and "predicted" in prediction_raw:
        prediction_raw = prediction_raw["predicted"]
    pos_file = pd.read_csv(args.pos_gt_path, dtype=str)
    neg_file = pd.read_csv(args.neg_gt_path, dtype=str)
    ground_truth_ = defaultdict(lambda: defaultdict())
    prediction_ = defaultdict(lambda: defaultdict())
    for user_id, item_id in zip(pos_file["user_id"], pos_file["item_id"]):
        ground_truth_[f"{user_id}_{item_id}"][item_id] = 1  # TODO if we are doing rating or something else change this
        prediction_[f"{user_id}_{item_id}"][item_id] = prediction_raw[user_id][item_id]
    for user_id, item_id, ref_item in zip(neg_file["user_id"], neg_file["item_id"], neg_file["ref_item"]):
        ground_truth_[f"{user_id}_{ref_item}"][item_id] = 0  # TODO if we are doing rating or something else change this
        prediction_[f"{user_id}_{ref_item}"][item_id] = prediction_raw[user_id][item_id]

    outfile = args.out_path
    results_ = get_results(prediction_, ground_truth_, ranking_metrics_, item_thrs, user_rates,
                           train_file=args.train_file_path,
                           item_file=args.item_file_path,
                           exp_dir=os.path.dirname(args.pred_path),
                           anchor_path=args.anchor_path,
                           neg_strategy=ng,
                           outfile_name=os.path.basename(outfile[:outfile.index(".txt")]))

    print(outfile)
    outfile_f = open(outfile, "w")
    for line in results_:
        json.dump(line, outfile_f)
        outfile_f.write("\n\n")
    outfile_f.close()
