from SBR.model.DeepCoNN import DeepCoNN
from SBR.model.mf_dot import MatrixFactorizatoinDotProduct
from SBR.model.bert_ffn_end_to_end import BertFFNUserTextProfileItemTextProfileEndToEnd


def get_model(config, user_info, item_info, device=None, dataset_config=None, exp_dir=None, test_only=False):
    if config['name'] == "MF":
        model = MatrixFactorizatoinDotProduct(config=config,
                                              n_users=user_info.shape[0],
                                              n_items=item_info.shape[0],
                                              device=device)
    elif config['name'] == "VanillaBERT_ffn_endtoend":
        model = BertFFNUserTextProfileItemTextProfileEndToEnd(model_config=config,
                                                              device=device,
                                                              dataset_config=dataset_config,
                                                              users=user_info,
                                                              items=item_info,
                                                              test_only=test_only)
    elif config['name'] == "DeepCoNN":
        model = DeepCoNN(config, exp_dir)
    else:
        raise ValueError(f"Model is not implemented! model.name = {config['name']}")
    return model
