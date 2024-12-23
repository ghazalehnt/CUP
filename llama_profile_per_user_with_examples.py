import random
import argparse

import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import csv
import os
import pandas as pd
import time
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_name = "Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = model.to(device)
examples_revs, example_profs, lens = [], [], []
MAX_LEN = 30000
remove_additional_length = 0
print(MAX_LEN)

def generate_test_prompt_all_rev(data_point, mode, num_ex, ex1=None, ex2=None):
    sys_prompt = "You are a book wizard, your goal is to find what aspects the users liked about the books.\n"
    syste_p_token_lens = 27
    if num_ex > 0:
        sys_prompt += ("Look at the following example(s):\n\n"
                       "## EXAMPLE 1 START\n"
                        f"Here are the user's reviews of the books the user read:\n"
                        f"'{ex1[0]}'\n"
                       )
        if mode == "kw":
            sys_prompt += "Generate some key phrases which can be used to characterize the type of book content described.\n"
        if mode == "abs":
            sys_prompt += "Generate an abstractive user profile written in the first-person narrative, based on the content of the book as described in the reviews.\n"

        sys_prompt += (f"'{ex1[1]}'\n"
                        "## EXAMPLE 1 END\n\n")
        syste_p_token_lens = 3440
    if num_ex > 1:
        sys_prompt += ("## EXAMPLE 2 START\n"
                        f"Here are the user's reviews of the books the user read:\n"
                        f"'{ex2[0]}'\n"
                       )
        if mode == "kw":
            sys_prompt += "Generate some key phrases which can be used to characterize the type of book content described.\n"
        if mode == "abs":
            sys_prompt += "Generate an abstractive user profile written in the first-person narrative, based on the content of the book as described in the reviews.\n"

        sys_prompt += (f"'{ex2[1]}'\n"
                        "## EXAMPLE 2 END\n\n")
        syste_p_token_lens = 6600

    sys_prompt += "You are a helpful assistant."

    user_instruct1 = "Here are the user's reviews of the books the user read: "
    user_instruct1_len = 15

    if mode == "kw":
        user_instruct2 = (".\n\nGenerate some key phrases which can be used to characterize the type of book content described.\n"
                          "Generate concise list of short key phrases in max 128 tokens total, print the comma-separated list in single line, "
                          "start right away with the keywords without introductory text.")
        user_instruct2_len = 53

    if mode == "abs":
        user_instruct2 = (".\n\nGenerate an abstractive user profile written in the first-person narrative, based on the content of the book as described in the reviews.\n"
                          "Generate a concise user profile in max 128 tokens total, "
                          "start right away with the user profile without introductory text.")
        user_instruct2_len = 53

    MAX_TEXT_LEN = MAX_LEN - (syste_p_token_lens + user_instruct1_len + user_instruct2_len + 20 + remove_additional_length) ## when there are errors
    data_point_trunc = tokenizer.decode(tokenizer(data_point, return_tensors="pt", max_length=MAX_TEXT_LEN, truncation=True).input_ids[0], skip_special_tokens=True)

    # Tokenize the full message as before (system and concatenated user instructions)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"{user_instruct1}{data_point_trunc}{user_instruct2}"}  # Decode back to full text
    ]

    return messages


def change_review_concat(text):
    text = text.replace("<ENDOFITEM>.", "Review:").replace("<ENDOFITEM>", "").replace("'", "")
    text = "Review: " + text
    return text


def get_examples(num_examples):
    global examples_revs, example_profs
    if num_examples == 0:
        return (None, None), (None, None)
    idx = random.sample(range(len(examples_revs)), k=num_examples)
    print(idx)
    if num_examples == 1:
        return (examples_revs[idx[0]], example_profs[idx[0]]), (None, None)
    return (examples_revs[idx[0]], example_profs[idx[0]]), (examples_revs[idx[1]], example_profs[idx[1]])


def summarize_text(text, mode, num_ex):
    # print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    # print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

    example1, example2 = get_examples(num_ex)
    inputs = tokenizer.apply_chat_template(generate_test_prompt_all_rev(text, mode, num_ex, example1, example2), add_generation_prompt=True, return_tensors="pt").to(device)

    lens.append(inputs.shape[-1])

    with torch.no_grad():
        try:
            summary_ids = model.generate(inputs, max_length=inputs.shape[-1]+200)
            summary = tokenizer.decode(summary_ids[0][inputs.shape[-1]:], skip_special_tokens=True).replace("\n", "").strip()
            del inputs
            del summary_ids
            torch.cuda.empty_cache()
            gc.collect()
        except:
            del inputs
            try:
                del summary_ids
            except:
                pass
            torch.cuda.empty_cache()
            gc.collect()
            summary = "ERROR"

    return summary


def read_input_file(file_path):
    data = pd.read_csv(file_path)
    data['text'] = data['text'].apply(lambda x: change_review_concat(x))
    return data


def main(mode, num_ex):
    ds = "amazon" #"goodreads"
    if ds == 'amazon':
        input_file_path = "users_profile_interaction.summary-interaction.reviewText_item_per_chunk_csTrue_nnTrue.csv"  # separated by <ENDOFITEM>.
    else:
        input_file_path = "users_profile_interaction.review_text_item_per_chunk_csTrue_nnTrue.csv"

    output_file_path = f"output_{ds}_per_user_rew-trunc-{MAX_LEN}_{num_ex}_random_example(fromgr)_{mode}.csv"
    err_users = None
    existing_users = {}
    if os.path.exists(output_file_path):
        df = pd.read_csv(output_file_path, dtype=str, header=None)
        err_users = list(df[df[1] == "ERROR"][0])
        df = df[~df[0].isin(err_users)].reset_index(drop=True)
        existing_users = df.set_index(0)[1].to_dict()
        output_file_path += "NEW"

    f = open(output_file_path, "w")
    writer = csv.writer(f)

    input_texts = read_input_file(input_file_path)

    start = time.time()
    all_sum = []
    cnt = 0
    for index, row in input_texts.iterrows():
        if err_users is not None and row.user_id not in err_users:
            summary = existing_users[row.user_id]
        else:
            summary = summarize_text(row.text, mode, num_ex)
            if summary != "ERROR":
                cnt += 1
            else:
                print(f"ERROR could not compute, {row.user_id} len: {lens[-1]}")
        all_sum.append([row.user_id, summary])
        if len(all_sum) == 5:
            print(f"{cnt} done")
            writer.writerows(all_sum)
            f.flush()
            all_sum = []

    print(f"Inference took {time.time() - start:.2f} seconds.")

    writer.writerows(all_sum)
    f.close()
    print(f"Summary saved to {output_file_path}")
    print(f"{cnt} done")

    print(max(lens))
    print(min(lens))
    print(sum(lens) / len(lens))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, default="kw", help='mode')
    parser.add_argument('--examples', '-e', type=int, default=0, help='number of examples')
    args, _ = parser.parse_known_args()

    mode = args.mode
    num_ex = args.examples

    if num_ex > 2:
        raise NotImplementedError("up to 2 examples")

    if num_ex > 0:
        examples_revs = [
            "Review: ... Review: ... , and so on", # example user #1
            "Review: ... Review: ... , and so on",  # example user #2
            "Review: ... Review: ... , and so on",  # example user #3
        #     ...
        ]

        if mode == "kw":
            example_profs = [
                "keyword profile user#1",
                "keyword profile user#2",
                "keyword profile user#3",
                #     ...
            ]
        if mode == "abs":
            example_profs = [
                "abstractive profile user#1",
                "abstractive profile user#2",
                "abstractive profile user#3",
                #     ...            ]

    main(mode, num_ex)



