"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"  # a 10B tokens sample of edu_fineweb
shard_size = int(1e8)  # 100M tokens per shard, total of 10B/100M = 100 shards

# create the local dir if it doesn't exist
DATA_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_DIR, exist_ok=True)

# download the dataset from HuggingFace
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# load the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot_token = enc._special_tokens["<|endoftext|>"]


def tokenize(doc):
    # tokenizes a single document and returns a numpy array of unit16 tokens
    tokens = [eot_token]  # the eot token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "token dictionary too large for unit16"
    return tokens_np.astype(np.uint16)


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


if __name__ == "__main__":
    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    cpu_cores_available = os.cpu_count() or 1
    nprocs = max(1, cpu_cores_available // 2)  # use half the cores
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        progress_bar = tqdm(
            total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
        )
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    DATA_DIR, f"edufineweb_{split}_{shard_index:06d}"
                )
                # split the document into whatever fits in the current shard; the remainder goes to the next shard
                space_left = shard_size - token_count
                all_tokens_np[token_count : token_count + space_left] = tokens[
                    :space_left
                ]
                write_datafile(filename, all_tokens_np)
                # update progess bar
                progress_bar.update(space_left)
                progress_bar.close()

                # begin new shard
                shard_index += 1
                progress_bar = tqdm(
                    total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                )
                # populate with leftovers of the current doc
                leftover_tokens = tokens[space_left:]
                all_tokens_np[0 : len(leftover_tokens)] = leftover_tokens
                token_count = len(leftover_tokens)
                # update progress bar
                progress_bar.update(len(leftover_tokens))

        # write any remaining tokens as the last shard
        if token_count > 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])
            progress_bar.close()
