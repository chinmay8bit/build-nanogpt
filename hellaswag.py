"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script (karpathy's run): 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)
- this script (my run): 10042 acc: 2872/10042=0.2860 acc_norm: 2969/10042=0.2957 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script (karpathy's run): 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)
- this script (my run): 10042 acc: 3858/10042=0.3842 acc_norm: 4914/10042=0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""

import os
import json
from typing import Tuple

import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel


DATA_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")


def download(split):
    """Downloads HellaSwah split to DATA_DIR"""
    os.makedirs(DATA_DIR, exist_ok=True)
    url = hellaswags[split]
    fname = os.path.join(DATA_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.isfile(fname):
        print(f"Downloading {url} to {fname}...")
        download_file(url, fname)


def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)

    Example HellaSwag json item:
    {"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}
    """
    ctx = example["ctx"]
    endings = example["endings"]
    label = example["label"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for ending in endings:
        ending_tokens = enc.encode(
            " " + ending
        )  # space is added since it will be appended to context
        tok_rows.append(ctx_tokens + ending_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(ending_tokens))
        data["ending_tokens"].append(ending_tokens)

    # collate into tensors, add padding where needed
    max_len = max(len(r) for r in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.bool)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(tok_row)] = torch.tensor(tok_row, dtype=torch.long)
        mask[i, : len(mask_row)] = torch.tensor(mask_row, dtype=torch.bool)

    return data, tokens, mask, label


def iterate_examples(split):
    download(split)
    fname = os.path.join(DATA_DIR, f"hellaswag_{split}.jsonl")
    with open(fname, "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


def get_most_likely_ending(
    tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor
) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
    """
    Helper function to get the most likely ending based on model logits for the tokens

    Returns:
    - pred: index of the most likely ending based on total loss
    - pred_norm: index of the most likely ending based on average token loss
    - sum_loss: tensor of total losses for each ending
    - avg_loss: tensor of average token losses for each ending
    """
    # Set up the logits, target and mask for loss calculation
    logits = logits[
        ..., :-1, :
    ].contiguous()  # we don't need the logit from the last token in each row
    target = tokens[..., 1:].contiguous()
    target_mask = mask[..., 1:].contiguous()

    flat_logits = logits.view(-1, logits.size(-1))
    flat_target = target.view(-1)
    losses = F.cross_entropy(flat_logits, flat_target, reduction="none")
    losses = losses.view_as(target)

    # sum up the losses where target_mask is set (will ignore context and padding)
    masked_losses = losses * target_mask
    sum_loss = masked_losses.sum(dim=-1)  # sum over sequence length
    avg_loss = sum_loss / target_mask.sum(dim=-1)

    # now we have the total loss and average token loss for each of the 4 completions
    # the completionn with the lowest loss is the most likely
    pred = torch.argmin(sum_loss, dim=-1).item()
    pred_norm = torch.argmin(avg_loss, dim=-1).item()

    return int(pred), int(pred_norm), sum_loss, avg_loss


@torch.no_grad()
def evaluate(model_type, device):
    torch.set_float32_matmul_precision("high")  # use TF32
    model = GPT2LMHeadModel.from_pretrained(model_type).to(device)
    model.compile()

    # set to eval mode
    model.eval()

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        logits: torch.Tensor = model(tokens).logits
        # get the predictons
        pred, pred_norm, sum_loss, avg_loss = get_most_likely_ending(
            tokens, mask, logits
        )

        # accumulate stats
        num_total += 1
        if pred == label:
            num_correct += 1
        if pred_norm == label:
            num_correct_norm += 1
        print(
            f"{num_total} acc: {num_correct}/{num_total}={num_correct/num_total:.4f} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}"
        )

        # debug: print a few examples, and the losses in each case
        if num_total < 10:
            print("---")
            print(f"Context:\n {example["ctx"]}")
            print("Endings:")
            for i, ending in enumerate(example["endings"]):
                print(
                    f"  {i}: (total loss: {sum_loss[i].item():.4f}, avg loss: {avg_loss[i].item():.4f}) {ending}"
                )
            print(f"predicted: {pred}, predicted (norm): {pred_norm}, actual: {label}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-type",
        type=str,
        default="gpt2",
        help="HuggingFace model type to use",
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="Device to run on"
    )
    args = parser.parse_args()

    evaluate(args.model_type, args.device)
