import collections
import random
import time
import os
import requests
import numpy as np
import json
import pickle
import implicit
from flask import Flask, jsonify, request
from helpers import parse_data, predict_user

app = Flask(__name__)


class RecFeatures(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 targets):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.targets = targets


class ProductTokenizer:
    def __init__(self, vocab_words):
        self.product2idx = {v: i for i, v in enumerate(vocab_words)}
        self.vocab_words = vocab_words

    def convert_tokens_to_ids(self, tokens):
        return [self.product2idx[t] if t in self.product2idx else self.product2idx['[UNK]'] for t in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.vocab_words[id] for id in ids]


vocab_words = []
with open('vocab.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        vocab_words.append(line.strip())

product_tokenizer = ProductTokenizer(vocab_words)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        del trunc_tokens[0]



MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, vocab_words):
    """Creates the predictions for the masked LM objective."""

    output_tokens = list(tokens)

    masked_lms = []
    covered_indexes = set()

    # mask last index (not including [SEP] and [TRANSACTION])
    index = len(tokens) - 3  # the -1 is [SEP]; the -2 is [TRANSACTION]
    covered_indexes.add(index)
    masked_token = "[MASK]"
    output_tokens[index] = masked_token
    masked_lms.append(MaskedLmInstance(
        index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


max_seq_length = 128
rng = random.Random(9000)


def preprocess(history):
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    tokens_a = []
    for transaction in history:
        for product in transaction:
            tokens_a.append(product)
        tokens_a.append("[TRANSACTION]")

    tokens_b = [product]
    tokens_b.append("[TRANSACTION]")

    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

    tokens = []
    segment_ids = []

    tokens.append("[CLS]")
    segment_ids.append(0)

    for t in tokens_a:
        tokens.append(t)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(tokens,
                                                                                   vocab_words)

    input_ids = product_tokenizer.convert_tokens_to_ids(tokens)
    masked_lm_ids = product_tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length, "%d != %d" % (
        len(input_ids), max_seq_length)
    assert len(input_mask) == max_seq_length, "%d != %d" % (
        len(input_mask), max_seq_length)
    assert len(segment_ids) == max_seq_length, "%d != %d" % (
        len(segment_ids), max_seq_length)

    masked_lm_positions = [0]
    print(masked_lm_positions)
    print(masked_lm_ids)

    print(input_ids)
    return {
        "inputs": {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "masked_lm_positions": [masked_lm_positions],
            "masked_lm_ids": masked_lm_ids
        }
    }


# baseline
baseline = {
    "recommended_products": [
        "4009f09b04",
        "15ccaa8685",
        "bf07df54e1",
        "3e038662c0",
        "4dcf79043e",
        "f4599ca21a",
        "5cb93c9bc5",
        "4a29330c8d",
        "439498bce2",
        "343e841aaa",
        "0a46068efc",
        "dc2001d036",
        "31dcf71bbd",
        "5645789fdf",
        "113e3ace79",
        "f098ee2a85",
        "53fc95e177",
        "080ace8748",
        "4c07cb5835",
        "ea27d5dc75",
        "cbe1cd3bb3",
        "1c257c1a1b",
        "f5e18af323",
        "5186e12ff4",
        "6d0f84a0ac",
        "f95785964a",
        "ad865591c6",
        "ac81544ebc",
        "de25bccdaf",
        "f43c12d228",
    ]
}


@app.route("/ready")
def ready():
    return "OK"


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json
        user_history = parse_data(data)
        app.logger.error(user_history)
        
        if not user_history:
            return jsonify(baseline)
        else:
            inp = preprocess(user_history)
            resp = requests.post(
                'http://localhost:8501/v1/models/model:predict', json=inp)
            outputs = resp.json()['outputs'][0]
            items_to_recommend = np.argsort(outputs)[::-1][:30]
            predictions = product_tokenizer.convert_ids_to_tokens(
                items_to_recommend)
            predictions = [pred for pred in predictions if pred[0] != '[']  # remove [SEP], [CLS], etc from predictions
            app.logger.error(predictions)

            predictions = predictions + [
                base_task
                for base_task in baseline["recommended_products"]
                if base_task not in predictions
            ]
            return jsonify({"recommended_products": predictions[:30]})
    except Exception as e:
        app.logger.error(e)
        return jsonify(baseline)


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host="0.0.0.0", debug=True, port=8000)
