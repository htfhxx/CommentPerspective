#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from models.inputters.dataset import PostResponseDataset
parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default="data/ES_newdata/")
parser.add_argument("--embed_file", type=str, default="data/ES_newdata/addcomments_small_embedding.txt")
parser.add_argument("--max_vocab_size", type=int, default=30000)
parser.add_argument("--min_len", type=int, default=5)
parser.add_argument("--max_len", type=int, default=300)
args = parser.parse_args()

vocab_file = os.path.join(args.data_dir, "vocab.json")
raw_train_file = os.path.join(args.data_dir, "music.train")
raw_valid_file = os.path.join(args.data_dir, "music.valid")
raw_test_file = os.path.join(args.data_dir, "music.test")
train_file = raw_train_file + ".pkl"
valid_file = raw_valid_file + ".pkl"
test_file = raw_test_file + ".pkl"

dataset = PostResponseDataset(
    max_vocab_size=args.max_vocab_size,
    min_len=args.min_len,
    max_len=args.max_len,
    embed_file=args.embed_file)

# Build vocabulary
print(' build_vocab ...')
dataset.build_vocab(raw_train_file)
print(' save_vocab ...')
dataset.save_vocab(vocab_file)

# Build examples
print(' Build examples valid_file...')
valid_examples = dataset.build_examples(raw_valid_file)
dataset.save_examples(valid_examples, valid_file)

print(' Build examples test_file...')
test_examples = dataset.build_examples(raw_test_file)
dataset.save_examples(test_examples, test_file)

print(' Build examples train_file...')
train_examples = dataset.build_examples(raw_train_file)
dataset.save_examples(train_examples, train_file)

print('finished! ')
