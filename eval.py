#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import sys
import json

from random import shuffle
from models.utils.metrics import Metric, bleu, distinct

NUM_MULTI_RESPONSES = 5


def evaluate_generation(results):

    #tgt = [result["response"].split(" ") for result in results]
    tgt = [result["response"].split() for result in results]
    tgt_multi = [x for x in tgt for _ in range(NUM_MULTI_RESPONSES)]
    preds = [
        list(map(lambda s: s.split(), result["preds"])) for result in results
        #list(map(lambda s: s.split(" "), result["preds"]))for result in results
    ]

    # Shuffle predictions
    for n in range(len(preds)):
        shuffle(preds[n])

    # Single response generation
    pred = [ps[0] for ps in preds]
    bleu1, bleu2 = bleu(pred, tgt)
    dist1, dist2 = distinct(pred)
    print("Random 1 candidate:   " + "BLEU-1/2: {:.3f}/{:.3f}   ".format(
        bleu1, bleu2) + "DIST-1/2: {:.3f}/{:.3f}".format(dist1, dist2))

    # Multiple response generation
    pred = [ps[:5] for ps in preds]
    pred = [p for ps in pred for p in ps]
    bleu1, bleu2 = bleu(pred, tgt_multi)
    dist1, dist2 = distinct(pred)
    print("Random {} candidates:   ".format(
        NUM_MULTI_RESPONSES) + "BLEU-1/2: {:.3f}/{:.3f}   ".format(bleu1, bleu2)
          + "DIST-1/2: {:.3f}/{:.3f}".format(dist1, dist2))


def main():
    result_file = sys.argv[1]
    with codecs.open(result_file, "r", encoding="utf-8") as fp:
        results = json.load(fp)
    evaluate_generation(results)


if __name__ == '__main__':
    main()
