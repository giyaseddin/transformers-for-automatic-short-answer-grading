import os
import json
from transformers import pipeline

_LABELS_ID2NAME = {
    0: "correct", 1: "correct_but_incomplete", 2: "contradictory", 3: "incorrect",
}

_LABELS_NAME2ID = {v: k for k, v in _LABELS_ID2NAME.items()}


def load_best_model(model_path=None):
    home_dir = os.path.dirname(os.path.abspath(__file__))
    if not model_path:
        config = json.load(open(os.path.join(home_dir, "config.json")))
        model_path = os.path.join(home_dir, config["best_model_path"])

    return pipeline(
        "text-classification", model=model_path, return_all_scores=True
    )


def format_results(raw_results):
    results = []
    for result in raw_results:
        score_dict = {}
        for score in result:
            score_dict[_LABELS_ID2NAME[int(score["label"][-1:])]] = "%.2f" % score["score"]

        results.append(score_dict)

    return results


def pre_process_body(context, question, ref_answer, answer):
    return " [SEP] ".join([
        context, question, ref_answer, answer
    ])
