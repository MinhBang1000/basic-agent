import os
import json
from typing import List, Dict, Any

# Assume you already have:
# - detect_prompt_injection(text: str, threshold: float=0.5) -> Dict[str, Any]
# from your Sentinel + ProtectAI code.

DATASET_PATH = "datasets/injecagent_retrieval_only_dataset_textonly.json"

# Disable torch.compile / inductor để tránh lỗi với ModernBERT
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------ Cấu hình model ------------------

_SENTINEL_MODEL_NAME = "qualifire/prompt-injection-sentinel"
_PROTECTAI_MODEL_NAME = "protectai/deberta-v3-base-prompt-injection"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_sentinel_tokenizer = AutoTokenizer.from_pretrained(_SENTINEL_MODEL_NAME)
_sentinel_model = AutoModelForSequenceClassification.from_pretrained(
    _SENTINEL_MODEL_NAME
).to(device).eval()

_protectai_tokenizer = AutoTokenizer.from_pretrained(_PROTECTAI_MODEL_NAME)
_protectai_model = AutoModelForSequenceClassification.from_pretrained(
    _PROTECTAI_MODEL_NAME
).to(device).eval()


def _find_positive_label_index(model) -> int:
    """
    Tự động tìm index class 'injection' từ id2label/label2id.
    Nếu không tìm được và là binary classifier, fallback = 1.
    """
    id2label = getattr(model.config, "id2label", None) or {}
    label2id = getattr(model.config, "label2id", None) or {}

    def norm(s: str) -> str:
        return str(s).lower().strip()

    positive_keywords = [
        "inject", "injection", "malicious", "attack",
        "jailbreak", "prompt_injection"
    ]

    # Ưu tiên id2label
    for idx, lab in id2label.items():
        lab_n = norm(lab)
        if any(k in lab_n for k in positive_keywords):
            return int(idx)

    # Thử label2id
    for lab, idx in label2id.items():
        lab_n = norm(lab)
        if any(k in lab_n for k in positive_keywords):
            return int(idx)

    # Binary classifier -> lấy class 1
    if getattr(model.config, "num_labels", None) == 2:
        return 1

    # fallback
    return 1


_SENTINEL_POS_IDX = _find_positive_label_index(_sentinel_model)
_PROTECTAI_POS_IDX = _find_positive_label_index(_protectai_model)


@torch.no_grad()
def _score_model(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    pos_idx: int,
) -> Dict[str, Any]:
    # Lấy max length hợp lý
    model_max = getattr(model.config, "max_position_embeddings", None)
    if model_max is None:
        model_max = 512  # default an toàn

    enc = tokenizer(
        text,
        truncation=True,
        max_length=model_max,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1)

    score = float(probs[pos_idx].item())
    pred_idx = int(torch.argmax(probs).item())
    return {
        "score": score,
        "pred_label": str(model.config.id2label.get(pred_idx, pred_idx)),
        "pos_label_index": pos_idx,
        "pos_label_name_guess": str(model.config.id2label.get(pos_idx, pos_idx)),
    }


def detect_prompt_injection(text: str, threshold: float = 0.5) -> Dict[str, Any]:
    sentinel = _score_model(
        text,
        _sentinel_tokenizer,
        _sentinel_model,
        _SENTINEL_POS_IDX,
    )
    protectai = _score_model(
        text,
        _protectai_tokenizer,
        _protectai_model,
        _PROTECTAI_POS_IDX,
    )

    sentinel_is = sentinel["score"] >= threshold
    protectai_is = protectai["score"] >= threshold

    return {
        "input": text,
        "sentinel": {
            "is_injection": sentinel_is,
            **sentinel,
        },
        "protectai": {
            "is_injection": protectai_is,
            **protectai,
        },
        "combined_decision": bool(sentinel_is or protectai_is),
        "threshold": threshold,
        "device": str(device),
    }




def load_tool_outputs(dataset_path: str) -> List[str]:
    """
    Load your generated dataset JSON and extract tool_output as a list of plain text strings.
    Expected JSON format: a list of objects with key "tool_output".
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of testcases.")

    tool_outputs: List[str] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        out = item.get("tool_output", "")
        # Ensure it is plain text
        if out is None:
            out = ""
        tool_outputs.append(str(out))

    return tool_outputs


def pretty_print_one(idx: int, result: Dict[str, Any]) -> None:
    print("=" * 80)
    print(f"Tool output #{idx}")
    print("Sentinel:", result.get("sentinel", {}))
    print("ProtectAI:", result.get("protectai", {}))
    print("Combined decision:", result.get("combined_decision", None))
    print("=" * 80)


def run_detector_on_dataset(
    dataset_path: str,
    threshold: float = 0.5,
    max_samples: int = 0,   # 0 means run all
) -> None:
    tool_outputs = load_tool_outputs(dataset_path)

    if max_samples and max_samples > 0:
        tool_outputs = tool_outputs[:max_samples]

    # Simple stats
    total = 0
    flagged_sentinel = 0
    flagged_protectai = 0
    flagged_combined = 0

    for idx, item in enumerate(tool_outputs, start=1):
        total += 1
        result = detect_prompt_injection(text=item, threshold=threshold)

        s = result.get("sentinel", {})
        p = result.get("protectai", {})

        if s.get("is_injection"):
            flagged_sentinel += 1
        if p.get("is_injection"):
            flagged_protectai += 1
        if result.get("combined_decision"):
            flagged_combined += 1

        pretty_print_one(idx, result)

    # Summary
    print("\n" + "#" * 80)
    print("SUMMARY")
    print(f"Dataset: {dataset_path}")
    print(f"Threshold: {threshold}")
    print(f"Total samples tested: {total}")
    print(f"Flagged by Sentinel: {flagged_sentinel}/{total}")
    print(f"Flagged by ProtectAI: {flagged_protectai}/{total}")
    print(f"Flagged by Combined: {flagged_combined}/{total}")
    print("#" * 80)


if __name__ == "__main__":
    run_detector_on_dataset(
        dataset_path=DATASET_PATH,
        threshold=0.5,
        max_samples=0  # set e.g. 5 to test only first 5
    )
