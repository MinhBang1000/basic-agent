#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Menu-based pipeline:
Benign Logs  -> Prefix Samples -> Train/Test JSONL
(for Layer-1 Anomaly Detection)

- Conversation-level shuffle + split
- Prefix expansion (N nodes -> N samples)
- Lowercase + strip + normalize whitespace (NO truncate)
- Deduplicate samples
"""

import json
import random
import hashlib
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple


# =========================
# Data structure
# =========================

@dataclass
class PrefixSample:
    sample_id: str
    origin_id: str
    source: str
    content: str
    label: int
    meta: Dict[str, Any]


# =========================
# Builder class
# =========================

class BenignPrefixDatasetBuilder:
    def __init__(
        self,
        test_ratio: float,
        seed: int,
        dedup: bool,
        sep_token: str = "[SEP]",
    ):
        self.test_ratio = test_ratio
        self.seed = seed
        self.dedup = dedup
        self.sep_token = sep_token
        random.seed(seed)

    # ---------- IO ----------

    def load_logs(self, path: Path) -> List[Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text.startswith("["):
            return json.loads(text)

        logs = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
        return logs

    # ---------- Preprocess ----------

    def normalize(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip().lower()

    def tag(self, name: str, text: str) -> str:
        return f"[{name}] {self.normalize(text)}"

    # ---------- Node extraction ----------

    def extract_nodes(self, conv: Dict[str, Any]) -> List[str]:
        nodes = []

        if conv.get("prompt"):
            nodes.append(self.tag("PROMPT", conv["prompt"]))

        for step in conv.get("reasoning_steps", []):
            node = step.get("node", "")
            typ = step.get("type", "")
            content = step.get("content", "")

            if not content or not content.strip():
                continue

            if node == "retrieve":
                nodes.append(self.tag("MEMORY", content))
            elif typ == "tool":
                tool = step.get("tool", "unknown")
                nodes.append(self.tag(f"TOOL:{tool}", self._tool_text(content)))

        if conv.get("final_answer"):
            nodes.append(self.tag("REASON", conv["final_answer"]))

        return nodes

    def _tool_text(self, content: Any) -> str:
        if isinstance(content, dict):
            return str(content.get("result", content))
        if isinstance(content, str):
            try:
                obj = json.loads(content)
                if isinstance(obj, dict):
                    return str(obj.get("result", obj))
            except Exception:
                pass
        return str(content)

    # ---------- Prefix expansion ----------

    def build_prefixes(self, conv: Dict[str, Any]) -> List[PrefixSample]:
        nodes = self.extract_nodes(conv)
        samples = []

        for i in range(1, len(nodes) + 1):
            content = f"\n{self.sep_token}\n".join(nodes[:i])
            samples.append(
                PrefixSample(
                    sample_id=f"{conv['id']}_p{i}",
                    origin_id=conv["id"],
                    source=conv.get("source", "unknown"),
                    content=content,
                    label=0,
                    meta={
                        "prefix_index": i,
                        "num_nodes": len(nodes),
                        "trace": conv.get("trace", []),
                    },
                )
            )
        return samples

    # ---------- Dedup ----------

    def deduplicate(self, samples: List[PrefixSample]) -> List[PrefixSample]:
        seen = set()
        out = []
        for s in samples:
            h = hashlib.sha256(s.content.encode("utf-8")).hexdigest()
            if h not in seen:
                seen.add(h)
                out.append(s)
        return out

    # ---------- Pipeline ----------

    def run(self, logs: List[Dict[str, Any]]) -> Tuple[List[PrefixSample], List[PrefixSample]]:
        random.shuffle(logs)
        n_test = max(1, int(len(logs) * self.test_ratio))

        test_logs = logs[:n_test]
        train_logs = logs[n_test:]

        train_samples, test_samples = [], []

        for c in train_logs:
            train_samples.extend(self.build_prefixes(c))
        for c in test_logs:
            test_samples.extend(self.build_prefixes(c))

        if self.dedup:
            train_samples = self.deduplicate(train_samples)
            test_samples = self.deduplicate(test_samples)

        return train_samples, test_samples


# =========================
# Menu
# =========================

def ask(prompt, cast=str, default=None):
    msg = f"{prompt}"
    if default is not None:
        msg += f" [{default}]"
    msg += ": "
    val = input(msg).strip()
    if not val and default is not None:
        return default
    return cast(val)


# =========================
# Main
# =========================

def main():
    print("\n=== Prefix Dataset Builder (Layer-1 AD) ===\n")

    input_path = Path(ask("Path to input logs (.jsonl or .json)", str))
    out_dir = Path(ask("Output directory", str, "out_dataset"))
    test_ratio = ask("Test ratio", float, 0.2)
    seed = ask("Random seed", int, 42)
    dedup = ask("Deduplicate samples? (y/n)", str, "y").lower() == "y"

    out_dir.mkdir(parents=True, exist_ok=True)

    builder = BenignPrefixDatasetBuilder(
        test_ratio=test_ratio,
        seed=seed,
        dedup=dedup,
    )

    logs = builder.load_logs(input_path)
    print(f"\nLoaded {len(logs)} conversations")

    train, test = builder.run(logs)

    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")

    with test_path.open("w", encoding="utf-8") as f:
        for s in test:
            f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")

    print("\n=== DONE ===")
    print(f"Train samples: {len(train)} -> {train_path}")
    print(f"Test  samples: {len(test)} -> {test_path}")

    if train:
        print("\n--- Preview (first train sample) ---\n")
        print(train[0].content[:1000])


if __name__ == "__main__":
    main()
