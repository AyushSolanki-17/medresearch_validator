"""
Simple data loader for NIH Chest X-ray dataset.
"""

import random
from datasets import load_dataset


class NIHDataLoader:
    def __init__(self, limit:int=100):
        stream = load_dataset(
            "BahaaEldin0/NIH-Chest-Xray-14",
            split="train",
            streaming=True
        )
        self.dataset = list(stream.take(limit))
        # self.dataset  = load_dataset("BahaaEldin0/NIH-Chest-Xray-14", split="train[:100]")

    def sample(self):
        """
        Returns one random sample from dataset.
        """
        sample = random.choice(self.dataset)

        return {
            "labels": sample.get("labels", []),
            "image": sample.get("image", None)
        }