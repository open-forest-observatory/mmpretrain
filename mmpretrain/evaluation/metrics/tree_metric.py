from mmengine.evaluator import BaseMetric
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings

from mmpretrain.registry import METRICS

@METRICS.register_module()
class TreeLevelAccuracy(BaseMetric):
    """Tree-level accuracy metric by aggregating predictions across multiple views."""

    def __init__(self, metadata_csv, level, classes, **kwargs):
        """
        Args:
            metadata_csv (str): Path to CSV containing metadata mapping images to trees
                and their ground-truth species labels.
                Must contain columns: ['image_id', 'tree_unique_id', 'species_l1', 'species_l2', 'species_l3', 'species_l4'].
            level (str): Species lumping level to use for evaluation ('l1', 'l2', 'l3', or 'l4').
            classes (list[str]): List of class names in the same order as dataset.
        """
        super().__init__(**kwargs)
        self.level = level
        self.classes = classes
        # Create a mapping from class name -> integer index
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        # Load metadata
        df = pd.read_csv(metadata_csv)
        df['image_id'] = df['image_id'].astype(str)
        df['tree_unique_id'] = df['tree_unique_id'].astype(str)
        
        # Use the species column corresponding to the specified level
        species_col = f'species_{level}'
        df[species_col] = df[species_col].astype(str)

        # Map each image_id -> tree_unique_id (for grouping predictions later)
        self.img2tree = dict(zip(df['image_id'], df['tree_unique_id']))

        # Map each tree_unique_id -> ground-truth label index
        self.tree2label = {
            row['tree_unique_id']: self.class_to_idx[row[species_col]]
            for _, row in df.iterrows()
        }

        self.results = []

    def process(self, data_batch, data_samples):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader. Currently unused.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for sample in data_samples:
            img_path = sample['img_path']
            img_id = img_path.split('/')[-1].split('.')[0]  # filename without extension
            # Convert prediction tensor to numpy array
            pred = sample['pred_score'].cpu().numpy()

            # Append prediction record with tree association
            self.results.append({
                'img_id': img_id,
                'tree_id': self.img2tree[img_id],
                'pred': pred
            })

    def compute_metrics(self, results):
        """Aggregate predictions per tree and compute accuracy.
        
        Args:
            results (list[dict]): The processed results of each batch.

        Returns:
            dict: Dictionary with tree-level accuracy values.
        """

        # For every tree (key) append predictions from all of its images to a single list (value)
        tree_preds = defaultdict(list)
        for r in results:
            tid = r['tree_id']
            tree_preds[tid].append(r['pred'])

        # Variables to track micro accuracy
        mean_correct = 0
        vote_correct = 0
        total = 0

        # Track macro (per-class) accuracy
        num_classes = len(self.classes)
        mean_correct_per_class = np.zeros(num_classes, dtype=int)
        mean_total_per_class = np.zeros(num_classes, dtype=int)

        vote_correct_per_class = np.zeros(num_classes, dtype=int)
        vote_total_per_class = np.zeros(num_classes, dtype=int)

        # Compute predictions per tree
        for tid, preds in tree_preds.items():
            preds = np.array(preds)
            gt = self.tree2label[tid]

            # 1. Mean-probability aggregation
            # Average predicted probabilities across all images and then select the class with highest mean probability
            mean_pred = preds.mean(axis=0)
            mean_label = np.argmax(mean_pred)

            total += 1
            mean_total_per_class[gt] += 1
            if mean_label == gt:
                mean_correct += 1  # micro
                mean_correct_per_class[gt] += 1  # macro

            # 2. Majority voting aggregation
            # Compute predicted label for each image and then find the most common label
            per_img_labels = np.argmax(preds, axis=1)
            # NOTE: If two classes have the same count (tie-breaking) argmax() picks the lowest index
            vote_label = np.bincount(per_img_labels).argmax()

            # micro
            if vote_label == gt:
                vote_correct += 1

            # macro
            vote_total_per_class[gt] += 1
            if vote_label == gt:
                vote_correct_per_class[gt] += 1

        # Identify classes with zero samples for macro-metric calculation
        excluded_classes = [self.classes[c] for c in range(num_classes)
                         if mean_total_per_class[c] == 0]
        if excluded_classes:
            warnings.warn(
                f"Excluded {len(excluded_classes)} classes from macro acc due to zero samples: {excluded_classes}",
                UserWarning,
            )

        # Compute macro accuracies by averaging per-class accuracies
        mean_macro = np.mean([
            mean_correct_per_class[c] / mean_total_per_class[c]
            for c in range(num_classes) if mean_total_per_class[c] > 0
        ])

        vote_macro = np.mean([
            vote_correct_per_class[c] / vote_total_per_class[c]
            for c in range(num_classes) if vote_total_per_class[c] > 0
        ])

        return {
            # Micro
            "tree_acc_mean_micro": mean_correct / total,
            "tree_acc_vote_micro": vote_correct / total,

            # Macro (per-class)
            "tree_acc_mean_macro": mean_macro,
            "tree_acc_vote_macro": vote_macro,
        }
