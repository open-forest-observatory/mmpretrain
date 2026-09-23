from mmengine.evaluator import BaseMetric
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings

from mmpretrain.registry import METRICS

@METRICS.register_module()
class TreeLevelAccuracy(BaseMetric):
    """Tree-level accuracy metric by aggregating predictions across multiple views."""

    def __init__(self, metadata_csv, classes, **kwargs):
        """
        Args:
            metadata_csv (str): Path to CSV containing metadata mapping images to trees
                and their ground-truth species labels.
                Must contain columns: ['image_path', 'class', 'tree_id', 'dataset_id'].
                - `image_path`: the absolute path to the image
                - `tree_id`: the string representation of the tree's unique ID within a dataset
                - `dataset_id`: the string representation of which dataset is being used
                - `class`: the groundtruth class of the tree. Note this is assumed to be the same across all rows which have the same `tree_id`-`dataset_id` pairing, but this is not checked.
            classes (list[str]): List of class names in the same order as dataset.
        """
        super().__init__(**kwargs)
        self.classes = classes
        # Create a mapping from class name -> integer index
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        # Load metadata
        df = pd.read_csv(metadata_csv)
        df['class'] = df['class'].astype(str)
        df['tree_id'] = df['tree_id'].astype(str)
        df['dataset_id'] = df['dataset_id'].astype(str)
        df['image_path'] = df['image_path'].astype(str)

        # Create a unique tree identifier by combining dataset_id and tree_id
        # Note: 'tree_id' alone is unique only to its dataset. The validation metadata
        # file includes trees from all datasets so there can be multiple trees with the same tree_id
        df['global_tree_id'] = df['dataset_id'] + '_' + df['tree_id']

        # Map each image_id -> global_tree_id (for grouping predictions later)
        self.img2tree = dict(zip(df['image_path'], df['global_tree_id']))

        # Map each global_tree_id -> ground-truth label index
        # Note that this will only take the information from the last row in each global_tree_id
        # but this is ok because all the chips from a given global_id should have the same class.
        self.tree2label = {
            row['global_tree_id']: self.class_to_idx[row["class"]]
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
            # Convert prediction tensor to numpy array
            pred = sample['pred_score'].cpu().numpy()

            # Append prediction record with tree association
            self.results.append({
                'img_path': img_path,
                'tree_id': self.img2tree[img_path],
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
            counts = np.bincount(per_img_labels)
            # Add tiny random noise for fair tie-breaking since np.argmax will always take the first index
            counts = counts + np.random.random(len(counts)) * 0.5
            vote_label = counts.argmax()

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