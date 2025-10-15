from mmengine.evaluator import BaseMetric
import pandas as pd
import numpy as np
from collections import defaultdict

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
            dict: Dictionary with tree-level accuracy as {'tree_acc': float}.
        """

        # For every tree (key) append predictions from all of its images to a single list (value)
        tree_preds = defaultdict(list)
        for r in results:
            tid = r['tree_id']
            tree_preds[tid].append(r['pred'])

        correct = 0
        total = 0
        # Combine per-image predictions into a single tree-level prediction
        # by averaging their softmax outputs (each image contributes equally).
        # The final tree label is the class with the highest mean confidence.
        # TODO: Experiment with confidence-weighted averaging
        for tid, preds in tree_preds.items():
            mean_pred = np.mean(preds, axis=0)
            pred_label = np.argmax(mean_pred)
            gt_label = self.tree2label[tid]  # ground-truth label

            if pred_label == gt_label:
                correct += 1
            total += 1

        return {'tree_acc': correct / total}