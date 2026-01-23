"""Prediction and Heatmap functions"""
import os
from pathlib import Path
import h5py
from trident import OpenSlideWSI, visualize_heatmap


class HeatmapVisualizer:
    def __init__(self, slide_id, slide_folder, features_folder, attn_scores_folder, results_dir):
        self.slide_id = slide_id #Slide filename
        self.slide_folder = slide_folder
        self.features_folder = features_folder
        self.results_dir = results_dir
        self.attn_scores_folder = attn_scores_folder #Attention scores for one slide

    @staticmethod
    def drop_extension(filepath):
        filename = Path(filepath)
        return filename.stem

    def _load_slide(self):
        slide_path = os.path.join(self.slide_folder, self.slide_id)
        slide = OpenSlideWSI(slide_path=slide_path, lazy_init=False)
        return slide

    def _load_attention_scores(self):
        attention_scores_path = os.path.join(self.attn_scores_folder, self.drop_extension(self.slide_id) + ".h5")
        with h5py.File(attention_scores_path, 'r') as f:
            attention_scores = f['attention_scores'][:]
        return attention_scores

    def _load_coords(self):
        features_path = os.path.join(self.features_folder, self.drop_extension(self.slide_id) + ".h5")
        with h5py.File(features_path, 'r') as f:
            coords = f['coords'][:]
            coords_attrs = dict(f['coords'].attrs)
            return coords, coords_attrs

    def run(self):
        slide = self._load_slide()
        coords, coords_attrs = self._load_coords()
        attention_scores = self._load_attention_scores()
        os.makedirs(self.results_dir, exist_ok=True)
        visualize_heatmap(
            wsi=slide,
            scores=attention_scores,
            coords=coords,
            vis_level=1,
            patch_size_level0=coords_attrs['patch_size_level0'],
            normalize=True,
            num_top_patches_to_save=20,
            output_dir=self.results_dir,
            filename="heatmap_" + self.drop_extension(self.slide_id) + ".png"
        )