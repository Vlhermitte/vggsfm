# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import hydra
import pycolmap
from omegaconf import DictConfig, OmegaConf
import warnings
warnings.filterwarnings("ignore")

from vggsfm.runners.runner import VGGSfMRunner
from vggsfm.datasets.demo_loader import DemoLoader
from vggsfm.utils.utils import seed_all_random_engines


@hydra.main(config_path="cfgs/", config_name="demo_dense")
def demo_fn(cfg: DictConfig):
    """
    Main function to run the VGGSfM demo. VGGSfMRunner is the main controller.
    """

    OmegaConf.set_struct(cfg, False)

    # Print configuration
    print("Model Config:", OmegaConf.to_yaml(cfg))

    # Configure CUDA settings
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Set seed for reproducibility
    seed_all_random_engines(cfg.seed)

    # Initialize VGGSfM Runner
    vggsfm_runner = VGGSfMRunner(cfg)

    # Load Data
    test_dataset = DemoLoader(
        SCENE_DIR=cfg.SCENE_DIR,
        img_size=cfg.img_size,
        normalize_cameras=False,
        load_gt=cfg.load_gt,
    )

    sequence_list = test_dataset.sequence_list

    seq_name = sequence_list[0]  # Run on one Scene

    # Load the data for the selected sequence
    batch, image_paths = test_dataset.get_data(
        sequence_name=seq_name, return_path=True
    )

    if cfg.OUTPUT_DIR is not None:
        output_dir = cfg.OUTPUT_DIR
    else:
        output_dir = batch[
            "scene_dir"
        ]  # which is also cfg.SCENE_DIR for DemoLoader

    images = batch["image"]
    masks = batch["masks"] if batch["masks"] is not None else None
    crop_params = (
        batch["crop_params"] if batch["crop_params"] is not None else None
    )

    # Cache the original data for visualization, so that we don't need to re-load many times
    original_images = batch["original_images"]

    sparse_reconstruction = pycolmap.Reconstruction(output_dir)

    predictions = dict()
    predictions["reconstruction"] = sparse_reconstruction

    prediction  = vggsfm_runner.extract_sparse_depth_and_point_from_reconstruction(predictions)

    # Run dense reconstruction
    predictions = vggsfm_runner.dense_reconstruct(
                    predictions, image_paths, original_images
                )

    vggsfm_runner.save_dense_depth_maps(predictions["depth_dict"], output_dir)

    print("Demo Finished Successfully")

    return True


if __name__ == "__main__":
    with torch.no_grad():
        demo_fn()
