import json
from dataclasses import dataclass
from pathlib import Path

import click
import cv2
import numpy as np
from PIL import Image
from clearml import Task, Logger
import matplotlib
from matplotlib import pyplot as plt
from ml_framework.anypick.actions.base import AnyPickRequest
from ml_framework.anypick.actions.predictions.detectron2_model import CropPolicy
from ml_framework.anypick.actions.predictions.selector import PredictionsActions
from ml_framework.anypick.actions.roi_2d.selector import ROI2DActions
from ml_framework.anypick.pipeline import AnyPickPipelineConfig, AnyPickPipeline
from ml_framework.training.train import TrainConfig
from ml_framework.utils.file_io import Scene
from omegaconf import OmegaConf
from tqdm import tqdm


@dataclass
class ImageEval:
    file: Path
    pred_keypoints: np.ndarray
    true_keypoints: np.ndarray

    def __post_init__(self):
        self.distance = _evaluate_closeness(self.true_keypoints, self.pred_keypoints)
        self.utility = _calculate_utility(self.distance)


def _resolve_model_file_path(path):
    if path.suffix == '':
        return path, 'model.pth'
    elif path.suffix == '.pth':
        return path.parent, path.parts[-1]
    else:
        raise ValueError("path to model needs to e a directory or end with `.pth`. "
                         f"Received {path}")


def _create_pipeline(model_path, roi_model_path):
    model_dir, model_name = _resolve_model_file_path(model_path)
    roi_model_dir, roi_model_name = _resolve_model_file_path(roi_model_path)

    config = AnyPickPipelineConfig()

    config.predictions.action = PredictionsActions.detectron2
    config.predictions.detectron2.crop_policy = CropPolicy.default
    config.predictions.detectron2.config = str(model_dir / 'config.yaml')
    config.predictions.detectron2.weights = str(model_dir / model_name)
    train_config: TrainConfig = OmegaConf.load(model_dir / 'train_config.yaml')
    config.predictions.detectron2.dataset_mapper_config = train_config.dataset_mapper

    config.roi_2d.action = ROI2DActions.detectron2
    config.roi_2d.detectron2.crop_policy = CropPolicy.none
    config.roi_2d.detectron2.config = str(roi_model_dir / 'config.yaml')
    config.roi_2d.detectron2.weights = str(roi_model_dir / roi_model_name)

    return AnyPickPipeline(config)


def _predict_image(pipeline, image):
    scene = Scene(image)
    anypick_request = AnyPickRequest(scene, None)
    anypick_response = pipeline.run_pipeline(anypick_request)
    return anypick_response.predictions


def _extract_keypoints(annotations):
    keypoints = []
    for annotation in annotations['annotations']:
        keypoint = np.array(annotation['keypoints'][:2])
        keypoints.append(keypoint)

    return np.stack(keypoints)


def _extract_top_keypoint_predictions(predictions):
    sorted_idx = np.argsort(predictions.scores)[-3:]
    return predictions.keypoints[sorted_idx, 0, :2]


def _evaluate_closeness(true_keypoints, pred_keypoints):
    diffs = true_keypoints[:, None] - pred_keypoints[None]
    distances = np.sqrt((diffs ** 2).sum(2))
    min_distances = np.min(distances, axis=0)
    min_distances = np.concatenate([min_distances, np.full(3 - len(min_distances), fill_value=np.inf)])

    return min_distances


def _produce_results(pipeline, files):
    results = []

    for file in tqdm(files):
        image = cv2.imread(file.as_posix(), cv2.IMREAD_GRAYSCALE)
        annotations_path = file.parent / f'{file.stem}.coco.json'

        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        true_keypoints = _extract_keypoints(annotations)

        predictions = _predict_image(pipeline, image)
        pred_keypoints = _extract_top_keypoint_predictions(predictions)

        results.append(ImageEval(
            file=file,
            true_keypoints=true_keypoints,
            pred_keypoints=pred_keypoints,
        ))
    return results


def _calculate_utility(distance):
    pick_probs = _pick_probability(distance)
    fail_probs = 1 - pick_probs
    utility = [1, 0.975, 0.95, 0]

    return (
        pick_probs[0] * utility[0]
        + (fail_probs[0] * pick_probs[1] * utility[1])
        + (np.product(fail_probs[:2]) * pick_probs[2] * utility[2])
        + (np.product(fail_probs) * utility[3])
    )


def _pick_probability(distance):
    variable_part = 1 / (1 + np.exp((distance - 45) / 7))
    return 0.98 * variable_part


def _visualize_two_models(main_results, secondary_results, model_names):
    _plot_graphs(main_results, secondary_results, model_names=model_names)
    _plot_faulty_images(main_results, secondary_results)

    Logger.current_logger().report_single_value("Model Utility",
                                                _utility(main_results).mean())
    Logger.current_logger().report_single_value("Secondary Model Utility",
                                                _utility(secondary_results).mean())


def _visualize_single_results(main_results, model_name):
    _plot_graphs(main_results, model_names=[model_name])
    _plot_faulty_images(main_results)

    Logger.current_logger().report_single_value("Model Utility", _utility(main_results).mean())


def _distance(results):
    return np.stack([x.distance for x in results])


def _utility(results):
    return np.array([x.utility for x in results])


def _plot_graphs(*results, model_names, clip_value=70):
    distances = [_distance(r) for r in results]
    for i in range(3):
        for model, distance in zip(model_names, distances):
            min_distance = np.clip(np.min(distance[:, :(i + 1)], axis=1), 0, clip_value)
            plt.hist(min_distance, bins=np.linspace(0, clip_value, 40))
        plt.legend(model_names)
        plt.title(f"Lowest Distance Top {i + 1} predictions")
        plt.show()


def _plot_image(*image_evals):
    image = cv2.imread(image_evals[0].file.as_posix(), cv2.IMREAD_GRAYSCALE)
    coloured_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for x, y in image_evals[0].true_keypoints:
        coloured_image[y-10:y+10, x-10:x+10] = (0, 255, 0)

    colour_list = [(255, 0, 0), (0, 0, 255)]
    for image_eval, colour in zip(image_evals, colour_list):
        pred_keypoints = image_eval.pred_keypoints.astype(int)
        for i, (x, y) in enumerate(pred_keypoints):
            o = 9 - i
            coloured_image[y-o:y+o, x-o:x+o] = colour

    Logger.current_logger().report_image(
        "Bad Images",
        image_evals[0].file.stem,
        iteration=0,
        image=coloured_image
    )


def _faulty_images_idx(result, utility_loss_ratio=0.8):
    utility = _utility(result)
    utility_lost = 1 - utility
    sorted_utility_loss = np.sort(utility_lost)[::-1]
    idx = np.searchsorted(np.cumsum(sorted_utility_loss),
                          utility_lost.sum() * utility_loss_ratio)
    idx = np.minimum(idx, len(sorted_utility_loss) - 1)
    threshold = sorted_utility_loss[idx]

    return np.where(utility_lost >= threshold)[0]


def _plot_faulty_images(*results):
    images_to_plot = set()
    for result in results:
        images_to_plot |= set(_faulty_images_idx(result))

    for idx in images_to_plot:
        _plot_image(*[r[idx] for r in results])


@click.command()
@click.option('--model-path', help='Path of the model to be evaluated')
@click.option('--secondary-model-path', default='',
              help='Path to second model to be evaluated and compared against. '
                   'If none is provided no comparison is made')
@click.option('--roi-model-path', help='Path to roi model')
@click.option('--dataset-path', help='Path to dataset location')
@click.option('--model-name', default='Model',
              help='Name of model. Used for graph labeling')
@click.option('--secondary-model-name', default='Secondary Model',
              help='Name of secondary model. Used for graph labeling')
def main(
        model_path,
        secondary_model_path,
        roi_model_path,
        dataset_path,
        model_name,
        secondary_model_name
):
    if Task.current_task() is None:
        Task.init(project_name='clustering/test')

    model_path = Path(model_path)
    secondary_model_path = Path(secondary_model_path) if secondary_model_path else None
    roi_model_path = Path(roi_model_path)
    dataset_path = Path(dataset_path)

    files = list(dataset_path.rglob("*.png"))

    main_pipeline = _create_pipeline(model_path, roi_model_path)
    main_results = _produce_results(main_pipeline, files)
    del main_pipeline

    if secondary_model_path is not None:
        secondary_pipeline = _create_pipeline(secondary_model_path, roi_model_path)
        secondary_results = _produce_results(secondary_pipeline, files)
        del secondary_pipeline

        _visualize_two_models(main_results, secondary_results, [model_name, secondary_model_name])
    else:
        _visualize_single_results(main_results, model_name)


if __name__ == "__main__":
    matplotlib.use("Agg")
    plt.rcParams['figure.dpi'] = 400
    main()
