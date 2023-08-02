from clearml import Task


task = Task.create(
    project_name='AIHub/eval',
    task_name='Eval',
    repo='git@github.com:ivanmasan/evaluation.git',
    branch='master',
    docker_args="--mount type=bind,source='/home/photoneo/Data/ai_hub',target='/home/ai_hub'",
    script='keypoint_eval.py',
    docker_bash_setup_script='export PYTHONPATH=$PYTHONPATH:ML_framework',
    docker='photoneo/ml_framework:v0.1.0-cu102'
)

params = {
    'Args/dataset_path': '/home/ai_hub/datasets/groceries/edelia/test/production',
    'Args/model_name': 'RCNN',
    'Args/model_path': '/home/ai_hub/output/training/instance/groceries/rohlik/20221207',
    'Args/roi_model_path': '/home/ai_hub/output/training/roiseg/box_finder_221110',
    'Args/secondary_model_name': 'M2F',
    'Args/secondary_model_path': '/home/ai_hub/output/training/instance/groceries/rohlik/production_m2f/model_final.pth'
}

task.set_parameters(params)
