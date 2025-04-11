# Le Robot Dataset Toolkit

This folder holds scripts to repair/update your Le Robot datasets.

This folder will help you:

- delete episodes
- repair a broken Le Robot dataset
- push your dataset to Hugging Face

## Pre-requisites

Install [uv](https://docs.astral.sh/uv/), a Python environment manager.
This handles the dependancies.

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Usage

### Delete episodes from your dataset

```bash
uv run repair_dataset.py --dataset-path <local_dataset_path> --indexes-to-delete <indexes_to_delete_comma_separated>
```

For example: uv run repair_dataset.py --dataset-path ~/phosphobot/recordings/lerobot_v2/example_dataset --indexes-to-delete 5,8

This will:

- delete the given parquets, videos associated to the deleted indexes
- correct the episode_index, frame_index and index
- rename the parquets/video files to order them
- compute the meta files
- upload your dataset to Hugging Face

### Repair a dataset

```bash
uv run repair_dataset.py --dataset-path <local_dataset_path>
```

For example: uv run repair_dataset.py --dataset-path ~/phosphobot/recordings/lerobot_v2/example_dataset

This will:

- correct the episode_index, frame_index and index
- rename the parquets/video files to order them
- compute the meta files
- upload your dataset to Hugging Face

### Upload to Hugging face

```bash
uv run push_dataset_to_hf.py <dataset_path> <dataset_name>
```

For example: uv run push_dataset_to_hf.py ~/phosphobot/recordings/lerobot_v2/example_dataset example_name

This will:

- upload the given dataset to hugging face and create a v2.0 branch with the same data

Note: make sure to have a hugging face token account and token

## Limitations

These scripts have been written with the v2.0 format in mind and are not yet compatible with v2.1
