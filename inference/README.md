# Inference scripts

These scripts will help you do the inference of a trained model to control your robot.

We assume you have:

- a SO-100 controlled by phosphobot
- a trained model uploaded to Hugging Face

We recommend you clone this repo to access and run the inference scripts.
You will also need to clone the lerobot repo.

```bash
git clone https://github.com/phospho-app/phosphobot.git
git clone https://github.com/huggingface/lerobot.git
```

# Setup a server

## Deploy an ACT model

If you use pip, just run

```bash
cd phosphobot/inference
pip install .
python ACT/server.py --model_id=<YOUR_HF_DATASET_NAME>
```

This will load your model and create an `/act` endpoint that expects the robot's current position and images. If using a local model, you can pass the path of a local model instead of a HF model with the "--model_id=..." flag.

This server script is intended to run on a machine with a GPU, separately from your phosphobot installation.

You can edit the URL and Port parameters in your phohsphobot Admin panel to connect your robot to the inference server. By default, these should be localhost and 8080.

## Inference on Pi0 models

Follow the instruction in the [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) repo to setup your server for inference.

# Call your inference server from a python script

We provide clients for ACT servers and Pi0 servers.
You can implement the `ActionModel` class with you own logic [here](phosphobot/am/models.py).

At this point, go to your phosphobot dashboard > docs and launch the auto/start endpoint that will communicate with the inference server to automatically control the robot.

You can stop at any time by calling the auto/stop endpoint.

### Notes

If running the inference script on a different machine than what you use to control your so-100 robot, you will need to change the URL in the admin panel. Set it to the URL of the machine running the inference script.

You can also change the port of the inference server by passing "--port=..."

If using a local model, you can pass the path of a local model instead of a HF model with the "--model_id=..." flag.
