# phosphobot

**phosphobot** is a community-driven platform that enables you to use action models to control your robot.

<div align="center">

<a href="https://pypi.org/project/phosphobot/"><img src="https://img.shields.io/pypi/v/phosphobot?style=flat-square&label=pypi+phospho" alt="phosphobot Python package on PyPi"></a>
<a href="https://www.ycombinator.com/companies/phospho"><img src="https://img.shields.io/badge/Y%20Combinator-W24-orange?style=flat-square" alt="Y Combinator W24"></a>
<a href="https://discord.gg/cbkggY6NSK"><img src="https://img.shields.io/discord/1106594252043071509" alt="phospho discord"></a>

</div>

## Overview

This repository contains demo code and community projects developed using the phospho starter pack. Whether you're a beginner or an experienced developer, you can explore existing projects or contribute your own creations.

## Getting started

### 1. Get a SO-100 robot

Purchase your Phospho starter pack at [robots.phospho.ai](https://robots.phospho.ai) or build your own robot following the instructions in [the SO-100 repo](https://github.com/TheRobotStudio/SO-ARM100).

### 2. Install the phosphobot server

```bash
# Install it this way
curl -fsSL https://raw.githubusercontent.com/phospho-app/phosphobot/main/install.sh | bash
# Start it this way
phosphobot run
# Upgrade it with brew or with apt
# sudo apt update && sudo apt install phosphobot
# brew update && brew upgrade phosphobot
```

### 3. Make your robot move for the first time!

Go to the webapp at `YOUR_SERVER_ADDRESS:YOUR_SERVER_PORT` (default is `localhost:80`) and click control.

You will be able to control your robot with:

- the keyboard
- a leader arm
- a Meta Quest if you have the phospho teleop app

### 4. Record a dataset

Record a 40 episodes dataset of the task you want the robot to learn.

Check out the [docs](https://docs.phospho.ai/basic-usage/dataset-recording) for more details.

### 5. Train an action model

Train an action model on the dataset you recorded.
If you want to train on your own machine, here are the requirements:

| GPU Memory | Model    | MPS Support |
| ---------- | -------- | ----------- |
| >16GB      | ACT      | ✅ Yes      |
| >70GB      | Gr00t n1 | ❌ No       |

To learn more about training action models for robotics, check out the [docs](https://docs.phospho.ai/basic-usage/training).

### 6. Use the model to control your robot

You can use the model you just trained to control your robot either:

- directly from the webapp
- from your own code using the HTTP API

Learn more [in the docs](https://docs.phospho.ai/basic-usage/inference).

## Examples

The `examples/` directory is the quickest way to see the toolkit in action. Check it out!
Proud of what you build? Share it with the community by opening a PR to add it to the `examples/` directory.

## Advanced Usage

You can directly call the phosphobot server from your own code, using the HTTP API and websocket API.

Go to the interactive docs of the API to use it interactively and learn more about it.
It is available at `YOUR_SERVER_ADDRESS:YOUR_SERVER_PORT/docs`. By default, it is available at `localhost:80/docs`.

We release new versions very often, so make sure to check the API docs for the latest features and changes.

## Supported Robots

We currently support the following robots:

- SO-100
- SO-101
- WX-250 by Trossen Robotics (beta)
- AgileX Piper (beta)

See this [README](phosphobot/README.md) for more details on how to add support for a new robots or open an issue.

## Join the Community

Connect with other developers and share your experience in our [Discord community](https://discord.gg/cbkggY6NSK)

## Support

- **Documentation**: Read the [documentation](https://docs.phospho.ai)
- **Community Support**: Join our [Discord server](https://discord.gg/cbkggY6NSK)
- **Issues**: Submit problems or suggestions through [GitHub Issues](https://github.com/phospho-app/phosphobot/issues)

## License

MIT License

---

Made with 💚 by the Phospho community
