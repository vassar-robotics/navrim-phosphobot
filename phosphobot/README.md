**phosphobot** – CLI Toolkit for Robot Teleoperation and Action Models
[![PyPI version](https://img.shields.io/pypi/v/phosphobot?style=flat-square)](https://pypi.org/project/phosphobot/)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?style=flat-square)](https://github.com/phospho-app/phosphobot)
[![Discord](https://img.shields.io/discord/1106594252043071509?style=flat-square)](https://discord.gg/cbkggY6NSK)

A simple, community-driven middleware for controlling robots, recording datasets, training action models.

All from your terminal or browser dashboard.

---

## Features

- **Easy Installation** via `pip` or the `uv` package manager
- **Web Dashboard**: Instant access to an interactive control panel for teleoperation
- **Dataset Recording**: Record expert demonstrations with a keyboard, in VR, or with a leader arm
- **Model Training & Inference**: Kick off training jobs or serve models through HTTP/WebSocket APIs

---

## Installation

### 1. Using pip

```bash
pip install phosphobot
```

### 2. Using [uv](https://github.com/astral-sh/uv)

If you already use `uv` to manage Python versions and deps:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add phosphobot to your project
uv add phosphobot
```

---

## Quick Start

Once installed, you can start the phosphobot server instantly.

```bash
# Verify installation and view info
phosphobot info

# Start the teleoperation server (default: localhost:80)
phosphobot run

# For custom port, e.g. 8080
phosphobot run --port 8080
```

If you’re managing via uv:

```bash
uv run phosphobot info
uv run phosphobot run
```

---

## Dashboard & Control

After launching the server, open your browser and navigate to:

```
http://<YOUR_SERVER_ADDRESS>:<PORT>/
```

By default, the address is [localhost:80](localhost:80)

Here you can:

- **Teleoperate** your robot via keyboard, leader arm, or Meta Quest
- **Record** demonstration datasets (40 episodes recommended)
- **Train** and **deploy** action models directly from the UI

---

## Start building

- **Docs**: Full user guide at [https://docs.phospho.ai](https://docs.phospho.ai)
- **Discord**: Join us on Discord for support and community chat: [https://discord.gg/cbkggY6NSK](https://discord.gg/cbkggY6NSK)
- **GitHub Repo**: [https://github.com/phospho-app/phosphobot](https://github.com/phospho-app/phosphobot)
- **Examples**: Browse [the examples](https://github.com/phospho-app/phosphobot/tree/main/examples)
- **Contribute**: Open a PR to expand the examples, support more robots, improve the tool

---

## Adding a New Robot

You can extend **phosphobot** by plugging in support for any custom robot. Just follow these steps:

1. **Clone the phosphobot repo and fetch submodules.** Make sure you have [git lfs](https://git-lfs.com) installed beforehand

   ```bash
   git clone https://github.com/phospho-app/phosphobot.git
   cd phosphobot
   git submodule update --init --recursive
   ```

2. **Install [uv](https://astral.sh/uv/)** to manage python dependencies. The recommended python version for dev is `3.10`

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install nvm and Node.js.** We recommend to manage Node versions via [nvm](https://github.com/nvm-sh/nvm).

   ```bash
   # Install nvm
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
   ```

   Then restart your terminal and verify:

   ```bash
   command -v nvm   # should output: nvm
   ```

   Finally, install the latest Node.js:

   ```bash
   nvm install node   # “node” is an alias for the latest version
   ```

4. **Build the app.** From the repo root, run:

   ```bash
   make
   ```

5. **Create your robot driver**

   1. In the directory `phosphobot/phosphobot/hardware` add a new file, e.g. `my_robot.py`. Inside, define a class inheriting from `BaseRobot`:

      ```python
      from phosphobot.hardware.base import BaseRobot

      class MyRobot(BaseRobot):
          def __init__(self, config):
              super().__init__(config)
              # Your initialization here

          ... # Implement the BaseRobot's abstract methods here
      ```

      We use pybullet as a robotics simulation backend. Make sure to add your robot's `urdf` in `phosphobot/resources/urdf`.

6. **Make your robot detectable**
   Open `phosphobot/phosphobot/robot.py` and locate the `RobotConnectionManager` class. Make sure your robot can be detected.

Build and run the app again and ensure your robot gets detected and can be moved. Happy with your changes? Open a pull request! We also recommend you look for testers on [our Discord](https://discord.gg/cbkggY6NSK).

---

## License

MIT License

Made with 💚 by the Phospho community.
