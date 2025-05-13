# Teleoperation server

## Setup and run

We instead use the [uv package manager](https://github.com/astral-sh/uv), which handles gracefully python version and dependencies. Think of uv like an alternative to pip, venv, and pyenv.

1. Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Pin python version
   In the teleop folder run:

```bash
uv python pin 3.10
```

3. Run the fastapi server

```bash
cd ..
make
```

### Run locally on port `8080` with simulation

```bash
cd ..
make local
```

### Run using the CLI

This program is a command line interface

```bash
# Display cool logo and check config
uv run teleop info
# Display help
uv run teleop --help
# Run on port 8080
uv run teleop run --port 8080
```

### How to add dependencies?

To install a new Python package, use uv:

```bash
cd teleop
sudo uv add numpy
```

Keep the dependencies lean, use optional dependency groups: `dev` and `test`. To add to one of those groups, do this:

```bash
sudo uv add mypy --optional dev
```

### Robot logic

The robot logic is in `hardware/base.py`

### GUI vs headless mode

In GUI mode, the FastAPI server launches `simulation/pybullet`.

#### Stop

If you have the error "Cannot connect to pybullet server" in GUI mode, do this

```bash
# Stop, then relaunch
make stop
make
```

And then relaunch.

## Build binary

We use [nuitka](https://github.com/Nuitka/Nuitka) to build the project into a binary.

```bash
cd ..
make build
```

This takes about 10min on a MacBook pro. The result is a binary `main.bin`

This binary only works for the same platform: eg MacOS. To compile it on Linux, you need to run a Linux machine.
