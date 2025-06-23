# Feetech STS3215 utility scripts

## Installation

1. [Clone or download this git repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)

```bash
git clone git@github.com:phospho-app/phosphobot.git
```

2. Install [uv.](https://docs.astral.sh/uv/)

MacOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows:

```pwh
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installing uv, restart your terminal.

3. You can now run the script `filename.py` with `un run filename.py`

## Initializing a Feetech STS3215 servomotor

This procedure is useful to replace a broken servomotor with a new one.

1. Connect the servo to the waveshare servo bus. Connect the servo bus to the power and to your computer using USB C.

2. Find out the motor bus of your waveshare servo bus.

```
cd scripts/feetech
uv run find_motor_bus.py
```

You can also use `phosphobot info` to do that.

3. Then, to initialize a servo with a certain id, you do :

```bash
uv run configure_motor.py \
  --port /dev/cu.usbmodem58FA0823771 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1
```

After running the command, you should hear a spinning sound and the motor should go

**Note:** Change the **port** depending on the value of `find_motor_bus.py` and and the **id** depending on what servo you're programming!

On the SO-100 and SO-101 robot, the ids of the motors start at 1 (base) and go up to 6 (gripper).

**Note**: On windows, you need to remove the breaklines and remove the backlashes:

```pwh
uv run configure_motor.py --port COM4 --brand feetech --model sts3215 --baudrate 1000000 --ID 1
```

4. After programming your servo and replacing it, you then need to **recalibrate** your robot arm. ([Example with SO-100 and phosphobot](https://www.youtube.com/watch?v=65DW8yLcRmM))
