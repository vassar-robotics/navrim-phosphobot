# phosphobot

A community-driven platform for robotics enthusiasts to share and explore creative projects built with the Phospho Junior Dev Kit.

<div align="center">

<a href="https://pypi.org/project/phosphobot/"><img src="https://img.shields.io/pypi/v/phosphobot?style=flat-square&label=pypi+phospho" alt="phosphobot Python package on PyPi"></a>
<a href="https://www.ycombinator.com/companies/phospho"><img src="https://img.shields.io/badge/Y%20Combinator-W24-orange?style=flat-square" alt="Y Combinator W24"></a>
<a href="https://discord.gg/cbkggY6NSK"><img src="https://img.shields.io/discord/1106594252043071509" alt="phospho discord"></a>

</div>

## Overview

This repository contains demo code and community projects developed using the Phospho Junior Dev Kit. Whether you're a beginner or an experienced developer, you can explore existing projects or contribute your own creations.

## Getting Started

1. **Get Your Dev Kit**: Purchase your Phospho Junior Dev Kit at [robots.phospho.ai](https://robots.phospho.ai). Unbox it and set it up following the instructions in the box.

2. **Control your Robot**: Donwload the Meta Quest app, connect it to your robot, start teleoperating it.

3. **Record a Dataset**: Record a dataset using the app. Do the same gesture 30 times to create a dataset.

4. **Install the Package**:

```bash
pip install --upgrade phosphobot
```

5. **Train a Model**: Use Le Robot to train a policy on the dataset you just recorded.

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

Add the `configs/policy/act_so100_phosphobot.yaml`file from this repository to the `lerobot/configs/policy` directory in the `lerobot` repository.

Launch the training script with the following command from the `lerobot` repository (change the device to `cuda` if you have an NVIDIA GPU or `cpu` if you don't have a GPU):

```bash
python lerobot/scripts/train.py \
  dataset_repo_id=YOUR_HF_DATASET_ID \
  policy=act_so100_phosphobot \
  env=so100_real \
  hydra.run.dir=outputs/train/act_so100_quickstart \
  hydra.job.name=act_so100_quickstart \
  device=mps \
  wandb.enable=false
```

5. **Test the Model**: Test the model you just trained using the following command:

```bash
TODO
```

## Next steps

- **Test the model**: Run the following code to test the model you just trained TODO
- **Join the Community**: Connect with other developers and share your experience in our [Discord community](https://discord.gg/cbkggY6NSK)

## Contributing

We welcome contributions from the community! Here's how you can participate:

1. Fork this repository
2. Create a new branch for your project
3. Add your code and documentation
4. Submit a Pull Request

Please ensure your code is well-documented and includes:

- Clear setup instructions
- Dependencies list
- Basic usage examples
- Any special requirements

## Community Projects

Explore projects created by our community members in the [code_examples](./code_examples) directory. Each project includes its own documentation and setup instructions.

## Support

- **Documentation**: Read the [documentation](https://docs.phospho.ai)
- **Community Support**: Join our [Discord server](https://discord.gg/cbkggY6NSK)
- **Issues**: Submit problems or suggestions through [GitHub Issues](https://github.com/phospho-app/phosphobot/issues)

## License

MIT License

---

Made with ❤️ by the Phospho community
