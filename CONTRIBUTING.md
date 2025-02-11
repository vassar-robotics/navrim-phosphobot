# Contributing

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

## Setup

This module use [uv](https://github.com/astral-sh/uv) for packaging.

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Generate the client code

Use [Fern](https://github.com/fern-api/fern) to create the client code from the `openapi.json` specs.
