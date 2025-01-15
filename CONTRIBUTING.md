# Contributing

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
