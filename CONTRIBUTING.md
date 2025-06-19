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

## Building from source

Look at the main README.md

## How to enable cloud features when building from source?

For development purposes, you may need to enable the phospho cloud features. If so, you'll need some secret token values that you need a member of the phospho team to share with you. Then, here is how to input them:

1. Create the file `dashboard/.env` with these variables:

```
VITE_SUPABASE_URL="..."
VITE_SUPABASE_KEY="..."
```

Make sure to rebuild the frontend (Vite bakes these variables in compiled code).

2. Create the file `phosphobot/resources/tokens.toml`

```
ENV = "dev"
SENTRY_DSN = "..."
POSTHOG_API_KEY = "..."
POSTHOG_HOST = "..."
SUPABASE_URL = "..."
SUPABASE_KEY = "..."
MODAL_API_URL = "..."
```
