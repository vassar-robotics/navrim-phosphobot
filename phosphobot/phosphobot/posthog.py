import os
import platform
import uuid

from posthog import Posthog

from phosphobot import __version__
from phosphobot.telemetry import TELEMETRY
from phosphobot.utils import get_home_app_path, get_tokens

tokens = get_tokens()
posthog = Posthog(
    api_key=tokens.POSTHOG_API_KEY,
    host=tokens.POSTHOG_HOST,
    super_properties={
        "env": tokens.ENV,
        "system_info": f"{platform.node()}_{platform.system()}_{platform.release()}",
        "phosphobot_version": __version__,
    },
)


def is_github_actions():
    return os.getenv("GITHUB_ACTIONS") == "true"


def get_or_create_unique_id(token_path):
    """
    Retrieve or generate a unique ID, storing it in a token file. This is an
    anonymous identifier for the user.

    Args:
        token_path (str): Path to the unique ID token file

    Returns:
        str: Unique identifier for the user
    """
    if is_github_actions():
        return "github_actions_ci"
    # Ensure the directory exists
    os.makedirs(os.path.dirname(token_path), exist_ok=True)

    # Check if token file exists
    if os.path.exists(token_path):
        # Read existing token
        with open(token_path, "r") as f:
            content = f.read().strip()
        return content

    # Generate a new unique ID if no token exists
    new_user_id = "phosphobot_" + str(uuid.uuid4())

    # Write the new ID to the token file
    with open(token_path, "w") as f:
        f.write(new_user_id)

    return new_user_id


# We disable posthog in dev environments
if not TELEMETRY or is_github_actions():
    posthog.disabled = True

session_id = str(uuid.uuid4())

posthog_details = {
    "session_id": session_id,
}
TOKEN_FILE = get_home_app_path() / "id.token"
user_id = get_or_create_unique_id(TOKEN_FILE)


def posthog_pageview(page: str) -> None:
    posthog.capture(
        distinct_id=user_id,
        event="$pageview",
        properties={
            **posthog_details,
            "$current_url": "http://phosphobot.local" + page,
        },
    )


def add_email_to_posthog(email: str | None) -> None:
    if posthog.disabled or not email:
        return

    posthog.identify(
        distinct_id=user_id,
        properties={"email": email},
    )
