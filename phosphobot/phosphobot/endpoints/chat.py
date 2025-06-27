from fastapi import APIRouter
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from phosphobot.agents.graph import chat_graph


router = APIRouter(tags=["chat"])

sdk = CopilotKitRemoteEndpoint(
    agents=[
        LangGraphAgent(
            name="chat_graph",
            description="This agent chats with user",
            graph=chat_graph,
        )
    ]
)
add_fastapi_endpoint(router, sdk, "/chat/tools")
