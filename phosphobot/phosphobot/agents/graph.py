from typing import Dict, List, Any, Optional

# Updated imports for LangGraph
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START

# Updated imports for CopilotKit
from copilotkit import CopilotKitState
from copilotkit.langchain import copilotkit_customize_config
from langgraph.types import Command
from typing_extensions import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from copilotkit.langgraph import copilotkit_exit

# Add imports for tool creation
from langchain_core.tools import tool
import random
from datetime import datetime


@tool
def get_weather(location: str) -> Dict[str, Any]:
    """
    Get simulated weather information for a given location.

    Args:
        location: The city or location to get weather for

    Returns:
        A dictionary containing simulated weather data
    """
    # Simulate weather conditions
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Thunderstorm", "Snowy", "Foggy"]

    # Generate random weather data
    temperature_c = round(random.uniform(-10, 35), 1)
    temperature_f = round((temperature_c * 9 / 5) + 32, 1)
    humidity = random.randint(30, 90)
    wind_speed_kmh = round(random.uniform(0, 40), 1)
    wind_speed_mph = round(wind_speed_kmh * 0.621371, 1)
    condition = random.choice(conditions)

    # Simulate UV index based on condition
    if condition in ["Sunny", "Partly Cloudy"]:
        uv_index = random.randint(6, 11)
    elif condition in ["Cloudy", "Foggy"]:
        uv_index = random.randint(1, 3)
    else:
        uv_index = random.randint(1, 5)

    # Create weather report
    weather_data = {
        "location": location,
        "timestamp": datetime.now().isoformat(),
        "temperature": {"celsius": temperature_c, "fahrenheit": temperature_f},
        "condition": condition,
        "humidity": f"{humidity}%",
        "wind": {
            "speed_kmh": wind_speed_kmh,
            "speed_mph": wind_speed_mph,
            "direction": random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
        },
        "uv_index": uv_index,
        "precipitation_chance": f"{random.randint(0, 100)}%",
        "visibility_km": round(random.uniform(1, 20), 1),
        "pressure_mb": random.randint(980, 1030),
        "feels_like": {
            "celsius": round(temperature_c + random.uniform(-3, 3), 1),
            "fahrenheit": round(temperature_f + random.uniform(-5, 5), 1),
        },
    }

    return weather_data


class AgentState(CopilotKitState):
    """
    Here we define the state of the agent

    In this instance, we're inheriting from CopilotKitState, which will bring in
    the CopilotKitState fields. We're also adding a custom field, `language`,
    which will be used to set the language of the agent.
    """


async def chat_node(state: AgentState, config: RunnableConfig):
    """
    Standard chat node based on the ReAct design pattern. It handles:
    - The model to use (and binds in CopilotKit actions and the tools defined above)
    - The system prompt
    - Getting a response from the model
    - Handling tool calls

    For more about the ReAct design pattern, see:
    https://www.perplexity.ai/search/react-agents-NcXLQhreS0WDzpVaS4m9Cg
    """

    # 1. Define the model
    print("state", state)
    print("config", config)
    model = ChatOpenAI(model="gpt-4o")

    # Define config for the model
    if config is None:
        config = RunnableConfig(recursion_limit=25)
    else:
        # Use CopilotKit's custom config functions to properly set up streaming
        config = copilotkit_customize_config(config)

    # 2. Bind the tools to the model
    model_with_tools = model.bind_tools(
        [*state["copilotkit"]["actions"], get_weather],
        # 2.1 Disable parallel tool calls to avoid race conditions,
        #     enable this for faster performance if you want to manage
        #     the complexity of running tool calls in parallel.
        parallel_tool_calls=False,
    )

    # 3. Define the system message by which the chat model will be run
    system_message = SystemMessage(
        content="You are a helpful assistant. You can use the following tools to help the user, including getting weather information for any location."
    )

    # 4. Run the model to generate a response
    response = await model_with_tools.ainvoke(
        [
            system_message,
            *state["messages"],
        ],
        config,
    )

    # 6. We've handled all tool calls, so we can end the graph.
    await copilotkit_exit(config)
    return Command(goto=END, update={"messages": response})


# Define a new graph
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.set_entry_point("chat_node")

# Add explicit edges, matching the pattern in other examples
workflow.add_edge(START, "chat_node")
workflow.add_edge("chat_node", END)

# Compile the graph
chat_graph = workflow.compile(checkpointer=MemorySaver())
