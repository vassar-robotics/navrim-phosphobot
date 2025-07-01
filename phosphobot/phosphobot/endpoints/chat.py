from fastapi import APIRouter
from copilotkit import CopilotKitRemoteEndpoint
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from phosphobot.agents.robot import build_robot_actions


def get_weather_handler(city: str):
    """
    Handler for getting weather information.
    Returns simulated weather data for the given city.
    """
    # Simple simulated weather data
    weather_data = {
        "city": city,
        "temperature": 22,  # Celsius
        "condition": "Partly Cloudy",
        "humidity": 65,
        "wind_speed": 12,  # km/h
        "description": f"The weather in {city} is currently partly cloudy with a temperature of 22°C."
    }
    return weather_data


router = APIRouter(tags=["chat"])

sdk = CopilotKitRemoteEndpoint(actions=build_robot_actions)
#     actions=[
#         Action(
#             name="get_weather",
#             handler=get_weather_handler,
#             description="Get the weather of certain place",
#             parameters=[
#                 {
#                     "name": "city",
#                     "type": "string",
#                     "description": "The name of the city"
#                 }
#             ]
#         )
#     ]
# )
add_fastapi_endpoint(router, sdk, "/chat/tools")
