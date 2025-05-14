import asyncio
import os
import subprocess

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger

from phosphobot.models import NetworkCredentials
from phosphobot.utils import background_task_log_exceptions, is_running_on_pi

router = APIRouter(tags=["networking"])


@background_task_log_exceptions
async def setup_hotspot_bg():
    """Background task for hotspot setup and activation"""
    try:
        # Check if setup has already been done
        if not os.path.exists("/etc/dnsmasq.d/hotspot.conf"):
            # Install dnsmasq if not already installed
            subprocess.run(["sudo", "apt-get", "install", "-y", "dnsmasq"], check=True)

            # Configure dnsmasq for DHCP
            dnsmasq_config = """interface=wlan0
dhcp-range=192.168.1.2,192.168.1.200,12h
"""
            # Write config using sudo
            subprocess.run(
                ["sudo", "tee", "/etc/dnsmasq.d/hotspot.conf"],
                input=dnsmasq_config.encode(),
                check=True,
            )

            # Restart dnsmasq to apply changes
            subprocess.run(["sudo", "systemctl", "restart", "dnsmasq"], check=True)

        # Delete any existing hotspot connection
        subprocess.run(
            ["sudo", "nmcli", "connection", "delete", "hotspot"], check=False
        )

        # Create the hotspot connection
        subprocess.run(
            [
                "sudo",
                "nmcli",
                "connection",
                "add",
                "type",
                "wifi",
                "ifname",
                "wlan0",
                "con-name",
                "hotspot",
                "ssid",
                "phosphobot",
                "mode",
                "ap",
            ],
            check=True,
        )

        # Secure the connection
        subprocess.run(
            [
                "sudo",
                "nmcli",
                "connection",
                "modify",
                "hotspot",
                "802-11-wireless.mode",
                "ap",
                "802-11-wireless.band",
                "bg",
                "ipv4.addresses",
                "192.168.1.1/24",
                "ipv4.method",
                "manual",
                "wifi-sec.key-mgmt",
                "wpa-psk",
                "wifi-sec.psk",
                "phosphobot123",
            ],
            check=True,
        )

        # Bring up the hotspot connection
        subprocess.run(["sudo", "nmcli", "connection", "up", "hotspot"], check=True)

    except Exception as e:
        logger.error(f"Error in hotspot background task: {str(e)}")


@background_task_log_exceptions
async def connect_to_network_bg(ssid: str, password: str):
    """Background task for network connection"""

    try:
        # Disconnect from current network
        subprocess.run(
            ["sudo", "nmcli", "device", "disconnect", "wlan0"],
            check=True,
        )

        # Wait for disconnect to complete
        await asyncio.sleep(3)

        # Connect to new network
        subprocess.run(
            [
                "sudo",
                "nmcli",
                "device",
                "wifi",
                "connect",
                ssid,
                "password",
                password,
            ],
            check=True,
        )

        logger.info(f"Connected to network: {ssid}")
        return

    except Exception as e:
        logger.warning(f"Error in network connection background task: {str(e)}")
        await setup_hotspot_bg()
        return


@router.post("/network/hotspot")
async def activate_hotspot(background_tasks: BackgroundTasks):
    """
    Endpoint to activate the hotspot on the Raspberry Pi.
    Returns immediately and performs setup in the background.
    """
    if not is_running_on_pi():
        raise HTTPException(
            status_code=400, detail="This endpoint is only available on Raspberry Pi"
        )
    background_tasks.add_task(setup_hotspot_bg)

    return {
        "status": "Hotspot activation started",
        "message": "The hotspot is being configured and will be available shortly",
        "SSID": "phosphobot",
        "Connect and visit": "http://phosphobot.local",
    }


@router.post("/network/connect")
async def switch_to_network(
    credentials: NetworkCredentials, background_tasks: BackgroundTasks
):
    """
    Endpoint to connect phosphobot to a new network.
    Returns immediately and performs connection in the background.
    Will fallback to the hotspot if it fails to connect.
    """
    if not is_running_on_pi():
        raise HTTPException(
            status_code=400, detail="This endpoint is only available on Raspberry Pi"
        )
    background_tasks.add_task(
        connect_to_network_bg, credentials.ssid, credentials.password
    )

    return {
        "status": "Connection process started, will fallback to hotspot if unsuccessful",
        "message": f"Attempting to connect to network: {credentials.ssid}",
    }
