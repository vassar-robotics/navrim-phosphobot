import socket
import json
import time
import requests  # pip install requests

# === CONFIGURATION ===
API_BASE_URL = "http://127.0.0.1:80"  # your FastAPI base
START_ENDPOINT = "/move/teleop/udp"
STOP_ENDPOINT = "/move/teleop/udp/stop"
BUFFER_SIZE = 1024
UDP_TIMEOUT = 2.0  # seconds


def start_udp_server():
    """Call FastAPI to (re)start the UDP server and return (host, port)."""
    resp = requests.post(API_BASE_URL + START_ENDPOINT)
    resp.raise_for_status()
    info = resp.json()
    host = info["host"]
    port = info["port"]
    print(f"[API] UDP server running on {host}:{port}")
    return host, port


def stop_udp_server():
    """Call FastAPI to stop the UDP server."""
    resp = requests.post(API_BASE_URL + STOP_ENDPOINT)
    resp.raise_for_status()
    print("[API] UDP server stopped")


def send_udp_and_receive(host: str, port: int, payload: dict):
    """Send `payload` as JSON over UDP and print the response (or error)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(UDP_TIMEOUT)

    data = json.dumps(payload).encode("utf-8")
    print(f"[UDP] Sending → {payload}")
    sock.sendto(data, (host, port))

    try:
        raw, addr = sock.recvfrom(BUFFER_SIZE)
        text = raw.decode("utf-8")
        try:
            resp = json.loads(text)
        except json.JSONDecodeError:
            print(f"[UDP] ✖ Non-JSON from {addr}: {text!r}")
            return

        if "error" in resp:
            print(f"[UDP] ✖ Error from server: {resp['error']}")
            if "detail" in resp:
                print("        detail:", resp["detail"])
        else:
            print(f"[UDP] ✓ Success from {addr}: {resp}")

    except socket.timeout:
        print(f"[UDP] ✖ No response within {UDP_TIMEOUT}s")
    finally:
        sock.close()


def main():
    # 1) Start (or get) the UDP server
    host, port = start_udp_server()

    # 2) Build an example payload
    example_payload = {
        "command": "teleop_move",
        "x": 0.5,
        "y": -0.2,
        "timestamp": time.time(),
    }

    # 3) Send over UDP & await a reply
    send_udp_and_receive(host, port, example_payload)

    # 4) (Optional) stop the UDP server when you’re done
    stop_udp_server()


if __name__ == "__main__":
    main()
