import socket
import json
import time

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5555
BUFFER_SIZE = 4096


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)

    example_data = {
        # intentionally malformed key for testing:
        "bad_field": "oops",
        "x": 0.5,
        "y": -0.2,
        "timestamp": time.time(),
    }
    payload = json.dumps(example_data).encode("utf-8")

    print(f"→ Sending: {example_data}")
    sock.sendto(payload, (SERVER_HOST, SERVER_PORT))

    try:
        data, addr = sock.recvfrom(BUFFER_SIZE)
        text = data.decode("utf-8")
        try:
            resp = json.loads(text)
        except json.JSONDecodeError:
            print(f"✖ Non‐JSON response from {addr}: {text!r}")
            return

        if "error" in resp:
            print(f"✖ Error from server: {resp['error']}")
            if "detail" in resp:
                print("   detail:", resp["detail"])
        else:
            print(f"✓ Success response from {addr}: {resp}")

    except socket.timeout:
        print("✖ No response (timeout)")

    finally:
        sock.close()


if __name__ == "__main__":
    main()
