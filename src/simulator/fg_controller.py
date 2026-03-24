import socket

class FGController:

    def __init__(self, host="localhost", port=5401):
        self.host = host
        self.port = port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

        print(f"[INFO] Connected to FlightGear Telnet ({host}:{port})")

    def set_property(self, path, value):
        cmd = f"set {path} {value}\r\n"
        self.sock.sendall(cmd.encode("ascii"))

    def set_chute_cmd_norm(self, deployed=True):
        value = 1 if deployed else 0
        self.set_property("/fdm/jsbsim/systems/chute/chute-cmd-norm", value)

    def close(self):
        self.sock.close()
        print("[INFO] Telnet connection closed")