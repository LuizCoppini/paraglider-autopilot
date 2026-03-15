import socket
from config import settings


class ParachuteController:

    def __init__(self):
        self.deployed = False

        self.host = "localhost"
        self.port = settings.TELNET_PORT

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self.sock.connect((self.host, self.port))
        except Exception as e:
            print("Telnet connection failed:", e)

    def deploy(self):
        if self.deployed:
            return

        try:
            command = "set /fdm/jsbsim/systems/chute/chute-cmd-norm 1\r\n"
            self.sock.send(command.encode())

            print("PARACHUTE DEPLOYED")

            self.deployed = True

        except Exception as e:
            print("Failed to deploy parachute:", e)