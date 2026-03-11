import socket


class FlightGearTelemetry:

    def __init__(self, port=5501):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("localhost", port))

    def receive(self):
        data, _ = self.sock.recvfrom(4096)
        return data