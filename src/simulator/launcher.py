import subprocess
from config import settings


class FlightGearLauncher:

    def start(self):

        command = [
            settings.FGFS_PATH,
            f"--fg-aircraft={settings.AIRCRAFT_PATH}",
            f"--aircraft={settings.AIRCRAFT}",
            f"--airport={settings.AIRPORT}",
            "--native-fdm=socket,out,2,localhost,5501,udp",
            "--disable-ai-traffic",
            "--disable-real-weather-fetch",
            "--log-level=alert"
        ]

        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        print("FlightGear started")