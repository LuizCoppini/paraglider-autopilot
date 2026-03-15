import subprocess
from config import settings


class FlightGearLauncher:

    def start(self):

        command = [
            settings.FGFS_PATH,

            f"--fg-aircraft={settings.AIRCRAFT_PATH}",
            f"--aircraft={settings.AIRCRAFT}",
            f"--airport={settings.AIRPORT}",

            "--lat=37.618805",
            "--lon=-122.375416",
            "--altitude=5000",
            "--heading=0",
            "--vc=0",

            "--native-fdm=socket,out,60,localhost,5501,udp",
            "--telnet=5401",

            "--prop:/sim/rendering/enabled=false",

            "--disable-sound",
            "--disable-hud",
            "--disable-panel",
            "--disable-ai-models",
            "--disable-clouds",
            "--disable-clouds3d",
            "--disable-random-objects",
            "--disable-terrasync",
            "--disable-real-weather-fetch",

            "--prop:/sim/frame-rate-throttle-hz=0",

            "--log-level=alert"
        ]

        subprocess.Popen(command)

        print("FlightGear started")