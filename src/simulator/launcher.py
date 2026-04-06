import subprocess
from config import settings


class FlightGearLauncher:
    def start(self, lat=None, lon=None, alt=None):
        # Se não for passado, usa o padrão do settings
        lat = lat if lat else "37.618805"
        lon = lon if lon else "-122.375416"
        alt = alt if alt else settings.START_ALT

        command = [
            settings.FGFS_PATH,
            f"--fg-aircraft={settings.AIRCRAFT_PATH}",
            f"--aircraft={settings.AIRCRAFT}",
            f"--airport={settings.AIRPORT}",
            f"--lat={lat}",
            f"--lon={lon}",
            f"--altitude={alt}",
            "--heading=0",
            "--vc=35", # Velocidade inicial de cruzeiro para estabilizar

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
            "--prop:/sim/frame-rate-throttle-hz=60",
            "--log-level=alert"
        ]

        subprocess.Popen(command)
        print(f"🚀 FlightGear started at Lat: {lat} Lon: {lon}")