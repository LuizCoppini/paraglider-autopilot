import jsbsim
import csv
from datetime import datetime
from pathlib import Path


def run_simulation():

    # caminho do repositório jsbsim
    fdm = jsbsim.FGFDMExec(r"D:\workspace\Pycharm\paraglider-autopilot\jsbsim")

    fdm.set_debug_level(0)

    # carrega aeronave
    fdm.load_model("c172p")

    # posição inicial
    fdm["ic/lat-gc-deg"] = 37.618805
    fdm["ic/long-gc-deg"] = -122.375416
    fdm["ic/h-sl-ft"] = 10000
    fdm["ic/vc-kts"] = 0

    fdm.run_ic()

    folder = Path("flight_records")
    folder.mkdir(exist_ok=True)

    filename = folder / f"flight_{datetime.now().timestamp()}.csv"

    with open(filename, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "time",
            "lat",
            "lon",
            "alt_ft",
            "roll",
            "pitch",
            "heading"
        ])

        sim_time = 0

        while sim_time < 540:

            fdm.run()

            lat = fdm["position/lat-gc-deg"]
            lon = fdm["position/long-gc-deg"]
            alt = fdm["position/h-sl-ft"]

            roll = fdm["attitude/phi-deg"]
            pitch = fdm["attitude/theta-deg"]
            heading = fdm["attitude/psi-deg"]

            writer.writerow([
                sim_time,
                lat,
                lon,
                alt,
                roll,
                pitch,
                heading
            ])

            sim_time += fdm.get_delta_t()

    print("Simulation finished")