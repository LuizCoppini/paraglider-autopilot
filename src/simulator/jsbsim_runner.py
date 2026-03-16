import jsbsim
import csv
from datetime import datetime
from pathlib import Path


JSBSIM_ROOT = r"D:\workspace\Pycharm\paraglider-autopilot\jsbsim"
AIRCRAFT_PATH = r"C:\Users\coppi\OneDrive\Documents\FlightGear\Aircraft"


def run_simulation(
        start_altitude_ft=12000,
        parachute_open_altitude_ft=3000,
        max_sim_time=2000
):

    # -------------------------------------------------
    # INICIALIZA JSBSIM
    # -------------------------------------------------

    fdm = jsbsim.FGFDMExec(JSBSIM_ROOT)
    fdm.set_debug_level(0)

    try:
        fdm.create_property("controls/engines/engine/throttle")
    except:
        pass

    fdm["controls/engines/engine/throttle"] = 0

    fdm.set_aircraft_path(AIRCRAFT_PATH)

    if not fdm.load_model("Parachutist"):
        raise RuntimeError("Failed to load aircraft")

    # timestep igual FlightGear
    fdm.set_dt(1 / 120)

    # -------------------------------------------------
    # CONDIÇÕES INICIAIS
    # -------------------------------------------------

    fdm["ic/lat-gc-deg"] = -26.238
    fdm["ic/long-gc-deg"] = -48.883
    fdm["ic/h-sl-ft"] = start_altitude_ft
    fdm["ic/vc-kts"] = 0
    fdm["ic/psi-true-deg"] = 0

    # parachute começa fechado
    fdm["systems/chute/chute-cmd-norm"] = 0

    # inicializa
    fdm.run_ic()

    parachute_open = False

    # -------------------------------------------------
    # CSV
    # -------------------------------------------------

    folder = Path("flight_records")
    folder.mkdir(exist_ok=True)

    filename = folder / f"parachute_flight_{datetime.now().timestamp()}.csv"

    with open(filename, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "time",
            "lat",
            "lon",
            "alt_ft",
            "vertical_speed_fps",
            "roll",
            "pitch",
            "heading",
            "parachute_open"
        ])

        sim_time = 0

        # -------------------------------------------------
        # LOOP SIMULAÇÃO
        # -------------------------------------------------

        while sim_time < max_sim_time:

            if not fdm.run():
                print("JSBSim stopped")
                break

            lat = fdm["position/lat-gc-deg"]
            lon = fdm["position/long-gc-deg"]
            alt = fdm["position/h-sl-ft"]

            roll = fdm["attitude/phi-deg"]
            pitch = fdm["attitude/theta-deg"]
            heading = fdm["attitude/psi-deg"]

            vertical_speed = fdm["velocities/h-dot-fps"]

            # -------------------------------------------------
            # ABRIR PARAQUEDAS (igual FlightGear telnet)
            # -------------------------------------------------

            if (not parachute_open) and alt <= parachute_open_altitude_ft:

                fdm["systems/chute/chute-cmd-norm"] = 1

                parachute_open = True
                print(f"Parachute opened at {alt:.0f} ft")

            writer.writerow([
                sim_time,
                lat,
                lon,
                alt,
                vertical_speed,
                roll,
                pitch,
                heading,
                parachute_open
            ])

            # -------------------------------------------------
            # TERMINAR SIMULAÇÃO
            # -------------------------------------------------

            if alt <= 1:
                print("Landed")
                break

            sim_time += fdm.get_delta_t()

    print("Simulation finished")
    print("Saved:", filename)


if __name__ == "__main__":

    run_simulation(
        start_altitude_ft=12000,
        parachute_open_altitude_ft=3000
    )