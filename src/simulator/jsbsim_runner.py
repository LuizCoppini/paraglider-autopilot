import jsbsim
import csv
from datetime import datetime
from pathlib import Path

JSBSIM_ROOT = r"D:\workspace\Pycharm\paraglider-autopilot\jsbsim"
AIRCRAFT_PATH = r"C:\Users\coppi\OneDrive\Documents\FlightGear\Aircraft"


def run_simulation(start_altitude_ft=8000, max_sim_time=1500):
    fdm = jsbsim.FGFDMExec(JSBSIM_ROOT)
    fdm.set_debug_level(0)
    fdm.set_aircraft_path(AIRCRAFT_PATH)

    if not fdm.load_model("Parachutist"):
        raise RuntimeError("Falha ao carregar o modelo")

    fdm.set_dt(1 / 120)

    # --- CONFIGURAÇÃO DO VENTO ---
    fdm["atmosphere/wind-north-fps"] = 0.0
    fdm["atmosphere/wind-east-fps"] = 25.0  # Vento lateral para deslocamento

    # CONDIÇÕES INICIAIS
    fdm["ic/lat-gc-deg"] = -26.238
    fdm["ic/long-gc-deg"] = -48.883
    fdm["ic/h-sl-ft"] = start_altitude_ft
    fdm["ic/psi-true-deg"] = 0

    # Velocidades iniciais (U = frente, W = baixo)
    # Importante para o modelo não nascer em "stall"
    fdm["ic/u-fps"] = 30.0
    fdm["ic/w-fps"] = 15.0

    fdm.run_ic()

    # --- CONFIGURAÇÃO ESPECÍFICA DO XML ---
    # 1. Definir o peso do skydiver (conforme Chute.xml)
    fdm["inertia/pointmass-weight-lbs[0]"] = 211.3

    # 2. Ativar o paraquedas (chute-cmd-norm gera chute-reef-pos-norm no Chute.xml)
    fdm["systems/chute/chute-cmd-norm"] = 1.0

    # Estabilizar o velame
    print("Estabilizando velame e skydiver...")
    for _ in range(600):
        fdm.run()

    folder = Path("flight_records")
    folder.mkdir(exist_ok=True)
    filename = folder / f"zigzag_xml_test_{datetime.now().timestamp()}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        # Heading e Roll para checar a curva, Aileron para checar o comando
        writer.writerow(["time", "lat", "lon", "alt_ft", "heading", "roll", "aileron", "elevator"])

        sim_time = 0
        while sim_time < max_sim_time:
            # LÓGICA DE ZIGUE-ZAGUE USANDO AILERON (conforme Controls.xml)
            # Alternamos o aileron entre -0.5 e 0.5 a cada 15 segundos
            if (sim_time % 30) < 15:
                aileron = -0.5  # Curva para esquerda
                elevator = 0.2  # Leve frenagem para estabilidade
            else:
                aileron = 0.5  # Curva para direita
                elevator = 0.2

            fdm["fcs/aileron-cmd-norm"] = aileron
            fdm["fcs/elevator-cmd-norm"] = elevator

            if not fdm.run(): break

            alt = fdm["position/h-sl-ft"]
            writer.writerow([
                sim_time,
                fdm["position/lat-gc-deg"],
                fdm["position/long-gc-deg"],
                alt,
                fdm["attitude/psi-deg"],
                fdm["attitude/phi-deg"],
                aileron,
                elevator
            ])

            if alt <= 1:
                print("Pousou!")
                break

            sim_time += fdm.get_delta_t()

    print(f"Simulação finalizada! Arquivo: {filename}")


if __name__ == "__main__":
    run_simulation()