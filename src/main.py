import time
import subprocess
import os
import sys
from simulator.launcher import FlightGearLauncher
from simulator.fdm_reader import start_reader
from simulator.jsbsim_runner import run_simulation

# 1: FlightGear | 2: JSBSim | 3: Treinamento Poetry
modo = 3


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if modo == 1:
        fg = FlightGearLauncher()
        fg.start()
        print("Waiting FlightGear start...")
        time.sleep(10)
        print("Starting telemetry reader...")
        start_reader()

    elif modo == 2:
        print("Iniciando simulação pura via JSBSim...")
        run_simulation()

    elif modo == 3:
        script_path = os.path.join(current_dir, "rl", "train_parachute.py")

        print(f"Iniciando treinamento via Poetry...")
        print(f"Caminho detectado: {script_path}")

        try:
            project_root = os.path.dirname(current_dir)
            # shell=True ajuda o Windows a encontrar o executável do poetry no PATH
            subprocess.run(["poetry", "run", "python", script_path], check=True, cwd=project_root)

        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar o treinamento: {e}")
        except FileNotFoundError:
            print("Erro: Comando 'poetry' não encontrado.")

    else:
        print("Modo inválido. Escolha 1, 2 ou 3.")


if __name__ == "__main__":
    main()