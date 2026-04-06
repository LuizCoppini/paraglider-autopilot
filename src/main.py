import time
import subprocess
import math
import os
import sys

from simulator.fgfs_validation import run_validation
from simulator.launcher import FlightGearLauncher
from simulator.fdm_reader import start_reader
from simulator.jsbsim_runner import run_simulation

# 1: FlightGear | 2: JSBSim | 3: Treinamento Poetry | 4: validação do modelo no flightgear
modo = 4
3

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

    elif modo == 4:
        print("--- INICIANDO VALIDAÇÃO AUTOMÁTICA NO FLIGHTGEAR ---")

        # 1. Configurações de Origem (Igual ao Treino - Posição 0)
        target_lat, target_lon = -26.2385, -48.884
        radius = 1500.0
        angle_pos = math.radians(0)  # Posição Norte
        start_lat = target_lat + (radius * math.cos(angle_pos)) / 111320.0
        start_lon = target_lon + (radius * math.sin(angle_pos)) / (111320.0 * math.cos(math.radians(target_lat)))
        start_alt = 9850  # Pés (Altitude do Treino)

        # 2. Abrir FlightGear
        fg = FlightGearLauncher()
        fg.start(lat=start_lat, lon=start_lon, alt=start_alt)

        print("Aguardando carregamento do simulador (20s)...")
        time.sleep(20)

        # 3. Iniciar Controle
        print("Iniciando controle PPO...")
        run_validation()

    else:
        print("Modo inválido. Escolha 1, 2 ou 3.")


if __name__ == "__main__":
    main()