import time
import subprocess
import math
import os
import sys

from simulator.fgfs_validation import run_validation
from simulator.launcher import FlightGearLauncher
from simulator.fdm_reader import start_reader
from simulator.jsbsim_runner import run_simulation

# 1: FlightGear (Manual)
# 2: JSBSim (Pure Physics)
# 3: Treinamento Clássico (Distância)
# 4: Validação Automática no FlightGear (PPO)
# 5: Treinamento Novo (Lógica de Cone de Descida)
modo = 4


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
        # Treinamento original baseado em distância simples
        script_path = os.path.join(current_dir, "rl", "train_parachute.py")
        run_poetry_train(script_path, current_dir)


    elif modo == 4:
        print("--- INICIANDO VALIDAÇÃO AUTOMÁTICA NO FLIGHTGEAR ---")
        # ATUALIZADO PARA MOJAVE (EDWARDS AFB)
        target_lat, target_lon = 34.9055, -117.8830
        radius = 1500.0
        angle_pos = math.radians(0)

        # O cálculo abaixo usará as coordenadas de Mojave para o início do voo
        start_lat = target_lat + (radius * math.cos(angle_pos)) / 111320.0
        start_lon = target_lon + (radius * math.sin(angle_pos)) / (111320.0 * math.cos(math.radians(target_lat)))
        start_alt = 9850
        fg = FlightGearLauncher()

        # O launcher agora enviará Mojave para o FlightGear na partida
        fg.start(lat=start_lat, lon=start_lon, alt=start_alt)

        print("Aguardando carregamento do simulador (20s)...")
        time.sleep(20)
        print("Iniciando controle PPO...")
        run_validation()

    elif modo == 5:
        # NOVO: Treinamento focado no Cone de Descida e Glide Ratio
        script_path = os.path.join(current_dir, "rl", "train_parachute_cone.py")
        print("--- INICIANDO TREINAMENTO COM LÓGICA DE CONE ---")
        run_poetry_train(script_path, current_dir)

    else:
        print("Modo inválido. Escolha entre 1, 2, 3, 4 ou 5.")


def run_poetry_train(script_path, current_dir):
    """Função auxiliar para executar scripts via Poetry"""
    print(f"Iniciando treinamento via Poetry...")
    print(f"Caminho detectado: {script_path}")
    try:
        project_root = os.path.dirname(current_dir)
        subprocess.run(["poetry", "run", "python", script_path], check=True, cwd=project_root)
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar o treinamento: {e}")
    except FileNotFoundError:
        print("Erro: Comando 'poetry' não encontrado.")


if __name__ == "__main__":
    main()