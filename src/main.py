import time
import subprocess
import math
import os
import sys
import random

from simulator.fgfs_validation import run_validation
from simulator.fgfs_validation_cone import run_validation as run_validation_cone
from simulator.fgfs_validation_cone_v2 import run_validation as run_validation_cone_v2
from simulator.launcher import FlightGearLauncher
from simulator.fdm_reader import start_reader
from simulator.jsbsim_runner import run_simulation

# 1: FlightGear (Manual)
# 2: JSBSim (Pure Physics)
# 3: Treinamento Clássico (Distância)
# 4: Validação no FlightGear — MODELO ANTIGO (distância simples), raio 1500m
# 5: Treinamento Novo (Lógica de Cone de Descida)
# 6: Validação no FlightGear — MODELO CONE v1 (GR=0.8, vento aleatório)
# 7: Validação no FlightGear — MODELO CONE v2 (GR=1.88, vento norte→sul)
modo = 7


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
        print("--- VALIDAÇÃO NO FLIGHTGEAR — MODELO ANTIGO (distância) ---")
        _start_validation(radius=1500.0, runner=run_validation)

    elif modo == 5:
        # Treinamento focado no Cone de Descida e Glide Ratio
        script_path = os.path.join(current_dir, "rl", "train_parachute_cone.py")
        print("--- INICIANDO TREINAMENTO COM LÓGICA DE CONE ---")
        run_poetry_train(script_path, current_dir)

    elif modo == 6:
        print("--- VALIDAÇÃO NO FLIGHTGEAR — MODELO CONE v1 ---")
        _start_validation(radius=4000.0, runner=run_validation_cone)

    elif modo == 7:
        print("--- VALIDAÇÃO NO FLIGHTGEAR — MODELO CONE v2 ---")
        _start_validation(radius=4000.0, runner=run_validation_cone_v2)

    else:
        print("Modo inválido. Escolha entre 1, 2, 3, 4, 5, 6 ou 7.")


def _start_validation(radius, runner):
    """
    Lança o FG numa posição aleatória ao redor do alvo de Mojave e
    chama o runner de validação informado (modelo antigo ou cone).
    """
    target_lat, target_lon = 34.9055, -117.8830

    angle_pos = random.uniform(0.0, 2.0 * math.pi)
    start_lat = target_lat + (radius * math.cos(angle_pos)) / 111320.0
    start_lon = target_lon + (radius * math.sin(angle_pos)) / (
        111320.0 * math.cos(math.radians(target_lat))
    )
    start_alt = 9850

    print(
        f"Voo 1 spawn aleatório: lat={start_lat:.5f}, lon={start_lon:.5f}, "
        f"raio={radius:.0f}m, ângulo={math.degrees(angle_pos):.1f}°"
    )

    fg = FlightGearLauncher()
    fg.start(lat=start_lat, lon=start_lon, alt=start_alt)

    print("Aguardando carregamento do simulador (20s)...")
    time.sleep(20)
    print("Iniciando controle PPO...")
    runner()


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
