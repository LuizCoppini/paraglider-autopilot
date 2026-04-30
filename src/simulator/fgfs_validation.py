"""
fgfs_validation.py — versão corrigida (v3).

CONTEXTO IMPORTANTE:
  flightgear_python.FDMConnection executa o callback em um PROCESSO FILHO
  (multiprocessing). Por isso, qualquer estado global criado no processo
  pai (modelo PPO, telnet, CSV, flags de reset) NÃO é visível dentro do
  callback. Toda a lógica precisa morar dentro do callback, no filho.

  Ao mesmo tempo, o callback NÃO pode bloquear (sleep longo) porque isso
  faz overflow do buffer UDP e deixa o processo filho em estado ruim.

A solução é uma máquina de estados dentro do callback, com deadlines:
  - FLYING:    voa normal, controla com PPO, loga CSV.
  - RESETTING: dispara comandos de freeze/teleporte/reinit no telnet e
               apenas marca quando voltar a operar (reset_done_at).
               Os pacotes recebidos durante essa janela são descartados
               sem bloquear.
  - Após reset_done_at, volta a FLYING com novo número de voo e
   um CSV novo.
"""

import math
import time
import csv
import socket
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
from flightgear_python.fg_if import FDMConnection
from stable_baselines3 import PPO

# --- CONFIGURAÇÕES ---
MODEL_PATH = r"D:\workspace\Pycharm\paraglider-autopilot\models\training_20260327_212226\parachute_model_final_4000_jumps.zip"
BASE_LOG_FOLDER = Path(r"D:\workspace\Pycharm\paraglider-autopilot\src\flight_records")

# Mojave / Edwards AFB
TARGET_LAT, TARGET_LON = 34.9055, -117.8830
START_ALT_FT = 9850
RADIUS_M = 1500
MOJAVE_GROUND_ALT = 2300  # ft MSL aprox.

MAX_FLIGHT_TIME = 1500
MAX_FLIGHTS = 10

# Quanto tempo o callback ignora pacotes depois de mandar reset, para o
# JSBSim estabilizar com as novas IC.
RESET_STABILIZATION_S = 12.0

# --- ESTADO (global do processo FILHO; recriado lá) ---
csv_writer = None
csv_file_handle = None
telnet_socket = None
model = None
last_control_time = 0.0
chute_deployed = False
start_time = None
current_actions = [0.0, 0.0]
flight_number = 1
current_session_folder = None

# Máquina de estados
PHASE_FLYING = "FLYING"
PHASE_RESETTING = "RESETTING"
phase = PHASE_FLYING
reset_done_at = 0.0  # epoch para sair de RESETTING
_last_reset_print = 0.0  # rate-limit do heartbeat


# ----------------------------------------------------------------------
# TELNET (criado dentro do processo filho)
# ----------------------------------------------------------------------
def connect_telnet():
    global telnet_socket
    if telnet_socket is not None:
        return
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect(("localhost", 5401))
        # Drena saudação sem nunca bloquear
        s.settimeout(0.3)
        try:
            while True:
                data = s.recv(4096)
                if not data:
                    break
        except Exception:
            pass
        s.settimeout(5)
        s.sendall(b"set /sim/time/warp-mode 0\r\n")
        s.sendall(b"set /sim/time/preset-adm-noon 1\r\n")
        telnet_socket = s
        print("[child] Telnet conectado.")
    except Exception as e:
        print(f"[child] Falha telnet: {e}")
        try:
            if s is not None:
                s.close()
        except Exception:
            pass
        telnet_socket = None


def send_telnet_cmd(cmd):
    global telnet_socket
    if telnet_socket is None:
        connect_telnet()
    if telnet_socket is None:
        return
    try:
        if not cmd.endswith("\n"):
            cmd = cmd + "\r\n"
        telnet_socket.sendall(cmd.encode())
    except Exception as e:
        print(f"[child] send_telnet_cmd falhou: {e}")
        try:
            telnet_socket.close()
        except Exception:
            pass
        telnet_socket = None


# ----------------------------------------------------------------------
# Disparo do reset — apenas envia os comandos. NÃO bloqueia.
# ----------------------------------------------------------------------
def trigger_reset():
    """
    Fecha CSV atual, manda freeze + teleporte + reinit + unfreeze para o
    FlightGear via telnet. Tudo é fire-and-forget; não espera FG aplicar
    aqui — quem espera é o callback (descartando pacotes) durante a
    janela RESETTING.
    """
    global csv_file_handle, csv_writer

    # Fecha CSV
    if csv_file_handle is not None:
        try:
            csv_file_handle.flush()
            csv_file_handle.close()
        except Exception:
            pass
        csv_file_handle = None
        csv_writer = None

    lat_off = RADIUS_M / 111320.0
    new_lat = TARGET_LAT + lat_off
    new_lon = TARGET_LON

    # Reset SIMPLES — só escrita direta em /position, /orientation,
    # /velocities. NÃO usar /fdm/jsbsim/ic/* + run_ic — combinado com
    # /position/* leva o JSBSim a estado numericamente corrompido.
    cmds = [
        # 1. Zera controles
        "set /controls/flight/aileron 0",
        "set /controls/flight/elevator 0",
        "set /controls/flight/rudder 0",
        "set /controls/flight/flaps 0",

        # 2. Recolhe paraquedas (cmd) — não usamos chute-deployed direto
        "set /fdm/jsbsim/systems/chute/chute-cmd-norm 0",

        # 3. Teleporte
        f"set /position/latitude-deg {new_lat}",
        f"set /position/longitude-deg {new_lon}",
        f"set /position/altitude-ft {START_ALT_FT}",

        # 4. Atitude
        "set /orientation/roll-deg 0",
        "set /orientation/pitch-deg 0",
        "set /orientation/heading-deg 180",

        # 5. Velocidades (body e world frame)
        "set /velocities/uBody-fps 35",
        "set /velocities/vBody-fps 0",
        "set /velocities/wBody-fps 0",
        "set /velocities/v-down-fps 0",
        "set /velocities/v-north-fps -35",
        "set /velocities/v-east-fps 0",

        # 6. Hora do dia
        "set /sim/time/preset-adm-noon 1",
        "set /sim/time/warp-mode 0",
    ]
    for c in cmds:
        send_telnet_cmd(c)


# ----------------------------------------------------------------------
# OBSERVAÇÃO
# ----------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_observation(fdm_data):
    lat = math.degrees(fdm_data.lat_rad)
    lon = math.degrees(fdm_data.lon_rad)
    dist = haversine(lat, lon, TARGET_LAT, TARGET_LON)
    v_ground = math.sqrt(fdm_data.v_north_ft_per_s ** 2 + fdm_data.v_east_ft_per_s ** 2)
    target_hdg = (90.0 - math.degrees(math.atan2(TARGET_LAT - lat, TARGET_LON - lon))) % 360
    bearing_err = (target_hdg - math.degrees(fdm_data.psi_rad) + 180) % 360 - 180
    return np.array([
        np.clip(dist / 5500, 0, 1),
        bearing_err / 180,
        v_ground / 60,
        -fdm_data.v_down_ft_per_s / 30,
        fdm_data.phi_rad,
        fdm_data.theta_rad,
    ], dtype=np.float32)


# ----------------------------------------------------------------------
# CALLBACK FDM — TUDO acontece aqui, no processo filho. NUNCA bloquear.
# ----------------------------------------------------------------------
def fdm_callback(fdm_data, event_pipe):
    global model, csv_writer, csv_file_handle, current_session_folder
    global last_control_time, start_time, chute_deployed, current_actions
    global flight_number, phase, reset_done_at

    try:
        now = time.time()

        # ---- FASE RESETTING: descarta pacotes até o deadline ----
        if phase == PHASE_RESETTING:
            if now < reset_done_at:
                # Heartbeat (rate-limited) só para você ver que o filho
                # continua vivo e que o FG ainda está mandando FDM.
                global _last_reset_print
                if now - _last_reset_print > 1.0:
                    rem = reset_done_at - now
                    cur_alt = fdm_data.alt_m * 3.28084
                    print(f"[child] ...estabilizando, {rem:.1f}s restantes (alt={cur_alt:.0f}ft)")
                    _last_reset_print = now
                return fdm_data
            # Fim do reset → próximo voo
            flight_number += 1
            chute_deployed = False
            start_time = now
            last_control_time = 0.0
            phase = PHASE_FLYING
            print(f"[child] --- Iniciando voo {flight_number} ---")
            # Cai para o caminho de FLYING abaixo

        # ---- INICIALIZAÇÃO (lazy, no filho) ----
        if model is None:
            print(f"[child] Carregando PPO: {MODEL_PATH}")
            model = PPO.load(MODEL_PATH)
            print("[child] PPO carregado.")

        if start_time is None:
            start_time = now

        # Descarta pacote inválido (FG manda lixo logo após reinit)
        for v in (fdm_data.lat_rad, fdm_data.lon_rad, fdm_data.alt_m,
                  fdm_data.v_north_ft_per_s, fdm_data.v_east_ft_per_s,
                  fdm_data.v_down_ft_per_s, fdm_data.psi_rad,
                  fdm_data.phi_rad, fdm_data.theta_rad):
            if not math.isfinite(v):
                return fdm_data

        elapsed = now - start_time
        alt_ft = fdm_data.alt_m * 3.28084
        lat = math.degrees(fdm_data.lat_rad)
        lon = math.degrees(fdm_data.lon_rad)

        # ---- DETECÇÃO DE EXPLOSÃO NUMÉRICA DO JSBSIM ----
        # Se altitude ou velocidades saem do envelope físico, o JSBSim
        # entrou em integrador degenerado. Aborta o voo ANTES que o lixo
        # contamine mais pacotes.
        physics_blew_up = (
            alt_ft < -500 or alt_ft > 30000
            or abs(fdm_data.v_down_ft_per_s) > 1500
            or abs(fdm_data.v_north_ft_per_s) > 1500
            or abs(fdm_data.v_east_ft_per_s) > 1500
        )
        if physics_blew_up:
            print(
                f"[child] >>> FÍSICA EXPLODIU no voo {flight_number} "
                f"(alt={alt_ft:.0f}ft, v_down={fdm_data.v_down_ft_per_s:.0f}ft/s). "
                f"Forçando reset."
            )
            if flight_number >= MAX_FLIGHTS:
                if csv_file_handle is not None:
                    try:
                        csv_file_handle.flush()
                        csv_file_handle.close()
                    except Exception:
                        pass
                import os
                os._exit(0)
            trigger_reset()
            phase = PHASE_RESETTING
            reset_done_at = time.time() + RESET_STABILIZATION_S
            return fdm_data

        # ---- 1. Deploy do paraquedas a 3 s ----
        if elapsed > 3.0 and not chute_deployed:
            send_telnet_cmd("set /fdm/jsbsim/systems/chute/chute-cmd-norm 1")
            chute_deployed = True
            print(f"[child] Voo {flight_number}: paraquedas aberto em t={elapsed:.1f}s.")

        # ---- 2. Controle PPO a 1 Hz ----
        if chute_deployed and (now - last_control_time >= 1.0):
            obs = get_observation(fdm_data)
            if np.all(np.isfinite(obs)):
                action, _ = model.predict(obs, deterministic=True)
                current_actions = [float(action[0]), float(action[1])]
                # NOTE: confirmar mapeamento de action[1]. No treino o
                # action_space é low=[-1,0], high=[1,1]. Se no env de
                # treino esse índice acionava freio/flap, troque
                # /controls/flight/elevator pela property correta.
                send_telnet_cmd(
                    f"set /controls/flight/aileron {current_actions[0]}\n"
                    f"set /controls/flight/elevator {current_actions[1]}"
                )
                last_control_time = now

        # ---- 3. CSV (abre lazy ao iniciar o voo) ----
        if csv_writer is None:
            if current_session_folder is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_session_folder = BASE_LOG_FOLDER / f"mojave_session_{ts}"
                current_session_folder.mkdir(parents=True, exist_ok=True)
            path = current_session_folder / f"flight_{flight_number:03d}.csv"
            csv_file_handle = open(path, "w", newline="")
            csv_writer = csv.writer(csv_file_handle)
            csv_writer.writerow(
                ["time", "lat", "lon", "alt_ft", "dist_m", "v_down", "hdg", "ail", "ele"]
            )
            print(f"[child] CSV criado: {path}")

        csv_writer.writerow([
            round(elapsed, 2),
            round(lat, 6),
            round(lon, 6),
            round(alt_ft, 2),
            round(haversine(lat, lon, TARGET_LAT, TARGET_LON), 2),
            round(fdm_data.v_down_ft_per_s, 2),
            round(math.degrees(fdm_data.psi_rad), 2),
            round(current_actions[0], 4),
            round(current_actions[1], 4),
        ])
        csv_file_handle.flush()

        # ---- 4. Critério de fim de voo ----
        ground_hit = (alt_ft < (MOJAVE_GROUND_ALT + 30) and chute_deployed)
        timeout = (elapsed > MAX_FLIGHT_TIME)

        if ground_hit or timeout:
            reason = "solo" if ground_hit else "timeout"
            print(f"[child] Voo {flight_number} encerrado ({reason}) em t={elapsed:.1f}s, alt={alt_ft:.0f}ft.")
            if flight_number >= MAX_FLIGHTS:
                print(f"[child] Todos os {MAX_FLIGHTS} voos completados. Encerrando.")
                if csv_file_handle is not None:
                    try:
                        csv_file_handle.flush()
                        csv_file_handle.close()
                    except Exception:
                        pass
                # mata o processo filho — o pai detecta via while-loop
                import os
                os._exit(0)
            else:
                trigger_reset()
                phase = PHASE_RESETTING
                reset_done_at = time.time() + RESET_STABILIZATION_S
                print(f"[child] Reset disparado, aguardando {RESET_STABILIZATION_S:.0f}s de estabilização.")

    except Exception:
        print("[child] Exceção no fdm_callback:")
        print(traceback.format_exc())

    return fdm_data


# ----------------------------------------------------------------------
# PROCESSO PAI: apenas sobe a conexão e fica vivo.
# ----------------------------------------------------------------------
def run_validation():
    print(f"Validador multi-voo | Meta: {MAX_FLIGHTS} voos | Mojave / dia")
    print("Modelo, telnet e CSV serão inicializados dentro do processo filho.")

    conn = FDMConnection(fdm_version=24)
    conn.connect_rx("localhost", 5501, fdm_callback)
    conn.start()
    print("[parent] FDM RX iniciado em 5501. Aguardando voos...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[parent] Interrompido pelo usuário.")
    finally:
        try:
            conn.stop()
        except Exception:
            pass


if __name__ == "__main__":
    run_validation()