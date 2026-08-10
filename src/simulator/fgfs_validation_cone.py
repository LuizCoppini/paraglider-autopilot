"""
fgfs_validation_cone.py — Validação do MODELO CONE no FlightGear.

Este arquivo é específico para o modelo treinado com o
parachute_cone_env.py (cone de descida + glide_ratio 0.8). Ele aplica:

  - Observação no formato cone: obs[0] = (dist - alt_AGL*0.8)/1000
    (alt em AGL para alinhar com o treino, que foi a nível do mar).
  - Normalização da obs via VecNormalize (vec_normalize_cone.pkl).
  - Rate-limit de 0.2/step no aileron, igual ao step() do treino.
  - Escrita direta em /fdm/jsbsim/fcs/aileron-cmd-norm e
    /fdm/jsbsim/fcs/elevator-cmd-norm (Parachutist não mapeia
    /controls/flight/* para o FCS).
  - Raio de spawn 4000 m (igual ao treino).

Para validar o modelo antigo (distância simples), use o
fgfs_validation.py.

Arquitetura:
  flightgear_python.FDMConnection roda o callback num processo FILHO
  (multiprocessing). Toda a lógica vive no callback. Nada bloqueia
  (state machine com deadlines). Em explosão numérica do JSBSim,
  matamos e relançamos o FlightGear (hard reset).
"""

import math
import time
import csv
import random
import socket
import subprocess
import sys
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
from flightgear_python.fg_if import FDMConnection
from stable_baselines3 import PPO

# --- CONFIGURAÇÕES ---
MODEL_PATH = r"D:\workspace\Pycharm\paraglider-autopilot\models\cone_method\training_20260406_200222\parachute_cone_model_final.zip"
VEC_NORMALIZE_PATH = r"D:\workspace\Pycharm\paraglider-autopilot\models\cone_method\training_20260406_200222\vec_normalize_cone.pkl"
BASE_LOG_FOLDER = Path(r"D:\workspace\Pycharm\paraglider-autopilot\src\flight_records")

# Mojave / Edwards AFB
TARGET_LAT, TARGET_LON = 34.9055, -117.8830
START_ALT_FT = 9850
RADIUS_M = 4000   # Mesmo raio do treino do cone (parachute_cone_env.reset)
MOJAVE_GROUND_ALT = 2300  # ft MSL aprox.
MOJAVE_GROUND_ALT_M = MOJAVE_GROUND_ALT * 0.3048  # ~700.8 m

# Treino do cone foi a nível do mar (target lat=-26.2385, lon=-48.884 — litoral
# brasileiro). Lá, h-sl-ft no pouso ≈ 0, então o cone fecha em zero. Em Mojave o
# solo está a 2300 ft — sem subtrair, o cone "termina" em raio 561 m e o modelo
# nunca consegue convergir. Usamos AGL na obs para alinhar com o que ele aprendeu.
USE_AGL_FOR_CONE = True

MAX_FLIGHT_TIME = 1500
MAX_FLIGHTS = 60

# Quanto tempo o callback ignora pacotes depois de mandar reset, para o
# JSBSim estabilizar com as novas IC.
RESET_STABILIZATION_S = 12.0
# Tempo de espera após relançar o FlightGear (boot completo).
HARD_RESET_WAIT_S = 25.0

# --- VENTO (igual ao treino: parachute_cone_env.reset linhas 79-85) ---
# O modelo cone foi treinado com vento de 4 níveis × 360° em direção. Em FG sem
# vento o modelo "compensa vento fantasma". Reproduzimos as mesmas condições.
WIND_SPEEDS_FPS = [4.0, 12.0, 25.0, 40.0]

# --- ESTADO (global do processo FILHO; recriado lá) ---
csv_writer = None
csv_file_handle = None
telnet_socket = None
model = None
norm_env = None  # VecNormalize wrapper para normalizar a obs (igual ao treino)
last_control_time = 0.0
chute_deployed = False
start_time = None
current_actions = [0.0, 0.0]
flight_number = 1
current_session_folder = None
wind_set_for_this_flight = False  # injeta vento uma vez por voo

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
# HARD RESET — mata e relança o processo do FlightGear.
# Necessário quando o JSBSim corrompe a memória interna (alt/v_down
# em valores impossíveis, sextilhões etc.) — soft reset via telnet
# não recupera porque /velocities/* são output do FDM, não input.
# ----------------------------------------------------------------------
def kill_flightgear():
    """Mata todos os processos fgfs.exe (Windows) ou fgfs (Unix)."""
    try:
        if sys.platform.startswith("win"):
            subprocess.run(
                ["taskkill", "/F", "/IM", "fgfs.exe"],
                capture_output=True, timeout=10,
            )
        else:
            subprocess.run(["pkill", "-9", "fgfs"], capture_output=True, timeout=10)
        print("[child] FlightGear morto.")
    except Exception as e:
        print(f"[child] Falha ao matar FG: {e}")


def launch_flightgear(lat, lon, alt_ft, heading_deg=0):
    """
    Relança o FlightGear na posição informada. Replica a linha de comando
    do simulator.launcher.FlightGearLauncher; preferimos hardcode aqui
    em vez de import para não criar dependência circular do filho.
    """
    try:
        from config import settings
        cmd = [
            settings.FGFS_PATH,
            f"--fg-aircraft={settings.AIRCRAFT_PATH}",
            f"--aircraft={settings.AIRCRAFT}",
            f"--airport={settings.AIRPORT}",
            f"--lat={lat}",
            f"--lon={lon}",
            f"--altitude={alt_ft}",
            f"--heading={heading_deg}",
            "--vc=35",
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
            "--log-level=alert",
        ]
        subprocess.Popen(cmd)
        print(
            f"[child] FG relançado em lat={lat:.5f} lon={lon:.5f} "
            f"heading={heading_deg:.1f}°"
        )
    except Exception as e:
        print(f"[child] Falha ao relançar FG: {e}")


# ----------------------------------------------------------------------
# Disparo do reset — apenas envia os comandos. NÃO bloqueia.
# ----------------------------------------------------------------------
def _bearing_deg(lat1, lon1, lat2, lon2):
    """Bearing inicial (graus, 0..360) de (lat1,lon1) para (lat2,lon2)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def set_random_wind():
    """
    Injeta vento aleatório, replicando o cenário do treino. Sorteia
    velocidade entre WIND_SPEEDS_FPS e direção em [0, 360).

    No treino (JSBSim puro Python) bastava setar atmosphere/wind-*-fps.
    No FG essa property é INPUT do JSBSim que é sobrescrita a cada frame
    a partir de /environment/wind-from-*-fps. Por isso setamos AMBAS.

    Convenção de sinais:
      - JSBSim: atmosphere/wind-north-fps  → +N = vento blowing TOWARD N
      - FG:     environment/wind-from-north-fps → +N = wind FROM N (TOWARD S)
      Estas duas têm sinal OPOSTO! Se quisermos "wind blowing toward north",
      JSBSim recebe +N e FG recebe -N (vento vindo do sul).
    """
    speed = random.choice(WIND_SPEEDS_FPS)
    direction = random.uniform(0.0, 360.0)
    rad = math.radians(direction)
    wind_n = -math.cos(rad) * speed   # JSBSim convention (treino)
    wind_e = -math.sin(rad) * speed
    # Property no JSBSim
    send_telnet_cmd(f"set /fdm/jsbsim/atmosphere/wind-north-fps {wind_n}")
    send_telnet_cmd(f"set /fdm/jsbsim/atmosphere/wind-east-fps {wind_e}")
    # Property no FG-level (fica persistente; o FG empurra para o JSBSim)
    # Sinal OPOSTO: wind-from-north significa "vento vem do norte (vai pro sul)",
    # ou seja, o vetor de wind é -N. Para fazer JSBSim wind-north = +N,
    # precisamos environment/wind-from-north = -N
    send_telnet_cmd(f"set /environment/wind-from-north-fps {-wind_n}")
    send_telnet_cmd(f"set /environment/wind-from-east-fps {-wind_e}")
    # Algumas versões do FG usam 'kt' para wind speed; vamos garantir
    # que o sistema de presets também não esteja com 0
    send_telnet_cmd(f"set /environment/config/boundary/entry[0]/wind-speed-kt {speed*0.5925}")
    print(f"[child] Vento injetado: {speed:.0f} fps de {direction:.0f}° "
          f"(JSBSim wind: N={wind_n:+.1f}, E={wind_e:+.1f})")


def random_spawn_around_target():
    """
    Sorteia um ponto na circunferência de raio RADIUS_M ao redor do alvo.
    Retorna (lat, lon, heading_deg) — heading aponta para o alvo.
    """
    angle = random.uniform(0.0, 2.0 * math.pi)
    new_lat = TARGET_LAT + (RADIUS_M * math.cos(angle)) / 111320.0
    new_lon = TARGET_LON + (RADIUS_M * math.sin(angle)) / (
        111320.0 * math.cos(math.radians(TARGET_LAT))
    )
    heading = _bearing_deg(new_lat, new_lon, TARGET_LAT, TARGET_LON)
    return new_lat, new_lon, heading


def trigger_reset():
    """
    Fecha CSV atual, manda os comandos de teleporte para uma posição
    aleatória ao redor do alvo. Tudo fire-and-forget; quem espera é o
    callback (descartando pacotes) durante a janela RESETTING.
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

    # Posição aleatória na circunferência de raio RADIUS_M
    new_lat, new_lon, heading = random_spawn_around_target()
    heading_rad = math.radians(heading)
    # Velocidade 35 fps no rumo do alvo (decomposta em N/E)
    v_north = 35.0 * math.cos(heading_rad)
    v_east = 35.0 * math.sin(heading_rad)

    print(
        f"[child] Reset → lat={new_lat:.5f}, lon={new_lon:.5f}, "
        f"heading={heading:.1f}° (rumo ao alvo)"
    )

    # Reset SIMPLES — só escrita direta em /position, /orientation,
    # /velocities. NÃO usar /fdm/jsbsim/ic/* + run_ic — combinado com
    # /position/* leva o JSBSim a estado numericamente corrompido.
    cmds = [
        # 1. Zera controles
        "set /controls/flight/aileron 0",
        "set /controls/flight/elevator 0",
        "set /controls/flight/rudder 0",
        "set /controls/flight/flaps 0",

        # 2. Recolhe paraquedas (cmd)
        "set /fdm/jsbsim/systems/chute/chute-cmd-norm 0",

        # 3. Teleporte
        f"set /position/latitude-deg {new_lat}",
        f"set /position/longitude-deg {new_lon}",
        f"set /position/altitude-ft {START_ALT_FT}",

        # 4. Atitude (heading aponta para o alvo)
        "set /orientation/roll-deg 0",
        "set /orientation/pitch-deg 0",
        f"set /orientation/heading-deg {heading}",

        # 5. Velocidades (body e world frame)
        "set /velocities/uBody-fps 35",
        "set /velocities/vBody-fps 0",
        "set /velocities/wBody-fps 0",
        "set /velocities/v-down-fps 0",
        f"set /velocities/v-north-fps {v_north}",
        f"set /velocities/v-east-fps {v_east}",

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
    """
    Observação do MODELO CONE — mesma fórmula de parachute_cone_env._get_obs():
      [0] cone_err = (dist - altitude_m * 0.8) / 1000, clipado em [-1, 1]
      [1] bearing_err / 180
      [2] v_ground / 60
      [3] v_down / 30 (h-dot-fps; positivo p/ cima — no FDM packet,
                      v_down_ft_per_s é positivo p/ baixo, então invertemos)
      [4] roll (phi_rad)
      [5] pitch (theta_rad)
    """
    lat = math.degrees(fdm_data.lat_rad)
    lon = math.degrees(fdm_data.lon_rad)
    alt_m_msl = fdm_data.alt_m
    # AGL para alinhar o cone com o treinamento (que foi a nível do mar)
    alt_m_for_cone = (
        max(0.0, alt_m_msl - MOJAVE_GROUND_ALT_M)
        if USE_AGL_FOR_CONE else alt_m_msl
    )
    dist = haversine(lat, lon, TARGET_LAT, TARGET_LON)
    raio_ideal = alt_m_for_cone * 0.8  # mesmo glide_ratio_target do treino
    cone_err = (dist - raio_ideal) / 1000.0

    v_ground = math.sqrt(fdm_data.v_north_ft_per_s ** 2 + fdm_data.v_east_ft_per_s ** 2)
    # h-dot-fps no JSBSim do treino: positivo p/ cima.
    # No pacote FDM, v_down_ft_per_s é positivo p/ baixo → invertemos.
    h_dot = -fdm_data.v_down_ft_per_s

    target_hdg = (90.0 - math.degrees(math.atan2(TARGET_LAT - lat, TARGET_LON - lon))) % 360
    bearing_err = (target_hdg - math.degrees(fdm_data.psi_rad) + 180) % 360 - 180

    return np.array([
        np.clip(cone_err, -1, 1),
        bearing_err / 180,
        v_ground / 60,
        h_dot / 30,
        fdm_data.phi_rad,
        fdm_data.theta_rad,
    ], dtype=np.float32)


# ----------------------------------------------------------------------
# CALLBACK FDM — TUDO acontece aqui, no processo filho. NUNCA bloquear.
# ----------------------------------------------------------------------
def fdm_callback(fdm_data, event_pipe):
    global model, norm_env, csv_writer, csv_file_handle, current_session_folder
    global last_control_time, start_time, chute_deployed, current_actions
    global flight_number, phase, reset_done_at, wind_set_for_this_flight

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
            wind_set_for_this_flight = False  # vai injetar vento novo
            phase = PHASE_FLYING
            print(f"[child] --- Iniciando voo {flight_number} ---")
            # Cai para o caminho de FLYING abaixo

        # ---- INICIALIZAÇÃO (lazy, no filho) ----
        if model is None:
            print(f"[child] Carregando PPO: {MODEL_PATH}")
            model = PPO.load(MODEL_PATH)
            print("[child] PPO carregado.")
            # Carrega o VecNormalize (estatísticas de normalização da obs).
            # Sem isso a obs vai pro modelo em escala completamente diferente
            # da que ele viu no treino → comportamento errático.
            try:
                from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
                import gymnasium as gym
                from gymnasium import spaces

                class _Stub(gym.Env):
                    observation_space = spaces.Box(low=-100, high=100,
                                                   shape=(6,), dtype=np.float32)
                    action_space = spaces.Box(low=np.array([-1, 0]),
                                              high=np.array([1, 1]),
                                              dtype=np.float32)
                    def reset(self, seed=None, options=None):
                        return np.zeros(6, dtype=np.float32), {}
                    def step(self, a):
                        return np.zeros(6, dtype=np.float32), 0.0, True, False, {}

                stub = DummyVecEnv([lambda: _Stub()])
                norm_env = VecNormalize.load(VEC_NORMALIZE_PATH, stub)
                norm_env.training = False
                norm_env.norm_reward = False
                print(f"[child] VecNormalize carregado. obs_mean={norm_env.obs_rms.mean}, obs_var={norm_env.obs_rms.var}")
            except Exception as e:
                print(f"[child] AVISO: falha ao carregar VecNormalize ({e}). Obs irá direto sem normalização.")
                norm_env = None

        if start_time is None:
            start_time = now

        # Injeta vento aleatório uma vez por voo, igual ao treino
        if not wind_set_for_this_flight:
            set_random_wind()
            wind_set_for_this_flight = True

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
            global telnet_socket
            print(
                f"[child] >>> FÍSICA EXPLODIU no voo {flight_number} "
                f"(alt={alt_ft:.0f}ft, v_down={fdm_data.v_down_ft_per_s:.0f}ft/s). "
                f"Hard reset: matando e relançando FG."
            )
            # Fecha CSV
            if csv_file_handle is not None:
                try:
                    csv_file_handle.flush()
                    csv_file_handle.close()
                except Exception:
                    pass
                csv_file_handle = None
                csv_writer = None
            # Fecha telnet (não vai responder enquanto FG estiver morto)
            if telnet_socket is not None:
                try:
                    telnet_socket.close()
                except Exception:
                    pass
                telnet_socket = None

            if flight_number >= MAX_FLIGHTS:
                import os
                os._exit(0)

            # HARD RESET: mata FG, espera, relança em nova posição
            kill_flightgear()
            time.sleep(3.0)
            new_lat, new_lon, new_heading = random_spawn_around_target()
            launch_flightgear(new_lat, new_lon, START_ALT_FT, new_heading)
            wind_set_for_this_flight = False  # FG novo → injeta vento de novo
            phase = PHASE_RESETTING
            reset_done_at = time.time() + HARD_RESET_WAIT_S
            return fdm_data

        # ---- 1. Deploy do paraquedas a 3 s ----
        if elapsed > 3.0 and not chute_deployed:
            send_telnet_cmd("set /fdm/jsbsim/systems/chute/chute-cmd-norm 1")
            chute_deployed = True
            print(f"[child] Voo {flight_number}: paraquedas aberto em t={elapsed:.1f}s.")

        # ---- 2. Controle PPO a 1 Hz ----
        # Espera 4 s entre o deploy do chute e o primeiro comando — igual
        # ao treino, que faz `for _ in range(480): self.fdm.run()` antes
        # de retornar a primeira obs (480 × 1/120 = 4 s).
        chute_settled = chute_deployed and (elapsed > 7.0)
        if chute_settled and (now - last_control_time >= 1.0):
            obs = get_observation(fdm_data)
            if np.all(np.isfinite(obs)):
                # Normaliza obs com as estatísticas do treino (VecNormalize).
                if norm_env is not None:
                    obs_for_model = norm_env.normalize_obs(obs)
                else:
                    obs_for_model = obs
                action, _ = model.predict(obs_for_model, deterministic=True)
                a0_raw, a1_raw = float(action[0]), float(action[1])

                # Rate-limit no aileron — IDÊNTICO ao parachute_cone_env.step()
                # max_rate = 0.2 por step.
                max_rate = 0.2
                a0 = max(current_actions[0] - max_rate,
                         min(current_actions[0] + max_rate, a0_raw))
                # Action space treino: low=[-1,0], high=[1,1]
                a0 = max(-1.0, min(1.0, a0))
                a1 = max(0.0, min(1.0, a1_raw))
                current_actions = [a0, a1]

                # Escreve DIRETO em fcs/aileron-cmd-norm e
                # fcs/elevator-cmd-norm — exatamente o que o treino
                # parachute_cone_env.step() faz (linhas 118-119 do env).
                # Pular /controls/flight/* evita qualquer scaling /
                # deadband / rate-limit do XML da aeronave Parachutist.
                send_telnet_cmd(
                    f"set /fdm/jsbsim/fcs/aileron-cmd-norm {a0}\n"
                    f"set /fdm/jsbsim/fcs/elevator-cmd-norm {a1}"
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