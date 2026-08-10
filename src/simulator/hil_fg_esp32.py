"""
hil_fg_esp32.py — Hardware-In-the-Loop runner: FlightGear ↔ ESP32 ↔ CSV.

Fluxo de UM voo completo:
  1. Abre serial pro ESP32 (em MODE_HIL) e valida handshake "HIL_READY".
  2. Lança o FlightGear (modelo Parachutist, JSBSim, speed-up=1, sim pausado).
  3. Abre socket UDP pra ler FDM nativo do FG (~30 Hz, assíncrono).
  4. Abre telnet pra escrever comandos no FG.
  5. Teleporta avião pra spawn aleatório ao redor do alvo de Mojave, vento
     configurado, paraquedas armado, despausa.
  6. Loop de controle a 1 Hz wall-clock (== 1 Hz sim, porque speed-up=1):
       a. Lê pacote FDM mais recente do socket UDP.
       b. Calcula as 6 observações RAW (igual ao parachute_cone_env_v3).
       c. Envia obs pro ESP32 via serial binário.
       d. Recebe ação (aileron, elevator) do ESP32.
       e. Aplica ação no JSBSim via telnet.
       f. Grava linha no CSV com timestamps + estado + obs + ação.
  7. Termina quando:
       - alt_agl <= 5 m (pousou)
       - alt_msl > 30 km ou NaN (física explodiu)
       - dist > 10 km (perdeu o alvo)
       - tempo de voo > MAX_FLIGHT_TIME_S

ESCOPO: um voo por execução do script.
Pra múltiplos voos, rode o script repetidamente (cada vez relança o FG).

DEPENDÊNCIAS: pyserial, flightgear_python NÃO é usado (UDP raw é suficiente).
"""

import os
import sys
import time
import math
import socket
import struct
import subprocess
import argparse
import csv
from datetime import datetime
from pathlib import Path

import serial


# =====================================================================
# CONFIG — ajuste pra seu ambiente
# =====================================================================

# >>> ESP32 — porta serial (Windows: COM3, COM4...; Linux: /dev/ttyUSB0)
ESP32_PORT = "COM3"
ESP32_BAUD = 115200

# >>> FlightGear
FG_EXE = r"C:\Program Files\FlightGear 2024.1\bin\fgfs.exe"
FG_AIRCRAFT_PATH = r"C:\Users\coppi\OneDrive\Documents\FlightGear\Aircraft"
FG_AIRCRAFT = "Parachutist"
FG_TELNET_PORT = 5401
FG_FDM_PORT = 5501

# Alvo (Mojave) — mesmo do treino cone v3
TARGET_LAT = 34.9055
TARGET_LON = -117.8830
MOJAVE_GROUND_ALT_M = 700.0  # ~2300 ft MSL no solo

# Geometria do voo (idem treino)
SPAWN_RADIUS_M = 4000.0
START_ALT_FT = 9850

# Vento (escolhe um dos 4 níveis do treino; ajuste a gosto pra testes)
WIND_SPEED_FPS = 8.0
WIND_DIR_DEG = 0.0  # vindo do norte

# GR do Parachutist (idêntico ao usado no treino — fixo na obs[0])
GLIDE_RATIO = 1.88

# Controle
STEP_DT_S = 1.0  # 1 Hz, matching ESP32
MAX_FLIGHT_TIME_S = 600  # 10 min wall = 10 min sim com speed-up=1

# Saída CSV
CSV_BASE_DIR = r"D:\workspace\Pycharm\paraglider-autopilot\src\flight_records"


# =====================================================================
# UTIL
# =====================================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Azimute de (lat1,lon1) para (lat2,lon2), em graus [0, 360)."""
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(lat2r)
    x = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


# =====================================================================
# ESP32 CLIENT — protocolo binário do main.cpp (MODE_HIL)
# =====================================================================

HEADER_SEND = bytes([0xAA, 0xBB])  # PC → ESP32
HEADER_RECV = bytes([0xCC, 0xDD])  # ESP32 → PC
OBS_DIM = 6
ACTION_DIM = 2
PKT_RECV_SZ = 2 + ACTION_DIM * 4


class ESP32Client:
    """Cliente serial pro ESP32 rodando o main.cpp em MODE_HIL."""

    def __init__(self, port=ESP32_PORT, baud=ESP32_BAUD, timeout=1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None

    def connect(self, wait_ready=True, ready_timeout=5.0):
        """Abre porta serial e (opcionalmente) espera 'HIL_READY' do ESP32."""
        print(f"[ESP32] Abrindo {self.port} a {self.baud} baud...")
        self.ser = serial.Serial(self.port, self.baud, timeout=0.2)
        time.sleep(2.0)  # ESP32 reinicia ao abrir a porta — espera reboot
        self.ser.reset_input_buffer()

        if wait_ready:
            print("[ESP32] Aguardando 'HIL_READY'...")
            deadline = time.time() + ready_timeout
            buf = b""
            while time.time() < deadline:
                chunk = self.ser.read(self.ser.in_waiting or 1)
                if chunk:
                    buf += chunk
                    if b"HIL_READY" in buf:
                        print("[ESP32] Handshake OK. ESP32 pronto.")
                        # drena resto do buffer
                        time.sleep(0.1)
                        self.ser.reset_input_buffer()
                        return True
            print("[ESP32] AVISO: handshake 'HIL_READY' não recebido. Continuando mesmo assim.")
            self.ser.reset_input_buffer()
            return False
        return True

    def step(self, obs):
        """
        Envia 6 floats de obs, lê 2 floats de ação. Bloqueante.
        Retorna numpy-like array [aileron, elevator] como tupla de float.
        Levanta TimeoutError se ESP32 não responder em <1s.
        """
        if self.ser is None:
            raise RuntimeError("Serial não conectado")
        payload = struct.pack(f"<{OBS_DIM}f", *obs)
        self.ser.write(HEADER_SEND + payload)
        self.ser.flush()

        deadline = time.time() + 1.0
        buf = b""
        while time.time() < deadline:
            chunk = self.ser.read(self.ser.in_waiting or 1)
            if chunk:
                buf += chunk
                idx = buf.find(HEADER_RECV)
                if idx >= 0 and len(buf) >= idx + PKT_RECV_SZ:
                    aileron, elevator = struct.unpack("<2f", buf[idx+2:idx+PKT_RECV_SZ])
                    return float(aileron), float(elevator)
        raise TimeoutError(f"ESP32 não respondeu em 1s. Buffer parcial: {len(buf)} bytes")

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None


# =====================================================================
# FDM UDP READER — pacote nativo do JSBSim/FG (não-bloqueante)
# =====================================================================

# FGNetFDM struct: 2 uint32 + 3 double + 17 float = 100 bytes (primeiros campos)
FDM_STRUCT = struct.Struct(">IIdddfffffffffffffffff")
FDM_FIELDS = [
    "version", "padding",
    "lon_rad", "lat_rad", "alt_m",
    "agl_m", "phi", "theta", "psi", "alpha", "beta",
    "phidot", "thetadot", "psidot",
    "vcas", "climb_rate",
    "v_north_fps", "v_east_fps", "v_down_fps",
    "v_body_u_fps", "v_body_v_fps", "v_body_w_fps",
]


class FDMReader:
    def __init__(self, host="0.0.0.0", port=FG_FDM_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self._last = None

    def read_latest(self):
        """Drena buffer UDP, retorna o pacote mais recente como dict."""
        latest = None
        while True:
            try:
                data, _ = self.sock.recvfrom(8192)
                latest = data
            except (BlockingIOError, socket.error):
                break
        if latest is None:
            return self._last
        if len(latest) < FDM_STRUCT.size:
            return self._last
        values = FDM_STRUCT.unpack_from(latest, 0)
        self._last = dict(zip(FDM_FIELDS, values))
        return self._last

    def wait_for_packet(self, timeout=10.0):
        """Espera primeiro pacote válido (versão >= 24)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            d = self.read_latest()
            if d is not None and d.get("version", 0) >= 24:
                return d
            time.sleep(0.05)
        return None

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


# =====================================================================
# TELNET CLIENT — comandos pro FG (escrita rápida, leitura raramente)
# =====================================================================

class FGProps:
    def __init__(self, host="127.0.0.1", port=FG_TELNET_PORT, timeout=1.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self, retries=60, delay=2.0):
        for i in range(retries):
            try:
                s = socket.create_connection((self.host, self.port), timeout=self.timeout)
                s.settimeout(self.timeout)
                self.sock = s
                self._read_until_prompt(timeout=2.0)
                return True
            except (ConnectionRefusedError, OSError, socket.timeout):
                time.sleep(delay)
        raise RuntimeError("Não consegui conectar telnet ao FG")

    def _read_until_prompt(self, timeout=None):
        if timeout is None:
            timeout = self.timeout
        self.sock.settimeout(timeout)
        buf = b""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                chunk = self.sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if b"/>" in buf:
                    return buf.decode("ascii", errors="ignore")
            except socket.timeout:
                break
        return buf.decode("ascii", errors="ignore")

    def set(self, prop, value):
        msg = f"set {prop} {value}\r\n".encode("ascii")
        self.sock.sendall(msg)
        try:
            self.sock.settimeout(0.1)
            self.sock.recv(4096)
        except (socket.timeout, BlockingIOError):
            pass
        finally:
            self.sock.settimeout(self.timeout)

    def close(self):
        if self.sock is None:
            return
        try:
            self.sock.sendall(b"quit\r\n")
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass
        self.sock = None


# =====================================================================
# LAUNCHER DO FG
# =====================================================================

def launch_fg(spawn_lat, spawn_lon, alt_ft, heading=0.0):
    if not os.path.exists(FG_EXE):
        raise FileNotFoundError(f"FGFS_PATH não existe: {FG_EXE!r}")

    cmd = [
        FG_EXE,
        f"--fg-aircraft={FG_AIRCRAFT_PATH}",
        f"--aircraft={FG_AIRCRAFT}",
        f"--lat={spawn_lat}",
        f"--lon={spawn_lon}",
        f"--altitude={alt_ft}",
        f"--heading={heading}",
        "--in-air",
        "--glideslope=0",
        "--vc=0",
        "--enable-freeze",
        "--prop:/sim/speed-up=1",
        f"--telnet={FG_TELNET_PORT}",
        f"--native-fdm=socket,out,30,localhost,{FG_FDM_PORT},udp",
        "--prop:/nasal/local_weather/enabled=0",
        "--prop:/environment/weather-scenario=Fair weather",
        "--prop:/environment/params/control=manual",
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
        "--log-level=alert",
        "--timeofday=noon",
        "--geometry=400x300",
    ]
    return subprocess.Popen(cmd)


def hard_kill_fg():
    try:
        subprocess.run(
            ["taskkill", "/F", "/IM", "fgfs.exe"],
            capture_output=True, timeout=10,
        )
    except Exception:
        pass


# =====================================================================
# OBSERVAÇÃO — IDÊNTICA à do parachute_cone_env_v3
# =====================================================================

def compute_obs(fdm):
    """
    Calcula as 6 dimensões da observação RAW (não normalizada — o VecNormalize
    está embarcado no ESP32).

    Espelha EXATAMENTE a observação do parachute_cone_env_v3:
      obs[0] = cone_err = (dist - alt_m * GR) / 1000   clip[-1, 1]
      obs[1] = bearing_err / 180   (alvo à direita > 0)
      obs[2] = vg / 60             (ground speed normalizada)
      obs[3] = -v_down / 30        (descendo → negativo)
      obs[4] = roll_rad
      obs[5] = pitch_rad
    """
    lat = math.degrees(fdm["lat_rad"])
    lon = math.degrees(fdm["lon_rad"])
    alt_m = fdm["alt_m"]
    psi_rad = fdm["psi"]
    heading_deg = math.degrees(psi_rad) % 360.0

    dist_m = haversine(lat, lon, TARGET_LAT, TARGET_LON)
    raio_ideal = alt_m * GLIDE_RATIO
    cone_err = (dist_m - raio_ideal) / 1000.0
    cone_err = max(-1.0, min(1.0, cone_err))

    tgt_brg = bearing_deg(lat, lon, TARGET_LAT, TARGET_LON)
    bearing_err = (tgt_brg - heading_deg + 540.0) % 360.0 - 180.0
    bearing_norm = bearing_err / 180.0

    # ground speed (norte/leste -> magnitude)
    vn = fdm["v_north_fps"]
    ve = fdm["v_east_fps"]
    vg_fps = math.sqrt(vn * vn + ve * ve)
    vg_norm = vg_fps / 60.0

    # vertical (h_dot, negativo = descendo no convencional)
    v_down_fps = fdm["v_down_fps"]
    hdot_norm = -v_down_fps / 30.0  # convertendo positive-down → negative-descent

    roll_rad = fdm["phi"]
    pitch_rad = fdm["theta"]

    obs = (cone_err, bearing_norm, vg_norm, hdot_norm, roll_rad, pitch_rad)
    return obs, {
        "lat": lat, "lon": lon, "alt_m": alt_m, "alt_agl_m": max(0.0, alt_m - MOJAVE_GROUND_ALT_M),
        "heading_deg": heading_deg, "dist_m": dist_m,
        "vg_fps": vg_fps, "v_down_fps": v_down_fps,
        "roll_rad": roll_rad, "pitch_rad": pitch_rad,
    }


# =====================================================================
# HIL RUNNER
# =====================================================================

def random_spawn():
    import random
    angle = random.uniform(0.0, 2.0 * math.pi)
    lat = TARGET_LAT + (SPAWN_RADIUS_M * math.cos(angle)) / 111320.0
    lon = TARGET_LON + (SPAWN_RADIUS_M * math.sin(angle)) / (
        111320.0 * math.cos(math.radians(TARGET_LAT))
    )
    return lat, lon


def run_one_flight(esp32: ESP32Client, csv_path: str):
    """Executa um voo completo HIL e retorna estatísticas."""
    import random

    # 1) Spawn aleatório, vento e heading
    spawn_lat, spawn_lon = random_spawn()
    heading_init = random.uniform(0.0, 360.0)

    # Vento como vetor JSBSim (idêntico ao parachute_cone_env_v3):
    #   wind_n = -cos(dir) * speed   (norte-fps)
    #   wind_e = -sin(dir) * speed   (leste-fps)
    # Convenção: dir = 0° significa vento VINDO DO NORTE → vetor aponta pra SUL
    # (wind_n negativo).
    rad_wind = math.radians(WIND_DIR_DEG)
    wind_n_fps = -math.cos(rad_wind) * WIND_SPEED_FPS
    wind_e_fps = -math.sin(rad_wind) * WIND_SPEED_FPS

    print("=" * 70)
    print("HIL FLIGHT")
    print("=" * 70)
    print(f"  spawn   = ({spawn_lat:.5f}, {spawn_lon:.5f})")
    print(f"  heading = {heading_init:.1f}°")
    print(f"  wind    = {WIND_SPEED_FPS:.1f} fps @ {WIND_DIR_DEG:.0f}°  "
          f"→ JSBSim (n,e) = ({wind_n_fps:+.2f}, {wind_e_fps:+.2f}) fps")
    print(f"  target  = ({TARGET_LAT}, {TARGET_LON}) (Mojave)")
    print(f"  alt0    = {START_ALT_FT} ft MSL")
    print(f"  csv     = {csv_path}")
    print()

    # 2) Mata FG anterior (se houver) e relança
    hard_kill_fg()
    time.sleep(3.0)

    fdm_reader = FDMReader(port=FG_FDM_PORT)
    print(f"[FG] UDP FDM listener aberto na porta {FG_FDM_PORT}.")

    fg_proc = launch_fg(spawn_lat, spawn_lon, START_ALT_FT, heading=heading_init)
    print("[FG] FG lançado, aguardando boot...")

    tn = FGProps()
    tn.connect(retries=60, delay=2.0)
    print("[FG] Telnet conectado.")

    # 3) Boot + setup — aguarda Nasal estabilizar
    time.sleep(20.0)
    try:
        # Pausa total (defesa contra Nasal despausar)
        tn.set("/sim/freeze/master", 1)
        tn.set("/sim/freeze/clock", 1)
        tn.set("/sim/freeze/fuel", 1)
        tn.set("/sim/pause", 1)
        tn.set("/sim/speed-up", 1)

        # >>> VENTO DIRETO NO JSBSIM <<<
        # Idêntico ao parachute_cone_env_v3.py (não usar /environment/* do
        # FG porque ele passa pela camada de weather que pode amplificar
        # ou modular o vento de forma incompatível com o treino).
        tn.set("/fdm/jsbsim/atmosphere/wind-north-fps", wind_n_fps)
        tn.set("/fdm/jsbsim/atmosphere/wind-east-fps", wind_e_fps)
        # Defesa adicional: zera o vento do FG environment pra não
        # somar com o do JSBSim.
        tn.set("/environment/wind-from-heading-deg", 0)
        tn.set("/environment/wind-speed-kt", 0)

        # Despausa
        tn.set("/sim/freeze/master", 0)
        tn.set("/sim/freeze/clock", 0)
        tn.set("/sim/freeze/fuel", 0)
        tn.set("/sim/pause", 0)

        # Reaplica vento APÓS despausar (o run_ic do JSBSim pode resetar)
        tn.set("/fdm/jsbsim/atmosphere/wind-north-fps", wind_n_fps)
        tn.set("/fdm/jsbsim/atmosphere/wind-east-fps", wind_e_fps)

        # Espera estabilizar e abre paraquedas
        time.sleep(2.0)
        tn.set("/fdm/jsbsim/systems/chute/chute-cmd-norm", 1)
        print("[FG] Paraquedas ARMADO. Aguardando estabilização (4s)...")
        time.sleep(4.0)

        # Reaplica vento mais uma vez antes do loop principal
        tn.set("/fdm/jsbsim/atmosphere/wind-north-fps", wind_n_fps)
        tn.set("/fdm/jsbsim/atmosphere/wind-east-fps", wind_e_fps)
    except Exception as e:
        print(f"[FG] Erro no setup: {e}")
        tn.close()
        fdm_reader.close()
        hard_kill_fg()
        raise

    # 4) Aguarda primeiro pacote FDM válido com altitude esperada
    print("[FG] Aguardando telemetria UDP estável...")
    first = fdm_reader.wait_for_packet(timeout=10.0)
    if first is None:
        print("[FG] ERRO: nenhum pacote FDM recebido.")
        tn.close()
        fdm_reader.close()
        hard_kill_fg()
        return None

    # 5) Abre CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "t_wall_s", "t_sim_s", "step",
        "lat", "lon", "alt_msl_m", "alt_agl_m", "heading_deg",
        "vg_fps", "v_down_fps", "roll_rad", "pitch_rad",
        "dist_target_m",
        "obs_cone_err", "obs_bearing_norm", "obs_vg_norm",
        "obs_hdot_norm", "obs_roll", "obs_pitch",
        "action_aileron", "action_elevator",
        "esp32_latency_ms",
    ])

    # 6) Loop principal
    print()
    print("[LOOP] Iniciando controle HIL @ 1 Hz...")
    print(f"{'step':>4} {'t_sim':>6} {'alt_agl':>8} {'dist':>6} "
          f"{'cone_e':>7} {'bear_e':>7} {'ail':>7} {'ele':>6} {'esp_ms':>6}")
    print("-" * 70)

    t_start_wall = time.time()
    step_num = 0
    aileron_last = 0.0
    elevator_last = 0.0
    landed = False
    end_reason = "timeout"

    try:
        while True:
            t_wall = time.time() - t_start_wall

            # Critério de timeout
            if t_wall > MAX_FLIGHT_TIME_S:
                end_reason = "timeout_wall"
                break

            step_num += 1

            # 6a) Lê estado mais recente do FG
            fdm = fdm_reader.read_latest()
            if fdm is None:
                print(f"[step {step_num}] sem pacote FDM, pulando...")
                time.sleep(STEP_DT_S)
                continue

            # Sanity check NaN / explosão
            alt_m = fdm["alt_m"]
            if not math.isfinite(alt_m) or alt_m > 30000.0 or alt_m < -200.0:
                print(f"[step {step_num}] EXPLOSÃO/NaN: alt={alt_m}")
                end_reason = "physics_explosion"
                break

            # 6b) Calcula obs
            obs, ctx = compute_obs(fdm)

            # Critérios de fim
            if ctx["alt_agl_m"] <= 5.0:
                landed = True
                end_reason = "landed"
            elif ctx["dist_m"] > 10000.0:
                end_reason = "lost_target"
                break

            # 6c) Envia obs pro ESP32, recebe ação
            t_esp_start = time.time()
            try:
                aileron, elevator = esp32.step(obs)
            except TimeoutError as e:
                print(f"[step {step_num}] ESP32 timeout: {e}")
                end_reason = "esp32_timeout"
                break
            esp32_latency_ms = (time.time() - t_esp_start) * 1000.0

            aileron_last = aileron
            elevator_last = elevator

            # 6d) Aplica ação no JSBSim via telnet
            try:
                tn.set("/fdm/jsbsim/fcs/aileron-cmd-norm", aileron)
                tn.set("/fdm/jsbsim/fcs/elevator-cmd-norm", elevator)
                # Reaplica vento a cada step (defesa contra reset/override
                # por subsistemas do FG). Custa ~2 telnet sets por step,
                # tempo total < 10ms.
                tn.set("/fdm/jsbsim/atmosphere/wind-north-fps", wind_n_fps)
                tn.set("/fdm/jsbsim/atmosphere/wind-east-fps", wind_e_fps)
            except Exception as e:
                print(f"[step {step_num}] erro no telnet: {e}")
                end_reason = "telnet_error"
                break

            # Pega sim_t (telnet — barato, 1 chamada)
            # OBS: comentado pra evitar overhead. Usa wall como referência.
            # Pode descomentar se quiser sim_t no CSV.
            t_sim = t_wall  # speed-up=1, então sim ≈ wall

            # 6e) Loga no CSV
            writer.writerow([
                round(t_wall, 3), round(t_sim, 3), step_num,
                round(ctx["lat"], 7), round(ctx["lon"], 7),
                round(ctx["alt_m"], 2), round(ctx["alt_agl_m"], 2),
                round(ctx["heading_deg"], 2),
                round(ctx["vg_fps"], 2), round(ctx["v_down_fps"], 2),
                round(ctx["roll_rad"], 4), round(ctx["pitch_rad"], 4),
                round(ctx["dist_m"], 2),
                round(obs[0], 5), round(obs[1], 5), round(obs[2], 5),
                round(obs[3], 5), round(obs[4], 5), round(obs[5], 5),
                round(aileron, 5), round(elevator, 5),
                round(esp32_latency_ms, 2),
            ])
            csv_file.flush()

            # Print periódico (a cada step)
            print(f"{step_num:>4} {t_wall:>6.1f} {ctx['alt_agl_m']:>8.0f} "
                  f"{ctx['dist_m']:>6.0f} {obs[0]:>+7.3f} {obs[1]:>+7.3f} "
                  f"{aileron:>+7.3f} {elevator:>6.3f} {esp32_latency_ms:>6.1f}")

            if landed:
                break

            # 6f) Cadência 1 Hz (real-time)
            elapsed = time.time() - t_start_wall - t_wall
            sleep_left = STEP_DT_S - elapsed
            if sleep_left > 0:
                time.sleep(sleep_left)

    except KeyboardInterrupt:
        end_reason = "user_interrupt"
        print("\n[LOOP] Interrompido pelo usuário.")
    finally:
        csv_file.close()
        tn.close()
        fdm_reader.close()
        hard_kill_fg()

    # 7) Resumo
    print()
    print("=" * 70)
    print(f"  FIM DO VOO — razão: {end_reason}")
    if step_num > 0:
        last_dist = ctx["dist_m"] if 'ctx' in locals() else float("nan")
        last_alt = ctx["alt_agl_m"] if 'ctx' in locals() else float("nan")
        last_vd = abs(ctx["v_down_fps"]) if 'ctx' in locals() else float("nan")
        print(f"  Steps:           {step_num}")
        print(f"  Tempo voo:       {t_wall:.1f} s")
        print(f"  Distância final: {last_dist:.1f} m")
        print(f"  Altitude final:  {last_alt:.1f} m AGL")
        print(f"  Vel. vert. fim:  {last_vd:.1f} fps")
    print(f"  CSV salvo em:    {csv_path}")
    print("=" * 70)

    return {
        "end_reason": end_reason,
        "steps": step_num,
        "duration_s": t_wall if step_num > 0 else 0.0,
        "csv_path": csv_path,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="HIL runner: FG ↔ ESP32 ↔ CSV")
    parser.add_argument("--port", default=ESP32_PORT, help="Porta serial do ESP32")
    parser.add_argument("--csv", default=None, help="Caminho do CSV (default: auto-timestamp)")
    parser.add_argument("--no-handshake", action="store_true",
                        help="Pula handshake HIL_READY (use se ESP32 já está rodando)")
    args = parser.parse_args()

    # CSV path
    if args.csv:
        csv_path = args.csv
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(CSV_BASE_DIR, f"hil_session_{ts}", "flight.csv")

    # 1) ESP32 PRIMEIRO — se ele não estiver, não adianta nem ligar FG
    esp32 = ESP32Client(port=args.port)
    try:
        esp32.connect(wait_ready=not args.no_handshake)
    except Exception as e:
        print(f"[ESP32] Falha ao conectar: {e}")
        print("Verifique que:")
        print("  - O ESP32 está plugado e a porta serial existe")
        print("  - O firmware foi compilado com #define MODE_HIL")
        print(f"  - A porta correta é {args.port} (verifique no Device Manager)")
        sys.exit(1)

    # 2) Testa 1 step de inferência com obs zerada (sanity check)
    print("[ESP32] Sanity check: enviando obs zerada...")
    try:
        ail, ele = esp32.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        print(f"[ESP32] OK. Resposta: aileron={ail:+.4f} elevator={ele:.4f}")
    except Exception as e:
        print(f"[ESP32] Sanity check FALHOU: {e}")
        esp32.close()
        sys.exit(1)
    print()

    # 3) Roda o voo
    try:
        result = run_one_flight(esp32, csv_path)
    finally:
        esp32.close()
        print("[ESP32] Porta serial fechada.")

    if result is None:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()