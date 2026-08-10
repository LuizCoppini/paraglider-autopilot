"""
train_parachute_cone_fg.py — treino do método CONE direto no FlightGear.

OBJETIVO:
  Treinar o PPO usando o FlightGear como engine de física, ao invés do
  JSBSim Python puro. Isso elimina o gap FG↔JSBSim por construção: o
  modelo aprende NO ambiente em que vai ser validado.

ARQUITETURA:
  - Gym env (FGConeEnv) que fala com FG via telnet (porta 5401).
  - Single-process, síncrono. Cada step manda action + lê estado.
  - Speed-up via /sim/speed-up (setado por telnet APÓS conectar).
  - Reset rápido via property writes (/position/*, /orientation/*).
  - Hard reset (kill+relaunch FG) só se detectar NaN ou explosão.
  - Taxa de ação = 1 Hz (compatível com ESP32 alvo).

COMO RODAR:
  Coloca em src/rl/train_parachute_cone_fg.py do seu projeto.
  Antes de rodar: feche TODAS as instâncias do FlightGear abertas.
    poetry run python src/rl/train_parachute_cone_fg.py
"""

import os
import sys
import time
import math
import socket
import struct
import subprocess
from datetime import datetime

# Adiciona src/ ao sys.path pra importar config.settings
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback


# =====================================================================
# CONFIG — ajuste pra seu ambiente
# =====================================================================

FG_EXE_FALLBACK = r"C:\Program Files\FlightGear 2024.1\bin\fgfs.exe"
FG_AIRCRAFT_PATH_FALLBACK = r"C:\Users\coppi\OneDrive\Documents\FlightGear\Aircraft"

try:
    from config import settings as _settings
    FG_EXE = _settings.FGFS_PATH
    FG_AIRCRAFT_PATH = getattr(_settings, "AIRCRAFT_PATH", None)
    FG_AIRCRAFT = getattr(_settings, "AIRCRAFT", "Parachutist")
    FG_AIRPORT = None  # ignoramos KSFO; vamos teleportar pra Mojave
    print(f"[config] FGFS_PATH = {FG_EXE}")
except Exception as _e:
    print(f"[config] settings não disponível ({_e}). Usando fallback.")
    FG_EXE = FG_EXE_FALLBACK
    FG_AIRCRAFT_PATH = FG_AIRCRAFT_PATH_FALLBACK
    FG_AIRCRAFT = "Parachutist"
    FG_AIRPORT = None

BASE_MODELS_PATH = r"D:\workspace\Pycharm\paraglider-autopilot\models\fg_native_method"

# Alvo Mojave
TARGET_LAT = 34.9055
TARGET_LON = -117.8830

# GR do Parachutist
GLIDE_RATIO = 1.88

# Spawn
SPAWN_RADIUS_M = 4000.0
START_ALT_FT = 9850
MOJAVE_GROUND_ALT_M = 700.0

# Vento — mesma faixa do treino JSBSim v2
WIND_SPEEDS_FPS = (4.0, 8.0, 14.0, 20.0)
WIND_DIR_DEG = 0.0

# Speed-up do FG (setado por telnet após conectar)
SPEED_UP = 8

# Telnet do FG (pra SET de comandos)
FG_HOST = "127.0.0.1"
FG_TELNET_PORT = 5401

# UDP FDM do FG (pra LER estado — assíncrono, sem round-trip)
FG_FDM_PORT = 5501

# >>> MODO DE EXECUÇÃO <<<
# Antes de tentar treinar, rode com SANITY_CHECK_ONLY=True. Esse modo:
#  - lança o FG
#  - despausa, aplica speed-up, abre paraquedas
#  - lê estado a cada 1s wall-clock por 60s
#  - printa altitude, velocidade, distância pra você VER se o avião
#    está caindo na velocidade esperada
# Se funcionar (avião descendo ~5-8 fps × speed-up), mude pra False e
# rode o treino real.
SANITY_CHECK_ONLY = False

# >>> TREINO — teste de sanidade rápido <<<
# 2k timesteps com 2-3 it/s = ~15 min compute + cold boots.
# Suficiente pra confirmar que a infra aguenta E ver o PPO começar a aprender.
TOTAL_TIMESTEPS = 2_000
MAX_EPISODE_STEPS = 100     # ~5min sim; previne avião perdido eterno
STEP_DT_SIM_S = 1.0
PREVENTIVE_COLD_BOOT_EVERY_N_EPS = 4  # cold boot proativo
MIN_AGL_TERMINATE = 50.0    # termina episódio bem antes do solo


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
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(lat2r)
    x = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


# =====================================================================
# FDM UDP READER — leitura assíncrona do estado físico via FGNetFDM
# =====================================================================
# FG envia estado físico continuamente via --native-fdm=socket,out,...
# Pacote binário ~408 bytes, big-endian. Parseamos só os campos que
# precisamos (primeiros 100 bytes). NÃO há round-trip; só lemos o último
# pacote do buffer UDP. Latência típica ~1ms.

# Layout de FGNetFDM (FlightGear/src/Network/net_fdm.hxx) primeiros 22
# fields, total 100 bytes:
#   version, padding (u32×2 = 8 bytes)
#   longitude_rad, latitude_rad, altitude_m (f64×3 = 24 bytes)
#   agl_m, phi, theta, psi, alpha, beta (f32×6 = 24 bytes)
#   phidot, thetadot, psidot (f32×3 = 12 bytes)
#   vcas, climb_rate (f32×2 = 8 bytes)
#   v_north, v_east, v_down (f32×3 = 12 bytes)
#   v_body_u, v_body_v, v_body_w (f32×3 = 12 bytes)
# 2 uint32 + 3 double + 17 float = 22 campos, 100 bytes (count os f's!)
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
    """Lê pacotes UDP FGNetFDM emitidos pelo FG. Non-blocking."""

    def __init__(self, host="0.0.0.0", port=FG_FDM_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        # Reuso da porta caso o FG anterior tenha deixado socket pendurado
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self._last = None

    def read_latest(self):
        """
        Drena todos os pacotes pendentes no buffer e retorna o mais
        recente como dict. Se não houver pacote novo, retorna o último
        cache (ou None se nunca leu).
        """
        latest_data = None
        while True:
            try:
                data, _ = self.sock.recvfrom(8192)
                latest_data = data
            except (BlockingIOError, socket.error):
                break
        if latest_data is None:
            return self._last
        if len(latest_data) < FDM_STRUCT.size:
            print(f"[FDM] pacote muito pequeno: {len(latest_data)} bytes "
                  f"(esperado >= {FDM_STRUCT.size})")
            return self._last
        values = FDM_STRUCT.unpack_from(latest_data, 0)
        if len(values) != len(FDM_FIELDS):
            print(f"[FDM] MISMATCH: struct deu {len(values)} valores, "
                  f"FDM_FIELDS tem {len(FDM_FIELDS)} — corrige!")
        self._last = dict(zip(FDM_FIELDS, values))
        return self._last

    def wait_for_packet(self, timeout=5.0):
        """Espera até receber pelo menos UM pacote ou timeout."""
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
# CLIENTE TELNET — modo interativo, parsing simples, reconnect auto
# =====================================================================
# Protocolo (modo interativo, default do FG):
#   "get /prop\r\n"  → "/prop = 'VALOR' (TIPO)\n/>"
#   "set /prop VAL\r\n" → "/>"
# O prompt "/>" sempre fecha. Procuramos por ele.

class FGProps:
    def __init__(self, host=FG_HOST, port=FG_TELNET_PORT, timeout=1.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self, retries=60, delay=2.0):
        last_err = None
        for i in range(retries):
            try:
                s = socket.create_connection((self.host, self.port), timeout=self.timeout)
                s.settimeout(self.timeout)
                self.sock = s
                # consome banner inicial — espera primeiro prompt
                self._read_until_prompt(timeout=3.0)
                return True
            except (ConnectionRefusedError, OSError, socket.timeout) as e:
                last_err = e
                try:
                    if s is not None:
                        s.close()
                except Exception:
                    pass
                time.sleep(delay)
        raise RuntimeError(f"Não consegui conectar telnet ao FG: {last_err}")

    def _read_until_prompt(self, timeout=None):
        """Lê do socket até receber o prompt '/>'. Retorna a string lida."""
        if timeout is None:
            timeout = self.timeout
        self.sock.settimeout(timeout)
        buf = b""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                chunk = self.sock.recv(4096)
                if not chunk:
                    raise ConnectionResetError("socket fechou")
                buf += chunk
                if b"/>" in buf:
                    return buf.decode("ascii", errors="ignore")
            except socket.timeout:
                break
        return buf.decode("ascii", errors="ignore")

    def _send(self, msg):
        if not msg.endswith("\n"):
            msg = msg + "\r\n"
        self.sock.sendall(msg.encode("ascii"))

    def _reconnect(self):
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None
        self.connect(retries=10, delay=1.0)

    def get(self, prop, _retry=True):
        try:
            self._send(f"get {prop}")
            resp = self._read_until_prompt()
            # parse linha "<prop> = 'V' (T)"
            for line in resp.split("\n"):
                line = line.strip()
                if "=" in line and not line.startswith("/>"):
                    val = line.split("=", 1)[1].strip()
                    val = val.split("(")[0].strip().strip("'").strip()
                    try:
                        return float(val)
                    except ValueError:
                        return val
            return None
        except (ConnectionResetError, OSError, socket.timeout) as e:
            if _retry:
                print(f"[telnet] get({prop}) falhou ({e}). Reconectando...")
                self._reconnect()
                return self.get(prop, _retry=False)
            raise

    def set(self, prop, value, _retry=True):
        try:
            self._send(f"set {prop} {value}")
            self._read_until_prompt()
        except (ConnectionResetError, OSError, socket.timeout) as e:
            if _retry:
                print(f"[telnet] set({prop}) falhou ({e}). Reconectando...")
                self._reconnect()
                return self.set(prop, value, _retry=False)
            raise

    def close(self):
        if self.sock is None:
            return
        try:
            self._send("quit")
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
        raise FileNotFoundError(
            f"FGFS_PATH não existe: {FG_EXE!r}. Edite config/settings.py."
        )

    cmd = [FG_EXE]
    if FG_AIRCRAFT_PATH:
        cmd.append(f"--fg-aircraft={FG_AIRCRAFT_PATH}")
    cmd.append(f"--aircraft={FG_AIRCRAFT}")
    if FG_AIRPORT:
        cmd.append(f"--airport={FG_AIRPORT}")

    cmd += [
        f"--lat={spawn_lat}",
        f"--lon={spawn_lon}",
        f"--altitude={alt_ft}",
        f"--heading={heading}",
        "--in-air",                # força avião a começar no ar
        "--glideslope=0",
        "--vc=0",
        "--enable-freeze",
        "--prop:/sim/speed-up=1",
        f"--telnet={FG_TELNET_PORT}",
        # >>> FDM via UDP nativo — leitura assíncrona, sem round-trip <<<
        # 30 Hz é mais que suficiente pra controle a 1 Hz.
        f"--native-fdm=socket,out,30,localhost,{FG_FDM_PORT},udp",
        # Render mínimo
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
        "--prop:/sim/frame-rate-throttle-hz=0",  # 0 = sem throttle, max FPS
        "--log-level=alert",
        "--timeofday=noon",
        "--geometry=400x300",  # janela mínima
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
# GYM ENV
# =====================================================================

class FGConeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32)

        self.target_lat = TARGET_LAT
        self.target_lon = TARGET_LON
        self.glide_ratio = GLIDE_RATIO
        self.spawn_radius_m = SPAWN_RADIUS_M
        self.start_alt_ft = START_ALT_FT
        self.wind_speeds = WIND_SPEEDS_FPS

        self.episode = 0
        self.steps = 0
        self.last_dist = None
        self.last_step_wall = None
        self.fg_process = None
        self.tn = None
        self.fdm = None  # FDMReader (UDP) — leitura rápida de estado

        self.lat = self.lon = self.alt_m = 0.0
        self.alt_agl_m = 0.0
        self.dist = 0.0
        self.current_wind_speed = 0.0
        self._last_vdown_fps = 0.0

        self._cold_boot()

    # -----------------------------------------------------------------

    def _random_spawn(self):
        angle = float(np.random.uniform(0.0, 2.0 * math.pi))
        lat = self.target_lat + (self.spawn_radius_m * math.cos(angle)) / 111320.0
        lon = self.target_lon + (
            self.spawn_radius_m * math.sin(angle)
        ) / (111320.0 * math.cos(math.radians(self.target_lat)))
        return lat, lon

    def _cold_boot(self):
        """Cold boot com 2 tentativas, sleeps generosos, retries altos."""
        for attempt in range(2):
            try:
                self._cold_boot_inner()
                return
            except Exception as e:
                print(f"[FGEnv] Cold boot tentativa {attempt+1} falhou: {e}")
                if attempt == 0:
                    print("[FGEnv] Vou tentar de novo com kill mais agressivo...")
                    hard_kill_fg()
                    time.sleep(8.0)
                else:
                    raise

    def _cold_boot_inner(self):
        print("[FGEnv] Cold boot do FlightGear...")
        if self.tn is not None:
            try:
                self.tn.close()
            except Exception:
                pass
            self.tn = None
        if self.fdm is not None:
            try:
                self.fdm.close()
            except Exception:
                pass
            self.fdm = None
        hard_kill_fg()
        time.sleep(5.0)  # tempo generoso pro Windows liberar processos

        # Abre o UDP receiver ANTES de lançar o FG (pra não perder pacotes iniciais)
        self.fdm = FDMReader(port=FG_FDM_PORT)
        print(f"[FGEnv] UDP FDM listener aberto na porta {FG_FDM_PORT}.")

        lat, lon = self._random_spawn()
        self.fg_process = launch_fg(lat, lon, self.start_alt_ft, heading=0.0)

        print("[FGEnv] Aguardando FG aceitar telnet (até 4.5 min)...")
        self.tn = FGProps()
        # 90 retries × 3s = 270s = 4.5 min de paciência
        self.tn.connect(retries=90, delay=3.0)
        print("[FGEnv] Telnet conectado.")

        # Espera FG estabilizar (Nasal, FDM, terreno) — sim PAUSADO
        time.sleep(20.0)

        # >>> FORÇA PAUSADO + speed-up SETADO mas sim ainda parado <<<
        # Tem que setar TODAS as freezes (--enable-freeze pode só ter
        # ativado master, mas clock/fuel/replay também podem travar sim).
        try:
            self.tn.set("/sim/freeze/master", 1)
            self.tn.set("/sim/freeze/clock", 1)
            self.tn.set("/sim/freeze/fuel", 1)
            self.tn.set("/sim/pause", 1)
            self.tn.set("/sim/speed-up", SPEED_UP)
            print(f"[FGEnv] TODAS freezes=1, speed-up pré-configurado em {SPEED_UP}x.")
        except Exception as e:
            print(f"[FGEnv] Falha pause/speed-up: {e}")

        # >>> SANITY CHECK <<<
        try:
            freeze = self.tn.get("/sim/freeze/master")
            speedup = self.tn.get("/sim/speed-up")
            alt = self.tn.get("/position/altitude-ft")
            agl = self.tn.get("/position/altitude-agl-ft")
            vdown = self.tn.get("/velocities/vertical-speed-fps")
            sim_t = self.tn.get("/sim/time/elapsed-sec")
            print(f"[SANITY] freeze={freeze} speed-up={speedup} "
                  f"alt={alt}ft agl={agl}ft v_down={vdown}fps sim_t={sim_t}")
        except Exception as e:
            print(f"[SANITY] read falhou: {e}")

        print("[FGEnv] FG pronto (PAUSADO). reset()/sanity vai despausar.")

    def _check_explosion(self):
        try:
            if not math.isfinite(self.lat) or not math.isfinite(self.lon):
                return True
            if not math.isfinite(self.alt_m):
                return True
            if self.alt_m > 30000.0 or self.alt_m < -200.0:
                return True
            if abs(self.lat) > 90.0:
                return True
        except Exception:
            return True
        return False

    # -----------------------------------------------------------------

    def _read_state(self):
        """Lê estado via UDP FDM (assíncrono, ~1ms)."""
        fdm = self.fdm.read_latest()
        if fdm is None:
            print("[FGEnv] _read_state: nenhum pacote FDM ainda.")
            self.lat = self.lon = self.alt_m = float("nan")
            return np.zeros(8, dtype=np.float32)

        lat = math.degrees(fdm["lat_rad"])
        lon = math.degrees(fdm["lon_rad"])
        alt_m = fdm["alt_m"]
        psi_rad = fdm["psi"]
        heading = math.degrees(psi_rad) % 360.0
        u = fdm["v_body_u_fps"]
        v = fdm["v_body_v_fps"]
        w = fdm["v_body_w_fps"]

        # Detecta NaN imediatamente — marca como explosão pra disparar cold boot
        for x in (lat, lon, alt_m, u, v, w):
            if not math.isfinite(x):
                self.lat = self.lon = self.alt_m = float("nan")
                return np.zeros(8, dtype=np.float32)

        self.lat = lat
        self.lon = lon
        self.alt_m = alt_m
        self.alt_agl_m = max(0.0, alt_m - MOJAVE_GROUND_ALT_M)
        self.dist = haversine(lat, lon, self.target_lat, self.target_lon)
        # cache para print no reset (v_down do FG é positive-down)
        self._last_vdown_fps = -fdm["v_down_fps"]

        cone_radius = self.alt_agl_m * self.glide_ratio
        cone_err = (self.dist - cone_radius) / 1000.0

        tgt_brg = bearing_deg(lat, lon, self.target_lat, self.target_lon)
        heading_err = (tgt_brg - heading + 540.0) % 360.0 - 180.0
        he_rad = math.radians(heading_err)

        obs = np.array([
            cone_err,
            self.dist / 5500.0,
            math.cos(he_rad),
            math.sin(he_rad),
            u / 50.0,
            v / 50.0,
            w / 50.0,
            self.alt_agl_m / 3000.0,
        ], dtype=np.float32)
        return obs

    # -----------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.steps = 0
        self.last_dist = None
        self.last_step_wall = None

        # Cold boot PREVENTIVO a cada N eps pra evitar acúmulo de NaN no JSBSim
        if self.episode > 1 and (self.episode - 1) % PREVENTIVE_COLD_BOOT_EVERY_N_EPS == 0:
            print(f"[FGEnv] Cold boot PREVENTIVO no ep {self.episode} "
                  f"(a cada {PREVENTIVE_COLD_BOOT_EVERY_N_EPS} eps).")
            self._cold_boot()

        lat, lon = self._random_spawn()
        wind_speed_fps = self.wind_speeds[self.episode % len(self.wind_speeds)]
        wind_speed_kt = wind_speed_fps * 0.5925
        heading = float(np.random.uniform(0.0, 360.0))
        self.current_wind_speed = wind_speed_fps

        try:
            # >>> SEQUÊNCIA CRÍTICA: PAUSA → CONFIG → UNPAUSE <<<
            # Importante: usar 0/1 (não "true"/"false") e liberar TODAS
            # as freezes, não só master.

            # 1) PAUSA o sim (TODAS as freezes)
            self.tn.set("/sim/freeze/master", 1)
            self.tn.set("/sim/freeze/clock", 1)
            self.tn.set("/sim/freeze/fuel", 1)
            self.tn.set("/sim/pause", 1)

            # 2) Reset controles
            self.tn.set("/fdm/jsbsim/systems/chute/chute-cmd-norm", 0)
            self.tn.set("/fdm/jsbsim/fcs/aileron-cmd-norm", 0)
            self.tn.set("/controls/flight/aileron", 0)
            self.tn.set("/controls/flight/elevator", 0)
            self.tn.set("/controls/flight/rudder", 0)

            # 3) Zera velocidades
            self.tn.set("/velocities/uBody-fps", 0)
            self.tn.set("/velocities/vBody-fps", 0)
            self.tn.set("/velocities/wBody-fps", 0)
            self.tn.set("/velocities/vertical-speed-fps", 0)

            # 4) Teleporta posição
            self.tn.set("/position/latitude-deg", lat)
            self.tn.set("/position/longitude-deg", lon)
            self.tn.set("/position/altitude-ft", self.start_alt_ft)

            # 5) Orientação
            self.tn.set("/orientation/heading-deg", heading)
            self.tn.set("/orientation/pitch-deg", 0)
            self.tn.set("/orientation/roll-deg", 0)

            # 6) Vento
            self.tn.set("/environment/wind-from-heading-deg", WIND_DIR_DEG)
            self.tn.set("/environment/wind-speed-kt", wind_speed_kt)

            # 7) Speed-up
            self.tn.set("/sim/speed-up", SPEED_UP)

            # 8) DESPAUSA — TODAS as freezes em 0
            sim_t_before = self.tn.get("/sim/time/elapsed-sec")
            self.tn.set("/sim/freeze/master", 0)
            self.tn.set("/sim/freeze/clock", 0)
            self.tn.set("/sim/freeze/fuel", 0)
            self.tn.set("/sim/pause", 0)
            self.tn.set("/sim/freeze/replay-state", 0)

            # 9) Espera 2s sim_t pro avião estabilizar do teleport
            time.sleep(2.0 / SPEED_UP)

            # >>> SANITY: sim_t avançou? <<<
            sim_t_after = self.tn.get("/sim/time/elapsed-sec")
            try:
                if float(sim_t_after) <= float(sim_t_before):
                    print(f"[FGEnv] AVISO: sim NÃO avançou "
                          f"({sim_t_before} → {sim_t_after}). "
                          f"Despause pode ter falhado.")
            except Exception:
                pass

            # 10) Abre paraquedas
            self.tn.set("/fdm/jsbsim/systems/chute/chute-cmd-norm", 1)

            # 11) Espera 4s sim_t pro chute abrir
            time.sleep(4.0 / SPEED_UP)

            # 12) Drena buffer UDP e espera dados FRESCOS (alt > 100m).
            # Sem isso, o read inicial pega pacote velho (pré-teleport).
            deadline = time.time() + 2.0  # max 2s wall esperando
            fresh = None
            while time.time() < deadline:
                self.fdm.read_latest()  # drena
                time.sleep(0.05)
                f = self.fdm.read_latest()
                if f is not None and math.isfinite(f["alt_m"]) and f["alt_m"] > 100.0:
                    fresh = f
                    break
            if fresh is None:
                print(f"[FGEnv] ep={self.episode}: dados UDP frescos não chegaram. "
                      f"Pode estar com NaN.")

        except Exception as e:
            print(f"[FGEnv] Reset falhou ({e}). Cold boot.")
            self._cold_boot()
            return self.reset(seed=seed, options=options)

        # >>> DIAGNÓSTICO PÓS-RESET <<<
        # Confirma que sim está despausado e avião está caindo
        try:
            t0 = time.monotonic()
            sim_t = self.tn.get("/sim/time/elapsed-sec")
            t_get = time.monotonic() - t0
            vd_raw = self.tn.get("/velocities/vertical-speed-fps")
            fz = self.tn.get("/sim/freeze/master")
            print(f"[RESET-DBG] ep={self.episode} 1 telnet get = {t_get*1000:.0f}ms | "
                  f"sim_t={sim_t} freeze={fz} v_down={vd_raw}fps")
        except Exception as e:
            print(f"[RESET-DBG] falhou: {e}")

        # Mede _read_state
        t0 = time.monotonic()
        obs = self._read_state()
        t_read = time.monotonic() - t0
        print(f"[RESET-DBG] _read_state = {t_read*1000:.0f}ms (7 telnet gets)")

        if self._check_explosion():
            print(f"[FGEnv] Explosão pós-reset (alt={self.alt_m:.1f}m). Cold boot.")
            self._cold_boot()
            return self.reset(seed=seed, options=options)

        print(
            f"[FGEnv] ep={self.episode} spawn=({lat:.4f},{lon:.4f}) "
            f"head={heading:.1f}° wind={wind_speed_fps:.0f}fps "
            f"dist0={self.dist:.0f}m alt0={self.alt_agl_m:.0f}m "
            f"vd0={self._last_vdown_fps:.1f}fps"
        )

        self.last_dist = self.dist
        self.last_step_wall = time.monotonic()
        return obs, {}

    def step(self, action):
        t_step_start = time.monotonic()
        ail = float(np.clip(action[0], -1.0, 1.0))
        try:
            t_a = time.monotonic()
            self.tn.set("/fdm/jsbsim/fcs/aileron-cmd-norm", ail)
            t_set = (time.monotonic() - t_a) * 1000
        except Exception as e:
            print(f"[FGEnv] step set falhou: {e}. Cold boot.")
            self._cold_boot()
            return np.zeros(8, dtype=np.float32), -10.0, True, False, {"error": "telnet"}

        # Cadência alvo
        target_wall_dt = STEP_DT_SIM_S / SPEED_UP
        if self.last_step_wall is not None:
            elapsed = time.monotonic() - self.last_step_wall
            sleep_left = target_wall_dt - elapsed
            if sleep_left > 0:
                time.sleep(sleep_left)
        t_sleep = (time.monotonic() - t_step_start) * 1000
        self.last_step_wall = time.monotonic()

        t_r = time.monotonic()
        obs = self._read_state()
        t_read = (time.monotonic() - t_r) * 1000
        self.steps += 1

        # Print diagnóstico nos primeiros 5 steps de cada episódio
        if self.steps <= 5:
            print(f"[STEP-DBG] ep={self.episode} step={self.steps} "
                  f"set={t_set:.0f}ms total_wait={t_sleep:.0f}ms "
                  f"read={t_read:.0f}ms total={(time.monotonic()-t_step_start)*1000:.0f}ms "
                  f"alt={self.alt_agl_m:.0f}m vd={self._last_vdown_fps:.1f}fps")

        if self._check_explosion():
            print(f"[FGEnv] Explosão durante step (alt={self.alt_m:.1f}m).")
            self._cold_boot()
            return obs, -50.0, True, False, {"explosion": True}

        # Reward (igual cone v2)
        cone_err_m = self.dist - self.alt_agl_m * self.glide_ratio
        reward = -abs(cone_err_m) / 1000.0
        if self.last_dist is not None:
            reward += (self.last_dist - self.dist) / 200.0
        self.last_dist = self.dist

        terminated = False
        truncated = False
        # Terminate ANTES do solo pra evitar JSBSim NaN
        if self.alt_agl_m <= MIN_AGL_TERMINATE:
            if   self.dist < 100:  reward += 100.0
            elif self.dist < 300:  reward += 50.0
            elif self.dist < 500:  reward += 20.0
            elif self.dist < 1000: reward += 5.0
            else:                  reward -= self.dist / 100.0
            terminated = True

        if self.steps >= MAX_EPISODE_STEPS:
            truncated = True

        return obs, float(reward), terminated, truncated, {}

    def close(self):
        if self.tn is not None:
            try:
                self.tn.close()
            except Exception:
                pass
            self.tn = None
        if self.fdm is not None:
            try:
                self.fdm.close()
            except Exception:
                pass
            self.fdm = None
        hard_kill_fg()


# =====================================================================
# DRIVER DE TREINO
# =====================================================================

def sanity_check():
    """
    Sobe o FG, despausa, abre o chute, MONITORA telemetria por 30s.
    Não teleporta. Não roda PPO. Pra entender o que o FG está fazendo.

    O QUE OBSERVAR:
      - /sim/time/elapsed-sec: TEM que crescer. Se ficar parado, o sim
        está congelado (apesar de freeze=false).
      - altitude-ft: TEM que diminuir após o chute abrir.
      - vertical-speed-fps: deve ficar em ~-5 a -20 fps (queda do chute).
    """
    print("=" * 70)
    print("SANITY CHECK — diagnóstico do FG sem PPO, sem teleport")
    print("=" * 70)
    raw_env = FGConeEnv()

    print("\n[sanity] FG está PAUSADO. Vamos:")
    print("        1) abrir chute (com sim pausado, comando fica armado)")
    print("        2) re-teleportar pra 9850ft (corrige altitude se boot falhou)")
    print("        3) zerar velocidades")
    print("        4) DESPAUSAR")
    print("        5) ler imediatamente — avião deve cair com chute aberto")
    try:
        # 1) chute armado
        raw_env.tn.set("/fdm/jsbsim/systems/chute/chute-cmd-norm", 1)
        # 2) altitude (pode estar baixa por causa do boot)
        raw_env.tn.set("/position/altitude-ft", START_ALT_FT)
        # 3) zera velocidades (parte do estado limpo)
        raw_env.tn.set("/velocities/uBody-fps", 0)
        raw_env.tn.set("/velocities/vBody-fps", 0)
        raw_env.tn.set("/velocities/wBody-fps", 0)
        raw_env.tn.set("/velocities/vertical-speed-fps", 0)
        # vento
        raw_env.tn.set("/environment/wind-from-heading-deg", WIND_DIR_DEG)
        raw_env.tn.set("/environment/wind-speed-kt", 8.0 * 0.5925)
        # speed-up baixo no sanity pra dar tempo de observar
        raw_env.tn.set("/sim/speed-up", 1)
        # 4) DESPAUSA
        raw_env.tn.set("/sim/freeze/master", "false")
        raw_env.tn.set("/sim/freeze/clock", "false")
        raw_env.tn.set("/sim/freeze/fuel", "false")
        raw_env.tn.set("/sim/pause", "false")
        print("[sanity] Comandos enviados. Sim despausado.")
    except Exception as e:
        print(f"[sanity] erro nos sets iniciais: {e}")
        raw_env.close()
        return

    print("\n[sanity] Monitorando 30 leituras (1/s wall-clock, speed-up=1x)...")
    print("  t  | sim_t  | alt(ft) | agl(ft) | v_down(fps) | chute | su | fz")
    print("-----|--------|---------|---------|-------------|-------|----|----")
    t0 = time.time()
    last_sim_t = None
    for i in range(30):
        try:
            sim_t = raw_env.tn.get("/sim/time/elapsed-sec")
            alt = raw_env.tn.get("/position/altitude-ft")
            agl = raw_env.tn.get("/position/altitude-agl-ft")
            vd = raw_env.tn.get("/velocities/vertical-speed-fps")
            chute = raw_env.tn.get("/fdm/jsbsim/systems/chute/chute-cmd-norm")
            su = raw_env.tn.get("/sim/speed-up")
            fz = raw_env.tn.get("/sim/freeze/master")
            elapsed = time.time() - t0

            def f(x, w=8, dec=1):
                try: return f"{float(x):{w}.{dec}f}"
                except: return str(x)

            print(f"{elapsed:4.1f} | {f(sim_t,6)} | {f(alt)} | {f(agl)} | "
                  f"{f(vd,8,2)} | {chute} | {su} | {fz}")
        except Exception as e:
            print(f"[sanity] erro leitura: {e}")
        time.sleep(1.0)

    print("\n[sanity] DIAGNÓSTICO:")
    print("  - sim_t cresceu? sim → FG rodando o FDM.")
    print("  - sim_t parado? não → FG está congelado por outra razão.")
    print("  - alt diminuiu? sim → chute funcionou, avião cai.")
    print("  - chute = 1? sim → comando aceito.")
    raw_env.close()


def train():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(BASE_MODELS_PATH, f"training_{timestamp}")
    checkpoint_dir = os.path.join(session_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    raw_env = FGConeEnv()
    env = DummyVecEnv([lambda: raw_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    checkpoint_cb = CheckpointCallback(
        save_freq=5_000,
        save_path=checkpoint_dir,
        name_prefix="parachute_fg_cone_model",
    )

    model = sb3.PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,         # rollout menor pra dar updates mais cedo
        batch_size=64,
        n_epochs=10,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=os.path.join(session_dir, "tensorboard"),
    )

    print("=" * 70)
    print("TREINO CONE — NATIVO NO FLIGHTGEAR  [TESTE DE SANIDADE]")
    print("=" * 70)
    print(f"  target           = ({TARGET_LAT}, {TARGET_LON})")
    print(f"  GR               = {GLIDE_RATIO}")
    print(f"  spawn radius     = {SPAWN_RADIUS_M:.0f}m  alt0 = {START_ALT_FT}ft")
    print(f"  ventos           = {WIND_SPEEDS_FPS} fps (dir {WIND_DIR_DEG}°)")
    print(f"  speed-up FG      = {SPEED_UP}x  (via telnet, NÃO boot prop)")
    print(f"  step dt          = {STEP_DT_SIM_S}s sim  (1 Hz controle, ESP32-compat)")
    print(f"  total timesteps  = {TOTAL_TIMESTEPS}  (teste curto)")
    print(f"  obs/reward       = cone v2 (idênticos ao JSBSim env)")
    print(f"  saída            = {session_dir}")
    print("=" * 70)

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_cb,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[main] Interrompido. Salvando progresso...")

    model.save(os.path.join(session_dir, "parachute_fg_cone_final"))
    env.save(os.path.join(session_dir, "vec_normalize_fg.pkl"))
    print(f"✅ Treino concluído. Arquivos em: {session_dir}")
    raw_env.close()


def main():
    if SANITY_CHECK_ONLY:
        sanity_check()
    else:
        train()


if __name__ == "__main__":
    main()