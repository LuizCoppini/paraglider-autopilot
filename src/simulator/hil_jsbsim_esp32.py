"""
hil_jsbsim_esp32.py — HIL com JSBSim isolado (SEM FlightGear).

OBJETIVO CIENTÍFICO:
  Comparar o mesmo ESP32 (mesmo firmware, mesmo serial, mesmo modelo cone v3)
  em duas configurações de engine física:

    hil_fg_esp32.py    — ESP32 ↔ FlightGear     (já testado: dist final ~4500m ✗)
    hil_jsbsim_esp32.py — ESP32 ↔ JSBSim isolado (este script)

  Se ESTE script convergir e o do FG não, fica provado empiricamente que o
  gap está na engine FG, não no ESP32, no protocolo serial, ou no real-time.
  Esse resultado vira evidência forte pra Seção 7.2 do artigo ENIAC e pra
  dissertação.

ARQUITETURA:
  Python loop em tempo real (1 Hz). A cada step:
    1. env._get_obs() → 6 floats RAW
    2. envia pro ESP32 via serial
    3. recebe ação (aileron, elevator)
    4. env.step(action) → JSBSim avança 120 ticks (1s sim)
    5. log CSV
    6. time.sleep(1.0 - wall_elapsed) → mantém 1 Hz wall-clock

  Reusa parachute_cone_env_v3.ParachuteConeEnvV3 SEM MODIFICAÇÃO (mesmo env
  do treino). O ESP32 (com cone v3 embarcado) processa as obs reais do env.

EXECUÇÃO:
  poetry run python src/simulator/hil_jsbsim_esp32.py --port COM3
  poetry run python src/simulator/hil_jsbsim_esp32.py --port COM3 --no-realtime
  poetry run python src/simulator/hil_jsbsim_esp32.py --port COM3 --wind 14

FIXES nesta versão (23/06/2026):
  [FIX-1] CSV_BASE_DIR agora resolve para src/flight_records do projeto.
          Antes ia para D:\flight_records (raiz do drive) por causa de "/".
  [FIX-2] Termination explícita quando alt_agl_m <= 1.0. JSBSim clipa
          h-agl-ft em 0 mas o env continua rodando, gerando "voo no chão"
          (~140 steps falsos que mascaravam a distância real do pouso).
"""

import argparse
import os
import sys
import time
import csv
import math
import random
from datetime import datetime

import numpy as np

# Adiciona src/ ao path pra importar rl.parachute_cone_env_v3
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from rl.parachute_cone_env_v3 import ParachuteConeEnvV3, haversine
except ModuleNotFoundError:
    from parachute_cone_env_v3 import ParachuteConeEnvV3, haversine

# Reusa ESP32Client do hil_fg_esp32. Se este script estiver em outro lugar,
# faz fallback inline.
try:
    from hil_fg_esp32 import ESP32Client
except ImportError:
    import serial
    import struct

    HEADER_SEND = bytes([0xAA, 0xBB])
    HEADER_RECV = bytes([0xCC, 0xDD])
    OBS_DIM = 6
    ACTION_DIM = 2
    PKT_RECV_SZ = 2 + ACTION_DIM * 4

    class ESP32Client:
        def __init__(self, port, baud=115200, timeout=1.0):
            self.port = port
            self.baud = baud
            self.timeout = timeout
            self.ser = None

        def connect(self, wait_ready=True, ready_timeout=5.0):
            print(f"[ESP32] Abrindo {self.port}...")
            self.ser = serial.Serial(self.port, self.baud, timeout=0.2)
            time.sleep(2.0)
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
                            time.sleep(0.1)
                            self.ser.reset_input_buffer()
                            print("[ESP32] Handshake OK.")
                            return True
                print("[ESP32] AVISO: handshake não recebido.")
                self.ser.reset_input_buffer()
            return False

        def step(self, obs):
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
                        ail, ele = struct.unpack("<2f", buf[idx+2:idx+PKT_RECV_SZ])
                        return float(ail), float(ele)
            raise TimeoutError("ESP32 não respondeu em 1s")

        def close(self):
            if self.ser:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None


# =====================================================================
# CONFIG
# =====================================================================

TARGET_LAT = 34.9055
TARGET_LON = -117.8830
MOJAVE_GROUND_ALT_M = 700.0
STEP_DT_S = 1.0  # 1 Hz wall-clock = 1 Hz sim (real-time)

# [FIX-1] Agora salva em src/flight_records do projeto, em vez de D:\flight_records.
CSV_BASE_DIR = os.path.join(_SRC_DIR, "flight_records")

# [FIX-2] Altura AGL (m) abaixo da qual encerra o voo. JSBSim clipa h-agl-ft
# em 0 mas o env mantém step()=False por algumas dezenas de ticks, gerando
# trajetória falsa "no chão". 1 m AGL é uma margem segura pro ruído de altímetro.
GROUND_CONTACT_AGL_M = 1.0


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="HIL JSBSim isolado + ESP32")
    parser.add_argument("--port", default="COM3", help="Porta serial do ESP32")
    parser.add_argument("--csv", default=None, help="Caminho do CSV (default: auto)")
    parser.add_argument("--no-handshake", action="store_true",
                        help="Pula handshake HIL_READY")
    parser.add_argument("--no-realtime", action="store_true",
                        help="Roda full-speed (sem time.sleep). Útil pra teste rápido.")
    parser.add_argument("--wind", type=float, default=8.0,
                        help="Vento em fps (default: 8). Valores do treino: 4/8/14/20")
    parser.add_argument("--pos-id", type=int, default=None,
                        help="Spawn position id [0-7]. Default: aleatório.")
    parser.add_argument("--max-steps", type=int, default=600,
                        help="Máximo de steps por voo (default: 600 = 10 min sim)")
    args = parser.parse_args()

    # CSV path
    if args.csv:
        csv_path = args.csv
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "rt" if not args.no_realtime else "fast"
        csv_path = os.path.join(
            CSV_BASE_DIR,
            f"hil_jsbsim_session_{ts}_{suffix}_wind{int(args.wind)}",
            "flight.csv",
        )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # ─── ESP32 ──────────────────────────────────────────────────────────
    esp32 = ESP32Client(port=args.port)
    try:
        esp32.connect(wait_ready=not args.no_handshake)
    except Exception as e:
        print(f"[ESP32] Falha ao conectar: {e}")
        sys.exit(1)

    print("[ESP32] Sanity check com obs zerada...")
    try:
        ail, ele = esp32.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        print(f"[ESP32] OK. Resposta: aileron={ail:+.4f} elevator={ele:.4f}")
    except Exception as e:
        print(f"[ESP32] Sanity check FALHOU: {e}")
        esp32.close()
        sys.exit(1)
    print()

    # ─── Cria env idêntico ao do treino ─────────────────────────────────
    print(f"[JSBSim] Criando ambiente cone v3 (wind={args.wind} fps)...")
    env = ParachuteConeEnvV3(
        TARGET_LAT, TARGET_LON,
        wind_speeds_fps=(args.wind,),  # vento fixo nesse valor
        # Demais parâmetros = defaults do treino:
        # glide_ratio_target=1.88, spawn_radius_m=4000, start_alt_ft=9850,
        # wind_dir_deg=0.0, random_initial_heading=True,
        # action_lag_ticks_range=(6, 24), gps_noise_m=3.0,
        # heading_noise_deg=0.5, alt_noise_m=2.0,
        # wind_gust_mag_pct=0.20, wind_gust_dir_deg=15.0, wind_gust_period_s=5.0
    )

    # Controla spawn position (se não especificado, aleatoriza)
    if args.pos_id is not None:
        # env usa (episode-1) // (125*n_winds) % 8 — pra forçar pos_id=N,
        # setamos episode=N+1.
        env.episode = args.pos_id
    else:
        env.episode = random.randint(0, 7)

    print(f"[JSBSim] Reset (4s settling do chute)...")
    obs, _ = env.reset()
    print("[JSBSim] Pronto. Estado inicial calculado.")

    # Lê estado inicial pra header
    fdm = env.fdm
    dist0 = haversine(
        fdm["position/lat-gc-deg"], fdm["position/long-gc-deg"],
        TARGET_LAT, TARGET_LON,
    )
    alt0_m = fdm["position/h-sl-ft"] * 0.3048

    print()
    print("=" * 70)
    print("HIL JSBSim ISOLADO + ESP32")
    print("=" * 70)
    print(f"  target    = ({TARGET_LAT}, {TARGET_LON}) (Mojave)")
    print(f"  wind      = {args.wind:.1f} fps norte→sul (direto no JSBSim)")
    print(f"  pos_id    = {env.current_pos_id}")
    print(f"  GR target = {env.glide_ratio_target}")
    print(f"  dist0     = {dist0:.0f} m | alt0 = {alt0_m:.0f} m MSL")
    print(f"  real-time = {not args.no_realtime} ({'1 Hz wall' if not args.no_realtime else 'full-speed'})")
    print(f"  max_steps = {args.max_steps}")
    print(f"  ground_thr= {GROUND_CONTACT_AGL_M} m AGL  [FIX-2]")
    print(f"  csv       = {csv_path}")
    print("=" * 70)

    # ─── CSV ────────────────────────────────────────────────────────────
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

    # Helper pra log + print (evita duplicação no termination block)
    def _log_step(step_num, t_wall, lat, lon, alt_m, alt_agl_m, heading,
                  vg_fps, v_down, roll, pitch, dist_m, obs,
                  aileron, elevator, esp_ms, tag=""):
        writer.writerow([
            round(t_wall, 3), step_num, step_num,
            round(lat, 7), round(lon, 7),
            round(alt_m, 2), round(alt_agl_m, 2), round(heading, 2),
            round(vg_fps, 2), round(v_down, 2),
            round(roll, 4), round(pitch, 4),
            round(dist_m, 2),
            round(float(obs[0]), 5), round(float(obs[1]), 5),
            round(float(obs[2]), 5), round(float(obs[3]), 5),
            round(float(obs[4]), 5), round(float(obs[5]), 5),
            round(aileron, 5), round(elevator, 5),
            round(esp_ms, 2),
        ])
        csv_file.flush()
        print(f"{step_num:>4} {t_wall:>6.1f} {alt_agl_m:>8.0f} {dist_m:>6.0f} "
              f"{float(obs[0]):>+7.3f} {float(obs[1]):>+7.3f} "
              f"{aileron:>+7.3f} {elevator:>6.3f} {esp_ms:>6.1f}{tag}")

    # ─── Loop principal ─────────────────────────────────────────────────
    print()
    print(f"[LOOP] Iniciando HIL @ {'1 Hz real-time' if not args.no_realtime else 'full-speed'}...")
    print(f"{'step':>4} {'t':>6} {'alt_agl':>8} {'dist':>6} "
          f"{'cone_e':>7} {'bear_e':>7} {'ail':>7} {'ele':>6} {'esp_ms':>6}")
    print("-" * 70)

    t_start = time.time()
    step_num = 0
    end_reason = "max_steps"
    last_dist = float("nan")
    last_alt_agl = float("nan")
    last_v_down = float("nan")
    t_wall = 0.0

    try:
        while step_num < args.max_steps:
            step_num += 1
            t_wall_step_start = time.time()
            t_wall = t_wall_step_start - t_start

            # 1) ESP32 inference (obs já está no scope do step anterior)
            obs_list = obs.tolist() if isinstance(obs, np.ndarray) else list(obs)
            t_esp = time.time()
            try:
                aileron, elevator = esp32.step(obs_list)
            except TimeoutError as e:
                print(f"[step {step_num}] ESP32 timeout: {e}")
                end_reason = "esp32_timeout"
                break
            esp_ms = (time.time() - t_esp) * 1000.0

            # 2) Avança JSBSim 1 segundo (120 ticks)
            action = np.array([aileron, elevator], dtype=np.float32)
            obs, reward, done, truncated, info = env.step(action)

            # 3) Lê estado pós-step direto do FDM (mais preciso que obs normalizada)
            fdm = env.fdm
            lat = fdm["position/lat-gc-deg"]
            lon = fdm["position/long-gc-deg"]
            alt_ft = fdm["position/h-sl-ft"]
            alt_m = alt_ft * 0.3048
            alt_agl_m = max(0.0, alt_m - MOJAVE_GROUND_ALT_M)
            heading = fdm["attitude/psi-deg"]
            vg_fps = fdm["velocities/vg-fps"]
            h_dot = fdm["velocities/h-dot-fps"]
            v_down = -h_dot  # converte h-dot (positive up) → v_down (positive down)
            roll = fdm["attitude/roll-rad"]
            pitch = fdm["attitude/pitch-rad"]
            dist_m = haversine(lat, lon, TARGET_LAT, TARGET_LON)

            last_dist = dist_m
            last_alt_agl = alt_agl_m
            last_v_down = v_down

            # [FIX-2] Termination forçada quando toca o solo ───────────
            # JSBSim clipa h-agl-ft em 0 mas o env mantém step()=False por
            # várias dezenas de ticks (não detecta v_down=0 imediatamente).
            # Sem este check, a "distância final" reportada é falsa porque
            # o vento continua "deslizando" o chute pelo solo.
            if alt_agl_m <= GROUND_CONTACT_AGL_M:
                end_reason = "ground_contact"
                _log_step(step_num, t_wall, lat, lon, alt_m, alt_agl_m, heading,
                          vg_fps, v_down, roll, pitch, dist_m, obs,
                          aileron, elevator, esp_ms, tag="  ← LANDED")
                break
            # ──────────────────────────────────────────────────────────

            # 4) Log CSV
            _log_step(step_num, t_wall, lat, lon, alt_m, alt_agl_m, heading,
                      vg_fps, v_down, roll, pitch, dist_m, obs,
                      aileron, elevator, esp_ms)

            if done:
                end_reason = "landed"
                break
            if truncated:
                end_reason = "truncated"
                break

            # 5) Cadência real-time (1 Hz)
            if not args.no_realtime:
                elapsed = time.time() - t_wall_step_start
                sleep_left = STEP_DT_S - elapsed
                if sleep_left > 0:
                    time.sleep(sleep_left)

    except KeyboardInterrupt:
        end_reason = "user_interrupt"
        print("\n[LOOP] Interrompido pelo usuário.")
    finally:
        csv_file.close()
        esp32.close()
        try:
            env.fdm = None  # libera handle
        except Exception:
            pass

    # ─── Resumo ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"  FIM DO VOO — razão: {end_reason}")
    print(f"  Steps:           {step_num}")
    print(f"  Tempo voo:       {t_wall:.1f} s")
    print(f"  Distância final: {last_dist:.1f} m  (dist0 = {dist0:.0f} m)")
    print(f"  Δ distância:     {last_dist - dist0:+.1f} m")
    print(f"  Altitude final:  {last_alt_agl:.1f} m AGL")
    print(f"  Vel. vert. fim:  {last_v_down:.1f} fps")
    print(f"  CSV salvo em:    {csv_path}")
    print("=" * 70)
    print()
    print("Comparação esperada com hil_fg_esp32.py:")
    print(f"  - Se dist_final << dist0  → ESP32 + JSBSim convergem (gap está no FG)")
    print(f"  - Se dist_final ~ dist0   → gap está no ESP32 + tempo real, não no FG")
    print(f"  - Se v_down_fim < 10 fps  → flare funciona aqui (não funcionou no FG)")


if __name__ == "__main__":
    main()