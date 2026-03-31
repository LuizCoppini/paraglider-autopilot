import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jsbsim
import math
import csv
import os
import random
from datetime import datetime

# Caminhos
JSBSIM_ROOT = r"D:\workspace\Pycharm\paraglider-autopilot\jsbsim"
AIRCRAFT_PATH = r"C:\Users\coppi\OneDrive\Documents\FlightGear\Aircraft"


class ParachuteEnv(gym.Env):
    def __init__(self, target_lat, target_lon):
        super().__init__()
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.episode = 0

        # Configurações de pastas
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(r"D:\workspace\Pycharm\paraglider-autopilot\src\flight_records",
                                    f"training_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_file = os.path.join(self.run_dir, "flight_log.csv")

        # Espaços Gym
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)

        self.fdm = None
        self.last_action = np.zeros(2)  # Para suavização
        self._init_log_files()

    def _init_log_files(self):
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["episode", "pos_id", "wind_speed_fps", "wind_dir_deg", "flight_time_s", "final_dist", "reward", "lat",
                 "lon"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.last_action = np.zeros(2)
        self._create_sim()

        # --- LÓGICA DE CENÁRIO REPRODUTÍVEL (4000 saltos) ---
        # 8 posições, 500 saltos cada. 125 saltos por força de vento.
        pos_id = ((self.episode - 1) // 500) % 8
        wind_type_id = ((self.episode - 1) // 125) % 4

        # 1. Posição (8 pontos a 4km de distância)
        angle_pos = math.radians(pos_id * 45)  # 0, 45, 90, 135...
        radius = 4000.0
        start_lat = self.target_lat + (radius * math.cos(angle_pos)) / 111320.0
        start_lon = self.target_lon + (radius * math.sin(angle_pos)) / (
                    111320.0 * math.cos(math.radians(self.target_lat)))

        # 2. Ventos (Baixo, Moderado, Alto, Muito Alto) em FPS
        wind_speeds = [4.0, 12.0, 25.0, 40.0]
        self.current_wind_speed = wind_speeds[wind_type_id]
        # Direção do vento varia conforme o episódio para cobrir 360º
        self.current_wind_dir = (self.episode * 73) % 360

        rad_wind = math.radians(self.current_wind_dir)
        self.wind_n = -math.cos(rad_wind) * self.current_wind_speed
        self.wind_e = -math.sin(rad_wind) * self.current_wind_speed

        # Setup JSBSim
        self.fdm["ic/lat-gc-deg"] = start_lat
        self.fdm["ic/long-gc-deg"] = start_lon
        self.fdm["ic/h-sl-ft"] = 9850
        self.fdm["ic/psi-true-deg"] = (self._get_bearing(start_lat, start_lon, self.target_lat, self.target_lon))

        self.fdm["atmosphere/wind-north-fps"] = self.wind_n
        self.fdm["atmosphere/wind-east-fps"] = self.wind_e
        self.fdm["ic/u-fps"] = 35.0
        self.fdm.run_ic()
        self.fdm["systems/chute/chute-cmd-norm"] = 1.0

        for _ in range(480): self.fdm.run()

        self.last_dist = haversine(self.fdm["position/lat-gc-deg"], self.fdm["position/long-gc-deg"], self.target_lat,
                                   self.target_lon)
        self.flight_time = 0
        self.total_reward = 0.0
        self.current_flight_telemetry = []
        self.current_pos_id = pos_id

        return self._get_obs(), {}

    def step(self, action):
        # --- 6) SUAVIZAÇÃO DE COMANDO (Rate Limiter) ---
        # Impede que o aileron mude mais que 0.2 por segundo (evita NaN)
        max_rate = 0.2
        action[0] = np.clip(action[0], self.last_action[0] - max_rate, self.last_action[0] + max_rate)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action

        self.fdm["fcs/aileron-cmd-norm"] = float(action[0])
        self.fdm["fcs/elevator-cmd-norm"] = float(action[1])

        success = True
        for _ in range(120):
            if not self.fdm.run(): success = False; break

        alt = self.fdm["position/h-sl-ft"]
        if not success or np.isnan(alt):
            return np.zeros(6, dtype=np.float32), -100.0, True, False, {}

        self.flight_time += 1
        obs = self._get_obs()
        dist = obs[0] * 5500.0
        lat, lon = self.fdm["position/lat-gc-deg"], self.fdm["position/long-gc-deg"]
        vs = self.fdm["velocities/h-dot-fps"]

        self.current_flight_telemetry.append([
            self.flight_time, lat, lon, alt, self.fdm["attitude/psi-deg"], vs, dist,
            action[0], action[1], self.current_wind_speed, self.current_wind_dir
        ])

        # --- 5) NOVA LÓGICA DE RECOMPENSA (FOCO EM POUSO SUAVE) ---
        reward = (self.last_dist - dist) * 10.0  # Aproximação básica
        # Sem penalidade de tempo aqui conforme solicitado

        done = bool(alt <= 10 or dist > 10000 or self.flight_time > 2000)

        if done:
            if alt <= 15 and dist < 100:
                # Bônus massivo para pouso suave (VS perto de 0)
                # Se VS for -20fps (duro), penaliza. Se for -5fps (suave), recompensa.
                landing_reward = 1000.0 / (abs(vs) + 1.0)
                reward += landing_reward
            self._save_logs(dist, lat, lon)

        self.total_reward += float(reward)
        self.last_dist = dist
        return obs, reward, done, False, {}

    def _save_logs(self, dist, lat, lon):
        # Log Geral
        with open(self.log_file, "a", newline="") as f:
            csv.writer(f).writerow([self.episode, self.current_pos_id, self.current_wind_speed,
                                    self.current_wind_dir, self.flight_time, dist, self.total_reward, lat, lon])

        # --- 3) SALVAR TRAJETÓRIA A CADA 50 EPISÓDIOS POR PASTA ---
        if self.episode % 50 == 0:
            pos_dir = os.path.join(self.run_dir, f"posicao_{self.current_pos_id}")
            os.makedirs(pos_dir, exist_ok=True)

            log_path = os.path.join(pos_dir, f"flight_ep_{self.episode}.csv")
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["time_s", "lat", "lon", "alt_ft", "heading", "vs_fps", "dist_m", "aileron", "elevator", "wind_spd",
                     "wind_dir"])
                writer.writerows(self.current_flight_telemetry)

    def _create_sim(self):
        self.fdm = jsbsim.FGFDMExec(JSBSIM_ROOT)
        self.fdm.set_aircraft_path(AIRCRAFT_PATH)
        self.fdm.load_model("Parachutist")
        self.fdm.set_dt(1 / 120)

    def _get_obs(self):
        try:
            lat, lon = self.fdm["position/lat-gc-deg"], self.fdm["position/long-gc-deg"]
            dist = haversine(lat, lon, self.target_lat, self.target_lon)
            bearing_err = (self._get_bearing(lat, lon, self.target_lat, self.target_lon) - self.fdm[
                "attitude/psi-deg"] + 180) % 360 - 180
            return np.array([np.clip(dist / 5500, 0, 1), bearing_err / 180, self.fdm["velocities/vg-fps"] / 60,
                             self.fdm["velocities/h-dot-fps"] / 30, self.fdm["attitude/roll-rad"],
                             self.fdm["attitude/pitch-rad"]], dtype=np.float32)
        except:
            return np.zeros(6, dtype=np.float32)

    def _get_bearing(self, lat1, lon1, lat2, lon2):
        off_x, off_y = lon2 - lon1, lat2 - lat1
        return (90.0 - math.degrees(math.atan2(off_y, off_x))) % 360


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))