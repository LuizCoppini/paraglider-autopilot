"""
parachute_cone_env_v2.py — Método CONE de descida (mantido), com:

  1. Vento funcional (bug do v1 corrigido — wind aplicado DEPOIS de
     run_ic e reaplicado a cada step).

  2. glide_ratio_target tratado como PROPRIEDADE DA AERONAVE, não
     constante mágica. Para o Parachutist, o GR real medido fica
     em torno de 1.88 (vg=30 fps / v_down=16 fps). O cone agora
     representa o ENVELOPE REAL do parafoil — não um cone artificial
     pequeno demais (como acontecia com o 0.8 do v1).

  3. Opcionalmente, o GR pode ser passado na observação como 7ª
     dimensão (`include_gr_in_obs=True`). Isso permite treinar UM
     modelo só que funciona em vários parafoils com GRs diferentes —
     basta passar o GR correto na obs no momento da inferência.

Método: o cone "ideal de descida" tem raio = altitude * gr. O agente
recebe `cone_err = (dist - raio_ideal) / 1000` e aprende a tracking.
A recompensa premia diminuir o erro do cone.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jsbsim
import math
import csv
import os
from datetime import datetime

JSBSIM_ROOT = r"D:\workspace\Pycharm\paraglider-autopilot\jsbsim"
AIRCRAFT_PATH = r"C:\Users\coppi\OneDrive\Documents\FlightGear\Aircraft"


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class ParachuteConeEnvV2(gym.Env):
    """
    Env do método cone, v2.

    Parâmetros:
      target_lat, target_lon : alvo geográfico
      glide_ratio_target     : INCLINAÇÃO DO CONE = GR da aeronave.
                               Para Parachutist usar ~1.88 (medido).
                               Para outras aeronaves, usar o GR delas.
      include_gr_in_obs      : se True, GR vira 7ª dim da obs e o
                               modelo treinado generaliza para outros
                               parafoils. Se False, modelo é dedicado
                               para um GR específico (mesmo do treino).
      spawn_radius_m, start_alt_ft : geometria do treino
      wind_speeds_fps        : tupla de níveis de vento que o env
                               sorteia entre os episódios

    Observação:
      Sem GR na obs (include_gr_in_obs=False, default) — 6 dims:
        [0] cone_err = (dist - alt_m * gr) / 1000, clipado em [-1, 1]
        [1] bearing_err / 180
        [2] vg / 60
        [3] h_dot / 30
        [4] roll_rad
        [5] pitch_rad

      Com GR na obs (include_gr_in_obs=True) — 7 dims:
        [0..5] iguais
        [6] gr / 3.0 (normalizado, GR típico de parafoil 0..3)

    Ação:
      [0] aileron-cmd-norm in [-1, 1]
      [1] elevator-cmd-norm in [0, 1]
    """

    def __init__(self, target_lat, target_lon,
                 glide_ratio_target=1.88,
                 include_gr_in_obs=False,
                 spawn_radius_m=4000.0,
                 start_alt_ft=9850,
                 wind_speeds_fps=(4.0, 12.0, 25.0, 40.0),
                 wind_dir_deg=0.0,
                 random_initial_heading=True):
        super().__init__()
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.glide_ratio_target = float(glide_ratio_target)
        self.include_gr_in_obs = bool(include_gr_in_obs)
        self.spawn_radius_m = spawn_radius_m
        self.start_alt_ft = start_alt_ft
        self.wind_speeds = list(wind_speeds_fps)
        # Vento sempre vem dessa direção (em graus, convenção meteorológica:
        # 0° = vento do norte indo pro sul, 90° = vento do leste indo pro
        # oeste, etc.). Por padrão norte-sul = 0°.
        self.wind_dir_deg = float(wind_dir_deg)
        # Se True, a proa inicial é aleatória [0, 360). Se False, aponta
        # para o alvo (comportamento dos envs antigos).
        self.random_initial_heading = bool(random_initial_heading)
        self.episode = 0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "with_gr_obs" if self.include_gr_in_obs else "fixed_gr"
        self.run_dir = os.path.join(
            r"D:\workspace\Pycharm\paraglider-autopilot\src\flight_records",
            f"training_cone_v2_{suffix}_gr{self.glide_ratio_target:.2f}_{timestamp}",
        )
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_file = os.path.join(self.run_dir, "flight_log.csv")

        self.action_space = spaces.Box(
            low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32
        )
        obs_dim = 7 if self.include_gr_in_obs else 6
        self.observation_space = spaces.Box(
            low=-100, high=100, shape=(obs_dim,), dtype=np.float32
        )

        self.fdm = None
        self.last_action = np.zeros(2)
        self.last_cone_error = 0.0
        self.wind_n = 0.0
        self.wind_e = 0.0
        self._init_log_files()

    def _init_log_files(self):
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "pos_id", "wind_speed_fps", "wind_dir_deg",
                "flight_time_s", "final_dist", "reward", "lat", "lon", "gr",
            ])

    def _get_cone_radius(self, altitude_m):
        return altitude_m * self.glide_ratio_target

    def _apply_wind(self):
        """Reaplica wind no JSBSim. Necessário porque o run_ic zerava."""
        self.fdm["atmosphere/wind-north-fps"] = self.wind_n
        self.fdm["atmosphere/wind-east-fps"] = self.wind_e

    def _create_sim(self):
        self.fdm = jsbsim.FGFDMExec(JSBSIM_ROOT)
        self.fdm.set_aircraft_path(AIRCRAFT_PATH)
        self.fdm.load_model("Parachutist")
        self.fdm.set_dt(1 / 120)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.last_action = np.zeros(2)
        self._create_sim()

        n_winds = len(self.wind_speeds)
        per_pos = 125 * n_winds
        pos_id = ((self.episode - 1) // per_pos) % 8
        wind_type_id = ((self.episode - 1) // 125) % n_winds

        angle_pos = math.radians(pos_id * 45)
        start_lat = self.target_lat + (
            self.spawn_radius_m * math.cos(angle_pos)
        ) / 111320.0
        start_lon = self.target_lon + (
            self.spawn_radius_m * math.sin(angle_pos)
        ) / (111320.0 * math.cos(math.radians(self.target_lat)))

        self.current_wind_speed = self.wind_speeds[wind_type_id]
        # Vento sempre vindo da mesma direção (default 0° = norte → sul).
        # A variação relativa entre aircraft e vento vem das 8 posições
        # de spawn diferentes — do ponto de vista do parafoil, o vento
        # "vem" de ângulos relativos diferentes a cada episódio mesmo
        # com o vetor de vento fixo no mundo.
        self.current_wind_dir = self.wind_dir_deg
        rad_wind = math.radians(self.current_wind_dir)
        self.wind_n = -math.cos(rad_wind) * self.current_wind_speed
        self.wind_e = -math.sin(rad_wind) * self.current_wind_speed

        # Proa inicial: aleatória (default) ou apontando pro alvo
        if self.random_initial_heading:
            initial_psi = float(self.np_random.uniform(0.0, 360.0))
        else:
            initial_psi = self._get_bearing(
                start_lat, start_lon, self.target_lat, self.target_lon
            )

        # IC primeiro (run_ic reseta atmosphere)
        self.fdm["ic/lat-gc-deg"] = start_lat
        self.fdm["ic/long-gc-deg"] = start_lon
        self.fdm["ic/h-sl-ft"] = self.start_alt_ft
        self.fdm["ic/psi-true-deg"] = initial_psi
        self.fdm["ic/u-fps"] = 35.0
        self.fdm.run_ic()

        # Vento DEPOIS de run_ic — esse era o bug do v1
        self._apply_wind()

        self.fdm["systems/chute/chute-cmd-norm"] = 1.0

        # Settling de 4 s já com vento ativo
        for _ in range(480):
            self.fdm.run()

        alt_m = self.fdm["position/h-sl-ft"] * 0.3048
        dist_ini = haversine(
            self.fdm["position/lat-gc-deg"],
            self.fdm["position/long-gc-deg"],
            self.target_lat, self.target_lon,
        )

        self.last_cone_error = abs(dist_ini - self._get_cone_radius(alt_m))
        self.flight_time = 0
        self.total_reward = 0.0
        self.current_flight_telemetry = []
        self.current_pos_id = pos_id

        return self._get_obs(), {}

    def step(self, action):
        max_rate = 0.2
        action[0] = np.clip(
            action[0],
            self.last_action[0] - max_rate,
            self.last_action[0] + max_rate,
        )
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action

        # Reaplica vento (paranoia barata)
        self._apply_wind()

        self.fdm["fcs/aileron-cmd-norm"] = float(action[0])
        self.fdm["fcs/elevator-cmd-norm"] = float(action[1])

        success = True
        for _ in range(120):
            if not self.fdm.run():
                success = False
                break

        alt_ft = self.fdm["position/h-sl-ft"]
        alt_m = alt_ft * 0.3048
        if not success or np.isnan(alt_ft):
            obs_dim = self.observation_space.shape[0]
            return np.zeros(obs_dim, dtype=np.float32), -100.0, True, False, {}

        self.flight_time += 1
        lat = self.fdm["position/lat-gc-deg"]
        lon = self.fdm["position/long-gc-deg"]
        dist_m = haversine(lat, lon, self.target_lat, self.target_lon)

        # Erro do cone — núcleo do método
        raio_ideal = self._get_cone_radius(alt_m)
        current_cone_error = abs(dist_m - raio_ideal)

        v_ground = self.fdm["velocities/vg-fps"]
        v_down = abs(self.fdm["velocities/h-dot-fps"])
        instant_gr = v_ground / v_down if v_down > 0.1 else 0.0

        # Telemetria pra log
        self.current_flight_telemetry.append([
            self.flight_time, lat, lon, alt_ft,
            self.fdm["attitude/psi-deg"], -v_down, dist_m,
            action[0], action[1], self.current_wind_speed, self.current_wind_dir,
            round(raio_ideal, 2), round(instant_gr, 3),
        ])

        # Recompensa idêntica ao cone v1: tracking do cone
        reward = (self.last_cone_error - current_cone_error) * 20.0
        reward -= (current_cone_error / 100.0)

        done = bool(alt_ft <= 10 or dist_m > 10000 or self.flight_time > 2000)

        if done:
            if alt_ft <= 15 and dist_m < 100:
                reward += 1000.0 / (v_down + 1.0)
            self._save_logs(dist_m, lat, lon)

        self.total_reward += float(reward)
        self.last_cone_error = current_cone_error

        return self._get_obs(), reward, done, False, {}

    def _save_logs(self, dist, lat, lon):
        with open(self.log_file, "a", newline="") as f:
            csv.writer(f).writerow([
                self.episode, self.current_pos_id, self.current_wind_speed,
                self.current_wind_dir, self.flight_time, dist,
                self.total_reward, lat, lon, self.glide_ratio_target,
            ])

        if self.episode % 50 == 0:
            pos_dir = os.path.join(self.run_dir, f"posicao_{self.current_pos_id}")
            os.makedirs(pos_dir, exist_ok=True)
            log_path = os.path.join(pos_dir, f"flight_ep_{self.episode}.csv")
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "time_s", "lat", "lon", "alt_ft", "heading", "vs_fps",
                    "dist_m", "aileron", "elevator", "wind_spd", "wind_dir",
                    "cone_radius_m", "glide_ratio",
                ])
                writer.writerows(self.current_flight_telemetry)

    def _get_obs(self):
        try:
            lat = self.fdm["position/lat-gc-deg"]
            lon = self.fdm["position/long-gc-deg"]
            alt_m = self.fdm["position/h-sl-ft"] * 0.3048
            dist = haversine(lat, lon, self.target_lat, self.target_lon)
            raio_ideal = self._get_cone_radius(alt_m)
            cone_err = (dist - raio_ideal) / 1000.0
            bearing_err = (
                self._get_bearing(lat, lon, self.target_lat, self.target_lon)
                - self.fdm["attitude/psi-deg"] + 180
            ) % 360 - 180

            base = [
                np.clip(cone_err, -1, 1),
                bearing_err / 180.0,
                self.fdm["velocities/vg-fps"] / 60.0,
                self.fdm["velocities/h-dot-fps"] / 30.0,
                self.fdm["attitude/roll-rad"],
                self.fdm["attitude/pitch-rad"],
            ]
            if self.include_gr_in_obs:
                base.append(self.glide_ratio_target / 3.0)
            return np.array(base, dtype=np.float32)
        except Exception:
            obs_dim = self.observation_space.shape[0]
            return np.zeros(obs_dim, dtype=np.float32)

    def _get_bearing(self, lat1, lon1, lat2, lon2):
        off_x, off_y = lon2 - lon1, lat2 - lat1
        return (90.0 - math.degrees(math.atan2(off_y, off_x))) % 360