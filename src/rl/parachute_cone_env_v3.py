"""
parachute_cone_env_v3.py — Cone v3 com cobertura sim-to-real mínima.

Mantém TUDO do v2 (método cone, vento norte→sul, proa aleatória, GR fixo
do Parachutist em 1.88) e adiciona 3 fatores que cobrem a transição
JSBSim → FlightGear → ESP32 + parafoil real:

  1. ACTION LAG: a ação enviada pelo agente leva 50-200 ms pra chegar nos
     servos (latência GPS read → ESP32 inference → PWM → servo response).
     Implementado aplicando a ação ANTERIOR durante os primeiros K ticks
     de cada step (K sorteado por episódio em U(6, 24) sim_ticks @ 120Hz).

  2. OBSERVATION NOISE: GPS (~3m), heading (~0.5°), altitude (~2m). O
     estado interno do JSBSim continua limpo, mas o que o AGENTE VÊ é
     ruidoso — exatamente como GPS NEO-6M, MPU6050, BMP180 reais.

  3. WIND GUSTS: o vento base (4/8/14/20 fps) é mantido como média do
     episódio, mas a cada 3-8 sim_seconds adiciona ±20% magnitude e
     ±15° direção. Cobre rajadas e turbulência não modeladas pelo
     vento constante do v2.

Tudo o que NÃO muda em relação ao v2:
  - Observação 6 dims (cone_err, bearing_err/180, vg/60, h_dot/30, roll, pitch)
  - Action 2 dims (aileron, elevator)
  - Reward (cone tracking)
  - GR fixo em 1.88 (Parachutist) — sem domain randomization de GR,
    porque a dissertação trata GR como propriedade da aeronave que vai
    ser re-treinada por aeronave
  - Vento base norte→sul, 4 níveis de velocidade
  - Proa inicial aleatória
  - 8 spawn positions ao redor do alvo

Isso significa que o validador FG (fgfs_validation_cone_v2.py) funciona
sem alterações pra avaliar modelos cone v3 — só trocar MODEL_PATH e
VEC_NORMALIZE_PATH no script de eval.
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


class ParachuteConeEnvV3(gym.Env):
    """
    Env do método cone v3. Idêntica em interface ao v2; adiciona
    parâmetros opcionais pra controlar a "robustificação" sim-to-real.

    Parâmetros novos vs v2:
      action_lag_ticks_range : (min, max) em sim_ticks (1 tick = 1/120 s).
                               Default (6, 24) = 50-200 ms.
      gps_noise_m            : desvio padrão do ruído GPS em metros.
                               Default 3.0 (típico NEO-6M com SBAS).
      heading_noise_deg      : σ do ruído de heading (compass/yaw).
                               Default 0.5 (típico MPU6050 fundido).
      alt_noise_m            : σ do ruído de altitude (barômetro).
                               Default 2.0 (típico BMP180).
      wind_gust_mag_pct      : amplitude da rajada como % do vento base.
                               Default 0.20 (±20%).
      wind_gust_dir_deg      : amplitude da rajada em direção (graus).
                               Default 15.0 (±15°).
      wind_gust_period_s     : periodicidade média entre rajadas (sim s).
                               Default 5.0 (re-sorteia U(3, 8) cada vez).
    """

    def __init__(self, target_lat, target_lon,
                 glide_ratio_target=1.88,
                 include_gr_in_obs=False,
                 spawn_radius_m=4000.0,
                 start_alt_ft=9850,
                 wind_speeds_fps=(4.0, 8.0, 14.0, 20.0),
                 wind_dir_deg=0.0,
                 random_initial_heading=True,
                 # >>> Novos parâmetros v3 (defaults pensados pra Parachutist + ESP32) <<<
                 action_lag_ticks_range=(6, 24),
                 gps_noise_m=3.0,
                 heading_noise_deg=0.5,
                 alt_noise_m=2.0,
                 wind_gust_mag_pct=0.20,
                 wind_gust_dir_deg=15.0,
                 wind_gust_period_s=5.0):
        super().__init__()
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.glide_ratio_target = float(glide_ratio_target)
        self.include_gr_in_obs = bool(include_gr_in_obs)
        self.spawn_radius_m = spawn_radius_m
        self.start_alt_ft = start_alt_ft
        self.wind_speeds = list(wind_speeds_fps)
        self.wind_dir_deg = float(wind_dir_deg)
        self.random_initial_heading = bool(random_initial_heading)

        # Sim-to-real params
        self.action_lag_ticks_range = action_lag_ticks_range
        self.gps_noise_m = float(gps_noise_m)
        self.heading_noise_deg = float(heading_noise_deg)
        self.alt_noise_m = float(alt_noise_m)
        self.wind_gust_mag_pct = float(wind_gust_mag_pct)
        self.wind_gust_dir_deg = float(wind_gust_dir_deg)
        self.wind_gust_period_s = float(wind_gust_period_s)

        self.episode = 0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "with_gr_obs" if self.include_gr_in_obs else "fixed_gr"
        self.run_dir = os.path.join(
            r"D:\workspace\Pycharm\paraglider-autopilot\src\flight_records",
            f"training_cone_v3_{suffix}_gr{self.glide_ratio_target:.2f}_{timestamp}",
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
        # Vento base do episódio (média)
        self.wind_n_base = 0.0
        self.wind_e_base = 0.0
        # Vento aplicado AGORA (base + rajada atual)
        self.wind_n = 0.0
        self.wind_e = 0.0
        # Controle de rajada
        self.next_gust_at_step = 0
        # Lag de ação do episódio
        self.action_lag_ticks = 0

        self._init_log_files()

    def _init_log_files(self):
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "pos_id", "wind_speed_fps", "wind_dir_deg",
                "flight_time_s", "final_dist", "reward", "lat", "lon", "gr",
                "action_lag_ticks",
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

    def _sample_new_gust(self):
        """
        Sorteia nova rajada: ±wind_gust_mag_pct na magnitude, ±wind_gust_dir_deg
        na direção, em torno do vento BASE do episódio.
        """
        # Recompõe magnitude e direção do BASE
        base_mag = math.sqrt(self.wind_n_base ** 2 + self.wind_e_base ** 2)
        # atan2 inverso: wind_n = -cos(dir), wind_e = -sin(dir)
        if base_mag < 1e-6:
            base_dir = 0.0
        else:
            base_dir = math.degrees(math.atan2(-self.wind_e_base, -self.wind_n_base))

        mag_jitter = float(self.np_random.uniform(
            -self.wind_gust_mag_pct, self.wind_gust_mag_pct
        ))
        dir_jitter = float(self.np_random.uniform(
            -self.wind_gust_dir_deg, self.wind_gust_dir_deg
        ))
        gust_mag = max(0.0, base_mag * (1.0 + mag_jitter))
        gust_dir = base_dir + dir_jitter

        rad = math.radians(gust_dir)
        self.wind_n = -math.cos(rad) * gust_mag
        self.wind_e = -math.sin(rad) * gust_mag

        # Próxima rajada em U(period*0.6, period*1.6) sim_seconds (1 step = 1 s)
        period = float(self.np_random.uniform(
            self.wind_gust_period_s * 0.6, self.wind_gust_period_s * 1.6
        ))
        self.next_gust_at_step = self.flight_time + max(1, int(round(period)))

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
        self.current_wind_dir = self.wind_dir_deg
        rad_wind = math.radians(self.current_wind_dir)
        # Vento BASE do episódio (média)
        self.wind_n_base = -math.cos(rad_wind) * self.current_wind_speed
        self.wind_e_base = -math.sin(rad_wind) * self.current_wind_speed
        # Inicializa vento aplicado = base (rajada=0 no t=0)
        self.wind_n = self.wind_n_base
        self.wind_e = self.wind_e_base

        # Lag de ação fixo PARA ESTE EPISÓDIO (sorteado uma vez)
        lo, hi = self.action_lag_ticks_range
        self.action_lag_ticks = int(self.np_random.integers(lo, hi + 1))

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

        # Vento DEPOIS de run_ic
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

        # Primeira rajada será aplicada em ~period segundos
        period = float(self.np_random.uniform(
            self.wind_gust_period_s * 0.6, self.wind_gust_period_s * 1.6
        ))
        self.next_gust_at_step = max(1, int(round(period)))

        return self._get_obs(), {}

    def step(self, action):
        # Rate limit (igual v2)
        max_rate = 0.2
        action[0] = np.clip(
            action[0],
            self.last_action[0] - max_rate,
            self.last_action[0] + max_rate,
        )
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Wind gusts: hora de re-sortear?
        if self.flight_time >= self.next_gust_at_step:
            self._sample_new_gust()

        # Reaplica vento atual (base + rajada)
        self._apply_wind()

        # >>> ACTION LAG <<<
        # Durante os primeiros action_lag_ticks ticks, aplica AÇÃO ANTERIOR.
        # Depois, aplica AÇÃO NOVA. Simula latência do pipeline.
        success = True

        # Fase 1: lag — ainda comandando ação anterior
        if self.action_lag_ticks > 0:
            self.fdm["fcs/aileron-cmd-norm"] = float(self.last_action[0])
            self.fdm["fcs/elevator-cmd-norm"] = float(self.last_action[1])
            for _ in range(self.action_lag_ticks):
                if not self.fdm.run():
                    success = False
                    break

        # Fase 2: ação nova aplicada
        if success:
            self.fdm["fcs/aileron-cmd-norm"] = float(action[0])
            self.fdm["fcs/elevator-cmd-norm"] = float(action[1])
            remaining = 120 - self.action_lag_ticks
            for _ in range(remaining):
                if not self.fdm.run():
                    success = False
                    break

        # Atualiza last_action (a nova é agora a anterior pro próximo step)
        self.last_action = action

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

        # Magnitude do vento atual (pra log/análise)
        current_wind_mag = math.sqrt(self.wind_n ** 2 + self.wind_e ** 2)

        # Telemetria pra log
        self.current_flight_telemetry.append([
            self.flight_time, lat, lon, alt_ft,
            self.fdm["attitude/psi-deg"], -v_down, dist_m,
            action[0], action[1], current_wind_mag, self.current_wind_dir,
            round(raio_ideal, 2), round(instant_gr, 3),
            self.action_lag_ticks,
        ])

        # Recompensa idêntica ao v2: tracking do cone
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
                self.action_lag_ticks,
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
                    "cone_radius_m", "glide_ratio", "action_lag_ticks",
                ])
                writer.writerows(self.current_flight_telemetry)

    def _get_obs(self):
        """
        Obs IDÊNTICA ao v2 em dimensionalidade e semântica, MAS com
        ruído de sensor aplicado nas leituras (GPS, heading, altitude).
        O estado interno do JSBSim continua limpo — só o que o agente
        VÊ é ruidoso. Simula exatamente o que vai acontecer no ESP32.
        """
        try:
            # Leituras "verdadeiras" do JSBSim
            lat_true = self.fdm["position/lat-gc-deg"]
            lon_true = self.fdm["position/long-gc-deg"]
            alt_m_true = self.fdm["position/h-sl-ft"] * 0.3048
            psi_true = self.fdm["attitude/psi-deg"]

            # >>> Aplica ruído de sensor (só na OBS, não no estado real) <<<
            # GPS: σ=gps_noise_m em metros. Converte pra deg.
            if self.gps_noise_m > 0:
                # 1 grau de lat ≈ 111320 m. 1 grau de lon ≈ 111320*cos(lat).
                noise_lat_m = float(self.np_random.normal(0.0, self.gps_noise_m))
                noise_lon_m = float(self.np_random.normal(0.0, self.gps_noise_m))
                lat = lat_true + noise_lat_m / 111320.0
                lon = lon_true + noise_lon_m / (
                    111320.0 * max(0.1, math.cos(math.radians(lat_true)))
                )
            else:
                lat = lat_true
                lon = lon_true

            # Altitude (BMP180)
            if self.alt_noise_m > 0:
                alt_m = alt_m_true + float(
                    self.np_random.normal(0.0, self.alt_noise_m)
                )
            else:
                alt_m = alt_m_true

            # Heading (compass/MPU6050)
            if self.heading_noise_deg > 0:
                psi = psi_true + float(
                    self.np_random.normal(0.0, self.heading_noise_deg)
                )
            else:
                psi = psi_true

            dist = haversine(lat, lon, self.target_lat, self.target_lon)
            raio_ideal = self._get_cone_radius(alt_m)
            cone_err = (dist - raio_ideal) / 1000.0
            bearing_err = (
                self._get_bearing(lat, lon, self.target_lat, self.target_lon)
                - psi + 180
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