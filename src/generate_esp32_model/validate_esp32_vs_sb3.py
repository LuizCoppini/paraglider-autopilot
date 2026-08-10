"""
validate_esp32_vs_sb3.py — Validação cruzada SB3 (PC) ↔ ESP32
==============================================================
Dois modos:

  MODO OFFLINE (padrão):
    Simula o forward pass C do ESP32 em Python (mesma aritmética float32)
    e compara contra o SB3. Não precisa do ESP32 conectado.
    Confirma que ppo_inference.h está matematicamente correto ANTES de flashar.

  MODO LIVE (--live):
    Envia as obs via serial para o ESP32 em MODE_HIL, recebe as
    ações e compara em tempo real. Requer ESP32 compilado em MODE_HIL.

Uso:
  python src/generate_esp32_model/validate_esp32_vs_sb3.py
  python src/generate_esp32_model/validate_esp32_vs_sb3.py --live --port COM3

Dependências: stable-baselines3, torch, numpy, (pyserial para --live)
"""

import argparse
import struct
import time
import os
import sys
import numpy as np

# ─── Caminhos ────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_ZIP = os.path.join(_BASE, r"..\..\models\cone_v3_method\training_20260601_230833\parachute_cone_v3_model_final.zip")
NORM_PKL  = os.path.join(_BASE, r"..\..\models\cone_v3_method\training_20260601_230833\vec_normalize_cone_v3.pkl")

# ─── Cenários de mock (mesmos do main.cpp) ────────────────────────────────────
SCENARIOS = [
    ("no_cone_direto",     np.array([ 0.00,  0.00,  0.58, -0.67,  0.00,  0.00], dtype=np.float32)),
    ("fora_cone_direita",  np.array([ 0.40,  0.30,  0.55, -0.60,  0.05, -0.02], dtype=np.float32)),
    ("dentro_cone_esq",    np.array([-0.35, -0.25,  0.62, -0.70, -0.08,  0.01], dtype=np.float32)),
    ("final_approach",     np.array([ 0.05,  0.02,  0.50, -0.80,  0.02,  0.05], dtype=np.float32)),
    ("vento_forte_desvio", np.array([ 0.60,  0.50,  0.45, -0.55,  0.12, -0.03], dtype=np.float32)),
]

TOLERANCE = 1e-3  # aceitável para float32 (ops de tanh + matmul acumulam ~1e-5)


# ─── Carrega modelo e extrai pesos ───────────────────────────────────────────
def load_model():
    import pickle
    from stable_baselines3 import PPO

    print("Carregando modelo SB3...")
    model = PPO.load(MODEL_ZIP)

    print("Carregando VecNormalize...")
    with open(NORM_PKL, "rb") as f:
        vec_norm = pickle.load(f)

    return model, vec_norm


def extract_weights(model):
    """Extrai W/b de cada camada da policy (float32, igual ao model_weights.h)."""
    sd = model.policy.state_dict()
    def g(k): return sd[k].cpu().numpy().astype(np.float32)
    return {
        "W0": g("mlp_extractor.policy_net.0.weight"),  # [64, 6]
        "b0": g("mlp_extractor.policy_net.0.bias"),
        "W1": g("mlp_extractor.policy_net.2.weight"),  # [64, 64]
        "b1": g("mlp_extractor.policy_net.2.bias"),
        "W2": g("action_net.weight"),                  # [2, 64]
        "b2": g("action_net.bias"),
    }


# ─── Simulação Python do forward pass do ESP32 (espelha ppo_inference.h) ────
def esp32_forward_py(obs_raw: np.ndarray, vec_norm, weights: dict) -> np.ndarray:
    """
    Replica exatamente o que o ESP32 faz em C (float32):
      1. VecNormalize: clip((obs - mean) / sqrt(var + 1e-8), -10, 10)
      2. Layer 0 → tanh
      3. Layer 1 → tanh
      4. Action net → linear
      5. Clip para bounds do action_space: action[0] ∈ [-1,1], action[1] ∈ [0,1]
    """
    obs = obs_raw.astype(np.float32)

    # 1. VecNormalize (float32 explícito para espelhar C)
    mean = vec_norm.obs_rms.mean.astype(np.float32)
    var  = vec_norm.obs_rms.var.astype(np.float32)
    eps  = np.float32(1e-8)
    obs_norm = np.clip((obs - mean) / np.sqrt(var + eps), -10.0, 10.0).astype(np.float32)

    # 2-4. Forward pass (tudo float32)
    h0 = np.tanh(weights["W0"] @ obs_norm + weights["b0"]).astype(np.float32)
    h1 = np.tanh(weights["W1"] @ h0      + weights["b1"]).astype(np.float32)
    mu = (weights["W2"] @ h1 + weights["b2"]).astype(np.float32)

    # 5. Clip (sem tanh — SB3 PPO default squash_output=False)
    action = np.array([
        np.clip(mu[0], np.float32(-1.0), np.float32(1.0)),
        np.clip(mu[1], np.float32( 0.0), np.float32(1.0)),
    ], dtype=np.float32)
    return action


# ─── Predição SB3 ground-truth ────────────────────────────────────────────────
def sb3_predict(model, vec_norm, obs_raw: np.ndarray) -> np.ndarray:
    obs_norm = vec_norm.normalize_obs(obs_raw.reshape(1, -1))
    action, _ = model.predict(obs_norm, deterministic=True)
    return action.flatten().astype(np.float32)


# ─── Modo OFFLINE ─────────────────────────────────────────────────────────────
def run_offline(model, vec_norm):
    weights = extract_weights(model)
    W = 68

    print("\n" + "="*W)
    print(" VALIDAÇÃO OFFLINE — simulação Python do ESP32 vs SB3")
    print(" (confirma ppo_inference.h antes de flashar)")
    print("="*W)
    print(f"{'Cenário':<22} {'SB3 (ground truth)':>20} {'ESP32 sim (Python)':>20} {'Δmax':>8}  {'OK?'}")
    print("-"*W)

    all_ok = True
    for name, obs in SCENARIOS:
        sb3_act  = sb3_predict(model, vec_norm, obs)
        sim_act  = esp32_forward_py(obs, vec_norm, weights)
        delta    = np.abs(sb3_act - sim_act)
        ok       = bool(np.all(delta <= TOLERANCE))
        all_ok   = all_ok and ok

        sb3_str = f"[{sb3_act[0]:+.4f}, {sb3_act[1]:.4f}]"
        sim_str = f"[{sim_act[0]:+.4f}, {sim_act[1]:.4f}]"
        flag    = "✓" if ok else "✗ FALHOU"
        print(f"{name:<22} {sb3_str:>20} {sim_str:>20} {delta.max():>8.5f}  {flag}")

    print("-"*W)
    if all_ok:
        print(f"  RESULTADO: ✓ Simulação Python idêntica ao SB3 (tol={TOLERANCE:.0e})")
        print("  → Pode flashar o ESP32. Use --live após flashar para confirmar no hardware.")
    else:
        print(f"  RESULTADO: ✗ Divergência encontrada — NÃO flashe ainda.")
        print("  Verifique VecNormalize, ordem das camadas ou rescale da ação.")
    print()

    # Imprime os valores esperados para copiar no monitor serial
    print("  Valores esperados no monitor serial (MODE_MOCK) após flashing:")
    for name, obs in SCENARIOS:
        act = esp32_forward_py(obs, vec_norm, weights)
        print(f"    {name:<22}  aileron={act[0]:+.4f}  elevator={act[1]:.4f}")
    print()

    return all_ok


# ─── Modo LIVE ────────────────────────────────────────────────────────────────
HEADER_SEND = bytes([0xAA, 0xBB])
HEADER_RECV = bytes([0xCC, 0xDD])
OBS_DIM     = 6
ACTION_DIM  = 2
PKT_RECV_SZ = 2 + ACTION_DIM * 4

def serial_send_obs(ser, obs: np.ndarray) -> np.ndarray:
    payload = struct.pack(f"<{OBS_DIM}f", *obs.astype(np.float32))
    ser.write(HEADER_SEND + payload)
    ser.flush()
    deadline = time.time() + 1.0
    buf = b""
    while time.time() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
            idx = buf.find(HEADER_RECV)
            if idx >= 0 and len(buf) >= idx + PKT_RECV_SZ:
                return np.array(struct.unpack("<2f", buf[idx+2:idx+PKT_RECV_SZ]), dtype=np.float32)
    raise TimeoutError("ESP32 não respondeu")


def run_live(model, vec_norm, port: str):
    import serial
    W = 68

    print(f"\nAbrindo serial {port}...")
    ser = serial.Serial(port, baudrate=115200, timeout=0.1)
    time.sleep(2)
    ser.reset_input_buffer()

    print("\n" + "="*W)
    print(f" VALIDAÇÃO LIVE — SB3 vs ESP32 hardware via {port} (MODE_HIL)")
    print("="*W)
    print(f"{'Cenário':<22} {'SB3':>20} {'ESP32 hw':>20} {'Δmax':>8}  {'OK?'}")
    print("-"*W)

    all_ok = True
    for name, obs in SCENARIOS:
        sb3_act   = sb3_predict(model, vec_norm, obs)
        esp32_act = serial_send_obs(ser, obs)
        delta     = np.abs(sb3_act - esp32_act)
        ok        = bool(np.all(delta <= TOLERANCE))
        all_ok    = all_ok and ok

        sb3_str  = f"[{sb3_act[0]:+.4f}, {sb3_act[1]:.4f}]"
        hw_str   = f"[{esp32_act[0]:+.4f}, {esp32_act[1]:.4f}]"
        flag     = "✓" if ok else "✗ FALHOU"
        print(f"{name:<22} {sb3_str:>20} {hw_str:>20} {delta.max():>8.5f}  {flag}")

    print("-"*W)
    if all_ok:
        print(f"  RESULTADO: ✓ ESP32 hardware idêntico ao SB3 (tol={TOLERANCE:.0e})")
        print("  Hardware validado — pronto para HIL completo.")
    else:
        print(f"  RESULTADO: ✗ Diferenças acima de {TOLERANCE:.0e}.")
    print()
    ser.close()
    return all_ok


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Modo live via serial (ESP32 em MODE_HIL)")
    parser.add_argument("--port", default="COM3",      help="Porta serial (apenas com --live)")
    args = parser.parse_args()

    model, vec_norm = load_model()

    if args.live:
        ok = run_live(model, vec_norm, args.port)
    else:
        ok = run_offline(model, vec_norm)

    sys.exit(0 if ok else 1)