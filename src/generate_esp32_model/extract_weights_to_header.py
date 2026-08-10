"""
extract_weights_to_header.py
============================
Rode este script no seu PC (onde tem PyTorch + SB3 instalados).

Ele carrega:
  - parachute_cone_v3_model_final.zip  (modelo SB3 PPO)
  - vec_normalize_cone_v3.pkl          (normalizador de obs)

E gera:
  - model_weights.h   -> coloque em src/ ou include/ do PlatformIO

Uso:
    python extract_weights_to_header.py

Dependências: torch, stable-baselines3, numpy
"""

import torch
import pickle
import numpy as np
import os
import struct

# ─── Caminhos ────────────────────────────────────────────────────────────────
import os
_BASE = os.path.dirname(os.path.abspath(__file__))  # pasta deste script

MODEL_ZIP  = os.path.join(_BASE, r"..\..\models\cone_v3_method\training_20260601_230833\parachute_cone_v3_model_final.zip")
NORM_PKL   = os.path.join(_BASE, r"..\..\models\cone_v3_method\training_20260601_230833\vec_normalize_cone_v3.pkl")
OUT_HEADER = os.path.join(_BASE, "model_weights.h")   # salva na mesma pasta do script

# ─── Carregar pesos da policy ─────────────────────────────────────────────────
print("Carregando pesos da policy...")
import zipfile, io

with zipfile.ZipFile(MODEL_ZIP) as zf:
    with zf.open("policy.pth") as f:
        state_dict = torch.load(io.BytesIO(f.read()), map_location="cpu")

print("Chaves encontradas em policy.pth:")
for k, v in state_dict.items():
    print(f"  {k:60s} shape={list(v.shape)}")

# ─── Extrair camadas da policy (ator) ─────────────────────────────────────────
# SB3 default ActorCriticPolicy net_arch=[64,64]:
# mlp_extractor.policy_net.0.weight  [64, 6]
# mlp_extractor.policy_net.0.bias    [64]
# mlp_extractor.policy_net.2.weight  [64, 64]
# mlp_extractor.policy_net.2.bias    [64]
# action_net.weight                  [2, 64]
# action_net.bias                    [2]

def get(key):
    return state_dict[key].numpy().astype(np.float32)

try:
    W0 = get("mlp_extractor.policy_net.0.weight")   # [64, 6]
    b0 = get("mlp_extractor.policy_net.0.bias")     # [64]
    W1 = get("mlp_extractor.policy_net.2.weight")   # [64, 64]
    b1 = get("mlp_extractor.policy_net.2.bias")     # [64]
    W2 = get("action_net.weight")                   # [2, 64]
    b2 = get("action_net.bias")                     # [2]
except KeyError:
    # Fallback: tenta nomes alternativos do SB3
    # Imprime chaves disponíveis para debug
    print("\nERRO: chaves esperadas não encontradas. Chaves disponíveis:")
    for k in state_dict.keys():
        print(f"  {k}")
    raise

print(f"\nArquitetura detectada:")
print(f"  Layer 0 (policy_net.0): in={W0.shape[1]}, out={W0.shape[0]}")
print(f"  Layer 1 (policy_net.2): in={W1.shape[1]}, out={W1.shape[0]}")
print(f"  Layer 2 (action_net):   in={W2.shape[1]}, out={W2.shape[0]}")

# ─── Carregar VecNormalize ────────────────────────────────────────────────────
print("\nCarregando VecNormalize...")
with open(NORM_PKL, "rb") as f:
    vec_norm = pickle.load(f)

obs_rms = vec_norm.obs_rms
obs_mean = obs_rms.mean.astype(np.float32)
obs_var  = obs_rms.var.astype(np.float32)
eps      = 1e-8

print(f"  obs_mean = {obs_mean}")
print(f"  obs_var  = {obs_var}")

# ─── Função auxiliar para formatar array como C ───────────────────────────────
def arr_to_c(arr: np.ndarray, name: str, indent="  ") -> str:
    flat = arr.flatten()
    rows = []
    for i in range(0, len(flat), 8):
        chunk = flat[i:i+8]
        rows.append(indent + ", ".join(f"{v:.8f}f" for v in chunk))
    shape_comment = "x".join(str(s) for s in arr.shape)
    return (
        f"// {name}  shape=[{shape_comment}]  ({flat.size} valores)\n"
        f"static const float {name}[{flat.size}] = {{\n"
        + ",\n".join(rows)
        + "\n};\n"
    )

# ─── Gerar header ─────────────────────────────────────────────────────────────
print(f"\nGerando {OUT_HEADER}...")

lines = []
lines.append("// model_weights.h  —  gerado por extract_weights_to_header.py")
lines.append("// NÃO edite manualmente.")
lines.append("#pragma once")
lines.append("")
lines.append(f"#define OBS_DIM   {W0.shape[1]}")
lines.append(f"#define HIDDEN0   {W0.shape[0]}")
lines.append(f"#define HIDDEN1   {W1.shape[0]}")
lines.append(f"#define ACTION_DIM {W2.shape[0]}")
lines.append("")

lines.append("// ── VecNormalize ──────────────────────────────────────────────")
lines.append(arr_to_c(obs_mean, "OBS_MEAN"))
lines.append(arr_to_c(obs_var,  "OBS_VAR"))

lines.append("// ── Pesos da rede (actor MLP) ─────────────────────────────────")
lines.append(arr_to_c(W0, "W0"))
lines.append(arr_to_c(b0, "b0"))
lines.append(arr_to_c(W1, "W1"))
lines.append(arr_to_c(b1, "b1"))
lines.append(arr_to_c(W2, "W2"))
lines.append(arr_to_c(b2, "b2"))

with open(OUT_HEADER, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

size_kb = os.path.getsize(OUT_HEADER) / 1024
print(f"✓ {OUT_HEADER} gerado ({size_kb:.1f} KB)")
print(f"\nCopie model_weights.h para a pasta src/ ou include/ do seu projeto PlatformIO.")