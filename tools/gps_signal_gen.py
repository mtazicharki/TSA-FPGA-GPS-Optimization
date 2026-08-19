"""
gps_signal_gen.py

Generateur de signal GPS L1 C/A synthetique (bande IF), parametrable pour
reproduire les deux scenarios de la these de Glauberto Albuquerque :

  - DANISH-like : signal terrestre, 1 bit ADC, Doppler +/-10 kHz,
                  avec injection de "Sample Errors" (SE) de 0% a ~50%.
  - INPE-like   : signal spatial (LEO 600 km, type CONASAT), 2 bits ADC,
                  Doppler jusqu'a +/-45 kHz, SNR balaye de 0 a 18 dB.

Ce module ne depend que de numpy. Il ne cherche pas a etre un simulateur
orbital complet : le Doppler est applique comme une rampe lineaire sur la
duree du bloc (1 ms), ce qui est suffisant pour tester un bloc d'acquisition
(recherche de code phase / frequence Doppler) comme acq_unified.cpp.

Auteur : genere pour Mohammed (UMONS, projet TSA-FPGA-GPS-Optimization)
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Generation du code C/A (Gold code GPS L1)
# ---------------------------------------------------------------------------

# Taps G2 (phase select) par PRN, 1..32 -- table standard ICD-GPS-200
_G2_TAPS = {
    1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9), 6: (2, 10),
    7: (1, 8), 8: (2, 9), 9: (3, 10), 10: (2, 3), 11: (3, 4), 12: (5, 6),
    13: (6, 7), 14: (7, 8), 15: (8, 9), 16: (9, 10), 17: (1, 4), 18: (2, 5),
    19: (3, 6), 20: (4, 7), 21: (5, 8), 22: (6, 9), 23: (1, 3), 24: (4, 6),
    25: (5, 7), 26: (6, 8), 27: (7, 9), 28: (8, 10), 29: (1, 6), 30: (2, 7),
    31: (3, 8), 32: (4, 9),
}


def generate_ca_code(prn: int) -> np.ndarray:
    """Retourne le code C/A du PRN donne (1..32), 1023 chips, valeurs +1/-1."""
    if prn not in _G2_TAPS:
        raise ValueError(f"PRN {prn} invalide (1..32 attendu)")

    tap1, tap2 = _G2_TAPS[prn]

    g1 = [1] * 10
    g2 = [1] * 10
    code = np.zeros(1023, dtype=np.int8)

    for i in range(1023):
        g1_out = g1[9]
        g2_out = (g2[tap1 - 1] ^ g2[tap2 - 1])
        # Convention alignee sur les fichiers PRN de reference du projet TSA_GNSS
        # (Ibrahima) : mappe {0,1} -> {-1,+1}
        code[i] = 2 * (g1_out ^ g2_out) - 1

        # LFSR G1 : polynome x^10+x^3+1
        new_g1 = g1[2] ^ g1[9]
        g1 = [new_g1] + g1[:9]

        # LFSR G2 : polynome x^10+x^9+x^8+x^6+x^3+x^2+1
        new_g2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g2 = [new_g2] + g2[:9]

    return code


# ---------------------------------------------------------------------------
# 2. Configuration de scenario
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    name: str
    fs_hz: float            # frequence d'echantillonnage
    if_hz: float            # frequence intermediaire (porteuse IF)
    adc_bits: int           # 1 ou 2
    prn: int
    doppler_hz: float       # decalage Doppler applique (centre de bande)
    duration_ms: float = 1.0
    snr_db: float | None = None      # None => pas de bruit ajoute
    sample_error_pct: float = 0.0    # 0..~50, uniquement pour ADC 1 bit


# Presets directement tires de la these (Table 10 + section 5.6)
DANISH_BASE = ScenarioConfig(
    name="DANISH", fs_hz=11.999e6, if_hz=3.563e6, adc_bits=1,
    prn=1, doppler_hz=3000.0, duration_ms=1.0,
)

INPE_BASE = ScenarioConfig(
    name="INPE", fs_hz=16.368e6, if_hz=4.092e6, adc_bits=2,
    prn=23, doppler_hz=42500.0, duration_ms=1.0,
)


# ---------------------------------------------------------------------------
# 3. Generation du signal IF
# ---------------------------------------------------------------------------

def _upsample_code(code: np.ndarray, n_samples: int, fs_hz: float,
                    code_rate_hz: float = 1.023e6) -> np.ndarray:
    """Suréchantillonne le code C/A (1023 chips) a la frequence Fs."""
    t = np.arange(n_samples) / fs_hz
    chip_idx = np.floor(t * code_rate_hz).astype(np.int64) % 1023
    return code[chip_idx]


def generate_if_signal(cfg: ScenarioConfig, rng: np.random.Generator | None = None
                        ) -> np.ndarray:
    """
    Genere un bloc de signal IF (float64, non quantifie) de duree cfg.duration_ms,
    contenant le C/A code du PRN cfg.prn module en BPSK sur la porteuse IF,
    avec le decalage Doppler cfg.doppler_hz, plus bruit si cfg.snr_db est fourni.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples = int(round(cfg.fs_hz * cfg.duration_ms * 1e-3))
    code = generate_ca_code(cfg.prn)
    code_seq = _upsample_code(code, n_samples, cfg.fs_hz)

    t = np.arange(n_samples) / cfg.fs_hz
    carrier_freq = cfg.if_hz + cfg.doppler_hz
    carrier = np.cos(2 * np.pi * carrier_freq * t)

    signal = code_seq.astype(np.float64) * carrier

    if cfg.snr_db is not None:
        sig_power = np.mean(signal ** 2)
        snr_linear = 10 ** (cfg.snr_db / 10.0)
        noise_power = sig_power / snr_linear
        noise = rng.normal(0.0, np.sqrt(noise_power), size=n_samples)
        signal = signal + noise

    return signal


# ---------------------------------------------------------------------------
# 4. Quantification ADC (1 bit ou 2 bits)
# ---------------------------------------------------------------------------

def quantize(signal: np.ndarray, bits: int) -> np.ndarray:
    """
    Quantifie le signal flottant selon la resolution ADC voulue.
      - 1 bit : sortie {+1, -1} (signe uniquement).
      - 2 bits : sortie {-3, -1, +1, +3} (schema sign-magnitude classique des
        front-ends GPS, seuil place a 1 sigma).
    """
    if bits == 1:
        out = np.where(signal >= 0, 1, -1).astype(np.int8)
    elif bits == 2:
        sigma = np.std(signal)
        threshold = sigma if sigma > 0 else 1.0
        sign = np.where(signal >= 0, 1, -1)
        mag = np.where(np.abs(signal) >= threshold, 3, 1)
        out = (sign * mag).astype(np.int8)
    else:
        raise ValueError("bits doit valoir 1 ou 2")
    return out


def inject_sample_errors(quantized: np.ndarray, se_pct: float,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Reproduit le protocole DANISH_SE de la these : inverse aleatoirement
    se_pct% des echantillons (uniquement pertinent pour un signal 1 bit,
    mais fonctionne aussi en inversant le signe pour du multi-bit).
    """
    if rng is None:
        rng = np.random.default_rng()
    if se_pct <= 0:
        return quantized.copy()

    out = quantized.copy()
    n = len(out)
    n_flip = int(round(n * se_pct / 100.0))
    idx = rng.choice(n, size=n_flip, replace=False)
    out[idx] = -out[idx]
    return out


# ---------------------------------------------------------------------------
# 5. Sauvegarde
# ---------------------------------------------------------------------------

def save_binary(samples: np.ndarray, path: Path) -> None:
    """Sauvegarde les echantillons en int8 brut, 1 octet par echantillon."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples.astype(np.int8).tofile(path)
