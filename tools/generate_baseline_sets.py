"""
generate_baseline_sets.py

Genere les deux jeux de signaux de test "equivalents" aux baselines de la
these de Glauberto Albuquerque :

  1) DANISH_SE : 1 bit, PRN 1, Doppler fixe dans +/-10 kHz, 14 fichiers avec
     un taux de Sample Error croissant (0% -> ~48%), comme decrit section 5.2.

  2) INPE      : 2 bits, PRN 23, Doppler ~42.5 kHz (proche du max theorique
     pour une orbite LEO a 600 km), 10 fichiers avec un SNR de 0 a 18 dB
     par pas de 2 dB, comme decrit section 5.6 (Table 10 + Fig. 36/37).

Chaque fichier est sauvegarde en binaire brut (1 octet signe = 1 echantillon,
int8), directement exploitable comme vecteur de test dans un testbench C/HLS
(ex: acq_unified.cpp), plus un fichier metadata.json qui documente les
parametres exacts de generation.

Une verification par correlation (recherche brute de phase/Doppler) est
executee a la fin pour s'assurer que le pic attendu est bien detectable dans
le cas le plus favorable (0% SE / SNR le plus haut).
"""

import json
from pathlib import Path

import numpy as np

from gps_signal_gen import (
    ScenarioConfig, DANISH_BASE, INPE_BASE,
    generate_ca_code, generate_if_signal, quantize, inject_sample_errors,
    save_binary,
)

OUT_DIR = Path("/mnt/user-data/outputs/gps_test_signals")
RNG = np.random.default_rng(seed=42)


# ---------------------------------------------------------------------------
# 1. Generation DANISH_SE (terrestre, 1 bit, Doppler +/-10kHz, SE 0-48%)
# ---------------------------------------------------------------------------

def build_danish_se_set():
    se_levels = np.linspace(0, 48, 14)  # 14 fichiers, comme dans la these
    metadata = []

    clean_if = generate_if_signal(DANISH_BASE, rng=RNG)  # pas de bruit gaussien
    clean_q = quantize(clean_if, DANISH_BASE.adc_bits)

    for i, se in enumerate(se_levels):
        noisy = inject_sample_errors(clean_q, se, rng=RNG)
        fname = OUT_DIR / "DANISH_SE" / f"danish_se_{i:02d}_SE{se:04.1f}pct.bin"
        save_binary(noisy, fname)

        metadata.append({
            "file": str(fname.relative_to(OUT_DIR)),
            "prn": DANISH_BASE.prn,
            "fs_hz": DANISH_BASE.fs_hz,
            "if_hz": DANISH_BASE.if_hz,
            "adc_bits": DANISH_BASE.adc_bits,
            "doppler_hz": DANISH_BASE.doppler_hz,
            "duration_ms": DANISH_BASE.duration_ms,
            "sample_error_pct": float(se),
            "n_samples": int(len(noisy)),
        })

    return metadata


def build_prn_ref_and_manifest(danish_meta):
    """
    Genere le fichier de reference PRN (signal local non module, requis par
    prn_stream dans acquisition_stsa_top) et un manifest CSV simple,
    facile a parser en C++ (evite d'avoir a parser le JSON en C++).
    """
    from gps_signal_gen import _upsample_code

    n_samples = danish_meta[0]["n_samples"]
    prn_code = generate_ca_code(DANISH_BASE.prn)
    prn_ref = _upsample_code(prn_code, n_samples, DANISH_BASE.fs_hz)

    ref_path = OUT_DIR / "PRN_REF" / f"prn_ref_PRN{DANISH_BASE.prn}_N{n_samples}.bin"
    save_binary(prn_ref, ref_path)

    manifest_path = OUT_DIR / "manifest_danish_se.csv"
    with open(manifest_path, "w") as f:
        f.write("file,prn_ref_file,prn,fs_hz,if_hz,adc_bits,doppler_hz,"
                "duration_ms,se_pct,n_samples\n")
        for m in danish_meta:
            f.write(f"{m['file']},{ref_path.relative_to(OUT_DIR)},{m['prn']},"
                    f"{m['fs_hz']},{m['if_hz']},{m['adc_bits']},{m['doppler_hz']},"
                    f"{m['duration_ms']},{m['sample_error_pct']},{m['n_samples']}\n")

    return ref_path, manifest_path


# ---------------------------------------------------------------------------
# 2. Generation INPE (spatial, 2 bits, Doppler ~42.5kHz, SNR 0-18dB pas de 2)
# ---------------------------------------------------------------------------

def build_inpe_set():
    snr_levels = list(range(0, 19, 2))  # 0,2,...,18 -> 10 fichiers
    metadata = []

    for i, snr in enumerate(snr_levels):
        cfg = ScenarioConfig(
            name="INPE", fs_hz=INPE_BASE.fs_hz, if_hz=INPE_BASE.if_hz,
            adc_bits=INPE_BASE.adc_bits, prn=INPE_BASE.prn,
            doppler_hz=INPE_BASE.doppler_hz, duration_ms=INPE_BASE.duration_ms,
            snr_db=float(snr),
        )
        sig = generate_if_signal(cfg, rng=RNG)
        q = quantize(sig, cfg.adc_bits)

        fname = OUT_DIR / "INPE" / f"inpe_{i:02d}_SNR{snr:02d}dB.bin"
        save_binary(q, fname)

        metadata.append({
            "file": str(fname.relative_to(OUT_DIR)),
            "prn": cfg.prn,
            "fs_hz": cfg.fs_hz,
            "if_hz": cfg.if_hz,
            "adc_bits": cfg.adc_bits,
            "doppler_hz": cfg.doppler_hz,
            "duration_ms": cfg.duration_ms,
            "snr_db": snr,
            "n_samples": int(len(q)),
        })

    return metadata


# ---------------------------------------------------------------------------
# 3. Validation rapide par correlation (brute force, non optimisee)
# ---------------------------------------------------------------------------

def validate_signal(bin_path: Path, cfg: ScenarioConfig,
                     doppler_search_hz=range(-45000, 45001, 500)):
    """
    Recherche brute (Serial Acquisition classique) de la phase de code et du
    Doppler sur le fichier donne, pour verifier qu'un pic net apparait bien
    pres du Doppler injecte et de la phase 0 (le code commence a l'echantillon 0).
    """
    samples = np.fromfile(bin_path, dtype=np.int8).astype(np.float64)
    n = len(samples)
    fs = cfg.fs_hz
    code = generate_ca_code(cfg.prn)

    t = np.arange(n) / fs
    best_peak = -1.0
    best_doppler = None

    # Recherche Doppler grossiere (pas de recherche de phase complete ici,
    # on correle juste avec le code aligne en phase 0 pour la verification)
    chip_idx = np.floor(t * 1.023e6).astype(np.int64) % 1023
    code_seq = code[chip_idx].astype(np.float64)

    for dopp in doppler_search_hz:
        carrier_i = np.cos(2 * np.pi * (cfg.if_hz + dopp) * t)
        carrier_q = np.sin(2 * np.pi * (cfg.if_hz + dopp) * t)
        I = np.sum(samples * code_seq * carrier_i)
        Q = np.sum(samples * code_seq * carrier_q)
        power = I ** 2 + Q ** 2
        if power > best_peak:
            best_peak = power
            best_doppler = dopp

    return best_doppler, best_peak


if __name__ == "__main__":
    print("Generation du set DANISH_SE (1 bit, PRN1, Doppler=%.0f Hz)..."
          % DANISH_BASE.doppler_hz)
    danish_meta = build_danish_se_set()

    print("Generation du fichier PRN de reference + manifest CSV (DANISH_SE)...")
    ref_path, manifest_path = build_prn_ref_and_manifest(danish_meta)

    print("Generation du set INPE (2 bits, PRN23, Doppler=%.0f Hz)..."
          % INPE_BASE.doppler_hz)
    inpe_meta = build_inpe_set()

    all_meta = {"DANISH_SE": danish_meta, "INPE": inpe_meta}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(all_meta, f, indent=2)

    print("\n--- Validation par correlation (brute force) ---")

    # Cas le plus favorable pour chaque set : SE=0% et SNR=18dB
    danish_clean = OUT_DIR / danish_meta[0]["file"]
    d_dopp, d_peak = validate_signal(danish_clean, DANISH_BASE,
                                      doppler_search_hz=range(-10000, 10001, 250))
    print(f"DANISH_SE (SE=0%)  -> Doppler detecte = {d_dopp} Hz "
          f"(injecte = {DANISH_BASE.doppler_hz} Hz), pic = {d_peak:.3e}")

    inpe_clean = OUT_DIR / inpe_meta[-1]["file"]  # dernier = SNR le plus haut
    cfg_check = ScenarioConfig(
        name="INPE", fs_hz=INPE_BASE.fs_hz, if_hz=INPE_BASE.if_hz,
        adc_bits=INPE_BASE.adc_bits, prn=INPE_BASE.prn,
        doppler_hz=INPE_BASE.doppler_hz, duration_ms=INPE_BASE.duration_ms,
    )
    i_dopp, i_peak = validate_signal(inpe_clean, cfg_check,
                                      doppler_search_hz=range(40000, 45001, 250))
    print(f"INPE (SNR=18dB)    -> Doppler detecte = {i_dopp} Hz "
          f"(injecte = {INPE_BASE.doppler_hz} Hz), pic = {i_peak:.3e}")

    print(f"\nFichiers ecrits dans : {OUT_DIR}")
    print(f"  - {len(danish_meta)} fichiers DANISH_SE")
    print(f"  - {len(inpe_meta)} fichiers INPE")
    print("  - metadata.json (parametres exacts de chaque fichier)")
    print(f"  - {ref_path.relative_to(OUT_DIR)} (reference PRN pour prn_stream)")
    print(f"  - {manifest_path.relative_to(OUT_DIR)} (manifest CSV pour le testbench C++)")
