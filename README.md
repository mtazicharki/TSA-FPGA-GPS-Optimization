# TSA-FPGA-GPS-Optimization

Optimisation FPGA d'un bloc d'acquisition GPS (algorithme **TSA/STSA**) pour application
Radio Occultation GNSS sur plateforme CubeSat.

Stage de Master 2 — Laboratoire SEMI, Université de Mons (UMONS), 2026
Encadrement : Jérôme Verscheuren (superviseur), Carlos Valderrama (chef de service)

---

## Contexte

Le bloc d'acquisition GPS développé initialement au laboratoire consommait **265 BRAM et
205 DSP** sur un FPGA Xilinx `xc7z020` (PYNQ-Z2), soit plus de 93 % des ressources
disponibles — rendant impossible l'intégration des blocs DMA et du wrapper processeur
nécessaires au fonctionnement complet du système.

Ce travail vise à réduire significativement cette consommation, tout en conservant les
performances de détection, afin de rendre viable une implémentation embarquée sur
CubeSat.

---

## Résultats principaux

| Ressource | Design original | Après optimisation | Gain |
|---|---:|---:|---:|
| BRAM | 265 | 127 | **−52 %** |
| DSP | 205 | 30 | **−85 %** |
| FF | 44 819 | 6 656 | −85 % |
| LUT | 49 733 | 14 462 | −70 % |
| Fmax | 137 MHz | 150 MHz | +9.5 % |

**Validation fonctionnelle** : campagne bit-exacte sur 14 signaux de test calibrés sur la
thèse de G. Albuquerque (protocole DANISH_SE) → **13/14 réussis (92.9 %)**, avec une
dégradation du pic de corrélation conforme à la loi théorique `(1−2p)²`.

---

## Organisation du dépôt

```
src/                      Sources HLS
├── acq_stsaV2_original.cpp    Architecture d'origine (référence)
├── acq_unified.cpp            Architecture "circuit pilote" (corrélateur partagé)
├── acq_blocks_pipeline.cpp    Architecture en 3 blocs synthétisables séparément
├── acq_blocks.h               Types partagés entre les 3 blocs
├── acq_stsaV2.h               Header commun (constantes, types virgule fixe)
└── dds_lut_rom.h              Table sinus/cosinus du DDS

testbench/                Bancs de test C++
├── test_acq_from_bin.cpp      Campagne automatisée sur les fichiers DANISH_SE
└── test_blocks.cpp            Test des 3 blocs, modes de décision A et B

tools/                    Génération des signaux de test (Python)
├── gps_signal_gen.py          Bibliothèque (code C/A, Doppler, bruit, quantification)
└── generate_baseline_sets.py  Génère les jeux DANISH_SE et INPE + manifest

test_signals/             Signaux de test synthétiques
├── DANISH_SE/                 14 fichiers, scénario terrestre 1 bit
├── INPE/                      10 fichiers, scénario spatial 2 bits
├── PRN_REF/                   Code PRN de référence
├── manifest_danish_se.csv     Manifest pour le testbench C++
└── metadata.json              Vérité terrain complète de chaque fichier

results/                  Résultats de campagnes de validation
report/                   Rapport de stage (LaTeX)
```

---

## Les trois architectures

| | Design original | Circuit pilote | 3 blocs séparés |
|---|---|---|---|
| Fichier | `acq_stsaV2_original.cpp` | `acq_unified.cpp` | `acq_blocks_pipeline.cpp` |
| Structure | Deux corrélateurs indépendants | Corrélateur partagé coarse/fine | Trois blocs synthétisés séparément |
| Seuil de décision | Variance normalisée | Seuil fixe uniquement | Les deux, au choix (paramètre) |
| Statut | Référence de départ | Validée (13/14 sur DANISH_SE) | Synthétisée, validation à refaire |

⚠️ `acq_unified.cpp` et `acq_stsaV2_original.cpp` définissent tous deux la fonction
`acquisition_stsa_top` — **ne jamais les compiler ensemble** (conflit de symbole).

### Architecture en 3 blocs

1. **`dds_mixer_top()`** — mélange DDS (Doppler × échantillons)
2. **`correlator_top()`** — corrélation signal × PRN, partagée coarse/fine
3. **`decision_top()`** — décision finale, avec paramètre `decision_mode` :
   - `DECISION_FIXED_THRESHOLD` : seuil fixe
   - `DECISION_VARIANCE_RELATIVE` : variance normalisée (méthode d'origine)

Les blocs communiquent par flux `hls::stream` et sont assemblés par
`acquisition_pipeline_top()` via `#pragma HLS DATAFLOW`.

---

## Utilisation

### Générer les signaux de test

```bash
cd tools
python3 generate_baseline_sets.py
```

Produit les jeux `DANISH_SE` (14 fichiers) et `INPE` (10 fichiers), plus le manifest et
les métadonnées de vérité terrain.

### Lancer la campagne de validation

```bash
export XILINX_HLS=/chemin/vers/Vitis_HLS/2023.2
g++ -std=c++17 -O2 -I$XILINX_HLS/include \
    testbench/test_acq_from_bin.cpp src/acq_unified.cpp -o test_acq
./test_acq test_signals/
```

Génère `resultats_danish_se.csv` avec, pour chaque fichier, le Doppler et la phase
détectés comparés aux valeurs injectées.

### Synthèse HLS

Sous Vitis HLS 2023.2, cible `xc7z020clg400-1` :
- Sources : les fichiers de `src/` correspondant à l'architecture choisie
- Top function : `acquisition_stsa_top` (monolithique), ou `dds_mixer_top` /
  `correlator_top` / `decision_top` pour une synthèse bloc par bloc

---

## Format des signaux de test

Flux binaire brut **int8** (1 octet = 1 échantillon signé).

| | DANISH_SE | INPE |
|---|---|---|
| PRN | 1 | 23 |
| Fréquence d'échantillonnage | 11.999 MHz | 16.368 MHz |
| Fréquence intermédiaire | 3.563 MHz | 4.092 MHz |
| Résolution | 1 bit (±1) | 2 bits (±1, ±3) |
| Doppler injecté | 3000 Hz | 42 500 Hz |
| Durée | 1 ms | 1 ms |
| Variable balayée | Sample Error 0 → 48 % | SNR 0 → 18 dB |

La convention de signe des codes C/A est alignée sur les fichiers PRN de référence du
projet TSA_GNSS (mapping `{0,1} → {−1,+1}`).

---

## Points ouverts

- **Seuil de décision par variance** : l'architecture `acq_unified.cpp` ne calcule pas la
  métrique de variance et ne peut donc pas rejeter fiablement l'absence de satellite.
  Réintroduit comme option dans l'architecture en 3 blocs, à valider.
- **Timing sur `correlator_top`** : violation résiduelle de −0.49 ns (le délai réel,
  7.79 ns, reste sous la période cible de 10 ns ; le dépassement vient de la marge
  conservative de l'outil). Sans impact fonctionnel, à revoir lors du placement-routage
  sous Vivado.
- **Scénario spatial INPE** : `N`, `FS` et `FREQUENCE_CENTRALE` sont des constantes de
  compilation fixées pour le scénario terrestre. Une resynthèse avec des constantes
  adaptées est nécessaire pour valider le scénario LEO 600 km.
- **Synthèse du wrapper complet** : les 3 blocs ont été synthétisés individuellement ;
  la synthèse de `acquisition_pipeline_top` assemblé reste à faire.

---

## Références

- G. L. A. Albuquerque, *Novel low complexity and power consumption GPS acquisition
  algorithm for space applications*, thèse de doctorat, UMONS.
- F. C. Silva et al., *Variance-Triggered Two-Step GPS Acquisition*, Sensors, 2019.
- G. L. A. Albuquerque et al., *Time-effective GPS Time Domain Signal Acquisition
  Algorithm*, IEEE, 2016.
