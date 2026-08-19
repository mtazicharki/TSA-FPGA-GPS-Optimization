//==============================================================================
// test_acq_from_bin.cpp
//
// Testbench de simulation C (csim, hors HLS) pour acquisition_stsa_top.
// Lit un manifest CSV (genere par generate_baseline_sets.py) decrivant une
// serie de fichiers .bin (signal RX quantifie) + un fichier PRN de reference,
// appelle acquisition_stsa_top() pour chacun, et compare le Doppler/la phase
// detectes au "ground truth" (valeurs injectees lors de la generation).
//
// Compilation (depuis le dossier contenant acq_unified.cpp, acq_stsaV2.h,
// dds_lut_rom.h et ce fichier) :
//
//   g++ -std=c++17 -O2 \
//       -I$XILINX_HLS/include \
//       test_acq_from_bin.cpp acq_unified.cpp -o test_acq
//
// $XILINX_HLS doit pointer vers l'installation Vitis HLS 2023.2, par exemple :
//   export XILINX_HLS=/tools/Xilinx/Vitis_HLS/2023.2
//
// Usage :
//   ./test_acq <chemin_vers_gps_test_signals>
//
// (le manifest attendu est <chemin>/manifest_danish_se.csv)
//==============================================================================
#include "acq_stsaV2.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Utilitaires
// ---------------------------------------------------------------------------

// Lit un fichier binaire brut int8 et retourne les valeurs en double.
static std::vector<double> read_bin_int8(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "ERREUR: impossible d'ouvrir " << path << std::endl;
        std::exit(1);
    }
    std::vector<char> raw((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    std::vector<double> out;
    out.reserve(raw.size());
    for (char c : raw) {
        out.push_back(static_cast<double>(static_cast<int8_t>(c)));
    }
    return out;
}

static axis_t make_axis(int32_t value, bool last) {
    axis_t pkt;
    pkt.data = value;
    pkt.keep = -1;
    pkt.strb = -1;
    pkt.last = last;
    return pkt;
}

struct ManifestRow {
    std::string file;
    std::string prn_ref_file;
    int prn;
    double fs_hz;
    double if_hz;
    int adc_bits;
    double doppler_hz;
    double duration_ms;
    double se_pct;
    int n_samples;
};

static std::vector<ManifestRow> read_manifest(const std::string &path) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "ERREUR: impossible d'ouvrir le manifest " << path << std::endl;
        std::exit(1);
    }
    std::vector<ManifestRow> rows;
    std::string line;
    std::getline(f, line);  // en-tete, ignoree

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string field;
        ManifestRow row;

        std::getline(ss, row.file, ',');
        std::getline(ss, row.prn_ref_file, ',');
        std::getline(ss, field, ','); row.prn = std::stoi(field);
        std::getline(ss, field, ','); row.fs_hz = std::stod(field);
        std::getline(ss, field, ','); row.if_hz = std::stod(field);
        std::getline(ss, field, ','); row.adc_bits = std::stoi(field);
        std::getline(ss, field, ','); row.doppler_hz = std::stod(field);
        std::getline(ss, field, ','); row.duration_ms = std::stod(field);
        std::getline(ss, field, ','); row.se_pct = std::stod(field);
        std::getline(ss, field, ','); row.n_samples = std::stoi(field);

        rows.push_back(row);
    }
    return rows;
}

// ---------------------------------------------------------------------------
// Execution d'un test unitaire
// ---------------------------------------------------------------------------

struct TestResult {
    int doppler_detected;
    int phase_detected;
    double peak;
    bool detected;
    int max_power;
    int mean_power;
    int doppler_error_hz;   // detecte - injecte
};

static TestResult run_one_test(const std::vector<double> &rx_samples,
                                const std::vector<double> &prn_samples) {
    if ((int)rx_samples.size() != N || (int)prn_samples.size() != N) {
        std::cerr << "ERREUR: taille de signal (" << rx_samples.size()
                   << ") ou PRN (" << prn_samples.size()
                   << ") differente de N=" << N << std::endl;
        std::exit(1);
    }

    hls::stream<axis_t> rx_stream;
    hls::stream<axis_t> prn_stream;
    hls::stream<axis_t> corr_out;

    for (int i = 0; i < N; i++) {
        rx_stream.write(make_axis((int32_t)rx_samples[i], i == N - 1));
        prn_stream.write(make_axis((int32_t)prn_samples[i], i == N - 1));
    }

    int doppler_out = 0, phase_out = 0;
    power_t peak_out = 0;
    bool detected_out = false;
    int max_power_out = 0, mean_power_out = 0;

    acquisition_stsa_top(
        rx_stream, prn_stream, corr_out,
        doppler_out, phase_out, peak_out, detected_out,
        max_power_out, mean_power_out
    );

    // Vide corr_out (pas exploite ici, juste pour eviter tout etat residuel).
    while (!corr_out.empty()) {
        corr_out.read();
    }

    TestResult r;
    r.doppler_detected = doppler_out;
    r.phase_detected = phase_out;
    r.peak = (double)peak_out;
    r.detected = detected_out;
    r.max_power = max_power_out;
    r.mean_power = mean_power_out;
    return r;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <chemin_vers_gps_test_signals>"
                   << std::endl;
        return 1;
    }
    std::string base_dir = argv[1];
    std::string manifest_path = base_dir + "/manifest_danish_se.csv";

    std::cout << "N attendu par le design HLS = " << N
              << " | FS = " << FS
              << " | FREQUENCE_CENTRALE = " << FREQUENCE_CENTRALE << std::endl;

    std::vector<ManifestRow> rows = read_manifest(manifest_path);
    std::cout << rows.size() << " fichiers a tester (DANISH_SE)." << std::endl;

    std::ofstream out_csv("resultats_danish_se.csv");
    out_csv << "file,se_pct,doppler_injecte,doppler_detecte,erreur_doppler_hz,"
               "phase_detectee,peak,detected,max_power,mean_power\n";

    int n_ok = 0;

    for (const auto &row : rows) {
        std::vector<double> rx = read_bin_int8(base_dir + "/" + row.file);
        // Le PRN de reference est le meme pour toutes les lignes de ce
        // manifest (toujours PRN1), on le recharge simplement a chaque
        // iteration pour rester generique si un manifest multi-PRN arrive.
        std::vector<double> prn = read_bin_int8(base_dir + "/" + row.prn_ref_file);

        TestResult res = run_one_test(rx, prn);
        int err = res.doppler_detected - (int)row.doppler_hz;
        bool ok = res.detected && (std::abs(err) <= (int)DOPPLER_BIN_FINE);
        n_ok += ok ? 1 : 0;

        std::cout << row.file
                  << " | SE=" << row.se_pct << "%"
                  << " | Doppler inj=" << row.doppler_hz
                  << " det=" << res.doppler_detected
                  << " (erreur=" << err << " Hz)"
                  << " | phase=" << res.phase_detected
                  << " | detected=" << (res.detected ? "oui" : "non")
                  << " | " << (ok ? "OK" : "ECHEC")
                  << std::endl;

        out_csv << row.file << "," << row.se_pct << ","
                << row.doppler_hz << "," << res.doppler_detected << ","
                << err << "," << res.phase_detected << "," << res.peak << ","
                << (res.detected ? 1 : 0) << "," << res.max_power << ","
                << res.mean_power << "\n";
    }

    std::cout << "\n" << n_ok << "/" << rows.size()
              << " tests reussis (Doppler detecte a +/- 1 bin fin pres)."
              << std::endl;
    std::cout << "Resultats detailles ecrits dans resultats_danish_se.csv"
              << std::endl;

    return 0;
}
