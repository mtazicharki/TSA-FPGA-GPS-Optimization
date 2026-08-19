#include "acq_blocks.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>

void acquisition_pipeline_top(
    hls::stream<axis_t> &rx_stream,
    hls::stream<axis_t> &prn_stream,
    int &doppler_out,
    int &phase_out,
    power_t &peak_out,
    bool &detected_out,
    decision_mode_t decision_mode,
    metric_t threshold_coarse,
    metric_t threshold_fine
);

static std::vector<int8_t> read_bin(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<int8_t> out;
    for (char c : raw) out.push_back((int8_t)c);
    return out;
}

static axis_t make_axis(int32_t v, bool last) {
    axis_t p; p.data = v; p.keep = -1; p.strb = -1; p.last = last; return p;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: test_blocks <rx.bin> <prn.bin>\n"; return 1; }

    auto rx  = read_bin(argv[1]);
    auto prn = read_bin(argv[2]);
    std::cout << "N attendu=" << N << " rx=" << rx.size() << " prn=" << prn.size() << std::endl;

    for (int mode_i = 0; mode_i < 2; mode_i++) {
        decision_mode_t mode = (mode_i == 0) ? DECISION_FIXED_THRESHOLD : DECISION_VARIANCE_RELATIVE;
        std::cout << "\n=== Mode " << (mode_i==0 ? "B (seuil fixe)" : "A (variance)") << " ===" << std::endl;

        hls::stream<axis_t> rx_s, prn_s;
        for (int i = 0; i < N; i++) {
            rx_s.write(make_axis(rx[i], i==N-1));
            prn_s.write(make_axis(prn[i], i==N-1));
        }

        int doppler_out=0, phase_out=0;
        power_t peak_out=0;
        bool detected_out=false;
        metric_t thr_coarse = (metric_t)SEUIL_VARIANCE_COARSE;
        metric_t thr_fine   = (metric_t)SEUIL_VARIANCE_FINE;

        acquisition_pipeline_top(rx_s, prn_s, doppler_out, phase_out, peak_out,
                                  detected_out, mode, thr_coarse, thr_fine);

        std::cout << "Doppler detecte = " << doppler_out
                   << " | phase = " << phase_out
                   << " | peak = " << (double)peak_out
                   << " | detected = " << (detected_out ? "oui" : "non") << std::endl;
    }
    return 0;
}
