#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xphoto/bm3d_image_denoising.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cstdio>

// ============================================================================
//                          BM3D PARAMETERS
// (defaults only: the values actually used are passed from config.json)
// ============================================================================

static float PARAM_SIGMA_DEFAULT = 10000.0f;
static float H_SCALE_16U = 1.0f;

static int BLOCK_STEP        = 2;     // >2 introduces visible banding
static int SEARCH_WINDOW     = 16;
static int NUM_SIMILAR_BLOCKS= 1200;
static int MATCHING_SHRINK   = 1;
static int GROUP_SIZE        = 256;
static int SLIDING_STEP      = 1;

static float BETA         = 0.001f;   // values around 5 produce a black output
static int   WAVELET      = cv::xphoto::HAAR;
static int   NORM_TYPE_16U= cv::NORM_L1;

static bool INVERT_16U_OUTPUT = true;

// ============================================================================

// Command-line parameter overrides: --key=value.
// This way bm3d.py can pass the values from config.json without recompiling.
// Accepted keys = names of the static variables above.
static void applyCliOverrides(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--", 0) != 0) continue;          // not an override
        auto eq = a.find('=');
        if (eq == std::string::npos) continue;
        std::string key = a.substr(2, eq - 2);
        std::string val = a.substr(eq + 1);

        try {
            if      (key == "PARAM_SIGMA_DEFAULT") PARAM_SIGMA_DEFAULT = std::stof(val);
            else if (key == "H_SCALE_16U")         H_SCALE_16U         = std::stof(val);
            else if (key == "BLOCK_STEP")          BLOCK_STEP          = std::stoi(val);
            else if (key == "SEARCH_WINDOW")       SEARCH_WINDOW       = std::stoi(val);
            else if (key == "NUM_SIMILAR_BLOCKS")  NUM_SIMILAR_BLOCKS  = std::stoi(val);
            else if (key == "MATCHING_SHRINK")     MATCHING_SHRINK     = std::stoi(val);
            else if (key == "GROUP_SIZE")          GROUP_SIZE          = std::stoi(val);
            else if (key == "SLIDING_STEP")        SLIDING_STEP        = std::stoi(val);
            else if (key == "BETA")                BETA                = std::stof(val);
            else if (key == "INVERT_16U_OUTPUT")   INVERT_16U_OUTPUT   = (val == "1" || val == "true");
            else {
                std::cerr << "[WARN] Unknown CLI parameter ignored: " << key << "\n";
                continue;
            }
            std::cout << "[CLI] " << key << " = " << val << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[WARN] Invalid value for " << key << ": '" << val << "'\n";
        }
    }
}

// ============================================================================

static void printDepth(const cv::Mat& m) {
    const char* d =
        (m.depth()==CV_8U  ? "CV_8U"  :
         m.depth()==CV_16U ? "CV_16U" :
         m.depth()==CV_16S ? "CV_16S" :
         m.depth()==CV_32F ? "CV_32F" : "OTHER");

    std::cerr << "[INFO] depth=" << d
              << " channels=" << m.channels()
              << " size=" << m.cols << "x" << m.rows << "\n";
}

// ============================================================================

int main(int argc, char* argv[]) {

    // Default values (relative to the working directory)
    std::string inputDir  = "data/input_tiff";
    std::string outputDir = "data/denoised_tiff";
    int startIdx = 1;
    int endIdx   = 11;

    // Positional arguments: only those NOT starting with "--".
    // Order: inputDir outputDir [startIdx endIdx]. --key=value flags can go anywhere.
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--", 0) != 0) pos.push_back(a);
    }
    if (pos.size() >= 2) {
        inputDir  = pos[0];
        outputDir = pos[1];
    }
    if (pos.size() >= 4) {
        startIdx = std::stoi(pos[2]);
        endIdx   = std::stoi(pos[3]);
    }

    // Optional BM3D parameter overrides from the CLI (--key=value).
    applyCliOverrides(argc, argv);

    std::cout << "[INFO] Input dir:  " << inputDir  << "\n";
    std::cout << "[INFO] Output dir: " << outputDir << "\n";
    std::cout << "[INFO] Slice range: " << startIdx << " → " << endIdx << "\n\n";

    for (int i = startIdx; i <= endIdx; ++i) {

        char inputPath[1024];
        char outputPath[1024];

        std::snprintf(inputPath,  sizeof(inputPath),
                      "%s/proj_%03d.tif", inputDir.c_str(), i);
        std::snprintf(outputPath, sizeof(outputPath),
                      "%s/proj_%03d.tif", outputDir.c_str(), i);

        std::cout << "[INFO] Processing " << inputPath << "\n";

        cv::Mat src = cv::imread(inputPath, cv::IMREAD_UNCHANGED);
        if (src.empty()) {
            std::cerr << "[ERROR] Cannot read " << inputPath << "\n";
            continue;
        }

        printDepth(src);

        try {
            float h = PARAM_SIGMA_DEFAULT * H_SCALE_16U;

            cv::Mat den;
            cv::xphoto::bm3dDenoising(
                src, den, h,
                BLOCK_STEP,
                SEARCH_WINDOW,
                NUM_SIMILAR_BLOCKS,
                MATCHING_SHRINK,
                GROUP_SIZE,
                SLIDING_STEP,
                BETA,
                NORM_TYPE_16U,
                cv::xphoto::BM3D_STEPALL,
                WAVELET
            );

            if (INVERT_16U_OUTPUT)
                den = 65535 - den;

            cv::imwrite(outputPath, den);
            std::cout << "[OK] Saved: " << outputPath << "\n\n";

        } catch (const std::exception& e) {
            std::cerr << "[BM3D ERROR] " << e.what() << "\n";
        }
    }

    std::cout << "[INFO] Batch done\n";
    return 0;
}
