/**
 * Created by vitaliy.kiselyov on 28.07.2020.
 * 
 * Класс фрэймворка нейронных сетей, основанный на библиотеке TensorRT.
 * Поддерживается загрузка нейронных сетей в формате ".onnx", ".uff" и CUDA engine файлов.
 * 
 * Для работы фрэймворка также необходима библиотека "neural-adapter": 
 *      https://github.com/vava94/neural-adapter
 * Neural-adapter обеспечивает разбор выходных данных нейронной сети и перевод их в удобный для отображения формат.
 * 
 */

#ifndef FEOFAN
#define FEOFAN

#include "neuralimage.hpp"

#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <NvInfer.h>
#include <string>

#ifdef _WIN32
#include "Windows.h"
#endif


 /// Внешняя функция из main
 void log(const std::string&, unsigned char level);

using namespace std;
using namespace nvinfer1;

class Feofan {

private:
#ifdef _WIN32
#ifdef _MSC_VER
#define WINCALL __cdecl
#endif // _MVS_VER
#elif __linux__
    typedef void* HINSTANCE;
#define WINCALL
#endif

    class CudaLogger : public ILogger
    {
        void log(Severity severity, const char* msg) override
        {
            switch(severity) {
                case nvinfer1::ILogger::Severity::kINFO:
                    ::log(std::string("CUDA: ") + msg, 0);
                    break;
                case nvinfer1::ILogger::Severity::kWARNING:
                    ::log(std::string("CUDA: ") + msg, 1);
                    break;
                case nvinfer1::ILogger::Severity::kERROR:
                    ::log(std::string("CUDA: ") + msg, 2);
                    break;
                case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                    ::log(std::string("CUDA: ") + msg, 2);
                    break;
                case Severity::kVERBOSE:
                    ::log(std::string("CUDA: ") + msg, 0);
                    break;
            }
        }
    } cuLogger;

    enum PrecisionType
    {
        FP32_PT = 0,		    /// 32-bit floating-point precision
        FP16_PT,		        /// 16-bit floating-point half precision
        INT8_PT,		        /// 8-bit integer precision
        FASTEST_PT,             /// Позволяет оптимизатору самому выбрать точность
        PRECISIONS_COUNT	/// Number of precision types defined
    };
    enum DeviceType
    {
        DEVICE_GPU = 0,			/**< GPU (if multiple GPUs are present, a specific GPU can be selected with cudaSetDevice() */
        DEVICE_DLA,				/**< Deep Learning Accelerator (DLA) Core 0 (only on Jetson Xavier) */
        DEVICE_DLA_0 = DEVICE_DLA,	/**< Deep Learning Accelerator (DLA) Core 0 (only on Jetson Xavier) */
        DEVICE_DLA_1,				/**< Deep Learning Accelerator (DLA) Core 1 (only on Jetson Xavier) */
        NUM_DEVICES				/**< Number of device types defined */
    };
    struct layerInfo
    {
        std::string name;
        nvinfer1::Dims dims;
        uint32_t size;
        int binding;
        float* CPU;
        float* CUDA;
    };

    struct NetworkDefinition {
        int pos;
        string filePath;
        string name;
        DeviceType deviceType;
        IExecutionContext *executionContext{nullptr};
        void **bindings;
    };

    enum class ModelFormat    {
        CAFFE,
        ONNX,
        UFF
    };


    std::vector<string> cocoLabels = {
            "person", "bicycle", "car", "motorbike", "aeroplane",
            "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird",
            "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon",
            "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "sofa", "pottedplant", "bed",
            "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    bool loaded;
    float *neuralResult;
    cudaStream_t cudaStream;
    function<void()> readyCallback;
    
    function<void(string)> neuralInfo;
    function<void(string, int)> log;
    

    IExecutionContext *executionContext{nullptr};
    int dlaCoresCount, networksCount;
    map<int, NeuralImage*> neuralImages;
    NetworkDefinition *networkDefinitions;
    string infoString, currentNetworkAdapter;
    vector<layerInfo> inputs, outputs;
    vector<PrecisionType> availablePrecisions;
    vector<string> availableAdapters;
    void **bindings;
    HINSTANCE neuralAdapter;

    /// Функции библиотеки neuraladapter
    size_t (WINCALL *getLayerHeight)(nvinfer1::Dims dims, std::string networkType) = nullptr;
    size_t (WINCALL *getLayerWidth)(nvinfer1::Dims dims, std::string networkType) = nullptr;
    int (WINCALL *parseDetectionOutput)(void **output, float **parsed, const std::string& networkType) = nullptr;
    void (WINCALL *setParam)(std::string paramName, std::string value) = nullptr;

    static inline nvinfer1::Dims validateDims( const nvinfer1::Dims& dims );

public:

    /**
    * @param logCallback            Функция обратного вызова для лога
    * @param neuralStatusCallback   Функция обратного вызова для информации о информации фрэймворка по работе нейронной сети
    */
    explicit Feofan(function<void(string, int)> logCallback = nullptr, 
        function<void(string)> neuralInfoCallback = nullptr);
    ~Feofan();
    void allocImage(int index, int width, int height, int channels);
    void applyDetections(int index);
    [[nodiscard]] int bindingsCount() const;
    [[nodiscard]] std::vector<std::string> getAdapters() const;
    [[nodiscard]] std::vector<layerInfo> getInputsInfo() const;
    [[nodiscard]] uint8_t *getImage() const;
    [[nodiscard]] std::vector<layerInfo> getOutputsInfo() const;
    [[nodiscard]] int layersCount() const;
    void loadNetwork(std::string networkPath);
    [[nodiscard]] std::string networkName() const;
    void newData(int index, uint8_t *data);
    bool neuralInit(std::string networkPath, const std::string& caffeProtoTxtPath = "");
    //[[nodiscard]] std::string precisionTypeName() const;
    string optimizeNetwork(string networkPath, DeviceType deviceType, PrecisionType precisionType);
    void processAll();
    void processData(int index);
    void selectBinding(std::string name, bool isInput);
    void setCurrentNetworkAdapter(std::string networkAdapter);
    void setNetworkReadyCallback(std::function<void()> callback);

};

#endif // !FEOFAN