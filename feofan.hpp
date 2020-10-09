    /**
 * Created by vava94 on 28.07.2020.
 * 
 * A neural network framework class based on the TensorRT library.
 * 
 * Supported file formats: ".onnx", ".uff" & CUDA engine (TRT native files).
 * Supported UNIX & Windows compilers.
 * 
 * For dynamic linking example requires "neural-adapter" library: 
 *      https://github.com/vava94/neural-adapter
 * Neural-adapter provides parsing of neural network output data and translating them into
 * a format convenient for display.
 * 
 * Please rerfer to https://github.com/vava94/feofan if using.
 * 
 */

/** Commit

Чистка кода.
Поддержка загрузки шрифтов.

*/

#ifndef FEOFAN
#define FEOFAN

#include "neuralimage.hpp"
#include "utils.hpp"

#include <cstdlib>
#include <map>
#include <memory>
#include <string>

#ifdef _WIN32
#include "Windows.h"
#endif

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
        std::string filePath, enginePath;
        std::string name;
        DeviceType deviceType;
        IExecutionContext *executionContext{nullptr};
        void **bindings;
    };

    enum class ModelFormat    {
        CAFFE,
        ONNX,
        UFF
    };


    std::vector<std::string> cocoLabels = {
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
    cudaStream_t cudaStream;
    std::function<void()> readyCallback;
    
    std::function<void(std::string)> neuralInfo;

    IExecutionContext *executionContext{nullptr};
    int dlaCoresCount, networksCount;
    std::map<int, NeuralImage*> neuralImages;
    std::string infoString, currentNetworkAdapter;
    std::vector<layerInfo> inputs, outputs;
    std::vector<NetworkDefinition> networkDefinitions;
    std::vector<PrecisionType> availablePrecisions;
    std::vector<std::string> availableAdapters;
    void **bindings;
    HINSTANCE neuralAdapter;

#ifdef  DYNAMIC_LINKING
    /// Neural-adapter library functions
    size_t(WINCALL* getLayerHeight)(nvinfer1::Dims dims, std::string networkType);
    size_t(WINCALL* getLayerWidth)(nvinfer1::Dims dims, std::string networkType);
    int (WINCALL* parseDetectionOutput)(void** output, float** parsed, const std::string& networkType);
    void (WINCALL* setParam)(std::string paramName, std::string value);
#else
    function<size_t(nvinfer1::Dims dims, std::string networkType)> getLayerHeight, getLayerWidth;
    function<int(void** output, float** parsed, const std::string& networkType)> parseDetectionOutput;
    function<void(std::string paramName, std::string value)> setParam;
#endif //  DYNAMIC_LINKING



    static inline Dims validateDims( const Dims& dims );

public:

    /**
    * @param logCallback            Функция обратного вызова для лога
    * @param neuralStatusCallback   Функция обратного вызова для информации о информации фрэймворка по работе нейронной сети
    */
    explicit Feofan(std::function<void(std::string, int)> logCallback = nullptr, 
        std::function<void(std::string)> neuralInfoCallback = nullptr);
    /**
    * Destructor.
    */
    ~Feofan();
    /**
    * The function reserves memory for data for further work with the neural network.
    * 
    * @param index - 'ID' of image. Number of source for example.
    * @param width - image width.
    * @param height - image height.
    * @param channels - number of color channels in the image.
    * 
    */
    void allocImage(int index, int width, int height, int channels);
    void applyDetections(int index);
    [[nodiscard]] int bindingsCount() const;
    [[nodiscard]] std::vector<std::string> getAdapters() const;
    [[nodiscard]] std::vector<layerInfo> getInputsInfo() const;
    [[nodiscard]] uint8_t *getImage() const;
    [[nodiscard]] std::vector<layerInfo> getOutputsInfo() const;
    [[nodiscard]] int layersCount() const;
    /**
    * The function loads a true type font file for name captions in detection mode.
    * 
    * @param ttfFont - font file.
    * 
    */
    void loadFont(FILE* ttfFont);
    /**
    * The function loads the neural network graph file.
    * 
    * @param networkPath - path to the network graph file.
    * 
    */
    void loadNetwork(std::string networkPath);
    [[nodiscard]] std::string networkName() const;
    void newData(int index, uint8_t *data);
    bool neuralInit(std::string networkPath, const std::string& caffeProtoTxtPath = "");
    //[[nodiscard]] std::string precisionTypeName() const;
    std::string optimizeNetwork(std::string networkPath, DeviceType deviceType, PrecisionType precisionType);
    /**
    * The function processes all data loaded into the framework.
    */
    void processAll();
    /**
    * The function processes data for a specific index.
    */
    void processData(int index);
    void setCurrentNetworkAdapter(std::string networkAdapter);
    void setNetworkReadyCallback(std::function<void()> callback);

#ifndef DYNAMIC_LINKING
    /**
    * Set a callback for a function that returns the height of the input layer.
    * 
    * @param callback - callback. Callback function requires layer dims and network name.
    */
    void setGetLayerHeightCallback(function<size_t(Dims dims, string networkType)> callback);
    /**
    * Set a callback for a function that returns the width of the input layer.
    *
    * @param callback - callback. Callback function requires layer dims and network name.
    */
    void setGetLayerWidthCallback(function<size_t(nvinfer1::Dims dims, std::string networkType)> callback);
    /**
    * Set a callback for a function that sets parameters to the neural network output parser.
    *
    * @param callback - callback. Callback function requires param name and value.
    */
    void setSetParamCallback(function<void(std::string paramName, std::string value)>);
#endif // !DYNAMIC_LINKING


};

#endif // !FEOFAN