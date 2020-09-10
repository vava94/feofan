/**
 * Created by vok on 28.07.2020.
 *
 * TODO: Включить поддержку DLA
 * TODO: Включить профайлер
 * TODO: Деструктор
 * TODO: Loading labels from file
 */

#include "feofan.hpp"

#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <NvCaffeParser.h>
#include <NvOnnxParser.h>
#include <thread>
#include <NvUffParser.h>
#include <fstream>
#include <NvInferPlugin.h>

#ifdef __linux__
    #include <dlfcn.h>
    const char __libname[] = "./libneuraladapter.so";
#elif _WIN32
    #ifdef _MSC_VER
        #include <Windows.h>
        string libname = "neuraladapter.dll",
            lastLibAction;
        const char* dlerror();
    #else
        #include <dlfcn.h>
    #endif
#endif

#if !_HAS_CXX17
    #ifdef _MSC_VER
        #ifndef _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
            #define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
        #endif
    #endif
    #include <experimental/filesystem>
    namespace fs = experimental::filesystem;
#else
    #include <filesystem>
    namespace fs = filesystem;
#endif

#define NN_TAG "NN: "

/// --- Platform dependent functions

/**
 * Dynamic library loading function.
 * @param path Library path.
 * @return pointer to loaded lib.
 */
HINSTANCE loadLib(string path);

/**
 * The function of loading a symbol from the library.
 * @param lib - library pointer.
 * @param symbol - function name.
 * @return pointer to function.
 */
void* loadSymbol(HINSTANCE lib, const char* symbol);

/// ----------------------------------------------


Feofan::Feofan(function<void(string, int)> logCallback, 
    function<void(string)> neuralInfoCallback) 
{

    bindings = nullptr;
    cudaStream = nullptr;
    dlaCoresCount = 0;
    loaded = false;
    networksCount = 0;
    neuralResult = nullptr;
    readyCallback = nullptr;

    getLayerHeight = nullptr;
    getLayerWidth = nullptr;
    parseDetectionOutput = nullptr;
    setParam = nullptr;

    
    if (logCallback) {
        utils::log = move(logCallback);
        utils::log(NN_TAG "Loading neural framework.", 0);
    }
    if(neuralInfoCallback) {
        neuralInfo = move(neuralInfoCallback);
    }

#ifdef DYNAMIC_LINKING
    /// Загрузка вспомогательной библиотеки для разбора результатов сетей
    neuralAdapter = loadLib(libname.data());
    if(!neuralAdapter) {
        if (utils::log) {
            utils::log(NN_TAG "Unable to load neural network adapter. Detection is disabled.", 1);
            utils::log(NN_TAG + string(dlerror()), 2);
        }
        if (neuralInfo) neuralInfo("Detection is disabled.");
        return;
    }
    /// Получение информации о доступных обработчиках
    auto _getAdapters = (vector<string>(WINCALL*)())loadSymbol(neuralAdapter, "getAdapters");
    if(!_getAdapters) {
        if (utils::log) {
            utils::log(dlerror(), 2);
        }
    } else {
        availableAdapters = _getAdapters();
        if(availableAdapters.empty()) {
            if (utils::log) utils::log("The neural network adapter is empty. Detection is disabled.", 1);
            if (neuralInfo) neuralInfo("Detection is disabled.");
            parseDetectionOutput = nullptr;
            return;
        } else {
            string _adapterNetworks = NN_TAG + string("Network use available:");
            for(const auto& _adapter : availableAdapters) {
                _adapterNetworks += "<br>&nbsp;&nbsp;&nbsp;&nbsp;" + _adapter + "</br>";
            }
            if (utils::log) utils::log(_adapterNetworks, 1);
            parseDetectionOutput = ((int(WINCALL*)(void **, float **, const string& networkType))
                    loadSymbol(neuralAdapter, "parseDetectionOutput"));
            if(!parseDetectionOutput) {
                if (utils::log) utils::log(dlerror(), 2);
                return;
            }
            getLayerHeight = (size_t(WINCALL*)(nvinfer1::Dims, string))loadSymbol(neuralAdapter, "getLayerHeight");
            if(!getLayerHeight) {
                if (utils::log) utils::log(dlerror(), 2);
                return;
            }
            getLayerWidth = (size_t(WINCALL*)(nvinfer1::Dims, string))loadSymbol(neuralAdapter, "getLayerWidth");
            if(!getLayerWidth) {
                if (utils::log) utils::log(dlerror(), 2);
                return;
            }
            setParam = (void(WINCALL*)(string, string))loadSymbol(neuralAdapter, "setParam");
            if(!setParam) {
                if (utils::log) utils::log(dlerror(), 2);
                return;
            }
        }
    }
#endif // DYNAMIC_LINKING

    /// Получение информации о доступных форматах точности устройства
    string _fastestPrecision;
    auto _builder = nvinfer1::createInferBuilder(cudaLogger);
    dlaCoresCount = _builder->getNbDLACores();
    if (dlaCoresCount) {
        if (utils::log) utils::log(NN_TAG "The device has support for DLA cores.", 1);
    }
    if (_builder->platformHasFastInt8()) {
        availablePrecisions.emplace_back(INT8_PT);
        _fastestPrecision = "INT8";
        if (utils::log) utils::log(NN_TAG "Available percision: INT8.", 1);
    }
    if (_builder->platformHasFastFp16()) {
        availablePrecisions.emplace_back(FP16_PT);
        if(_fastestPrecision.empty()) {
            _fastestPrecision = "FP16";
            if (utils::log) utils::log(NN_TAG "Available percision: FP16.", 1);
        }
    }
    availablePrecisions.emplace_back(FP32_PT);
    if (utils::log) utils::log(NN_TAG "Available percision: FP32.", 1);
    if(_fastestPrecision.empty()) {
        _fastestPrecision = "FP32";
    }
    if(neuralInfo) {
        infoString = "TensorRT: " + to_string(NV_TENSORRT_MAJOR) + "." + to_string(NV_TENSORRT_MINOR) + "." +
                     to_string(NV_TENSORRT_PATCH) + " | " + _fastestPrecision + " | ";
        neuralInfo(infoString + "  0.00 FPS");
    }
    loaded = true;

}

void Feofan::allocImage(int index, int width, int height, int channels) {

    if(!loaded) return;
    auto _image = new NeuralImage(index, width, height, channels);
    neuralImages.emplace(index, _image);

}

int Feofan::bindingsCount() const {
    /// TODO: Количество точек
    //if (!cudaEngine) return -1;
    return 4;
}

vector<string> Feofan::getAdapters() const{
    return availableAdapters;
}

vector<Feofan::layerInfo> Feofan::getInputsInfo() const {
    return inputs;
}

uint8_t *Feofan::getImage() const {
    return neuralImages.at(0)->data();
}

vector<Feofan::layerInfo> Feofan::getOutputsInfo() const {
    return outputs;
}

int Feofan::layersCount() const {
    /// TODO: Количество слоёв
    //if(!cudaEngine) return -1;
    return 0;
}

void Feofan::loadNetwork(string networkPath) {
    new thread(bind(&Feofan::neuralInit, this, placeholders::_1, placeholders::_1), networkPath);
}

void Feofan::newData(int index, uint8_t *data) {
    if(!loaded) return;
    neuralImages.at(index)->newData(data);
}

string Feofan::networkName() const {
    /// TODO: Имя сети
    //if (!cudaEngine) return "Сеть не загружена.";
    return "cudaEngine->getName()";
}

bool Feofan::neuralInit(string networkPath, const string& caffeProtoTxtPath) {
    
    /// TODO: add loaded network to the vector of network definitions
    
    ICudaEngine *_cudaEngine;
    string _enginePath;
    char *_engineStream;
    FILE *_cacheFile;
    size_t  _engineSize;

    if (utils::log) utils::log(NN_TAG "Loading network...", 0);
    if (utils::log) utils::log(NN_TAG "Path: " + networkPath, 0);

    /// Проверка типа файла. Если ".engine" - загрузка, иначе - парсинг и оптимизация.
    if(networkPath.length() < 7 || networkPath.compare(networkPath.length() - 7, 7, ".engine") != 0) {
        _enginePath = optimizeNetwork(networkPath, DeviceType::DEVICE_GPU, PrecisionType::FP32_PT);
    } else {
        _enginePath = networkPath;
    }

    if(_enginePath.empty()) {
        if (utils::log) utils::log("Error loading network: \"enginePath is empty\".", 2);
        return false;
    }

    IRuntime *_runtime = createInferRuntime(cudaLogger);

    if (!_runtime) {
        return false;
    }

    if ((_engineSize = fs::file_size(_enginePath.data())) == 0){
        return false;
    }
    _engineStream = (char*)malloc(_engineSize);

    _cacheFile = fopen(_enginePath.data(), "rb");
    if(fread(_engineStream, 1, _engineSize, _cacheFile) != _engineSize) {
        return false;
    }
    fclose(_cacheFile);

    if(cudaStreamCreateWithFlags(&cudaStream, cudaStreamDefault) != cudaSuccess) {
        return false;
    }

    _cudaEngine = _runtime->deserializeCudaEngine(_engineStream, _engineSize);
    if (!_cudaEngine) {
        return false;
    }

    executionContext = _cudaEngine->createExecutionContext();

    for(int _i = 0; _i < _cudaEngine->getNbBindings(); _i++) {
        layerInfo _layerInfo;
        void *cpuBind = nullptr;
        void *gpuBind = nullptr;
        Dims _dims = validateDims(_cudaEngine->getBindingDimensions(_i));
        /** TODO: Для onnx:
         *	    if( modelType == MODEL_ONNX )
         *	        inputDims = shiftDims(inputDims);   // change NCHW to CHW if EXPLICIT_BATCH set
         */
        size_t _bindSize = _cudaEngine->getMaxBatchSize() * sizeof(float);
        for (int _j = 0; _j < _dims.nbDims; _j++) {
            _bindSize = _bindSize * _dims.d[_j];
        }

        if (!utils::cudaAllocMapped((void**)&cpuBind, (void**)&gpuBind, _bindSize) ) {
            return false;
        }

        _layerInfo.CPU = (float *) cpuBind;
        _layerInfo.CUDA = (float *) gpuBind;
        _layerInfo.size = _bindSize;
        _layerInfo.name = _cudaEngine->getBindingName(_i);
        _layerInfo.dims = _dims;
        _layerInfo.binding = _cudaEngine->getBindingIndex(_layerInfo.name.data());
        if(_cudaEngine->bindingIsInput(_layerInfo.binding)) {
            inputs.emplace_back(_layerInfo);
            if (utils::log) utils::log(NN_TAG"Input layer: " + _layerInfo.name, 0);
        } else {
            outputs.emplace_back(_layerInfo);
            if (utils::log) utils::log(NN_TAG"Output layer: " + _layerInfo.name, 0);
        }
    }

    const size_t _bindingsSize = sizeof(void*) * _cudaEngine->getNbBindings();
    bindings = (void**)malloc(_bindingsSize);
    if(!bindings) {
        if (utils::log) utils::log(NN_TAG"Memory allocation error for the bindings. Size: " + to_string(_bindingsSize) + ".",2);
        return false;
    }
    memset(bindings, 0, _bindingsSize);
    for (const auto& _input: inputs) {
        bindings[_input.binding] = (void*)_input.CUDA;
    }
    int _n = 0;
    for(const auto& _output: outputs) {
        bindings[_output.binding] = (void*)_output.CUDA;
        _n ++;
    }

    if (utils::log) {
        char _memStr[20] = { 0 };
        sprintf(_memStr, "%.2f", _cudaEngine->getDeviceMemorySize() * 1.0 / pow(1024, 2));
        utils::log(NN_TAG + string("Neural network loaded: ") + string(_cudaEngine->getName()), 0);
        utils::log(NN_TAG + string("Number of layers: ") + to_string(_cudaEngine->getNbLayers()), 0);
        utils::log(NN_TAG + string("Number of bindings: ") + to_string(_cudaEngine->getNbBindings()), 0);
        utils::log(NN_TAG + string("Memory used: ") + _memStr + string(" MB"), 0);
    }

    if(readyCallback) {
        readyCallback();
    }

    /// ПОКА ТОЛЬКО YOLO 3
    setCurrentNetworkAdapter(string("Yolo v3"));

    return true;
}

string Feofan::optimizeNetwork(string modelPath, DeviceType deviceType, PrecisionType precisionType) {
    /**
     * TODO: автоматический поиск caffe.prototxt файла по имени caffe файла
     * TODO: оптимизация engine
     */

    auto _builder = createInferBuilder(cudaLogger);
    bool _isEngine = false;
    function<void()> _destroyParser;
    const auto _explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto _network = _builder->createNetworkV2(_explicitBatch);
    NetworkDefinition _networkDefinition;
    string _enginePath;
    ICudaEngine *_cudaEngine;

    /// Определение типа загружаемого файла
    /// ONNX
    if(modelPath.compare(modelPath.length() - 5, 5, ".onnx") == 0) {

        using namespace nvonnxparser;
        auto _onnxParser = createParser(*_network, cudaLogger);
        if (!_onnxParser->parseFromFile(modelPath.data(), 3)) {
            if (utils::log) utils::log(NN_TAG"Error parsing '.onnx' file.", 2);
            _onnxParser->destroy();
            return _enginePath;
        }
        _destroyParser = [_onnxParser]() {
            _onnxParser->destroy();
        };
        _enginePath = modelPath.substr(0,modelPath.length() - 5);
        auto _p = _enginePath.find_last_of('/') + 1;
        _networkDefinition.name = _enginePath.substr(_p,modelPath.length() - _p);
        _network->setName(_networkDefinition.name.data());
        _enginePath += "_onnx";

    }
    /// CAFFE
    else if(modelPath.compare(modelPath.length() - 6, 6, ".caffe") == 0) {

        /// Loading ".prototxt" file by model name
        string _prototxt = modelPath.substr(0, modelPath.length() - 6) + ".prototxt";

        if (!fs::exists(_prototxt)) {
            if (utils::log) utils::log(NN_TAG "File '" + _prototxt + "' not found.", 2);
            return "";
        }

        using namespace nvcaffeparser1;
        auto _caffeParser = createCaffeParser();
        const auto _blobNameToTensor = _caffeParser->parse(_prototxt.data(), modelPath.data(),
                                                           *_network, nvinfer1::DataType::kFLOAT);
        if(!_blobNameToTensor) {
            if (utils::log) utils::log(NN_TAG "Error parsing '.caffe' file.", 2);
            _caffeParser->destroy();
            shutdownProtobufLibrary();
            return "";
        }
        _destroyParser = [_caffeParser]() {
            _caffeParser->destroy();
            shutdownProtobufLibrary();
        };
        _enginePath = modelPath.substr(0,modelPath.length() - 6);
        auto _p = _enginePath.find_last_of('/') + 1;
        _networkDefinition.name = _enginePath.substr(_p,modelPath.length() - _p);
        _network->setName(_networkDefinition.name.data());
        _enginePath += "_caffe";

    }
    /// UFF
    else if(modelPath.compare(modelPath.length() - 4, 4, ".uff") == 0) {

        using namespace nvuffparser;
        auto _uffParser = createUffParser();
        if (!_uffParser->parse(modelPath.data(), *_network)){
            if (utils::log) utils::log(NN_TAG"Error parsung '.uff' file.", 2);
            shutdownProtobufLibrary();
            _uffParser->destroy();
            return _enginePath;
        }
        _destroyParser = [_uffParser]() {
            _uffParser->destroy();
            shutdownProtobufLibrary();
        };
        _enginePath = modelPath.substr(0,modelPath.length() - 4);
        auto _p = _enginePath.find_last_of('/') + 1;
        _networkDefinition.name = _enginePath.substr(_p,modelPath.length() - _p);
        _network->setName(_networkDefinition.name.data());
        _enginePath += "_uff";

    }
        /// ENGINE
    else if(modelPath.compare(modelPath.length() - 7, 7, ".engine") == 0) {
        _isEngine = true;
        _enginePath = modelPath;
    }
        /// Underfined
    else {
        if (utils::log) utils::log(NN_TAG"Unsupported model type. Supported file types:"
            "<br>&nbsp;&nbsp;&nbsp;&nbsp;caffe</br>"
            "<br>&nbsp;&nbsp;&nbsp;&nbsp;engine</br>"
            "<br>&nbsp;&nbsp;&nbsp;&nbsp;onnx</br>"
            "<br>&nbsp;&nbsp;&nbsp;&nbsp;uff</br>", 2);
        return _enginePath;
    }
    /// Запись в имя файла информации о точности
    if(!_isEngine) {
        _enginePath += "_" + to_string(NV_TENSORRT_MAJOR) + to_string(NV_TENSORRT_MINOR) +
                       to_string(NV_TENSORRT_PATCH);
        if (precisionType == FP32_PT ||
           (precisionType == FASTEST_PT && availablePrecisions[0] == FP32_PT))
        {
            _enginePath += "_FP32";
        }
        else if (precisionType == FP16_PT ||
                (precisionType == FASTEST_PT && availablePrecisions[0] == FP16_PT))
        {
            _enginePath += "_FP16";
        }
        else if (precisionType == INT8_PT ||
                (precisionType == FASTEST_PT && availablePrecisions[0] == INT8_PT)) {
            _enginePath += "_INT8";
        }
        else
        {
            if (utils::log) utils::log("Can't set percision.", 2);
            return "";
        }
        _enginePath += "_GPU";
        _enginePath += ".engine";

    }

    /// Созданипе engine файла
    if(!_isEngine && !fs::exists(_enginePath)) {

        auto _config = _builder->createBuilderConfig();
        /// Выбор оптимизации
        if(availablePrecisions[0] == FP16_PT) {
            _config->setFlag(nvinfer1::BuilderFlag::kFP16);
        } else if(availablePrecisions[0] == INT8_PT) {
            _config->setFlag(nvinfer1::BuilderFlag::kINT8);
        }

        /// Проверка и подключение DLA
        if(_builder->getNbDLACores() > 0) {
            _config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
            _config->setDLACore(0);
            _config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);

        } else {
            _config->setInt8Calibrator(nullptr);
            _config->setDefaultDeviceType(nvinfer1::DeviceType::kGPU);
        }

        _config->setMaxWorkspaceSize(fs::file_size(modelPath));
        _cudaEngine = _builder->buildEngineWithConfig(*_network, *_config);

        if(_cudaEngine) {
            auto _iHostMemory = _cudaEngine->serialize();
            ofstream _ofStream(_enginePath.data(), ios::binary);
            _ofStream.write((const char *)_iHostMemory->data(), _iHostMemory->size());
        } else {
            if (utils::log) utils::log("Engine creation error.", 2);
        }

    }

    _destroyParser();
    _network->destroy();
    _builder->destroy();
    return _enginePath;
}

void Feofan::processAll() {
    for (int _i = 0; _i < neuralImages.size(); _i ++) {
        processData(_i);
    }
}

void Feofan::processData(int index) {
    int
        _inputWidth = getLayerWidth(inputs[0].dims, currentNetworkAdapter),
        _inputHeight = getLayerHeight(inputs[0].dims, currentNetworkAdapter);
    neuralImages[index]->prepareTensor(cudaStream,
                                       inputs[0].CUDA,
                                       _inputWidth,
                                       _inputHeight,
                                       256.f);
    executionContext->enqueueV2(bindings, cudaStream, nullptr);
    cudaStreamSynchronize(cudaStream);

    auto _numDetections = parseDetectionOutput(&bindings[1], &neuralResult, currentNetworkAdapter);
    float
        _scaleX = (float)neuralImages[index]->width() / (float)_inputWidth,
        _scaleY = (float)neuralImages[index]->height() / (float)_inputHeight;
    double _cx, _cy, _cz;
    dim3 _color;
    if(_numDetections  == 0) return;
    auto _detSize = sizeof(float) * 6;
    char percent[6];
    for (int _i = 0; _i < _numDetections; _i ++) {

        if(neuralResult[_i * _detSize + 5] > 0.3) {

            _cx = sin(neuralResult[_i * _detSize + 4] * 4.5 * 3.14 / 180);
            _cy = sin((neuralResult[_i * _detSize + 4] + 26) * 4.5 * 3.14 / 180);
            _cz = sin((neuralResult[_i * _detSize + 4] + 53) * 4.5 * 3.14 / 180);
            if (_cx < 0) _cx += 1;
            if (_cy < 0) _cy += 1;
            if (_cz < 0) _cz += 1;
            _color.x = (unsigned int) (_cx * 255.);
            _color.y = (unsigned int) (_cy * 255.);
            _color.z = (unsigned int) (_cz * 255.);
            sprintf(percent, "%.2f", neuralResult[_i * _detSize + 5] * 100);
            neuralImages[index]->cudaDrawBox(
                    (int)(neuralResult[_i * _detSize] * _scaleX),
                    (int)(neuralResult[_i * _detSize + 2] * _scaleX),
                    (int)(neuralResult[_i * _detSize + 1] * _scaleY),
                    (int)(neuralResult[_i * _detSize + 3] * _scaleY),
                    _color,
                    cudaStream,
                    3,
                    cocoLabels[neuralResult[_i * _detSize + 4]] + " " +
                    percent + "%");
        }
    }
    cudaStreamSynchronize(cudaStream);

}

void Feofan::selectBinding(string name, bool isInput) {
    /// TODO
}

void Feofan::setCurrentNetworkAdapter(string networkAdapter) {
    if (utils::log) utils::log(NN_TAG "Selected neural adapter for " + networkAdapter, 0);
    currentNetworkAdapter = move(networkAdapter);
    setParam(currentNetworkAdapter + ".InputSize", to_string(getLayerWidth(inputs[0].dims, currentNetworkAdapter)));
}

#ifndef DYNAMIC_LINKING

void Feofan::setGetLayerHeightCallback(function<size_t(nvinfer1::Dims dims, string networkType)> callback) {
    getLayerHeight = callback;
}

void Feofan::setGetLayerWidthCallback(function<size_t(nvinfer1::Dims dims, string networkType)> callback) {
    getLayerWidth = callback;
}
#endif // DYNAMIC_LINKING

void Feofan::setNetworkReadyCallback(function<void()> callback) {
    readyCallback = move(callback);
}

inline nvinfer1::Dims Feofan::validateDims( const nvinfer1::Dims& dims ) {
    if (dims.nbDims == nvinfer1::Dims::MAX_DIMS) {
        return dims;
    }
    nvinfer1::Dims _outDims = dims;
    for (int _i = _outDims.nbDims; _i < nvinfer1::Dims::MAX_DIMS; _i++) {
        _outDims.d[_i] = 1;
    }
    return _outDims;
}

#ifndef DYNAMIC_LINKING

void Feofan::setSetParamCallback(function<void(string paramName, string value)> callback) {
    setParam = callback;
}
#endif

Feofan::~Feofan() {

}


/// Вспомогательные функции загрузки библиотек
HINSTANCE loadLib(string path) {

#ifdef __linux__
    return dlopen(path.data(), RTLD_NOW);
#elif _WIN32
    lastLibAction = "load lib " + string(path);
    wstring wc(path.length(), L'#');
    mbstowcs(&wc[0], path.data(), path.length());
    return  LoadLibraryW(wc.data());
#endif

}

void* loadSymbol(HINSTANCE lib, const char *symbol) {

#ifdef _MSC_VER
    lastLibAction = "load symbol " + string(symbol);
    auto _ls = GetProcAddress(lib, symbol);
    return (void*)_ls;
#else
    return dlsym(lib, symbol);
#endif

}

#ifdef _MSC_VER
const char * dlerror() {
    string _errorString = "Error at action \"" + lastLibAction + "\"";
    return move(_errorString.data());
}
#endif
