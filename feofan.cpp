/**
 * Created by vok on 28.07.2020.
 *
 * TODO: Включить поддержку DLA
 * TODO: Включить профайлер
 * TODO: Деструктор
 *
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
    namespace fs = std::experimental::filesystem;
#else
    #include <filesystem>
    namespace fs = std::filesystem;
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

    
    if (logCallback) {
        log = move(logCallback);
        log(NN_TAG "Загрузка нейронного фрэймворка.", 0);
    }
    if(neuralInfoCallback) {
        neuralInfo = move(neuralInfoCallback);
    }
#ifdef DYNAMIC_LINKING
    /// Загрузка вспомогательной библиотеки для разбора результатов сетей
    neuralAdapter = loadLib(libname.data());
    if(!neuralAdapter) {
        log(NN_TAG"Невозможно загрузить адаптер нейронных сетей. Детектирование отключено.", 1);
        neuralInfo("Детектирование отключено.");
        log(NN_TAG + std::string(dlerror()), 2);
        return;
    }
    /// Получение информации о доступных обработчиках
    auto _getAdapters = (vector<string>(WINCALL*)())loadSymbol(neuralAdapter, "getAdapters");
    if(!_getAdapters) {
        log(dlerror(), 2);
    } else {
        availableAdapters = _getAdapters();
        if(availableAdapters.empty()) {
            log("Адаптер нейронных сетей пуст. Детектирование невозможно.", 1);
            neuralInfo("Детектирование отключено.");
            parseDetectionOutput = nullptr;
            return;
        } else {
            std::string _adapterNetworks = NN_TAG + std::string("Доступно использование сетей:");
            for(const auto& _adapter : availableAdapters) {
                _adapterNetworks += "<br>&nbsp;&nbsp;&nbsp;&nbsp;" + _adapter + "</br>";
            }
            log(_adapterNetworks, 1);
            parseDetectionOutput = ((int(WINCALL*)(void **, float **, const std::string& networkType))
                    loadSymbol(neuralAdapter, "parseDetectionOutput"));
            if(!parseDetectionOutput) {
                log(dlerror(), 2);
                return;
            }
            getLayerHeight = (size_t(WINCALL*)(nvinfer1::Dims, std::string))loadSymbol(neuralAdapter, "getLayerHeight");
            if(!getLayerHeight) {
                log(dlerror(), 2);
                return;
            }
            getLayerWidth = (size_t(WINCALL*)(nvinfer1::Dims, std::string))loadSymbol(neuralAdapter, "getLayerWidth");
            if(!getLayerWidth) {
                log(dlerror(), 2);
                return;
            }
            setParam = (void(WINCALL*)(std::string, std::string))loadSymbol(neuralAdapter, "setParam");
            if(!setParam) {
                log(dlerror(), 2);
                return;
            }
        }
    }
#endif // DYNAMIC_LINKING

    /// Получение информации о доступных форматах точности устройства
    std::string _fastestPrecision;
    auto _builder = nvinfer1::createInferBuilder(cuLogger);
    dlaCoresCount = _builder->getNbDLACores();
    if (dlaCoresCount) {
        log(NN_TAG "Устройство имеет поддержку DLA ядер.", 1);
    }
    if (_builder->platformHasFastInt8()) {
        availablePrecisions.emplace_back(INT8_PT);
        _fastestPrecision = "INT8";
        log(NN_TAG "Доступна точность INT8.", 1);
    }
    if (_builder->platformHasFastFp16()) {
        availablePrecisions.emplace_back(FP16_PT);
        if(_fastestPrecision.empty()) {
            _fastestPrecision = "FP16";
            log(NN_TAG "Доступна точность FP16.", 1);
        }
    }
    availablePrecisions.emplace_back(FP32_PT);
    if(_fastestPrecision.empty()) {
        _fastestPrecision = "FP32";
    }
    if(neuralInfo) {
        infoString = "TensorRT: " + std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR) + "." +
                     std::to_string(NV_TENSORRT_PATCH) + " | " + _fastestPrecision + " | ";
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

std::vector<std::string> Feofan::getAdapters() const{
    return availableAdapters;
}

std::vector<Feofan::layerInfo> Feofan::getInputsInfo() const {
    return inputs;
}

uint8_t *Feofan::getImage() const {
    return neuralImages.at(0)->data();
}

std::vector<Feofan::layerInfo> Feofan::getOutputsInfo() const {
    return outputs;
}

int Feofan::layersCount() const {
    /// TODO: Количество слоёв
    //if(!cudaEngine) return -1;
    return 0;
}

void Feofan::loadNetwork(std::string networkPath) {
    new std::thread(std::bind(&Feofan::neuralInit, this, std::placeholders::_1, std::placeholders::_1), networkPath);
}

void Feofan::newData(int index, uint8_t *data) {
    if(!loaded) return;
    neuralImages.at(index)->newData(data);
}

std::string Feofan::networkName() const {
    /// TODO: Имя сети
    //if (!cudaEngine) return "Сеть не загружена.";
    return "cudaEngine->getName()";
}

bool Feofan::neuralInit(std::string networkPath, const std::string& caffeProtoTxtPath) {
    
    /// TODO: add loaded network to the vector of network definitions
    
    ICudaEngine *_cudaEngine;
    string _enginePath;
    char *_engineStream;
    FILE *_cacheFile;
    size_t  _engineSize;

    if (log) log(NN_TAG "Loading network...", 0);
    if (log) log(NN_TAG "Path: " + networkPath, 0);

    /// Проверка типа файла. Если ".engine" - загрузка, иначе - парсинг и оптимизация.
    if(networkPath.compare(networkPath.length() - 7, 7, ".engine") != 0) {
        _enginePath = optimizeNetwork(networkPath, DeviceType::DEVICE_GPU, PrecisionType::FP32_PT);
    } else {
        _enginePath = networkPath;
    }

    if(_enginePath.empty()) {
        if (log) log("Error loading network: \"enginePath is empty\".", 2);
        return false;
    }

    IRuntime *_runtime = createInferRuntime(cuLogger);

    if (!_runtime) {
        return false;
    }

    if ( (_engineSize = fs::file_size(_enginePath.data())) == 0){
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

        if( !cudaAllocMapped((void**)&cpuBind, (void**)&gpuBind, _bindSize) ) {
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

            log(NN_TAG"Input layer: " + _layerInfo.name, 0);
        } else {
            outputs.emplace_back(_layerInfo);
            log(NN_TAG"Output layer: " + _layerInfo.name, 0);
        }
    }

    const size_t _bindingsSize = sizeof(void*) * _cudaEngine->getNbBindings();
    bindings = (void**)malloc(_bindingsSize);
    if(!bindings) {
        log(NN_TAG"Memory allocation error for the bindings. Size: " + std::to_string(_bindingsSize) + ".",2);
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

    if (log) {
        char _memStr[20] = { 0 };
        std::sprintf(_memStr, "%.2f", _cudaEngine->getDeviceMemorySize() * 1.0 / std::pow(1024, 2));
        log(NN_TAG + std::string("Neural network loaded: ") + std::string(_cudaEngine->getName()), 0);
        log(NN_TAG + std::string("Number of layers: ") + std::to_string(_cudaEngine->getNbLayers()), 0);
        log(NN_TAG + std::string("Number of bindings: ") + std::to_string(_cudaEngine->getNbBindings()), 0);
        log(NN_TAG + std::string("Memory used: ") + _memStr + std::string(" MB"), 0);
    }

    if(readyCallback) {
        readyCallback();
    }

    /// ПОКА ТОЛЬКО YOLO 3
    setCurrentNetworkAdapter(std::string("Yolo v3"));

    return true;
}

string Feofan::optimizeNetwork(string modelPath, DeviceType deviceType, PrecisionType precisionType) {
    /**
     * TODO: автоматический поиск caffe.prototxt файла по имени caffe файла
     * TODO: оптимизация engine
     */

    auto _builder = nvinfer1::createInferBuilder(cuLogger);
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
        auto _onnxParser = createParser(*_network, cuLogger);
        if (!_onnxParser->parseFromFile(modelPath.data(), 3)) {
            log(NN_TAG"Ошибка разбора onnx файла.", 2);
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
    /*else if(modelPath.compare(modelPath.length() - 6, 6, ".caffe") == 0) {

        if(caffeProtoTxtPath.empty()) {
            log(NN_TAG"Не указан путь к prototxt файлу caffe модели.", 2);
            return false;
        }
        using namespace nvcaffeparser1;
        auto _caffeParser = createCaffeParser();
        const auto _blobNameToTensor = _caffeParser->parse(caffeProtoTxtPath.data(), modelPath.data(),
                                                           *_network, nvinfer1::DataType::kFLOAT);
        if(!_blobNameToTensor) {
            log(NN_TAG "Ошибка разбора caffe файла.", 2);
            _caffeParser->destroy();
            shutdownProtobufLibrary();
            return false;
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

    } */
    /// UFF
    else if(modelPath.compare(modelPath.length() - 4, 4, ".uff") == 0) {

        using namespace nvuffparser;
        auto _uffParser = createUffParser();
        if (!_uffParser->parse(modelPath.data(), *_network)){
            log(NN_TAG"Ошибка разбора uff файла.", 2);
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
        log(NN_TAG"Неподдерживаемый тип модели. Поддерживаемые типы файлов:"
            "<br>&nbsp;&nbsp;&nbsp;&nbsp;caffe</br>"
            "<br>&nbsp;&nbsp;&nbsp;&nbsp;engine</br>"
            "<br>&nbsp;&nbsp;&nbsp;&nbsp;onnx</br>"
            "<br>&nbsp;&nbsp;&nbsp;&nbsp;uff</br>", 2);
        return _enginePath;
    }
    /// Запись в имя файла информации о точности
    if(!_isEngine) {
        _enginePath += "_" + std::to_string(NV_TENSORRT_MAJOR) + std::to_string(NV_TENSORRT_MINOR) +
                       std::to_string(NV_TENSORRT_PATCH);
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
            log("Невозможно уствновить точность.", 2);
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
            std::ofstream _ofStream(_enginePath.data(), std::ios::binary);
            _ofStream.write((const char *)_iHostMemory->data(), _iHostMemory->size());
        } else {
            log("Ошибка создания engine.", 2);
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

            _cx = std::sin(neuralResult[_i * _detSize + 4] * 4.5 * 3.14 / 180);
            _cy = std::sin((neuralResult[_i * _detSize + 4] + 26) * 4.5 * 3.14 / 180);
            _cz = std::sin((neuralResult[_i * _detSize + 4] + 53) * 4.5 * 3.14 / 180);
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

void Feofan::selectBinding(std::string name, bool isInput) {
    /// TODO
}

void Feofan::setCurrentNetworkAdapter(std::string networkAdapter) {
    log(NN_TAG"Выбран нейронный адаптер для " + networkAdapter, 0);
    currentNetworkAdapter = std::move(networkAdapter);
    setParam(currentNetworkAdapter + ".InputSize", std::to_string(getLayerWidth(inputs[0].dims, currentNetworkAdapter)));
}

#ifndef DYNAMIC_LINKING

void Feofan::setGetLayerHeightCallback(function<size_t(nvinfer1::Dims dims, std::string networkType)> callback) {
    getLayerHeight = callback;
}

void Feofan::setGetLayerWidthCallback(function<size_t(nvinfer1::Dims dims, std::string networkType)> callback) {
    getLayerWidth = callback;
}
#endif // DYNAMIC_LINKING

void Feofan::setNetworkReadyCallback(std::function<void()> callback) {
    readyCallback = std::move(callback);
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

void Feofan::setSetParamCallback(function<void(std::string paramName, std::string value)> callback) {
    setParam = callback;
}
#endif

Feofan::~Feofan() {

}


/// Вспомогательные функции загрузки библиотек
HINSTANCE loadLib(std::string path) {

#ifdef __linux__
    return dlopen(path.data(), RTLD_NOW);
#elif _WIN32
    lastLibAction = "load lib " + string(path);
    std::wstring wc(path.length(), L'#');
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
    return std::move(_errorString.data());
}
#endif
