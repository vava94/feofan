/**
 * Created by vok on 28.07.2020.
 *
 * TODO: Включить поддержку DLA
 * TODO: Включить профайлер
 * TODO: Деструктор
 * TODO: Loading labels from file
 * TODO: Посмотреть примеры и настроить калибратор
 */

#include "feofan.hpp"

#include <cmath>
#include <cuda.h>
#include <memory.h>
#include <NvCaffeParser.h>
#include <NvOnnxParser.h>
#include <thread>
#include <NvUffParser.h>
#include <fstream>

using namespace nvinfer1;
using namespace std;
#define NN_TAG "NN: "

/********************************************************************
 * Dynamic lib load section
 ********************************************************************/
#ifdef __linux__
    #include <dlfcn.h>
    const char libname[] = "./libneuraladapter.so";
#elif _WIN32
    const std::string libname = "neuraladapter.dll";
    std::string lastLibAction;
    #ifdef _MSC_VER
        #include <Windows.h>
        const char* dlerror();
    #else
        #include <dlfcn.h>
    #endif
#endif
/********************************************************************
 * Filesystem dependencies
 *******************************************************************/
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

/********************************************************************
 * Platform dependent functions
 *******************************************************************/
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
///******************************************************************

/**
 * Constructor
 * @param logCallback callback for logs
 * @param neuralInfoCallback callback for network infos (ex FPS)
 */
Feofan::Feofan(function<void(string, int)> logCallback,
    function<void(string)> neuralInfoCallback)
{

    cudaStream = nullptr;
    dlaCoresCount = 0;
    loaded = false;
    networksCount = 0;
    readyCallback = nullptr;
    getLayerHeight = nullptr;
    getLayerWidth = nullptr;
    parseDetectionOutput = nullptr;
    setParam = nullptr;

    if (logCallback) {
        utils::Log = move(logCallback);
        utils::Log(NN_TAG "Loading neural framework.", 0);
    }
    if(neuralInfoCallback) {
        neuralInfo = move(neuralInfoCallback);
    }

#ifdef DYNAMIC_LINKING
    /// Загрузка вспомогательной библиотеки для разбора результатов сетей
    neuralAdapter = loadLib(libname);
    if(!neuralAdapter) {
        if (utils::Log) {
            utils::Log(NN_TAG "Unable to load neural network adapter. Detection is disabled.", 1);
            auto _a1 = dlerror();
            std::string _err(_a1);
            utils::Log(NN_TAG + _err, 2);
            delete[] _a1;
        }
        if (neuralInfo) neuralInfo("Detection is disabled.");
        return;
    }
    /// Получение информации о доступных обработчиках
    auto _getAdapters = (vector<string>(WINCALL*)())loadSymbol(neuralAdapter, "getAdapters");
    if(!_getAdapters) {
        if (utils::Log) {
            auto _a1 = dlerror();
            std::string _err(_a1);
            utils::Log(NN_TAG + _err, 2);
            delete[] _a1;
        }
    } else {
        availableAdapters = _getAdapters();
        if(availableAdapters.empty()) {
            if (utils::Log) utils::Log("The neural network adapter is empty. Detection is disabled.", 1);
            if (neuralInfo) neuralInfo("Detection is disabled.");
            parseDetectionOutput = nullptr;
            return;
        } else {
            string _adapterNetworks = NN_TAG + string("Network use available:");
            for(const auto& _adapter : availableAdapters) {
                _adapterNetworks += "<br>&nbsp;&nbsp;&nbsp;&nbsp;" + _adapter + "</br>";
            }
            if (utils::Log) utils::Log(_adapterNetworks, 1);
            parseDetectionOutput = ((int(WINCALL*)(void **, float **, const string& networkType))
                    loadSymbol(neuralAdapter, "parseDetectionOutput"));
            if(!parseDetectionOutput) {
                if (utils::Log) {
                    auto _a1 = dlerror();
                    std::string _err(_a1);
                    utils::Log(NN_TAG + _err, 2);
                    delete[] _a1;
                }
                return;
            }
            getLayerHeight = (size_t(WINCALL*)(nvinfer1::Dims, string))loadSymbol(neuralAdapter, "getLayerHeight");
            if(!getLayerHeight) {
                if (utils::Log){
                    auto _a1 = dlerror();
                    std::string _err(_a1);
                    utils::Log(NN_TAG + _err, 2);
                    delete[] _a1;
                }
                return;
            }
            getLayerWidth = (size_t(WINCALL*)(nvinfer1::Dims, string))loadSymbol(neuralAdapter, "getLayerWidth");
            if(!getLayerWidth) {
                if (utils::Log) {
                    auto _a1 = dlerror();
                    std::string _err(_a1);
                    utils::Log(NN_TAG + _err, 2);
                    delete[] _a1;
                }
                return;
            }
            setParam = (void(WINCALL*)(string, string))loadSymbol(neuralAdapter, "setParam");
            if(!setParam) {
                if (utils::Log) {
                    auto _a1 = dlerror();
                    std::string _err(_a1);
                    utils::Log(NN_TAG + _err, 2);
                    delete[] _a1;
                }
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
        if (utils::Log) utils::Log(NN_TAG "The device has support for DLA cores.", 1);
    }
    if (_builder->platformHasFastInt8()) {
        availablePrecisions.emplace_back(precisionsStr[2]);
        if (utils::Log) utils::Log(NN_TAG "Available precision: INT8.", 1);
        _fastestPrecision = "INT8";
    }
    if (_builder->platformHasFastFp16()) {
        availablePrecisions.emplace_back(precisionsStr[1]);
        if (utils::Log) utils::Log(NN_TAG "Available precision: FP16.", 1);
        if(_fastestPrecision.empty()) {
            _fastestPrecision = "FP16";
        }
    }
    availablePrecisions.emplace_back(precisionsStr[0]);
    if (utils::Log) utils::Log(NN_TAG "Available precision: FP32.", 1);
    if(_fastestPrecision.empty()) {
        _fastestPrecision = "FP32";
    }
    if(neuralInfo) {
        infoString = "TensorRT: " + to_string(NV_TENSORRT_MAJOR) + "." + to_string(NV_TENSORRT_MINOR) + "." +
                     to_string(NV_TENSORRT_PATCH) + " | " + _fastestPrecision + " | ";
        neuralInfo(infoString + "  0.00 FPS");
    }


    if(cudaStreamCreateWithFlags(&cudaStream, cudaStreamDefault) != cudaSuccess) {
        if (utils::Log) utils::Log(NN_TAG "Error creating cuda stream.", 2);
        return;
    }

    loaded = true;

}

void Feofan::allocImage(int index, int width, int height, int channels) {

    if(!loaded) return;
    auto _image = new NeuralImage(index, width, height, channels);
    neuralImages.emplace(index, _image);

}

void Feofan::closeNetwork(int pos) {

    networkDefinitions[pos].executionContext->destroy();
    networkDefinitions[pos].engine->destroy();

    cudaFreeHost(networkDefinitions[pos].input.CPU);
    cudaFree(networkDefinitions[pos].input.CUDA);

    for (const auto &mLayer : networkDefinitions[pos].outputs) {
        cudaFreeHost(mLayer.CPU);
        cudaFree(mLayer.CUDA);
    }

    networkDefinitions[pos].outputs.clear();
    delete networkDefinitions[pos].bindings;
    networkDefinitions.erase(networkDefinitions.begin() + pos);
    readyCallback(pos, false);
}

vector<string> Feofan::getAdapters() const {
    return availableAdapters;
}

const std::vector<std::string>& Feofan::getAvailablePrecisions() {
    return availablePrecisions;
}

uint8_t *Feofan::getImage() const {
    //TODO: Получение изображения по позиции
    return neuralImages.at(0)->data();
}

Feofan::NetworkDefinition Feofan::getNetworkDefinition(int pos) {
    return networkDefinitions.at(pos);
}

void Feofan::loadFont(FILE* fontFile) {
    NeuralImage::loadFont(fontFile, 12);
}

void Feofan::loadNetwork(string networkPath) {
    new thread(bind(&Feofan::neuralInit, this, placeholders::_1, placeholders::_1), networkPath);
}

void Feofan::newData(int index, uint8_t *data) {
    if(!loaded) return;
    neuralImages.at(index)->newData(data);
}

string Feofan::networkName(int networkIndex) const {

    if (networkDefinitions.size() <= networkIndex) return "Сеть не загружена.";
    return networkDefinitions[networkIndex].name;
}

void Feofan::neuralInit(const string& networkPath, const string& caffeProtoTxtPath) {

    /// TODO: add loaded network to the vector of network definitions
    const int mPos = (int)networkDefinitions.size();
    size_t mMemFree0, mMemFree2, mMemTotal, mMemAllocated;

    char *mEngineStream;
    FILE *mCacheFile;
    ICudaEngine *mCudaEngine;
    IExecutionContext *mExecutionContext;
    IRuntime *mRuntime;
    NetworkDefinition mNetworkDefinition;
    size_t mBindingsSize, mEngineSize;
    string mEnginePath;
    vector<layerInfo> mInputs, mOutputs;

    if (utils::Log) utils::Log(NN_TAG "***************************************", 0);
    if (utils::Log) utils::Log(NN_TAG "Loading network...", 0);
    if (utils::Log) utils::Log(NN_TAG "Path: " + networkPath, 0);
    if (utils::Log) utils::Log(NN_TAG "***************************************", 0);
    /// Проверка типа файла. Если ".engine" - загрузка, иначе - парсинг и оптимизация.
    if(networkPath.length() < 7 || networkPath.compare(networkPath.length() - 7, 7, ".engine") != 0) {
        mEnginePath = optimizeNetwork(networkPath, DeviceType::DEVICE_GPU, PrecisionType::FP32_PT);
    } else {
        mEnginePath = networkPath;
    }

    if(mEnginePath.empty()) {
        if (utils::Log) utils::Log(NN_TAG "Error loading network: \"enginePath is empty\".", 2);
        readyCallback(mPos, false);
        return;
    }

    mRuntime = createInferRuntime(cudaLogger);
    cuMemGetInfo(&mMemFree0, &mMemTotal);
    if (!mRuntime) {
        if (utils::Log) utils::Log(NN_TAG "Error creating infer runtime.", 2);
        readyCallback(mPos, false);
        return;
    }

    if ((mEngineSize = fs::file_size(mEnginePath.data())) == 0){
        if (utils::Log) utils::Log(NN_TAG "Error loading network: \"engine file size is 0\".", 2);
        mRuntime->destroy();
        readyCallback(mPos, false);
        return;
    }
    mEngineStream = new char[mEngineSize];
    //TODO: remove
    //mEngineStream = (char*)malloc(mEngineSize);

    mCacheFile = fopen(mEnginePath.data(), "rb");
    if(fread(mEngineStream, 1, mEngineSize, mCacheFile) != mEngineSize) {
        if (utils::Log) utils::Log(NN_TAG "Error reading engine file.", 2);
        readyCallback(mPos, false);
        mRuntime->destroy();
        delete[] mEngineStream;
        fclose(mCacheFile);
        return;
    }
    fclose(mCacheFile);


    mCudaEngine = mRuntime->deserializeCudaEngine(mEngineStream, mEngineSize);
    if (!mCudaEngine) {
        if (utils::Log) utils::Log(NN_TAG "Error creating cuda engine.", 2);
        readyCallback(mPos, false);
        mRuntime->destroy();
        delete[] mEngineStream;
        fclose(mCacheFile);
        return;
    }

    mExecutionContext = mCudaEngine->createExecutionContext();

    for(int _i = 0; _i < mCudaEngine->getNbBindings(); _i++) {
        layerInfo _layerInfo;
        void *cpuBind = nullptr;
        void *gpuBind = nullptr;
        Dims _dims = validateDims(mCudaEngine->getBindingDimensions(_i));
        /** TODO: Для onnx:
         *	    if( modelType == MODEL_ONNX )
         *	        inputDims = shiftDims(inputDims);   // change NCHW to CHW if EXPLICIT_BATCH set
         */
        size_t _bindSize = mCudaEngine->getMaxBatchSize() * sizeof(float);
        for (int _j = 0; _j < _dims.nbDims; _j++) {
            _bindSize = _bindSize * _dims.d[_j];
        }

        if (!utils::cudaAllocMapped((void**)&cpuBind, (void**)&gpuBind, _bindSize) ) {
            readyCallback(mPos, false);
            return;
        }

        _layerInfo.CPU = (float *) cpuBind;
        _layerInfo.CUDA = (float *) gpuBind;
        _layerInfo.size = _bindSize;
        _layerInfo.name = mCudaEngine->getBindingName(_i);
        _layerInfo.dims = _dims;
        _layerInfo.binding = mCudaEngine->getBindingIndex(_layerInfo.name.data());
        if(mCudaEngine->bindingIsInput(_layerInfo.binding)) {
            mInputs.emplace_back(_layerInfo);
            if (utils::Log) utils::Log(NN_TAG"Input layer: " + _layerInfo.name, 0);
        } else {
            mOutputs.emplace_back(_layerInfo);
            if (utils::Log) utils::Log(NN_TAG"Output layer: " + _layerInfo.name, 0);
        }
    }

    mBindingsSize = sizeof(void*) * mCudaEngine->getNbBindings();
    mNetworkDefinition.bindings = (void**)malloc(mBindingsSize);
    if (!mNetworkDefinition.bindings) {
        if (utils::Log) utils::Log(NN_TAG"Memory allocation error for the bindings. Size: " + to_string(mBindingsSize) + ".",2);
        //TODO: деинитиализация переменных
        readyCallback(mPos, false);
        return;
    }
    memset(mNetworkDefinition.bindings, 0, mBindingsSize);
    for (const auto& mInput: mInputs) {
        mNetworkDefinition.bindings[mInput.binding] = (void*)mInput.CUDA;

    }
    int _n = 0;
    for(const auto& mOutput: mOutputs) {
        mNetworkDefinition.bindings[mOutput.binding] = (void*)mOutput.CUDA;
        _n ++;
    }
    mRuntime->destroy();
    cuMemGetInfo(&mMemFree2, &mMemTotal);
    mMemAllocated = mMemFree0 - mMemFree2;
    /// Вывод информации о загруженной сети
    if (utils::Log) {
        char _memStr[20] = { 0 };
        //size_t memAlloc = mCudaEngine->getDeviceMemorySize() + mInputs[0].size;


        utils::Log(NN_TAG "***************************************", 0);
        sprintf(_memStr, "%.2f", (mMemAllocated) * 1.0 / pow(1024, 2));
        utils::Log(NN_TAG + string("Neural network loaded: ") + string(mCudaEngine->getName()), 0);
        utils::Log(NN_TAG + string("Number of layers: ") + to_string(mCudaEngine->getNbLayers()), 0);
        utils::Log(NN_TAG + string("Number of bindings: ") + to_string(mCudaEngine->getNbBindings()), 0);
        utils::Log(NN_TAG + string("Memory used: ") + _memStr + string(" MB"), 0);
        utils::Log(NN_TAG "***************************************", 0);
    }

    auto mI  = mCudaEngine->getMaxBatchSize();
    /// Запись информации в networkDefinition
    mNetworkDefinition.name = mCudaEngine->getName();
    mNetworkDefinition.enginePath = mEnginePath;
    mNetworkDefinition.filePath = networkPath;
    mNetworkDefinition.executionContext = mExecutionContext;
    mNetworkDefinition.engine = mCudaEngine;
    mNetworkDefinition.nbLayers = mCudaEngine->getNbLayers();
    mNetworkDefinition.input = mInputs[0];
    mNetworkDefinition.outputs.insert(mNetworkDefinition.outputs.cbegin(), mOutputs.cbegin(), mOutputs.cend());
    mNetworkDefinition.memory = mMemAllocated;
    networkDefinitions.emplace_back(mNetworkDefinition);

    if(readyCallback) {
        readyCallback(mPos, true);
    }



    /// ПОКА ТОЛЬКО YOLO 3
    setCurrentNetworkAdapter(mPos, string("Yolo v3"));


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
    string _enginePath;
    ICudaEngine *_cudaEngine;

    /// Определение типа загружаемого файла
    /// ONNX
    if(modelPath.compare(modelPath.length() - 5, 5, ".onnx") == 0) {

        using namespace nvonnxparser;
        auto _onnxParser = createParser(*_network, cudaLogger);
        if (!_onnxParser->parseFromFile(modelPath.data(), 3)) {
            if (utils::Log) utils::Log(NN_TAG"Error parsing '.onnx' file.", 2);
            _onnxParser->destroy();
            return _enginePath;
        }
        _destroyParser = [_onnxParser]() {
            _onnxParser->destroy();
        };
        _enginePath = modelPath.substr(0,modelPath.length() - 5);
        auto _p = _enginePath.find_last_of('/') + 1;
        auto mName = _enginePath.substr(_p,modelPath.length() - _p);
        _network->setName(mName.data());
        _enginePath += "_onnx";

    }
    /// CAFFE
    else if(modelPath.compare(modelPath.length() - 6, 6, ".caffe") == 0) {

        /// Loading ".prototxt" file by model name
        string _prototxt = modelPath.substr(0, modelPath.length() - 6) + ".prototxt";

        if (!fs::exists(_prototxt)) {
            if (utils::Log) utils::Log(NN_TAG "File '" + _prototxt + "' not found.", 2);
            return "";
        }

        using namespace nvcaffeparser1;
        auto _caffeParser = createCaffeParser();
        const auto _blobNameToTensor = _caffeParser->parse(_prototxt.data(), modelPath.data(),
                                                           *_network, nvinfer1::DataType::kFLOAT);
        if(!_blobNameToTensor) {
            if (utils::Log) utils::Log(NN_TAG "Error parsing '.caffe' file.", 2);
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
        auto mName = _enginePath.substr(_p,modelPath.length() - _p);
        _network->setName(mName.data());
        _enginePath += "_caffe";

    }
    /// UFF
    else if(modelPath.compare(modelPath.length() - 4, 4, ".uff") == 0) {

        using namespace nvuffparser;
        auto _uffParser = createUffParser();
        if (!_uffParser->parse(modelPath.data(), *_network)){
            if (utils::Log) utils::Log(NN_TAG"Error parsung '.uff' file.", 2);
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
        auto mName = _enginePath.substr(_p,modelPath.length() - _p);
        _network->setName(mName.data());
        _enginePath += "_uff";

    }
        /// ENGINE
    else if(modelPath.compare(modelPath.length() - 7, 7, ".engine") == 0) {
        _isEngine = true;
        _enginePath = modelPath;
    }
        /// Underfined
    else {
        if (utils::Log) utils::Log(NN_TAG"Unsupported model type. Supported file types:"
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
           (precisionType == FASTEST_PT && availablePrecisions[0] == "FP32_PT"))
        {
            _enginePath += "_FP32";
        }
        else if (precisionType == FP16_PT ||
                (precisionType == FASTEST_PT && availablePrecisions[0] == "FP16_PT"))
        {
            _enginePath += "_FP16";
        }
        else if (precisionType == INT8_PT ||
                (precisionType == FASTEST_PT && availablePrecisions[0] == "INT8_PT")) {
            _enginePath += "_INT8";
        }
        else
        {
            if (utils::Log) utils::Log("Can't set percision.", 2);
            return "";
        }
        _enginePath += "_GPU";
        _enginePath += ".engine";

    }

    /// Созданипе engine файла
    if(!_isEngine && !fs::exists(_enginePath)) {

        auto _config = _builder->createBuilderConfig();
        /// Выбор оптимизации
        if(precisionType == FP16_PT ||
           (precisionType == FASTEST_PT && availablePrecisions[0] == "FP16_PT")) {
            _config->setFlag(nvinfer1::BuilderFlag::kFP16);
        } else if(precisionType == INT8_PT ||
                  (precisionType == FASTEST_PT && availablePrecisions[0] == "INT8_PT")) {
            _config->setFlag(nvinfer1::BuilderFlag::kINT8);
        } else

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
            if (utils::Log) utils::Log("Engine creation error.", 2);
            _enginePath.clear();
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

void Feofan::processData(int networkIndex) {
    int
        _inputWidth = getLayerWidth(networkDefinitions[networkIndex].input.dims, currentNetworkAdapter),
        _inputHeight = getLayerHeight(networkDefinitions[networkIndex].input.dims, currentNetworkAdapter);
    neuralImages[networkIndex]->lockImage();
    neuralImages[networkIndex]->prepareTensor(cudaStream,
                                       networkDefinitions[networkIndex].input.CUDA,
                                       _inputWidth,
                                       _inputHeight,
                                       255.f);
    networkDefinitions[networkIndex].executionContext->enqueueV2(networkDefinitions[networkIndex].bindings, cudaStream, nullptr);
    cudaStreamSynchronize(cudaStream);

    auto _numDetections = parseDetectionOutput(&networkDefinitions[networkIndex].bindings[1], neuralImages[networkIndex]->detectionsDataPtr(), currentNetworkAdapter);
    if (_numDetections) {
        neuralImages[networkIndex]->applyDetections(_numDetections, NeuralImage::BOX | NeuralImage::CONFIDENCE | NeuralImage::LABEL, 3);
    }
    neuralImages[networkIndex]->unlockImage();
}

void Feofan::setCurrentNetworkAdapter(int networkIndex, string networkAdapter) {
    if (utils::Log) utils::Log(NN_TAG "Selected neural adapter for " + networkAdapter, 0);
    currentNetworkAdapter = move(networkAdapter);
    setParam(currentNetworkAdapter + ".InputSize", to_string(getLayerWidth(networkDefinitions[networkIndex].input.dims, currentNetworkAdapter)));
}

#ifndef DYNAMIC_LINKING

void Feofan::setGetLayerHeightCallback(function<size_t(nvinfer1::Dims dims, string networkType)> callback) {
    getLayerHeight = callback;
}

void Feofan::setGetLayerWidthCallback(function<size_t(nvinfer1::Dims dims, string networkType)> callback) {
    getLayerWidth = callback;
}
#endif // DYNAMIC_LINKING

void Feofan::setNetworkReadyCallback(function<void(int, bool)> callback) {
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
/// ---------------------------------------------------------------------------------------------------------
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
    const char *_ch = new const char[_errorString.size() + 1]();
    memset((void *) _ch, 0,  _errorString.size() + 1);
    memcpy((void *) _ch, (void *) _errorString.c_str(), _errorString.size());
    return _ch;
}
#endif
