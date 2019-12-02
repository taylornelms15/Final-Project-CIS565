 /*
 *
 */
#include "ModelImporter.h"
#include "../utils/randInt8Calibrator.h"
#include "../cuda_utilities/cudaMappedMemory.h"
#include "../utils/filesystem.h"


/*
* Tensor RT parsers 
* These will help us import our models
*/
#include <NvCaffeParser.h>
#include <NvOnnxParser.h>
#include <NvUffParser.h>
#include <NvInferPlugin.h>
#include <NvInfer.h>

/*
* For file parsing and caching engine files
*/
#include <iostream>
#include <fstream>
#include <map>
#include <assert.h>

/*
* 
*/
#define CREATE_INFER_BUILDER nvinfer1::createInferBuilder
#define CREATE_INFER_RUNTIME nvinfer1::createInferRuntime


//
// Parsing for speed optmizations
const char* precisionTypeToStr( precisionType type )
{
	switch(type)
	{
		case TYPE_DISABLED:	return "DISABLED";
		case TYPE_FASTEST:	return "FASTEST";
		case TYPE_FP32:	return "FP32";
		case TYPE_FP16:	return "FP16";
		case TYPE_INT8:	return "INT8";
	}
}

// 
// parsing for speed optimizations
precisionType precisionTypeFromStr( const char* str )
{
	if( !str )
		return TYPE_DISABLED;

	for( int n=0; n < NUM_PRECISIONS; n++ )
	{
		if( strcasecmp(str, precisionTypeToStr((precisionType)n)) == 0 )
			return (precisionType)n;
	}

	return TYPE_DISABLED;
}

//
// 
static inline nvinfer1::DataType precisionTypeToTRT( precisionType type )
{
	switch(type)
	{
		case TYPE_FP16:	return nvinfer1::DataType::kHALF;
		case TYPE_INT8:	return nvinfer1::DataType::kINT8;
	}

	return nvinfer1::DataType::kFLOAT;
}

//
//
//
static inline bool isFp16Enabled( nvinfer1::IBuilder* builder )
{
	return builder->getFp16Mode();
}

//
//
//
static inline bool isInt8Enabled( nvinfer1::IBuilder* builder )
{
	return builder->getInt8Mode();
}

//
//
//
static inline const char* dataTypeToStr( nvinfer1::DataType type )
{
	switch(type)
	{
		case nvinfer1::DataType::kFLOAT:	return "FP32";
		case nvinfer1::DataType::kHALF:	return "FP16";
		case nvinfer1::DataType::kINT8:	return "INT8";
		case nvinfer1::DataType::kINT32:	return "INT32";
	}

	printf(LOG_TRT "warning -- unknown nvinfer1::DataType (%i)\n", (int)type);
	return "UNKNOWN";
}

//
//
//
static inline const char* dimensionTypeToStr( nvinfer1::DimensionType type )
{
	switch(type)
	{
		case nvinfer1::DimensionType::kSPATIAL:	 return "SPATIAL";
		case nvinfer1::DimensionType::kCHANNEL:	 return "CHANNEL";
		case nvinfer1::DimensionType::kINDEX:	 return "INDEX";
		case nvinfer1::DimensionType::kSEQUENCE: return "SEQUENCE";
	}

	printf(LOG_TRT "warning -- unknown nvinfer1::DimensionType (%i)\n", (int)type);
	return "UNKNOWN";
}

//
//
// What does this do?
static inline nvinfer1::Dims validateDims( const nvinfer1::Dims& dims )
{
	if( dims.nbDims == nvinfer1::Dims::MAX_DIMS )
		return dims;
	
	nvinfer1::Dims dims_out = dims;

	// TRT doesn't set the higher dims, so make sure they are 1
	for( int n=dims_out.nbDims; n < nvinfer1::Dims::MAX_DIMS; n++ )
		dims_out.d[n] = 1;

	return dims_out;
}

//
//
// 
const char* deviceTypeToStr( deviceType type )
{
	switch(type)
	{
		case DEVICE_GPU:	return "GPU";	
	}
}

//
//
// nano only has GPU
deviceType deviceTypeFromStr( const char* str )
{
	if( !str )
		return DEVICE_ERROR;

	return DEVICE_GPU;
}

static inline nvinfer1::DeviceType deviceTypeToTRT( deviceType type )
{
	switch(type)
	{
		case DEVICE_GPU:	return nvinfer1::DeviceType::kGPU;
	}
}


const char* modelTypeToStr( modelType format )
{
	switch(format)
	{
		// case MODEL_CAFFE:	return "caffe";
		case MODEL_ONNX:	return "ONNX";
		case MODEL_UFF:	return "UFF";
		case MODEL_ERROR: return "ERROR";
	}
}

modelType modelTypeFromStr( const char* str )
{
	if( !str )
		return MODEL_ERROR;

	// if( strcasecmp(str, "caffe") == 0 )
	// 	return MODEL_CAFFE;
	else if( strcasecmp(str, "onnx") == 0 )
		return MODEL_ONNX;
	else if( strcasecmp(str, "uff") == 0 )
		return MODEL_UFF;

	return MODEL_ERROR;
}

//---------------------------------------------------------------------

// constructor
ModelImporter::ModelImporter()
{
	TRTEngine  = NULL; // our tensorRT engine
	TRTInfer   = NULL; 
	TRTContext = NULL;
	TRTStream  = NULL;

	TRTWidth          = 0;
	TRTHeight         = 0;
	TRTInputSize      = 0;
	TRTMaxBatchSize   = 0;
	TRTInputCPU       = NULL;
	TRTInputCUDA      = NULL;
	TRTEnableDebug    = false;
	TRTEnableProfiler = false;

	TRTModelType        = MODEL_ERROR;
	TRTPrecision 	   = TYPE_FASTEST;
	TRTDevice    	   = DEVICE_GPU;
	TRTAllowGPUFallback = false;

}


// Destructor
ModelImporter::~ModelImporter()
{
	if( TRTEngine != NULL )
	{
		TRTEngine->destroy();
		TRTEngine = NULL;
	}
		
	if( TRTInfer != NULL )
	{
		TRTInfer->destroy();
		TRTInfer = NULL;
	}
}




// EnableDebug
void ModelImporter::EnableDebug()
{
	TRTEnableDebug = true;
}


// 
// figure out based on our model and device what we can do
// nano supports them all
//
std::vector<precisionType> ModelImporter::DetectNativePrecisions( deviceType device )
{
	// what type of precision can we run / optmize
	std::vector<precisionType> types;

	// create our necessary logger
	Logger logger;

	// create a temporary builder for querying the supported types
	nvinfer1::IBuilder* builder = CREATE_INFER_BUILDER(logger);
	
	// return error
	if( !builder )
	{
		printf(LOG_TRT "QueryNativePrecisions() failed to create TensorRT IBuilder instance\n");
		return types;
	}

	// FP32 is supported on all platforms
	types.push_back(TYPE_FP32);

	// detect fast (native) FP16
	if( builder->platformHasFastFp16() ){
		types.push_back(TYPE_FP16);
		printf(LOG_TRT "fp16 detected on platform\n");
	}

	// detect fast (native) INT8
	if( builder->platformHasFastInt8() ){
		types.push_back(TYPE_INT8);
		printf(LOG_TRT "int8 detected on platform\n");
	}

	// Now we have all of our supported types that we can run at
	const uint32_t numTypes = types.size();

	printf(LOG_TRT "native precisions detected for %s:  ", deviceTypeToStr(device));
 	
 	// iterate through printing what is supported from our model
	for( uint32_t n=0; n < numTypes; n++ )
	{
		printf("%s", precisionTypeToStr(types[n]));

		if( n < numTypes - 1 )
			printf(", ");
	}

	printf("\n");

	// destroy our temp builder after displaying
	builder->destroy();

	// return our vector of supported types
	return types;
}

static bool ContainsPrecision( const std::vector<precisionType>& types, precisionType type )
{
	const uint32_t numTypes = types.size();

	for( uint32_t n=0; n < numTypes; n++ )
	{
		if( types[n] == type )
			return true;
	}

	return false;
}


// FindFastestPrecision
precisionType ModelImporter::FindFastestPrecision( deviceType device )
{
	// get a list of available precision
	std::vector<precisionType> types = DetectNativePrecisions(device);

	// figure out the fastest that we can use
	if( ContainsPrecision(types, TYPE_INT8) )
		return TYPE_INT8;
	else if( ContainsPrecision(types, TYPE_FP16) )
		return TYPE_FP16;
	else
		return TYPE_FP32;
}


// Create our model
bool ModelImporter::ProfileModel(const std::string& deployFile,			    // name for caffe prototxt
					    const std::string& modelFile,			   		// name for model 
					    const char* input, 
					    const Dims3& inputDims,
					    const std::vector<std::string>& outputs,    	// network outputs
					    unsigned int maxBatchSize,			   			// batch size - NB must be at least as large as the batch we want to run with
					    precisionType precision, 
					    deviceType device, 
					    bool allowGPUFallback,
					    nvinfer1::IInt8Calibrator* calibrator, 	
					    std::ostream& gieModelStream)			   		// output stream for the GIE model
{
	// from the website we follow the necessary steps to build our TRT
	// https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html
	
	/*
	* Step 1 create our parsers depending on incoming model
	* Step 2 creaet our builder same for all networks
	*/

	// create our all necessary gGlogger
	nvinfer1::IBuilder* builder = CREATE_INFER_BUILDER(gLogger);

	// create our network builder which can be used on any model
	nvinfer1::INetworkDefinition* network = builder->createNetwork();

	// set debug flags
	builder->setDebugSync(TRTEnableDebug);

	// TODO figure out why this is here
	builder->setMinFindIterations(3);	// allow time for TX1 GPU to spin up
	builder->setAverageFindIterations(2);

	// 
	printf(LOG_TRT "device %s, loading %s %s\n", deviceTypeToStr(device), deployFile.c_str(), modelFile.c_str());
	
	//
	// depending on our model we need to create and parse differently
	// this is all explained in the documentation

	// create our tensorRt parsor for our onnx model
	if( TRTModelType == MODEL_ONNX )
	{

		nvonnxparser::IParser* onnx_parser = nvonnxparser::createParser(*network, gLogger);
		// check that it worked currently after reading lots of ML
		// stuff I learned that ONNX is a fucking mess so likely
		// version mismatch
		//
		if( !onnx_parser )
		{
			printf(LOG_TRT "failed to create nvonnxparser::IParser instance\n");
			return false;
		}

		// if we succeed we can bring in our ONNX model 
		//
		if( !onnx_parser->parseFromFile(modelFile.c_str(), (int)nvinfer1::ILogger::Severity::kWARNING) )
		{
			printf(LOG_TRT "failed to parse ONNX model '%s'\n", modelFile.c_str());
			return false;
		}
	}
	else if( TRTModelType == MODEL_UFF )
	{
		// create parser instance
		nvuffparser::IUffParser* parser = nvuffparser::createUffParser();
		// check that it worked
		if( !parser )
		{
			printf(LOG_TRT "failed to create UFF parser\n");
			return false;
		}
		
		// register input
		// Not sure what input and output mean
		// TODO figure input string out
		if( !parser->registerInput(input, inputDims, nvuffparser::UffInputOrder::kNCHW) )
		{
			printf(LOG_TRT "failed to register input '%s' for UFF model '%s'\n", input, modelFile.c_str());
			return false;
		}
		
		// TODO
		// figure out relevance of this string
		if( !parser->registerOutput("MarkOutput_0") )
			printf(LOG_TRT "failed to register output '%s' for UFF model '%s'\n", "MarkOutput_0", modelFile.c_str());

		
		// parse network
		if( !parser->parse(modelFile.c_str(), *network, nvinfer1::DataType::kFLOAT) )
		{
			printf(LOG_TRT "failed to parse UFF model '%s'\n", modelFile.c_str());
			return false;
		}
	
	}
	else
	{
		printf(LOG_TRT "invalid model '%s'\n",modelFile.c_str());
		return false;
	}

	// extract the dimensions of the network input blobs
	// Batch size specifies the the batch size for which tensorRT will optimize at, at run time a smaller batch size may be chosen
	// 
	//
	std::map<std::string, nvinfer1::Dims3> inputDimensions;

	// TODO figure this out
	// for( int i=0, n=network->getNbInputs(); i < n; i++ )
	// {
	// 	nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3&&>(network->getInput(i)->getDimensions());
	// 	inputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
	// 	std::cout << LOG_TRT << "retrieved Input tensor \"" << network->getInput(i)->getName() << "\":  " << dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
	// }


	// display progress of the engine
	printf(LOG_TRT "device %s, configuring CUDA engine\n", deviceTypeToStr(device));
	
	/*
	* Step 3 build an engine
	*/ 
	// 
	// set batch size and workspace	
	builder->setMaxBatchSize(maxBatchSize);
	
	// workspace tells us much memory we are allowed to use
	// may vary depending on other apps
	builder->setMaxWorkspaceSize(16 << 20);


	// set up the builder for the desired precision
	if( precision == TYPE_INT8 )
	{
		builder->setInt8Mode(true);
		//builder->setFp16Mode(true);		// TODO:  experiment for benefits of both INT8/FP16
		
		if( !calibrator )
		{
			calibrator = new randInt8Calibrator(1, TRTCacheCalibrationPath, inputDimensions);
			printf(LOG_TRT "warning:  device %s using INT8 precision with RANDOM calibration\n", deviceTypeToStr(device));
		}

		builder->setInt8Calibrator(calibrator);

		return false;
	}
	else if( precision == TYPE_FP16 )
	{
		builder->setFp16Mode(true);
	}
	

	builder->setDefaultDeviceType(deviceTypeToTRT(device));

	if( allowGPUFallback )
		builder->allowGPUFallback(true);
	
	if( device != DEVICE_GPU )
	{
		printf(LOG_TRT "device %s is not supported in TensorRT %u.%u\n", deviceTypeToStr(device), NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
		return false;
	}

	// build CUDA engine
	printf(LOG_TRT "device %s, building FP16:  %s\n", deviceTypeToStr(device), isFp16Enabled(builder) ? "ON" : "OFF"); 
	printf(LOG_TRT "device %s, building INT8:  %s\n", deviceTypeToStr(device), isInt8Enabled(builder) ? "ON" : "OFF"); 
	printf(LOG_TRT "device %s, building CUDA engine (this may take a few minutes the first time a network is loaded)\n", deviceTypeToStr(device));

	/*
	* step 4 build the engine 
	* serialize the engine and store in disk for later reuse
	* if we store on disk we do not have to load every time
	*/
	nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
	
	if( !engine )
	{
		printf(LOG_TRT "device %s, failed to build CUDA engine\n", deviceTypeToStr(device));
		return false;
	}

	printf(LOG_TRT "device %s, completed building CUDA engine\n", deviceTypeToStr(device));

	// serialize the engine, then close everything down
	nvinfer1::IHostMemory* serMem = engine->serialize();

	// check for failures
	if( !serMem )
	{
		printf(LOG_TRT "device %s, failed to serialize CUDA engine\n", deviceTypeToStr(device));
		return false;
	}

	// store model to disk so we do not have to create every single time!
	gieModelStream.write((const char*)serMem->data(), serMem->size());

	//
	printf(LOG_TRT "device %s, completed building CUDA engine preparing to clean upw\n",deviceTypeToStr(device));

	// clean up

	network->destroy();
	engine->destroy();
	builder->destroy();

	printf(LOG_TRT "clean up successful\n");

	return true;
}

					   
// Our top level call where the magic happens
bool ModelImporter::LoadNetwork( const char* prototxt_path_, const char* model_path_, const char* mean_path, 
					    const char* input_blob, const Dims3& input_dims,
					    const std::vector<std::string>& output_blobs, 
					    uint32_t maxBatchSize, precisionType precision,
				   	    deviceType device, bool allowGPUFallback,
					    nvinfer1::IInt8Calibrator* calibrator, cudaStream_t stream )
{
	if( !model_path_ )
		return false;

	printf(LOG_TRT "loading NVIDIA plugins...\n");

	bool loadedPlugins = initLibNvInferPlugins(&gLogger, "");

	if( !loadedPlugins ){
		printf(LOG_TRT "failed to load NVIDIA plugins\n");
		return 0;
	}
	else
		printf(LOG_TRT "completed loading NVIDIA plugins.\n");


	/*
	 * verify valid parameters
	 */
	const std::string model_path    = locateFile(model_path_);
	const std::string prototxt_path = locateFile(prototxt_path_ != NULL ? prototxt_path_ : "");

	const std::string model_ext = fileExtension(model_path_);
	const modelType   model_fmt = modelTypeFromStr(model_ext.c_str());

	printf(LOG_TRT "detected model format - %s  (extension '.%s')\n", modelTypeToStr(model_fmt), model_ext.c_str());

	// TODO add error handling

	TRTModelType = model_fmt;


	/*
	 * if the precision is left unspecified, detect the fastest
	 * probably just let it be fastest?
	 */
	printf(LOG_TRT "desired precision specified for %s: %s\n", deviceTypeToStr(device), precisionTypeToStr(precision));

	if( precision == TYPE_FASTEST )
	{
		if( !calibrator )
			printf(LOG_TRT "requested fasted precision for device %s without providing valid calibrator, disabling INT8\n", deviceTypeToStr(device));

		precision = FindFastestPrecision(device);
		printf(LOG_TRT "selecting fastest native precision for %s:  %s\n", deviceTypeToStr(device), precisionTypeToStr(precision));
	}
	else
	{
		// assert, force me to choose
		assert(precision == TYPE_FASTEST);
	}



	/*
	 * attempt to load network from cache before profiling with tensorRT
	 */
	std::stringstream gieModelStream;
	gieModelStream.seekg(0, gieModelStream.beg);

	char cache_prefix[512];
	char cache_path[512];

	sprintf(cache_prefix, "%s.%u.%u.%s.%s", model_path.c_str(), maxBatchSize, (uint32_t)allowGPUFallback, deviceTypeToStr(device), precisionTypeToStr(precision));
	sprintf(cache_path, "%s.calibration", cache_prefix);
	TRTCacheCalibrationPath = cache_path;
	
	sprintf(cache_path, "%s.engine", cache_prefix);
	TRTCacheEnginePath = cache_path;	
	printf(LOG_TRT "attempting to open engine cache file %s\n", TRTCacheEnginePath.c_str());
	
	std::ifstream cache( TRTCacheEnginePath );

	// if file is not present then we load and store for next useage
	if( !cache )
	{
		printf(LOG_TRT "cache file not found, profiling network model on device %s\n", deviceTypeToStr(device));
		//std::cout << "model path " << model_path << std::endl;
		if( model_path.size() == 0 )
		{
			printf(LOG_TRT "error:  model file '%s' was not found.\n", model_path_);
			return 0;
		}

		if( !ProfileModel(prototxt_path, model_path, input_blob, input_dims,
						 output_blobs, maxBatchSize, precision, device, 
						 allowGPUFallback, calibrator, gieModelStream) )
		{
			printf(LOG_TRT "device %s, failed to load %s\n", deviceTypeToStr(device), model_path_);
			return 0;
		}
	
		printf(LOG_TRT "network profiling complete, writing engine cache to %s\n", TRTCacheEnginePath.c_str());
		std::ofstream outFile;
		outFile.open(TRTCacheEnginePath);
		outFile << gieModelStream.rdbuf();
		outFile.close();
		gieModelStream.seekg(0, gieModelStream.beg);
		printf(LOG_TRT "device %s, completed writing engine cache to %s\n", deviceTypeToStr(device), TRTCacheEnginePath.c_str());
	}
	// we have a configuration already so load and go
	else
	{
		printf(LOG_TRT "loading network profile from engine cache... %s\n", TRTCacheEnginePath.c_str());
		gieModelStream << cache.rdbuf();
		cache.close();
	}

	// 
	printf(LOG_TRT "device %s, %s loaded\n", deviceTypeToStr(device), model_path.c_str());
	

	/*
	 * create runtime inference engine execution context
	 */
	nvinfer1::IRuntime* infer = CREATE_INFER_RUNTIME(gLogger);
	
	if( !infer )
	{
		printf(LOG_TRT "device %s, failed to create InferRuntime\n", deviceTypeToStr(device));
		return 0;
	}



	// how many bytes
	gieModelStream.seekg(0, std::ios::end);
	const int modelSize = gieModelStream.tellg();
	gieModelStream.seekg(0, std::ios::beg);

	// TODO I think this should be in bytes...
	void* modelMem = malloc(modelSize * sizeof(char));

	if( !modelMem )
	{
		printf(LOG_TRT "failed to allocate %i bytes to deserialize model\n", modelSize);
		return 0;
	}

	// read model
	gieModelStream.read((char*)modelMem, modelSize);
	
	// deserialize the engine
	// which prepares it for inference
	nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(modelMem, modelSize, NULL);
	
	free(modelMem);


	if( !engine )
	{
		printf(LOG_TRT "device %s, failed to create CUDA engine\n", deviceTypeToStr(device));
		return 0;
	}
	
	// illustrated in step 2.5 create our execution context
	nvinfer1::IExecutionContext* context = engine->createExecutionContext();
	
	if( !context )
	{
		printf(LOG_TRT "device %s, failed to create execution context\n", deviceTypeToStr(device));
		return 0;
	}

	// if debug provided add it to the context engine
	if( TRTEnableDebug )
	{
		printf(LOG_TRT "device %s, enabling context debug sync.\n", deviceTypeToStr(device));
		context->setDebugSync(true);
	}

	printf(LOG_TRT "device %s, CUDA engine context initialized with %u bindings\n", deviceTypeToStr(device), engine->getNbBindings());
	
	// save off our classes for inference
	TRTInfer   = infer;
	TRTEngine  = engine;
	TRTContext = context;
	
	// set stream
	SetStream(stream);


	/*
	 * print out binding info
	 */
	const int numBindings = engine->getNbBindings();
	
	for( int n=0; n < numBindings; n++ )
	{
		printf(LOG_TRT "binding -- index   %i\n", n);

		const char* bind_name = engine->getBindingName(n);

		printf("               -- name    '%s'\n", bind_name);
		printf("               -- type    %s\n", dataTypeToStr(engine->getBindingDataType(n)));
		printf("               -- in/out  %s\n", engine->bindingIsInput(n) ? "INPUT" : "OUTPUT");

		const nvinfer1::Dims bind_dims = engine->getBindingDimensions(n);

		printf("               -- # dims  %i\n", bind_dims.nbDims);
		
		for( int i=0; i < bind_dims.nbDims; i++ )
			printf("               -- dim #%i  %i (%s)\n", i, bind_dims.d[i], dimensionTypeToStr(bind_dims.type[i]));
	}


	/*
	 * determine dimensions of network input bindings
	 */
	const int inputIndex = engine->getBindingIndex(input_blob);
	
	//
	printf(LOG_TRT "binding to input 0 %s  binding index:  %i\n", input_blob, inputIndex);
	
	//
	nvinfer1::Dims inputDims = validateDims(engine->getBindingDimensions(inputIndex));

	//
	size_t inputSize = maxBatchSize * DIMS_C(inputDims) * DIMS_H(inputDims) * DIMS_W(inputDims) * sizeof(float);
	printf(LOG_TRT "binding to input 0 %s  dims (b=%u c=%u h=%u w=%u) size=%zu\n", input_blob, maxBatchSize, DIMS_C(inputDims), DIMS_H(inputDims), DIMS_W(inputDims), inputSize);
	

	/*
	 * allocate shared memory
	 */
	if( !cudaAllocMapped((void**)&TRTInputCPU, (void**)&TRTInputCUDA, inputSize) )
	{
		printf(LOG_TRT "failed to alloc CUDA mapped memory for tensor input, %zu bytes\n", inputSize);
		return false;
	}
	
	// safe off our settings
	TRTInputSize    = inputSize;
	TRTWidth        = DIMS_W(inputDims);
	TRTHeight       = DIMS_H(inputDims);
	TRTMaxBatchSize = maxBatchSize;
	

	/*
	 * setup network output buffers
	 */
	const int numOutputs = output_blobs.size();
	
	//
	// what is a network output blob and buffer?
	for( int n=0; n < numOutputs; n++ )
	{
		const int outputIndex = engine->getBindingIndex(output_blobs[n].c_str());
		printf(LOG_TRT "binding to output %i %s  binding index:  %i\n", n, output_blobs[n].c_str(), outputIndex);

		//
		nvinfer1::Dims outputDims = validateDims(engine->getBindingDimensions(outputIndex));


		size_t outputSize = maxBatchSize * DIMS_C(outputDims) * DIMS_H(outputDims) * DIMS_W(outputDims) * sizeof(float);
		printf(LOG_TRT "binding to output %i %s  dims (b=%u c=%u h=%u w=%u) size=%zu\n", n, output_blobs[n].c_str(), maxBatchSize, DIMS_C(outputDims), DIMS_H(outputDims), DIMS_W(outputDims), outputSize);
	
		// allocate output memory 
		void* outputCPU  = NULL;
		void* outputCUDA = NULL;
		
		// allocate more shared memory
		if( !cudaAllocMapped((void**)&outputCPU, (void**)&outputCUDA, outputSize) )
		{
			printf(LOG_TRT "failed to alloc CUDA mapped memory for tensor output, %zu bytes\n", outputSize);
			return false;
		}
	
		outputLayer l;
		
		l.CPU  = (float*)outputCPU;
		l.CUDA = (float*)outputCUDA;
		l.size = outputSize;


		DIMS_W(l.dims) = DIMS_W(outputDims);
		DIMS_H(l.dims) = DIMS_H(outputDims);
		DIMS_C(l.dims) = DIMS_C(outputDims);


		l.name = output_blobs[n];
		TRTOutputs.push_back(l);
	}
	
	DIMS_W(TRTInputDims) = DIMS_W(inputDims);
	DIMS_H(TRTInputDims) = DIMS_H(inputDims);
	DIMS_C(TRTInputDims) = DIMS_C(inputDims);

	TRTPrototxtPath     = prototxt_path;
	TRTModelPath        = model_path;
	TRTInputBlobName    = input_blob;
	TRTPrecision        = precision;
	TRTDevice           = device;
	TRTAllowGPUFallback = allowGPUFallback;

	if( mean_path != NULL )
		TRTMeanPath = mean_path;
	
	printf(LOG_TRT "device %s, %s initialized.\n", deviceTypeToStr(device), TRTModelPath.c_str());
	return true;
}


// to optimize owe want to cuda stream everything
// for debg purposes allow non streaming
// if we just block then its normal cuda which is useful for benchmarking and debugging
cudaStream_t ModelImporter::CreateStream( bool nonBlocking )
{
	uint32_t flags = cudaStreamDefault;

	if( nonBlocking )
		flags = cudaStreamNonBlocking;

	cudaStream_t stream = NULL;

	if( CUDA_FAILED(cudaStreamCreateWithFlags(&stream, flags)) )
		return NULL;

	SetStream(stream);
	return stream;
}


// SetStream
void ModelImporter::SetStream( cudaStream_t stream )
{
	TRTStream = stream;
}	

