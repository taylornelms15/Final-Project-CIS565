 
#ifndef __TENSOR_NET_H__
#define __TENSOR_NET_H__

namespace nvinfer1 { class IInt8Calibrator; }

// includes
#include <NvInfer.h>

#include <vector>
#include <sstream>
#include <math.h>


typedef nvinfer1::DimsCHW Dims3;

#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]



/**
 * Default maximum batch size
 */
#define DEFAULT_MAX_BATCH_SIZE  1

/**
 * Prefix used for tagging printed log output from TensorRT.
 */
#define LOG_TRT "[TRT]   "


/**
 * Enumeration for indicating the types that TRT supports
 */
enum precisionType
{
	TYPE_DISABLED = 0,	/**< Unknown, unspecified, or disabled type */
	TYPE_FASTEST,		/**< The fastest detected precision should be use (i.e. try INT8, then FP16, then FP32) */
	TYPE_FP32,		/**< 32-bit floating-point precision (FP32) */
	TYPE_FP16,		/**< 16-bit floating-point half precision (FP16) */
	TYPE_INT8,		/**< 8-bit integer precision (INT8) */
	NUM_PRECISIONS
};

/**
 * string to type
 * 
 */
const char* precisionTypeToStr( precisionType type );

/**
 * type to string
 *
 */
precisionType precisionTypeFromStr( const char* str );

/**
 * 
 */
enum deviceType
{
	DEVICE_GPU = 0,	// 
	DEVICE_ERROR				
};

/**
 * return a string from text
 */
const char* deviceTypeToStr( deviceType type );

/**
 * Parse the device type from a string.
 */
deviceType deviceTypeFromStr( const char* str );

/**
 * Enumeration indicating the format of the model that's
 * imported in TensorRT (either caffe, ONNX, or UFF).
 * @ingroup tensorNet
 */
enum modelType
{
	/**< caffemodel  not yet done*/
	MODEL_ONNX = 0,		/**< ONNX */
	MODEL_UFF,			/**< UFF */
	MODEL_ERROR
};

/**
 * Stringize function that returns modelType in text.
 */
const char* modelTypeToStr( modelType type );

/**
 * Parse the model format from a string.
 */
modelType modelTypeFromStr( const char* str );


/**
 *  class whose single job is to load ML models and transform them
 *  into tensorRT acclerated models.
 */
class ModelImporter
{
public:
	/**
	 * Destory
	 */
	virtual ~ModelImporter();
	

	/**
	 * Load a new network instance (this variant is used for UFF models)
	 * @param prototxt File path to the deployable network prototxt
	 * @param model File path to the caffemodel 
	 * @param mean File path to the mean value binary proto (NULL if none)
	 * @param input_blob The name of the input blob data to the network.
	 * @param input_dims The dimensions of the input blob (used for UFF).
	 * @param output_blobs List of names of the output blobs from the network.
	 * @param maxBatchSize The maximum batch size that the network will be optimized for.
	 */
	bool LoadNetwork( const char* prototxt, const char* model, const char* mean,
				   const char* input_blob, const Dims3& input_dims, 
				   const std::vector<std::string>& output_blobs,
				   uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
				   precisionType precision=TYPE_FASTEST,
				   deviceType device=DEVICE_GPU, bool allowGPUFallback=true,
				   nvinfer1::IInt8Calibrator* calibrator=NULL, cudaStream_t stream=NULL );

	/**
	 * Manually enable layer profiling times.	
	 */
	void EnableLayerProfiler();

	/**
	 * Manually enable debug messages and synchronization.
	 */
	void EnableDebug();

	/**
 	 * Return true if GPU fallback is enabled.
	 */
	inline bool AllowGPUFallback() const				{ return TRTAllowGPUFallback; }

	/**
 	 * Retrieve the device being used for execution.
	 */
	inline deviceType GetDevice() const				{ return TRTDevice; }

	/**
	 * Retrieve the type of precision being used.
	 */
	inline precisionType GetPrecision() const			{ return TRTPrecision; }

	/**
	 * Check if a particular precision is being used.
	 */
	inline bool IsPrecision( precisionType type ) const	{ return (TRTPrecision == type); }

	/**
	 * Determine the fastest native precision on a device.
	 */
	static precisionType FindFastestPrecision( deviceType device=DEVICE_GPU );

	/**
	 * Detect the precisions supported natively on a device.
	 */
	static std::vector<precisionType> DetectNativePrecisions( deviceType device=DEVICE_GPU );

	/**
	 * Retrieve the stream that the device is operating on.
	 */
	inline cudaStream_t GetStream() const				{ return TRTStream; }

	/**
	 * Create and use a new stream for execution.
	 */
	cudaStream_t CreateStream( bool nonBlocking=true );

	/**
	 * Set the stream that the device is operating on.
	 */
	void SetStream( cudaStream_t stream );

	/**
	 * Retrieve the path to the network prototxt file.
	 */
	inline const char* GetPrototxtPath() const			{ return TRTPrototxtPath.c_str(); }

	/**
	 * Retrieve the path to the network model file.
	 */
	inline const char* GetModelPath() const				{ return TRTModelPath.c_str(); }

	/**
	 * Retrieve the format of the network model.
	 */
	inline modelType GetModelType() const				{ return TRTModelType; }

	/**
	 * Return true if the model is of the specified format.
	 */
	inline bool IsModelType( modelType type ) const		{ return (TRTModelType == type); }

protected:

	/**
	 * Constructor.
	 */
	ModelImporter();
			  
	/**
	 * Create and output an optimized network model
	 * @note this function is automatically used by LoadNetwork, but also can 
	 *       be used individually to perform the network operations offline.
	 * @param deployFile name for network prototxt
	 * @param modelFile name for model
	 * @param outputs network outputs
	 * @param maxBatchSize maximum batch size 
	 * @param modelStream output model stream
	 */
	bool ProfileModel( const std::string& deployFile, const std::string& modelFile,
					const char* input, const Dims3& inputDims,
				    const std::vector<std::string>& outputs, uint32_t maxBatchSize, 
				    precisionType precision, deviceType device, bool allowGPUFallback,
				    nvinfer1::IInt8Calibrator* calibrator, std::ostream& modelStream);

	/**
	 * Logger class for GIE info/warning/errors
	 * Taken from tensorRT guide this is necessary for building
	 * https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html
	 */
	class Logger : public nvinfer1::ILogger			
	{
		void log( Severity severity, const char* msg ) override
		{
			if( severity != Severity::kINFO )
				printf(LOG_TRT "%s\n", msg);
		}
	} gLogger;

	// TODO add profiling tools
	
protected:

	/* Member Variables */
	std::string TRTPrototxtPath;
	std::string TRTModelPath;
	std::string TRTMeanPath;
	std::string TRTInputBlobName;
	std::string TRTCacheEnginePath;
	std::string TRTCacheCalibrationPath;

	deviceType    TRTDevice;
	precisionType TRTPrecision;
	modelType     TRTModelType;
	cudaStream_t  TRTStream;

	nvinfer1::IRuntime* TRTInfer;
	nvinfer1::ICudaEngine* TRTEngine;
	nvinfer1::IExecutionContext* TRTContext;
	
	uint32_t TRTWidth;
	uint32_t TRTHeight;
	uint32_t TRTInputSize;
	float*   TRTInputCPU;
	float*   TRTInputCUDA;

	uint32_t TRTMaxBatchSize;
	bool	 TRTEnableProfiler;
	bool     TRTEnableDebug;
	bool	 TRTAllowGPUFallback;

	Dims3 TRTInputDims;
	
	struct outputLayer
	{
		std::string name;
		Dims3 dims;
		uint32_t size;
		float* CPU;
		float* CUDA;
	};
	
	std::vector<outputLayer> TRTOutputs;
};

#endif
