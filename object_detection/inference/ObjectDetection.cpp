
 
#include "ObjectDetection.h"

#include "../cuda_utilities/imageNet.cuh"
#include "../cuda_utilities/detectNet.cuh"

#include "../cuda_utilities/cudaMappedMemory.h"
#include "../cuda_utilities/cudaUtility.h"

#include "../utils/filesystem.h"
#include <assert.h>

// #define OUTPUT_CVG  0	// Caffe has output coverage (confidence) heat map
// #define OUTPUT_BBOX 1	// Caffe has separate output layer for bounding box

#define OUTPUT_UFF  0	// UFF has primary output containing detection results
#define OUTPUT_NUM	1	// UFF has secondary output containing one detection count

#define CHECK_NULL_STR(x)	(x != NULL) ? x : "NULL"



// constructor
ObjectDetection::ObjectDetection( float meanPixel ) : ModelImporter()
{
	TRTCoverageThreshold = DETECTNET_DEFAULT_THRESHOLD;
	TRTMeanPixel         = meanPixel;
	TRTCustomClasses     = 0;
	TRTNumClasses        = 0;

	TRTClassColors[HOST]   = NULL; // cpu ptr
	TRTClassColors[DEVICE]   = NULL; // gpu ptr
	
	TRTDetectionSets[HOST] = NULL; // cpu ptr
	TRTDetectionSets[DEVICE] = NULL; // gpu ptr
	TRTDetectionSet     = 0;
	TRTMaxDetections    = 0;
}


// destructor
ObjectDetection::~ObjectDetection()
{
	if( TRTDetectionSets != NULL )
	{
		// shared memory so freeing the host frees the device
		CUDA(cudaFreeHost(TRTDetectionSets[HOST]));
		
		TRTDetectionSets[HOST] = NULL;
		TRTDetectionSets[DEVICE] = NULL;
	}
	
	if( TRTClassColors != NULL )
	{
		// shared memory so freeing the host frees the device
		CUDA(cudaFreeHost(TRTClassColors[HOST]));
		
		TRTClassColors[HOST] = NULL;
		TRTClassColors[DEVICE] = NULL;
	}
}


// Create (UFF)
ObjectDetection* ObjectDetection::CreateUFF( const char* model, const char* class_labels, float threshold, 
						const char* input, const Dims3& inputDims, 
						const char* output, const char* numDetections,
						uint32_t maxBatchSize, precisionType precision,
				   		deviceType device, bool allowGPUFallback )
{
	ObjectDetection* net = new ObjectDetection();
	
	if( !net )
		return NULL;

	// see what we imprted
	printf("\n");
	printf("ObjectDetection -- loading ObjectDetection network model from:\n");
	printf("          -- model        %s\n", CHECK_NULL_STR(model));
	printf("          -- input_blob   '%s'\n", CHECK_NULL_STR(input));
	printf("          -- output_blob  '%s'\n", CHECK_NULL_STR(output));
	printf("          -- output_count '%s'\n", CHECK_NULL_STR(numDetections));
	printf("          -- class_labels %s\n", CHECK_NULL_STR(class_labels));
	printf("          -- threshold    %f\n", threshold);
	printf("          -- batch_size   %u\n\n", maxBatchSize);
	
	// enable debug flag maybe make a macro
	//net->EnableDebug();
	
	// create list of output names	
	std::vector<std::string> output_blobs;

	if( output != NULL )
		output_blobs.push_back(output);

	// TODO num detections
	if( numDetections != NULL )
		output_blobs.push_back(numDetections);
	
	// load the model this is based on the model maker loader
	if( !net->LoadNetwork(NULL, model, NULL, input, inputDims, output_blobs,maxBatchSize, precision, device, allowGPUFallback) )
	{
		printf("detectNet -- failed to initialize.\n");
		return NULL;
	}

	printf(LOG_TRT " allocating ... \n" );

	// allocate detection sets
	if( !net->allocDetections() )
		return NULL;


	printf(LOG_TRT " loading class descriptors \n" );

	// load class descriptions
	net->loadClassDesc(class_labels);
	net->defaultClassDesc();
	
	// set default class colors
	if( !net->defaultColors() )
		return NULL;

	printf(LOG_TRT "setting threshold \n");

	// set the specified threshold
	net->SetThreshold(threshold);

	printf(LOG_TRT "threshold set! exiting function\n");

	return net;
}


// These are all pre trained models I have. Will get more for testing
ObjectDetection* ObjectDetection::CreateModel( NetworkType networkType)
{
	// for now we make this default values later we can maybe do some CLI
	precisionType precision = TYPE_FASTEST;
	deviceType device = DEVICE_GPU;
	bool allowGPUFallback = true;
	float threshold=DETECTNET_DEFAULT_THRESHOLD;
	uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE;


	if( networkType == SSD_INCEPTION_V2 )
		return CreateUFF("networks/SSD-Inception-v2/ssd_inception_v2_coco.uff", "networks/SSD-Inception-v2/ssd_coco_labels.txt", threshold, "Input", Dims3(3,300,300), "NMS", "NMS_1", maxBatchSize, precision, device, allowGPUFallback);
	else if( networkType == SSD_MOBILENET_V1 )
		return CreateUFF("networks/SSD-Mobilenet-v1/ssd_mobilenet_v1_coco.uff", "networks/SSD-Mobilenet-v1/ssd_coco_labels.txt", threshold, "Input", Dims3(3,300,300), "Postprocessor", "Postprocessor_1", maxBatchSize, precision, device, allowGPUFallback);
	else if( networkType == SSD_MOBILENET_V2 )
		return CreateUFF("networks/SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.uff", "networks/SSD-Mobilenet-v2/ssd_coco_labels.txt", threshold, "Input", Dims3(3,300,300), "NMS", "NMS_1", maxBatchSize, precision, device, allowGPUFallback);
	else
		return NULL;
}


// NetworkTypeFromStr
ObjectDetection::NetworkType ObjectDetection::NetworkTypeFromStr( const char* modelName )
{
	if( !modelName )
		return ObjectDetection::ERROR;

	ObjectDetection::NetworkType type = ObjectDetection::ERROR;

	if( strcasecmp(modelName, "ssd-inception") == 0 )
		type = ObjectDetection::SSD_INCEPTION_V2;
	else if( strcasecmp(modelName, "ssd-mobilenet-v1") == 0 )
		type = ObjectDetection::SSD_MOBILENET_V1;
	else if( strcasecmp(modelName, "ssd-mobilenet-v2") == 0 )
		type = ObjectDetection::SSD_MOBILENET_V2;

	return type;
}


// Create
ObjectDetection* ObjectDetection::Create( NetworkType model )
{
	// add if else for type of model
	return CreateModel( model ); 
}
	

// allocDetections
bool ObjectDetection::allocDetections()
{
	// determine max detections
	// TODO FIX THIS OR W/e CAN NOT USE ONNX
	if( IsModelType(MODEL_UFF) )
	{
		printf("W = %u  H = %u  C = %u\n", DIMS_W(TRTOutputs[OUTPUT_UFF].dims), DIMS_H(TRTOutputs[OUTPUT_UFF].dims), DIMS_C(TRTOutputs[OUTPUT_UFF].dims));
		TRTMaxDetections = DIMS_H(TRTOutputs[OUTPUT_UFF].dims) * DIMS_C(TRTOutputs[OUTPUT_UFF].dims);
	}
	else if( IsModelType(MODEL_ONNX) )
	{
		TRTMaxDetections = 1;
		TRTNumClasses = 1;
		printf("detectNet -- using ONNX model\n");
	}
	else{
		assert(0);
	}

	printf("detectNet -- maximum bounding boxes:  %u\n", TRTMaxDetections);

	// allocate array to store detection results
	const size_t det_size = sizeof(Detection) * TRTNumDetectionSets * TRTMaxDetections;
	
	// allocate shared memory
	if( !cudaAllocMapped((void**)&TRTDetectionSets[HOST], (void**)&TRTDetectionSets[DEVICE], det_size) )
		return false;
	
	memset(TRTDetectionSets[HOST], 0, det_size);
	return true;
}

	
// defaultColors
// this is hard 
bool ObjectDetection::defaultColors()
{
	const uint32_t numClasses = GetNumClasses();
	
	if( !cudaAllocMapped((void**)&TRTClassColors[HOST], (void**)&TRTClassColors[DEVICE], numClasses * sizeof(float4)) )
		return false;

	// programatically generate the class color map
		// https://github.com/dusty-nv/pytorch-segmentation/blob/16882772bc767511d892d134918722011d1ea771/datasets/sun_remap.py#L90
		#define bitget(byteval, idx)	((byteval & (1 << idx)) != 0)

		for( int i=0; i < numClasses; i++ )
		{
			int r = 0;
			int g = 0;
			int b = 0;
			int c = i;

			for( int j=0; j < 8; j++ )
			{
				r = r | (bitget(c, 0) << 7 - j);
				g = g | (bitget(c, 1) << 7 - j);
				b = b | (bitget(c, 2) << 7 - j);
				c = c >> 3;
			}

			TRTClassColors[0][i*4+0] = r;
			TRTClassColors[0][i*4+1] = g;
			TRTClassColors[0][i*4+2] = b;
			TRTClassColors[0][i*4+3] = DETECTNET_DEFAULT_ALPHA; 

//			printf(LOG_TRT "color %02i  %3i %3i %3i %3i\n", i, (int)r, (int)g, (int)b, (int)DETECTNET_DEFAULT_ALPHA);
		}
		// blue colors, except class 1 is green
		for( uint32_t n=0; n < numClasses; n++ )
		{
			if( n != 1 )
			{
				TRTClassColors[0][n*4+0] = 0.0f;	// r
				TRTClassColors[0][n*4+1] = 200.0f;	// g
				TRTClassColors[0][n*4+2] = 255.0f;	// b
				TRTClassColors[0][n*4+3] = DETECTNET_DEFAULT_ALPHA;	// a
			}
			else
			{
				TRTClassColors[0][n*4+0] = 0.0f;	// r
				TRTClassColors[0][n*4+1] = 255.0f;	// g
				TRTClassColors[0][n*4+2] = 175.0f;	// b
				TRTClassColors[0][n*4+3] = 75.0f;	// a
			}
		}

	return true;
}


// defaultClassDesc
void ObjectDetection::defaultClassDesc()
{
	const uint32_t numClasses = GetNumClasses();
	
	printf(LOG_TRT "number of classes %d\n",numClasses);

	// assign defaults to classes that have no info
	for( uint32_t n=TRTClassDesc.size(); n < numClasses; n++ )
	{
		char syn_str[10];
		sprintf(syn_str, "n%08u", TRTCustomClasses);
		printf(LOG_TRT "default classes %d\n",n);
		char desc_str[16];
		sprintf(desc_str, "class #%u", TRTCustomClasses);

		TRTClassSynset.push_back(syn_str);
		TRTClassDesc.push_back(desc_str);

		TRTCustomClasses++;
	}
}


/* loadClassDesc
*
* a parser that goes through and gets our data set
*/
bool ObjectDetection::loadClassDesc( const char* filename )
{
	if( !filename )
		return false;
	
	// locate the file
	const std::string path = locateFile(filename);

	if( path.length() == 0 )
	{
		printf("detectNet -- failed to find %s\n", filename);
		return false;
	}

	// open the file
	FILE* f = fopen(path.c_str(), "r");
	
	if( !f )
	{
		printf("detectNet -- failed to open %s\n", path.c_str());
		return false;
	}
	
	// read class descriptions
	char str[512];

	while( fgets(str, 512, f) != NULL )
	{
		const int syn = 9;  // length of synset prefix (in characters)
		const int len = strlen(str);
		
		if( len > syn && str[0] == 'n' && str[syn] == ' ' )
		{
			str[syn]   = 0;
			str[len-1] = 0;
	
			const std::string a = str;
			const std::string b = (str + syn + 1);
	
			//printf("a=%s b=%s\n", a.c_str(), b.c_str());

			TRTClassSynset.push_back(a);
			TRTClassDesc.push_back(b);
		}
		else if( len > 0 )	// no 9-character synset prefix (i.e. from DIGITS snapshot)
		{
			char a[10];
			sprintf(a, "n%08u", TRTCustomClasses);

			//printf("a=%s b=%s (custom non-synset)\n", a, str);
			TRTCustomClasses++;

			if( str[len-1] == '\n' )
				str[len-1] = 0;

			TRTClassSynset.push_back(a);
			TRTClassDesc.push_back(str);
		}
	}
	
	fclose(f);
	
	printf("detectNet -- loaded %zu class info entries\n", TRTClassDesc.size());
	
	if( TRTClassDesc.size() == 0 )
		return false;

	if( IsModelType(MODEL_UFF) )
		TRTNumClasses = TRTClassDesc.size();

	printf("detectNet -- number of object classes:  %u\n", TRTNumClasses);
	TRTClassPath = path;	
	return true;
}



// Detect
int ObjectDetection::Detect( float* input, uint32_t width, uint32_t height, Detection** detections, uint32_t overlay )
{
	Detection* det = TRTDetectionSets[HOST] + TRTDetectionSet * GetMaxDetections();

	if( detections != NULL )
		*detections = det;

	TRTDetectionSet++;

	if( TRTDetectionSet >= TRTNumDetectionSets )
		TRTDetectionSet = 0;
	
	return Detect(input, width, height, det, overlay);
}

	
// Detect
int ObjectDetection::Detect( float* rgba, uint32_t width, uint32_t height, Detection* detections, uint32_t overlay )
{
	if( !rgba || width == 0 || height == 0 || !detections )
	{
		printf(LOG_TRT "detectNet::Detect( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return -1;
	}

	// PROFILER_BEGIN(PROFILER_PREPROCESS);

	// preprocess setp we need to convert our image based on what it was trained on
	// This will likely be different for EVERY network 
	if( IsModelType(MODEL_UFF) )
	{
		if( CUDA_FAILED(cudaPreImageNetNormBGR((float4*)rgba, width, height, TRTInputCUDA, TRTWidth, TRTHeight,
										  make_float2(-1.0f, 1.0f), GetStream())) )
		{
			printf(LOG_TRT "detectNet::Detect() -- cudaPreImageNetNorm() failed\n");
			return -1;
		}
	}
	else{
		return false;
	}

	// PROFILER_END(PROFILER_PREPROCESS);
	// PROFILER_BEGIN(PROFILER_NETWORK);

	// process with TensorRT
	void* inferenceBuffers[] = { TRTInputCUDA, TRTOutputs[HOST].CUDA, TRTOutputs[DEVICE].CUDA };
	
	// this is synchronous
	// consider changing to asynchronous
	// context->enqueue(batchSize, buffers, stream, nullptr);
	// may need some ring buffer to do this properly
	// and pipeline our loop
	if( !TRTContext->execute(1, inferenceBuffers) )
	{
		printf(LOG_TRT "detectNet::Detect() -- failed to execute TensorRT context\n");
		return -1;
	}
	
	// PROFILER_END(PROFILER_NETWORK);
	// PROFILER_BEGIN(PROFILER_POSTPROCESS);

	// post-processing / clustering
	int numDetections = 0;

	if( IsModelType(MODEL_UFF) )
	{		
		const int rawDetections = *(int*)TRTOutputs[OUTPUT_NUM].CPU;
		const int rawParameters = DIMS_W(TRTOutputs[OUTPUT_UFF].dims);

		// filter the raw detections by thresholding the confidence
		for( int n=0; n < rawDetections; n++ )
		{
			float* object_data = TRTOutputs[OUTPUT_UFF].CPU + n * rawParameters;

			if( object_data[2] < TRTCoverageThreshold )
				continue;

			detections[numDetections].Instance   = numDetections; //(uint32_t)object_data[0];
			detections[numDetections].ClassID    = (uint32_t)object_data[1];
			detections[numDetections].Confidence = object_data[2];
			detections[numDetections].Left       = object_data[3] * width;
			detections[numDetections].Top        = object_data[4] * height;
			detections[numDetections].Right      = object_data[5] * width;
			detections[numDetections].Bottom	  = object_data[6] * height;

			if( detections[numDetections].ClassID >= TRTNumClasses )
			{
				printf(LOG_TRT "detectNet::Detect() -- detected object has invalid classID (%u)\n", detections[numDetections].ClassID);
				detections[numDetections].ClassID = 0;
			}

			if( strcmp(GetClassDesc(detections[numDetections].ClassID), "void") == 0 )
				continue;

			numDetections += clusterDetections(detections, numDetections);
		}

		// sort the detections by confidence value
		sortDetections(detections, numDetections);
	}

	// PROFILER_END(PROFILER_POSTPROCESS);

	// render the overlay
	if( overlay != 0 && numDetections > 0 )
	{
		if( !Overlay(rgba, width, height, detections, numDetections, overlay) )
			printf(LOG_TRT "detectNet::Detect() -- failed to render overlay\n");
	}
	
	// wait for GPU to complete work			
	CUDA(cudaDeviceSynchronize());

	// return the number of detections
	return numDetections;
}


// clusterDetections (UFF)
int ObjectDetection::clusterDetections( Detection* detections, int n, float threshold )
{
	if( n == 0 )
		return 1;

	// test each detection to see if it intersects
	for( int m=0; m < n; m++ )
	{
		if( detections[n].Intersects(detections[m], threshold) )	// TODO NMS or different threshold for same classes?
		{
			// if the intersecting detections have different classes, pick the one with highest confidence
			// otherwise if they have the same object class, expand the detection bounding box
			if( detections[n].ClassID != detections[m].ClassID )
			{
				if( detections[n].Confidence > detections[m].Confidence )
				{
					detections[m] = detections[n];

					detections[m].Instance = m;
					detections[m].ClassID = detections[n].ClassID;
					detections[m].Confidence = detections[n].Confidence;					
				}
			}
			else
			{
				detections[m].Expand(detections[n]);
				detections[m].Confidence = fmaxf(detections[n].Confidence, detections[m].Confidence);
			}

			return 0; // merged detection
		}
	}

	return 1;	// new detection
}


// sortDetections (UFF)
void ObjectDetection::sortDetections( Detection* detections, int numDetections )
{
	if( numDetections < 2 )
		return;

	// order by area (descending) or confidence (ascending)
	for( int i=0; i < numDetections-1; i++ )
	{
		for( int j=0; j < numDetections-i-1; j++ )
		{
			if( detections[j].Area() < detections[j+1].Area() ) //if( detections[j].Confidence > detections[j+1].Confidence )
			{
				const Detection det = detections[j];
				detections[j] = detections[j+1];
				detections[j+1] = det;
			}
		}
	}

	// renumber the instance ID's
	for( int i=0; i < numDetections; i++ )
		detections[i].Instance = i;	
}


// Overlay
bool ObjectDetection::Overlay( float* input, uint32_t width, uint32_t height, Detection* detections, uint32_t numDetections, uint32_t flags )
{
	// PROFILER_BEGIN(PROFILER_VISUALIZE);

	if( flags == 0 )
	{
		printf(LOG_TRT "detectNet -- Overlay() was called with OVERLAY_NONE, returning false\n");
		return false;
	}

	// bounding box overlay
	if( flags & OVERLAY_BOX )
	{
		if( CUDA_FAILED(cudaDetectionOverlay((float4*)input, width, height, detections, numDetections, (float4*)TRTClassColors[1])) )
			return false;
	}
	
	// PROFILER_END(PROFILER_VISUALIZE);
	return true;
}
