
 
#include "detectNet.h"
#include "imageNet.cuh"

#include "cudaMappedMemory.h"
#include "cudaFont.h"

#include "commandLine.h"
#include "filesystem.h"


// #define OUTPUT_CVG  0	// Caffe has output coverage (confidence) heat map
// #define OUTPUT_BBOX 1	// Caffe has separate output layer for bounding box

#define OUTPUT_UFF  0	// UFF has primary output containing detection results
#define OUTPUT_NUM	1	// UFF has secondary output containing one detection count

#define CHECK_NULL_STR(x)	(x != NULL) ? x : "NULL"



// constructor
ObjectDetection::ObjectDetection( float meanPixel ) : ModelImporter()
{
	mCoverageThreshold = DETECTNET_DEFAULT_THRESHOLD;
	mMeanPixel         = meanPixel;
	mCustomClasses     = 0;
	mNumClasses        = 0;

	mClassColors[HOST]   = NULL; // cpu ptr
	mClassColors[DEVICE]   = NULL; // gpu ptr
	
	mDetectionSets[HOST] = NULL; // cpu ptr
	mDetectionSets[DEVICE] = NULL; // gpu ptr
	mDetectionSet     = 0;
	mMaxDetections    = 0;
}


// destructor
ObjectDetection::~ObjectDetection()
{
	if( mDetectionSets != NULL )
	{
		// shared memory so freeing the host frees the device
		CUDA(cudaFreeHost(mDetectionSets[HOST]));
		
		mDetectionSets[HOST] = NULL;
		mDetectionSets[DEVICE] = NULL;
	}
	
	if( mClassColors != NULL )
	{
		// shared memory so freeing the host frees the device
		CUDA(cudaFreeHost(mClassColors[HOST]));
		
		mClassColors[HOST] = NULL;
		mClassColors[DEVICEw] = NULL;
	}
}


// Create (UFF)
static detectNet* ObjectDetection::CreateUFF( const char* model, const char* class_labels, float threshold, 
						const char* input, const Dims3& inputDims, 
						const char* output, const char* numDetections,
						uint32_t maxBatchSize, precisionType precision,
				   		deviceType device, bool allowGPUFallback )
{
	detectNet* net = new detectNet();
	
	if( !net )
		return NULL;

	// see what we imprted
	printf("\n");
	printf("detectNet -- loading detection network model from:\n");
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
	// look into null params
	// TODO
	if( !net->LoadNetwork(NULL, model, NULL, input, inputDims, output_blobs,maxBatchSize, precision, device, allowGPUFallback) )
	{
		printf("detectNet -- failed to initialize.\n");
		return NULL;
	}
	
	// allocate detection sets
	if( !net->allocDetections() )
		return NULL;

	// load class descriptions
	net->loadClassDesc(class_labels);
	net->defaultClassDesc();
	
	// set default class colors
	if( !net->defaultColors() )
		return NULL;

	// set the specified threshold
	net->SetThreshold(threshold);

	return net;
}


// These are all pre trained models I have. Will get more for testing
static detectNet* ObjectDetection::CreateModel( NetworkType networkType)
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
		return detectNet::ERROR;

	ObjectDetection::NetworkType type = ObjectDetection::ERROR;

	else if( strcasecmp(modelName, "ssd-inception") == 0 )
		type = detectNet::SSD_INCEPTION_V2;
	else if( strcasecmp(modelName, "ssd-mobilenet-v1") == 0 )
		type = detectNet::SSD_MOBILENET_V1;
	else if( strcasecmp(modelName, "ssd-mobilenet-v2") == 0 )
		type = detectNet::SSD_MOBILENET_V2;

	return type;
}


// Create
ObjectDetection::ObjectDetection* ObjectDetection::Create( NetworkType model )
{
	ObjectDetection* net = new detectNet();
	
	if( !net )
		return NULL;

	if( !net->CreateModel( model ) )
		return NULL;

	return net;
}
	

// allocDetections
bool ObjectDetection::allocDetections()
{
	// determine max detections
	// TODO FIX THIS OR W/e CAN NOT USE ONNX
	if( IsModelType(MODEL_UFF) )
	{
		printf("W = %u  H = %u  C = %u\n", DIMS_W(mOutputs[OUTPUT_UFF].dims), DIMS_H(mOutputs[OUTPUT_UFF].dims), DIMS_C(mOutputs[OUTPUT_UFF].dims));
		mMaxDetections = DIMS_H(mOutputs[OUTPUT_UFF].dims) * DIMS_C(mOutputs[OUTPUT_UFF].dims);
	}
	else if( IsModelType(MODEL_ONNX) )
	{
		mMaxDetections = 1;
		mNumClasses = 1;
		printf("detectNet -- using ONNX model\n");
	}
	else if{
		assert(0);
	}

	printf("detectNet -- maximum bounding boxes:  %u\n", mMaxDetections);

	// allocate array to store detection results
	const size_t det_size = sizeof(Detection) * mNumDetectionSets * mMaxDetections;
	
	// allocate shared memory
	if( !cudaAllocMapped((void**)&mDetectionSets[0], (void**)&mDetectionSets[1], det_size) )
		return false;
	
	memset(mDetectionSets[0], 0, det_size);
	return true;
}

	
// defaultColors
// this is hard 
bool ObjectDetection::defaultColors()
{
	const uint32_t numClasses = GetNumClasses();
	
	if( !cudaAllocMapped((void**)&mClassColors[0], (void**)&mClassColors[1], numClasses * sizeof(float4)) )
		return false;

	// if there are a large number of classes (MS COCO)
	// programatically generate the class color map
	if( numClasses > 10 )
	{
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

			mClassColors[0][i*4+0] = r;
			mClassColors[0][i*4+1] = g;
			mClassColors[0][i*4+2] = b;
			mClassColors[0][i*4+3] = DETECTNET_DEFAULT_ALPHA; 

			printf(LOG_TRT "color %02i  %3i %3i %3i %3i\n", i, (int)r, (int)g, (int)b, (int)alpha);
		}
	}
	else
	{
		// blue colors, except class 1 is green
		for( uint32_t n=0; n < numClasses; n++ )
		{
			if( n != 1 )
			{
				mClassColors[0][n*4+0] = 0.0f;	// r
				mClassColors[0][n*4+1] = 200.0f;	// g
				mClassColors[0][n*4+2] = 255.0f;	// b
				mClassColors[0][n*4+3] = DETECTNET_DEFAULT_ALPHA;	// a
			}
			else
			{
				mClassColors[0][n*4+0] = 0.0f;	// r
				mClassColors[0][n*4+1] = 255.0f;	// g
				mClassColors[0][n*4+2] = 175.0f;	// b
				mClassColors[0][n*4+3] = 75.0f;	// a
			}
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
	for( uint32_t n=mClassDesc.size(); n < numClasses; n++ )
	{
		char syn_str[10];
		sprintf(syn_str, "n%08u", mCustomClasses);
		printf(LOG_TRT "default classes %d\n",n);
		char desc_str[16];
		sprintf(desc_str, "class #%u", mCustomClasses);

		mClassSynset.push_back(syn_str);
		mClassDesc.push_back(desc_str);

		mCustomClasses++;
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

			mClassSynset.push_back(a);
			mClassDesc.push_back(b);
		}
		else if( len > 0 )	// no 9-character synset prefix (i.e. from DIGITS snapshot)
		{
			char a[10];
			sprintf(a, "n%08u", mCustomClasses);

			//printf("a=%s b=%s (custom non-synset)\n", a, str);
			mCustomClasses++;

			if( str[len-1] == '\n' )
				str[len-1] = 0;

			mClassSynset.push_back(a);
			mClassDesc.push_back(str);
		}
	}
	
	fclose(f);
	
	printf("detectNet -- loaded %zu class info entries\n", mClassDesc.size());
	
	if( mClassDesc.size() == 0 )
		return false;

	if( IsModelType(MODEL_UFF) )
		mNumClasses = mClassDesc.size();

	printf("detectNet -- number of object classes:  %u\n", mNumClasses);
	mClassPath = path;	
	return true;
}



// Detect
int ObjectDetection::Detect( float* input, uint32_t width, uint32_t height, Detection** detections, uint32_t overlay )
{
	Detection* det = mDetectionSets[0] + mDetectionSet * GetMaxDetections();

	if( detections != NULL )
		*detections = det;

	mDetectionSet++;

	if( mDetectionSet >= mNumDetectionSets )
		mDetectionSet = 0;
	
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
		if( CUDA_FAILED(cudaPreImageNetNormBGR((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
										  make_float2(-1.0f, 1.0f), GetStream())) )
		{
			printf(LOG_TRT "detectNet::Detect() -- cudaPreImageNetNorm() failed\n");
			return -1;
		}
	}
	else if( IsModelType(MODEL_ONNX) )
	{
		// downsample, convert to band-sequential RGB, and apply pixel normalization, mean pixel subtraction and standard deviation
		if( CUDA_FAILED(cudaPreImageNetNormMeanRGB((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight, 
										   make_float2(0.0f, 1.0f), 
										   make_float3(0.485f, 0.456f, 0.406f),
										   make_float3(0.229f, 0.224f, 0.225f), 
										   GetStream())) )
		{
			printf(LOG_TRT "imageNet::PreProcess() -- cudaPreImageNetNormMeanRGB() failed\n");
			return false;
		}
	}
	else if{
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
		const int rawDetections = *(int*)mOutputs[OUTPUT_NUM].CPU;
		const int rawParameters = DIMS_W(mOutputs[OUTPUT_UFF].dims);

		// filter the raw detections by thresholding the confidence
		for( int n=0; n < rawDetections; n++ )
		{
			float* object_data = mOutputs[OUTPUT_UFF].CPU + n * rawParameters;

			if( object_data[2] < mCoverageThreshold )
				continue;

			detections[numDetections].Instance   = numDetections; //(uint32_t)object_data[0];
			detections[numDetections].ClassID    = (uint32_t)object_data[1];
			detections[numDetections].Confidence = object_data[2];
			detections[numDetections].Left       = object_data[3] * width;
			detections[numDetections].Top        = object_data[4] * height;
			detections[numDetections].Right      = object_data[5] * width;
			detections[numDetections].Bottom	  = object_data[6] * height;

			if( detections[numDetections].ClassID >= mNumClasses )
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
	else if( IsModelType(MODEL_ONNX) )
	{
		float* coord = mOutputs[0].CPU;

		coord[0] = ((coord[0] + 1.0f) * 0.5f) * float(width);
		coord[1] = ((coord[1] + 1.0f) * 0.5f) * float(height);
		coord[2] = ((coord[2] + 1.0f) * 0.5f) * float(width);
		coord[3] = ((coord[3] + 1.0f) * 0.5f) * float(height);

		printf(LOG_TRT "detectNet::Detect() -- ONNX -- coord (%f, %f) (%f, %f)  image %ux%u\n", coord[0], coord[1], coord[2], coord[3], width, height);

		detections[numDetections].Instance   = numDetections;
		detections[numDetections].ClassID    = 0;
		detections[numDetections].Confidence = 1;
		detections[numDetections].Left       = coord[0];
		detections[numDetections].Top        = coord[1];
		detections[numDetections].Right      = coord[2];
		detections[numDetections].Bottom	  = coord[3];	
	
		numDetections++;
	}

	PROFILER_END(PROFILER_POSTPROCESS);

	// render the overlay
	if( overlay != 0 && numDetections > 0 )
	{
		if( !Overlay(rgba, rgba, width, height, detections, numDetections, overlay) )
			printf(LOG_TRT "detectNet::Detect() -- failed to render overlay\n");
	}
	
	// wait for GPU to complete work			
	CUDA(cudaDeviceSynchronize());

	// return the number of detections
	return numDetections;
}


// clusterDetections (UFF)
// TODO check if being called, it is
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


// from detectNet.cu
cudaError_t cudaDetectionOverlay( float4* input, float4* output, uint32_t width, uint32_t height, detectNet::Detection* detections, int numDetections, float4* colors );

// Overlay
bool ObjectDetection::Overlay( float* input, float* output, uint32_t width, uint32_t height, Detection* detections, uint32_t numDetections, uint32_t flags )
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
		if( CUDA_FAILED(cudaDetectionOverlay((float4*)input, (float4*)output, width, height, detections, numDetections, (float4*)mClassColors[1])) )
			return false;
	}

	// class label overlay
	if( (flags & OVERLAY_LABEL) || (flags & OVERLAY_CONFIDENCE) )
	{
		static cudaFont* font = NULL;

		// make sure the font object is created
		if( !font )
		{
			font = cudaFont::Create(adaptFontSize(width));
	
			if( !font )
			{
				printf(LOG_TRT "detectNet -- Overlay() was called with OVERLAY_FONT, but failed to create cudaFont()\n");
				return false;
			}
		}

		// draw each object's description
		std::vector< std::pair< std::string, int2 > > labels;

		for( uint32_t n=0; n < numDetections; n++ )
		{
			const char* className  = GetClassDesc(detections[n].ClassID);
			const float confidence = detections[n].Confidence * 100.0f;
			const int2  position   = make_int2(detections[n].Left+5, detections[n].Top+3);
			
			if( flags & OVERLAY_CONFIDENCE )
			{
				char str[256];

				if( (flags & OVERLAY_LABEL) && (flags & OVERLAY_CONFIDENCE) )
					sprintf(str, "%s %.1f%%", className, confidence);
				else
					sprintf(str, "%.1f%%", confidence);

				labels.push_back(std::pair<std::string, int2>(str, position));
			}
			else
			{
				// overlay label only
				labels.push_back(std::pair<std::string, int2>(className, position));
			}
		}

		font->OverlayText((float4*)input, width, height, labels, make_float4(255,255,255,255));
	}
	
	// PROFILER_END(PROFILER_VISUALIZE);
	return true;
}


// OverlayFlagsFromStr
uint32_t ObjectDetection::OverlayFlagsFromStr( const char* str_user )
{
	if( !str_user )
		return OVERLAY_BOX;

	// copy the input string into a temporary array,
	// because strok modifies the string
	const size_t str_length = strlen(str_user);

	if( str_length == 0 )
		return OVERLAY_BOX;

	char* str = (char*)malloc(str_length + 1);

	if( !str )
		return OVERLAY_BOX;

	strcpy(str, str_user);

	// tokenize string by delimiters ',' and '|'
	const char* delimiters = ",|";
	char* token = strtok(str, delimiters);

	if( !token )
	{
		free(str);
		return OVERLAY_BOX;
	}

	// look for the tokens:  "box", "label", and "none"
	uint32_t flags = OVERLAY_NONE;

	while( token != NULL )
	{
		//printf("%s\n", token);

		if( strcasecmp(token, "box") == 0 )
			flags |= OVERLAY_BOX;
		else if( strcasecmp(token, "label") == 0 || strcasecmp(token, "labels") == 0 )
			flags |= OVERLAY_LABEL;
		else if( strcasecmp(token, "conf") == 0 || strcasecmp(token, "confidence") == 0 )
			flags |= OVERLAY_CONFIDENCE;

		token = strtok(NULL, delimiters);
	}	

	free(str);
	return flags;
}


// SetClassColor
void ObjectDetection::SetClassColor( uint32_t classIndex, float r, float g, float b, float a )
{
	if( classIndex >= GetNumClasses() || !mClassColors[0] )
		return;
	
	const uint32_t i = classIndex * 4;
	
	mClassColors[HOST][i+0] = r;
	mClassColors[HOST][i+1] = g;
	mClassColors[HOST][i+2] = b;
	mClassColors[HOST][i+3] = a;
}


// SetOverlayAlpha
void ObjectDetection::SetOverlayAlpha( float alpha )
{
	const uint32_t numClasses = GetNumClasses();

	for( uint32_t n=0; n < numClasses; n++ )
		mClassColors[HOST][n*4+3] = alpha;
}
