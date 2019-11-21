
 
#ifndef __OBJECT_DETECTION_H__
#define __OBJECT_DETECTION_H__


#include "ModelImporter.h" 

/**
 * Default value of the minimum detection threshold
 */
#define DETECTNET_DEFAULT_THRESHOLD 0.5f

/**
 * Default alpha blending value used during overlay
 */
#define DETECTNET_DEFAULT_ALPHA 120


/**
 * Object recognition with TensorRT acceleration
 */
class ObjectDetection : public ModelImporter
{
public:
	/**
	 * Object Detection result.
	 */
	struct Detection
	{
		// Object Info
		uint32_t Instance;	// Index of this unique object instance
		uint32_t ClassID;
		float Confidence;

		// Bounding Box Coordinates in pixel format
		float Left;
		float Right;
		float Top;
		float Bottom;

		/**< Calculate the width of the object */
		inline float Width() const											{ return Right - Left; }

		/**< Calculate the height of the object */
		inline float Height() const											{ return Bottom - Top; }

		/**< Calculate the area of the object */
		inline float Area() const											{ return Width() * Height(); }

		/**< Calculate the width of the bounding box */
		static inline float Width( float x1, float x2 )							{ return x2 - x1; }

		/**< Calculate the height of the bounding box */
		static inline float Height( float y1, float y2 )							{ return y2 - y1; }

		/**< Calculate the area of the bounding box */
		static inline float Area( float x1, float y1, float x2, float y2 )			{ return Width(x1,x2) * Height(y1,y2); }

		/**< Return the center of the object */
		inline void Center( float* x, float* y ) const							{ if(x) *x = Left + Width() * 0.5f; if(y) *y = Top + Height() * 0.5f; }

		/**< Return true if the coordinate is inside the bounding box */
		inline bool Contains( float x, float y ) const							{ return x >= Left && x <= Right && y >= Top && y <= Bottom; }
		
		/**< Return true if the bounding boxes intersect and exceeds area % threshold */	
		inline bool Intersects( const Detection& det, float areaThreshold=0.0f ) const  { return (IntersectionArea(det) / fmaxf(Area(), det.Area()) > areaThreshold); }
	
		/**< Return true if the bounding boxes intersect and exceeds area % threshold */	
		inline bool Intersects( float x1, float y1, float x2, float y2, float areaThreshold=0.0f ) const  { return (IntersectionArea(x1,y1,x2,y2) / fmaxf(Area(), Area(x1,y1,x2,y2)) > areaThreshold); }
	
		/**< Return the area of the bounding box intersection */
		inline float IntersectionArea( const Detection& det ) const					{ return IntersectionArea(det.Left, det.Top, det.Right, det.Bottom); }

		/**< Return the area of the bounding box intersection */
		inline float IntersectionArea( float x1, float y1, float x2, float y2 ) const	{ if(!Overlaps(x1,y1,x2,y2)) return 0.0f; return (fminf(Right, x2) - fmaxf(Left, x1)) * (fminf(Bottom, y2) - fmaxf(Top, y1)); }

		/**< Return true if the bounding boxes overlap */
		inline bool Overlaps( const Detection& det ) const						{ return !(det.Left > Right || det.Right < Left || det.Top > Bottom || det.Bottom < Top); }
		
		/**< Return true if the bounding boxes overlap */
		inline bool Overlaps( float x1, float y1, float x2, float y2 ) const			{ return !(x1 > Right || x2 < Left || y1 > Bottom || y2 < Top); }
		
		/**< Expand the bounding box if they overlap (return true if so) */
		inline bool Expand( float x1, float y1, float x2, float y2 ) 	     		{ if(!Overlaps(x1, y1, x2, y2)) return false; Left = fminf(x1, Left); Top = fminf(y1, Top); Right = fmaxf(x2, Right); Bottom = fmaxf(y2, Bottom); return true; }
		
		/**< Expand the bounding box if they overlap (return true if so) */
		inline bool Expand( const Detection& det )      							{ if(!Overlaps(det)) return false; Left = fminf(det.Left, Left); Top = fminf(det.Top, Top); Right = fmaxf(det.Right, Right); Bottom = fmaxf(det.Bottom, Bottom); return true; }

		/**< Reset all member variables to zero */
		inline void Reset()													{ Instance = 0; ClassID = 0; Confidence = 0; Left = 0; Right = 0; Top = 0; Bottom = 0; } 								

		/**< Default constructor */
		inline Detection()													{ Reset(); }
	};

	/**
	 * Overlay flags useful for debugging.
	 */
	enum OverlayFlags
	{
		OVERLAY_NONE       = 0,			/**< No overlay. */
		OVERLAY_BOX        = (1 << 0),	/**< Overlay the object bounding boxes */
	};

	/*
	* host or device pointers 
	*/ 
	enum Ownership
	{
		HOST=0,
		DEVICE=1
	};
	
	/**
	 * Network choice enumeration.
	 */
	enum NetworkType
	{
		SSD_MOBILENET_V1=0,	/**< SSD Mobilenet-v1 UFF model, trained on MS-COCO */
		SSD_MOBILENET_V2,	/**< SSD Mobilenet-v2 UFF model, trained on MS-COCO */
		SSD_INCEPTION_V2,	/**< SSD Inception-v2 UFF model, trained on MS-COCO */
		ERROR
	};

	/**
	 * Parse a string to one of the built-in pretrained models.
	 */
	static NetworkType NetworkTypeFromStr( const char* model_name );

	/**
	 * Parse a string sequence into OverlayFlags enum.
	 */
	static uint32_t OverlayFlagsFromStr( const char* flags );

	static ObjectDetection* CreateModel( NetworkType networkType);

	static ObjectDetection* CreateUFF( const char* model, const char* class_labels, float threshold, 
						const char* input, const Dims3& inputDims, 
						const char* output, const char* numDetections,
						uint32_t maxBatchSize, precisionType precision,
				   		deviceType device, bool allowGPUFallback );

	/**
	 * Load a new network instance by parsing the command line.
	 */
	static ObjectDetection* Create( ObjectDetection::NetworkType model );

	/**
	 * Destroy
	 */
	virtual ~ObjectDetection();

	/**
	 * Detect object locations from an RGBA image, returning an array containing the detection results.
	 * @param[in]  input float4 RGBA input image in CUDA device memory.
	 * @param[in]  width width of the input image in pixels.
	 * @param[in]  height height of the input image in pixels.
	 * @param[out] detections pointer that will be set to array of detection results (residing in shared CPU/GPU memory)
	 * @param[in]  overlay bitwise OR combination of overlay flags (@see OverlayFlags and @see Overlay()), or OVERLAY_NONE.
	 * @returns    The number of detected objects, 0 if there were no detected objects, and -1 if an error was encountered.
	 */
	int Detect( float* input, uint32_t width, uint32_t height, Detection** detections, uint32_t overlay=OVERLAY_BOX );
	
	/**
	 * Detect object locations in an RGBA image, into an array of the results allocated by the user.
	 * @param[in]  input float4 RGBA input image in CUDA device memory.
	 * @param[in]  width width of the input image in pixels.
	 * @param[in]  height height of the input image in pixels.
	 * @param[out] detections pointer to user-allocated array that will be filled with the detection results.
	 *                        @see GetMaxDetections() for the number of detection results that should be allocated in this buffer.
	 * @param[in]  overlay bitwise OR combination of overlay flags (@see OverlayFlags and @see Overlay()), or OVERLAY_NONE.
	 * @returns    The number of detected objects, 0 if there were no detected objects, and -1 if an error was encountered.
	 */
	int Detect( float* input, uint32_t width, uint32_t height, Detection* detections, uint32_t overlay=OVERLAY_BOX );
	
	/**
	 * Draw the detected bounding boxes overlayed on an RGBA image.
	 * @note Overlay() will automatically be called by default by Detect(), if the overlay parameter is true 
	 * @param input float4 RGBA input image in CUDA device memory.
	 * @param output float4 RGBA output image in CUDA device memory.
	 * @param detections Array of detections allocated in CUDA device memory.
	 */
	bool Overlay( float* input, float* output, uint32_t width, uint32_t height, Detection* detections, uint32_t numDetections, uint32_t flags );
	
	/**
	 * Retrieve the minimum threshold for detection.
	 */
	inline float GetThreshold() const							{ return TRTCoverageThreshold; }

	/**
	 * Set the minimum threshold for detection.
	 */
	inline void SetThreshold( float threshold ) 					{ TRTCoverageThreshold = threshold; }

	/**
	 * Retrieve the maximum number of simultaneous detections the network supports.
	 * Knowing this is useful for allocating the buffers to store the output detection results.
	 */
	inline uint32_t GetMaxDetections() const					{ return TRTMaxDetections; } 
		
	/**
	 * Retrieve the number of object classes supported in the detector
	 */
	inline uint32_t GetNumClasses() const						{ return TRTNumClasses; }

	/**
	 * Retrieve the description of a particular class.
	 */
	inline const char* GetClassDesc( uint32_t index )	const		{ return TRTClassDesc[index].c_str(); }
	
	/**
	 * Retrieve the class synset category of a particular class.
	 */
	inline const char* GetClassSynset( uint32_t index ) const		{ return TRTClassSynset[index].c_str(); }
	
	/**
 	 * Retrieve the path to the file containing the class descriptions.
	 */
	inline const char* GetClassPath() const						{ return TRTClassPath.c_str(); }

	/**
	 * Retrieve the RGBA visualization color a particular class.
	 */
	inline float* GetClassColor( uint32_t classIndex ) const		{ return TRTClassColors[0] + (classIndex*4); }

	/**
	 * Set the visualization color of a particular class of object.
	 */
	void SetClassColor( uint32_t classIndex, float r, float g, float b, float a=255.0f );
	
	/**
 	 * Set overlay alpha blending value for all classes (between 0-255).
	 */
	void SetOverlayAlpha( float alpha );

	
protected:

	// constructor
	ObjectDetection( float meanPixel=0.0f );

	bool allocDetections();
	bool defaultColors();
	void defaultClassDesc();
	bool loadClassDesc( const char* filename );

	bool init( const char* prototxt_path, const char* model_path, const char* mean_binary, const char* class_labels, 
			 float threshold, const char* input, const char* coverage, const char* bboxes, uint32_t maxBatchSize, 
			 precisionType precision, deviceType device, bool allowGPUFallback );
	
	int clusterDetections( Detection* detections, uint32_t width, uint32_t height );
	int clusterDetections( Detection* detections, int n, float threshold=0.75f );

	void sortDetections( Detection* detections, int numDetections );

	float  TRTCoverageThreshold;
	float* TRTClassColors[2];
	float  TRTMeanPixel;

	std::vector<std::string> TRTClassDesc;
	std::vector<std::string> TRTClassSynset;

	std::string TRTClassPath;
	uint32_t    TRTCustomClasses;
	uint32_t	  TRTNumClasses;

	Detection* TRTDetectionSets[2];	// list of detections, mNumDetectionSets * mMaxDetections
	uint32_t   TRTDetectionSet;	// index of next detection set to use
	uint32_t	 TRTMaxDetections;	// number of raw detections in the grid

	static const uint32_t TRTNumDetectionSets = 16; // size of detection ringbuffer
};


#endif
