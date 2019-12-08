

#ifndef __ROS_DEEP_LEARNING_IMAGE_CONVERTER_H_
#define __ROS_DEEP_LEARNING_IMAGE_CONVERTER_H_

#include <sensor_msgs/Image.h>


/**
 * GPU image conversion
 */
class imageConverter
{
public:
	/**
	 * Constructor
	 */
	imageConverter();

	/**
	 * Destructor
	 */
	~imageConverter();

	/**
	 * Convert to 32-bit RGBA floating point
	 */
	bool Convert( const sensor_msgs::ImageConstPtr& input );

	/**
	 * Convert to ROS sensor_msgs::Image message
	 */
	bool Convert( sensor_msgs::Image& msg, const std::string& encoding );

	/**
	 * Resize the memory (if necessary)
	 */
	bool Resize( uint32_t width, uint32_t height );

	/**
	 * Retrieve the converted image width
	 */
	inline uint32_t GetWidth() const		{ return mWidth; }

	/**
	 * Retrieve the converted image height
	 */
	inline uint32_t GetHeight() const		{ return mHeight; }

	/**
	 * Retrieve the size of the converted image (in bytes)
	 */
	inline size_t GetSize() const			{ return mSize; }

	/**
	 * Retrieve the GPU pointer of the converted image
	 */
	inline float* ImageGPU() const		{ return mOutputGPU; }

private:

	uint32_t mWidth;
	uint32_t mHeight;
	size_t   mSize;

	uint8_t* mInputCPU;
	uint8_t* mInputGPU;

	float*   mOutputCPU;
	float*   mOutputGPU;
};

#endif

