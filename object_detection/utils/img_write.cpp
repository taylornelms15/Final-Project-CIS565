
#include "img_write.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "write_img.h"

#define LOG_IMAGE "[image] "


// limit_pixel
static inline unsigned char limit_pixel( float pixel, float max_pixel )
{
	if( pixel < 0 )
		pixel = 0;

	if( pixel > max_pixel )
		pixel = max_pixel;

	return (unsigned char)pixel;
}


// saveImageRGBA
bool saveImageRGBA( const char* filename, float4* cpu, int width, int height, float max_pixel, int quality )
{
	// validate parameters
	if( !filename || !cpu || width <= 0 || height <= 0 )
	{
		printf(LOG_IMAGE "saveImageRGBA() - invalid parameter\n");
		return false;
	}
	
	if( quality < 1 )
		quality = 1;

	if( quality > 100 )
		quality = 100;
	
	// allocate memory for the uint8 image
	const size_t stride = width * sizeof(unsigned char) * 4;
	const size_t size   = stride * height;
	unsigned char* img  = (unsigned char*)malloc(size);

	if( !img )
	{
		printf(LOG_IMAGE "failed to allocate %zu bytes to save %ix%i image '%s'\n", size, width, height, filename);
		return false;
	}

	// convert image from float to uint8
	const float scale = 255.0f / max_pixel;

	for( int y=0; y < height; y++ )
	{
		const size_t yOffset = y * stride;

		for( int x=0; x < width; x++ )
		{
			const size_t offset = yOffset + x * sizeof(unsigned char) * 4;
			const float4 pixel  = cpu[y * width + x];

			img[offset + 0] = limit_pixel(pixel.x * scale, max_pixel);
			img[offset + 1] = limit_pixel(pixel.y * scale, max_pixel);
			img[offset + 2] = limit_pixel(pixel.z * scale, max_pixel);
			img[offset + 3] = limit_pixel(pixel.w * scale, max_pixel);
		}
	}

	// // save the image
	int save_result = 0;

	save_result = stbi_write_jpg(filename, width, height, 4, img, quality);

	// check the return code
	if( !save_result )
	{
		printf(LOG_IMAGE "failed to save %ix%i image to '%s'\n", width, height, filename);
		free(img);
		return false;
	}

	free(img);
	return true;
}
