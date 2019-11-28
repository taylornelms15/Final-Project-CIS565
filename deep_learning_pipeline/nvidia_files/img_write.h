#ifndef __IMAGE_IO_H_
#define __IMAGE_IO_H_

#include "../cuda_utilities/cudaUtility.h"

/**
 * Save a float4 RGBA image to disk.
 *
 * Supported image file formats by saveImageRGBA() include:  
 *
 *   - JPEG
 *   - PNG
 *   - TGA
 *   - BMP
 *   - HDR
 *
 * @param filename Desired path of the image file to save to disk.
 * @param cpu Pointer to the buffer containing the image in CPU address space.
 * @param width Width of the image in pixels.
 * @param height Height of the image in pixels.
 * @param max_pixel The maximum pixel value of this image, by default it's 255 for images in the range of 0-255.
 *                  If your image is in the range of 0-1, pass 1.0 as this value.  Then the pixel values of the
 *                  image will be rescaled appropriately to be stored on disk (which expects a range of 0-255).
 * @param quality Indicates the compression quality level (between 1 and 100) to be applied for JPEG and PNG images.
 *                A level of 1 correponds to reduced quality and maximum compression.
 *                A level of 100 corresponds to maximum quality and reduced compression.
 *                By default a level of 100 is used for maximum quality and reduced compression. 
 *                Note that this quality parameter only applies to JPEG and PNG, other formats will ignore it.
 * @ingroup image
 */
bool saveImageRGBA( const char* filename, float4* cpu, int width, int height, float max_pixel=255.0f, int quality=100 );

#endif
