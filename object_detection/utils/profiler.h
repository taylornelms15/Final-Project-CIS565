#ifndef __PROFILER_H__
#define __PROFILER_H__



/**
 * Enumeration for queries
 */
enum profilertype
{
	INFERENCE_BEGIN =0,
	INFERENCE_END,
	PREPROCESS_BEGIN,
	PREPROCESS_END,
	POSTPROCESS_BEGIN,
	POSTPROCESS_END,
	DETECT_BEGIN,
	DETECT_END,
	NUM_PROFILERS
};


void create_events();


void profiler_begin(profilertype type, cudaStream_t stream=0);


void profiler_end(profilertype type, cudaStream_t stream=0);


void destroy_events();


#endif
