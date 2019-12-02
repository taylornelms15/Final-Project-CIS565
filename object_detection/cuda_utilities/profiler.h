#ifndef __PROFILER_H__
#define __PROFILER_H__

#include <assert.h>
#include <stdio.h>
#include "cudaUtility.h"

/**
 * Prefix used for tagging printed log output from TensorRT.
 * @ingroup tensorNet
 */
#define LOG_PRF "[PRF]  "

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
	NUM_PROFILERS
};

// each profiler has a start and end event
cudaEvent_t cuda_evt_table[NUM_PROFILERS];

inline void create_events()
{
	for(int i = 0; i < NUM_PROFILERS; i++)
	{
		CUDA(cudaEventCreate(&cuda_evt_table[i]));
	}
}

inline void profiler_begin(profilertype type, cudaStream_t stream=0)
{
	cudaEvent_t evt = cuda_evt_table[type];
	
	assert(evt == NULL);

	CUDA(cudaEventRecord(evt,stream)); 
}

inline void profiler_end(profilertype type, cudaStream_t stream=0)
{
	float accumulated_time;
	cudaEvent_t end_evt = cuda_evt_table[type];
	cudaEvent_t start_evt = cuda_evt_table[type-1];
	
	assert(end_evt != NULL);
	assert(start_evt != NULL);

	CUDA(cudaEventRecord(end_evt,stream)); 
	CUDA(cudaEventElapsedTime(accumulated_time,start_evt,end_evt));
	
	printf(LOG_PRF "%f ms\n",accumulated_time);
}

inline void destroy_events()
{
	for(int i = 0; i < NUM_PROFILERS; i++)
	{
		CUDA(cudaEventDestroy(cuda_evt_table[i]));
	}
}

#endif