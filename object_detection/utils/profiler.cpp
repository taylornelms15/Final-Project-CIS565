#include <assert.h>
#include <stdio.h>
#include "../cuda_utilities/cudaUtility.h"
#include "profiler.h"
/**
 * Prefix used for tagging printed log output from TensorRT.
 * @ingroup tensorNet
 */
#define LOG_PRF "[PRF]  "


// each profiler has a start and end event
static cudaEvent_t cuda_evt_table[NUM_PROFILERS];
// call as many times as you want now without causing bad shit to happen
static int created = 0;

void create_events()
{
	if(!created){	
	for(int i = 0; i < NUM_PROFILERS; i++)
	{
		CUDA(cudaEventCreate(&cuda_evt_table[i]));
	}
	created = 1;
	}
}

void profiler_begin(profilertype type, cudaStream_t stream)
{
	cudaEvent_t evt = cuda_evt_table[type];
	
	assert(evt != NULL);

	CUDA(cudaEventRecord(evt,stream)); 
}

void profiler_end(profilertype type, cudaStream_t stream)
{
	float accumulated_time;
	cudaEvent_t end_evt = cuda_evt_table[type];
	cudaEvent_t start_evt = cuda_evt_table[type-1];
	
	assert(end_evt != NULL);
	assert(start_evt != NULL);
	CUDA(cudaStreamWaitEvent(stream,start_evt,0));
	CUDA(cudaEventRecord(end_evt,stream)); 
	CUDA(cudaEventElapsedTime(&accumulated_time,start_evt,end_evt));
	
	printf(LOG_PRF "%f ms\n",accumulated_time);
}

void destroy_events()
{
	if(created){	
	for(int i = 0; i < NUM_PROFILERS; i++)
	{
		CUDA(cudaEventDestroy(cuda_evt_table[i]));
	}
	created = 0;
	}
}

