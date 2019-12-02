#ifndef __PROFILER_H__
#define __PROFILER_H__


inline void create_events();


inline void profiler_begin(profilertype type, cudaStream_t stream=0);


inline void profiler_end(profilertype type, cudaStream_t stream=0);


inline void destroy_events();


#endif