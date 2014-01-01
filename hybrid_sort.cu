#include <iostream>
#include <xmmintrin.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <boost/timer/timer.hpp>
#include <test/test_util.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include "util.h"
#include "cpu_sort.h"

const int cacheFactor = 4; //what is the most suitable cache size?
const size_t chunkFactor = 10;
const size_t dlFactor = 20;

void gpu_sort(float *data, size_t dataLen, size_t blockLen);
void gpu_sort_serial(float *data, size_t dataLen, size_t blockLen);
void gpu_sort(float *data, rsize_t dataLen);
float *cpu_sort_sse_parallel(DoubleBuffer<float> &data, rsize_t dataLen);
void hybrid_sort(float *data, size_t dataLen);

int main(int argc, char **argv)
{
	rsize_t dataLen = (1 << dlFactor) * chunkFactor; //default length of sorted data
	int seed = 1023;  //default seed for generate random data sequence
	std::cout << omp_get_max_threads() << std::endl;
	CommandLineArgs args(argc, argv);
	args.GetCmdLineArgument("l", dataLen);
	args.GetCmdLineArgument("s", seed);
	std::cout << dataLen << " " << seed << "\n";
	float *data = new float[dataLen];
	GenerateData(seed, data, dataLen);
	args.DeviceInit();
	hybrid_sort(data, dataLen);
	//resultTest(cpu_sort_sse_parallel(hdata, dataLen), dataLen);
	//resultTest(mergeSortInBlockParallel(dataIn, dataOut, dataLen), dataLen);
	//gpu_sort(dataIn, dataLen, dataLen >> 2);
	//gpu_sort_serial(dataIn, dataLen, dataLen >>2);
	delete [] data;
		/*#pragma omp parallel
	{
		omp_set_nested(1);
#pragma omp single nowait
		std::cout << "single run" << omp_get_nested() << std::endl;
		gpu_sort(data, dataLen);
		#pragma omp single
		  resultTest(data, dataLen);
		  #pragma omp parallel
		std::cout << omp_get_thread_num();
		}*/
	return 0;
}

//using stream to overlap kernal excution and data transfer between CPU and GPU.
//all sorting task broken to 2 parts, the first will overlap data upload to GPU,
//the second will overlap data download from CPU.
void gpu_sort(float *data, size_t dataLen, size_t blockLen)
{
	boost::timer::auto_cpu_timer t;
	double begin, end;
	begin = omp_get_wtime();
	int blockNum = dataLen / blockLen;
	size_t blockBytes = sizeof(float) * blockLen;
	cudaStream_t *streams = new cudaStream_t[blockNum];
	for (int i = 0; i < blockNum; ++i)
		cudaStreamCreate(&streams[i]);
    cub::DoubleBuffer<float> d_keys;
    cub::CachingDeviceAllocator cda;
    cda.DeviceAllocate((void**) &d_keys.d_buffers[0], sizeof(float) * dataLen);
    cda.DeviceAllocate((void**) &d_keys.d_buffers[1], sizeof(float) * dataLen);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys,
								   blockLen);
	cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaMemcpyAsync(d_keys.d_buffers[d_keys.selector], data, blockBytes, cudaMemcpyHostToDevice, streams[0]);
	int remain_to_upload = blockNum - 1;
	int upload_loop = remain_to_upload >> 1;
	size_t offset = 0;
	size_t up_offset = blockLen;
	for (int i = 0; i < upload_loop; ++i)
	{
		cub::DoubleBuffer<float> chunk(d_keys.d_buffers[d_keys.selector] + offset, d_keys.d_buffers[d_keys.selector ^ 1] + offset);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, chunk, blockLen, 0, 32, streams[i]);
		int upload_blocks = 2 + remain_to_upload % 2;
		cudaMemcpyAsync(d_keys.d_buffers[d_keys.selector] + up_offset, data + up_offset, upload_blocks * blockBytes, cudaMemcpyHostToDevice, streams[i + 1]);
		remain_to_upload -= upload_blocks;
		up_offset += upload_blocks * blockLen;
		offset += blockLen;
	}
	int selector = d_keys.selector ^ 1;
	int remain_to_donwload = upload_loop;
	size_t down_offset = 0;
	for (int i = upload_loop; i < blockNum; ++i)
	{
		cub::DoubleBuffer<float> chunk(d_keys.d_buffers[d_keys.selector] + offset, d_keys.d_buffers[d_keys.selector ^ 1] + offset);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, chunk, blockLen, 0, 32, streams[i]);
		int dowload_blocks = 1 + (remain_to_donwload > 1);
		cudaMemcpyAsync(data + down_offset, d_keys.d_buffers[selector] + down_offset, dowload_blocks * blockBytes, cudaMemcpyDeviceToHost, streams[i - 1]);
		remain_to_donwload -= (dowload_blocks - 1);
		down_offset += dowload_blocks * blockLen;
		offset += blockLen;
	}
	cudaMemcpyAsync(data + dataLen - blockLen, d_keys.d_buffers[selector] + dataLen - blockLen, sizeof(float) * blockLen, cudaMemcpyDeviceToHost, streams[blockNum - 1]);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float sort_time;
	cudaEventElapsedTime(&sort_time, start, stop);
	std::cout << "time used on gpu sort loop: " << sort_time << std::endl;
	for (size_t offset = 0; offset < dataLen; offset += blockLen)
		resultTest(data + offset, blockLen);
	for (int i = 0; i < blockNum; ++i)
		cudaStreamDestroy(streams[i]);
	end = omp_get_wtime();
	std::cout << "timing using openmp function: " << end - begin << std::endl;
}

void gpu_sort_serial(float *data, size_t dataLen, size_t blockLen)
{
    cub::DoubleBuffer<float> d_keys;
    cub::CachingDeviceAllocator cda;
    cda.DeviceAllocate((void**) &d_keys.d_buffers[0], sizeof(float) * dataLen);
    cda.DeviceAllocate((void**) &d_keys.d_buffers[1], sizeof(float) * dataLen);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys,
								   blockLen);
	cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
	cudaEvent_t start, stop;
	float sort_time = 0, transfer_time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaMemcpyAsync(d_keys.d_buffers[d_keys.selector], data, sizeof(float) * dataLen, cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&transfer_time, start, stop);
	std::cout << "time used for host to device transfer: " << transfer_time << std::endl;
	cudaMemsetAsync(d_keys.d_buffers[d_keys.selector ^ 1], 0, sizeof(float) * dataLen);
	for (size_t offset = 0; offset < dataLen; offset += blockLen)
	{
		float stime;
		cudaEventRecord(start, 0);
	    cub::DoubleBuffer<float> chunk(d_keys.d_buffers[d_keys.selector] + offset, d_keys.d_buffers[d_keys.selector ^ 1] + offset);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, chunk, blockLen);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&stime, start, stop);
		sort_time += stime;
		//d_keys.selector = chunk.selector;
	}
	std::cout << "average time used for block sort:" << sort_time * (blockLen * 1.0 / dataLen) << std::endl;
	d_keys.selector ^= 1;
	cudaEventRecord(start, 0);
	cudaMemcpyAsync(data, d_keys.Current(), sizeof(float) * dataLen, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&transfer_time, start, stop);
	std::cout << "time used for device to host transfer: " << transfer_time << std::endl;
	for (size_t offset = 0; offset < dataLen; offset += blockLen)
		resultTest(data + offset, blockLen);
}

void gpu_sort(float *data, rsize_t dataLen)
{
    cub::DoubleBuffer<float> d_keys;
    cub::CachingDeviceAllocator cda;
    cda.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(float) * dataLen);
    cda.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(float) * dataLen);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys,
								   dataLen);
    cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
    cudaMemcpy(d_keys.d_buffers[d_keys.selector], data,
			   sizeof(float) * dataLen, cudaMemcpyHostToDevice);
    cudaMemset(d_keys.d_buffers[d_keys.selector ^ 1], 0,
			   sizeof(float) * dataLen);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys,
								   dataLen);
    cudaMemcpy(data, d_keys.Current(), sizeof(float) * dataLen,
			   cudaMemcpyDeviceToHost);
}

float *cpu_sort_sse_parallel(DoubleBuffer<float> &data, rsize_t dataLen)
{
	boost::timer::auto_cpu_timer t;
	const rsize_t blockSize = cacheSizeInByte() / (cacheFactor * sizeof(float));
    std::cout << "selected block size: " << blockSize << std::endl;
	if (blockSize)
	{
		int w, selector = 0;
		updateMergeSelcetor(&selector, blockSize);
		//if serialized, then this single code block is enough
#pragma omp parallel private(w) firstprivate(selector)
		{
			int threads = omp_get_num_threads();
			w = omp_get_thread_num();
			rsize_t chunkSize = dataLen / threads;
			rsize_t chunkStart = w * chunkSize;
			for (rsize_t offset = chunkStart; offset < chunkStart + chunkSize;
				 offset += blockSize)
			{
				DoubleBuffer<float> chunk(data.buffers[data.selector] + offset, data.buffers[data.selector ^ 1] + offset);
				mergeSort(chunk, blockSize);
			}
			DoubleBuffer<float> chunk(data.buffers[selector] + chunkStart,
									  data.buffers[selector ^ 1] + chunkStart);
			//chunk.selector = selector ^ data.selector;
			registerSortIteration(chunk, blockSize * 2, chunkSize, chunkSize);
#pragma omp single
			data.selector = chunk.selector;
		}
		
		rsize_t blockNum = (dataLen / blockSize);
#pragma omp parallel private(w)
		{
			int threads = omp_get_num_threads();
			rsize_t blocksPerThread = blockNum /threads;
			w = omp_get_thread_num();
			rsize_t quantileStart = w * blocksPerThread;
			rsize_t chunkSize = blockSize * blocksPerThread;
			rsize_t chunkStart = quantileStart * blockSize;
			multiWayMerge(data, dataLen, chunkSize, blockSize, quantileStart,
						  quantileStart + blocksPerThread);
			//data.selector ^= 1;	 
			for (rsize_t offset = chunkStart; offset < chunkStart + chunkSize;
				 offset += blockSize)
			{
				DoubleBuffer<float> chunk(data.buffers[data.selector] + offset,
										  data.buffers[data.selector ^ 1] + offset);
				chunk.selector ^= 1;
				mergeSort(chunk, blockSize);
			}
		}
		updateMergeSelcetor(&data.selector, blockSize);
		data.selector ^= 1;
	}
	return data.Current();
}

void mergeStage(DoubleBuffer<float> &data, size_t dataLen, size_t chunkSize,
				size_t blockSize)
{
	boost::timer::auto_cpu_timer t;
	updateMergeSelcetor(&data.selector, blockSize);
#pragma omp parallel 
	{
#pragma omp for 
		for (size_t cOffset = 0; cOffset < dataLen; cOffset += chunkSize)
		{
			for (size_t bOffset = cOffset; bOffset < cOffset + chunkSize;
				 bOffset += blockSize)
			{
				DoubleBuffer<float> block(data.buffers[0] + bOffset,
										  data.buffers[1] + bOffset);
				mergeSort(block, blockSize);
			}
			DoubleBuffer<float> chunk(data.buffers[data.selector] + cOffset,
									  data.buffers[data.selector ^ 1] + cOffset);
			registerSortIteration(chunk, blockSize * 2, chunkSize, chunkSize);
		}
	}
	updateMergeSelcetor(&data.selector, chunkSize);
}

void multiWayStage(DoubleBuffer<float> &data, size_t dataLen, size_t chunkSize,
				   size_t blockSize)
{
	boost::timer::auto_cpu_timer t;
#pragma omp parallel
	{
#pragma omp for
		for (size_t cOffset = 0; cOffset < dataLen; cOffset += chunkSize)
		{
			multiWayMergeGeneral(data, dataLen, chunkSize, blockSize,
								 cOffset, cOffset + chunkSize);
		}
	}
	data.selector ^= 1;
#pragma omp parallel
	{
#pragma omp for
		for (size_t cOffset = 0; cOffset < dataLen; cOffset += chunkSize)
			for (size_t boffset = cOffset; boffset < cOffset + chunkSize;
				 boffset += blockSize)
			{
				DoubleBuffer<float> block(data.buffers[data.selector] + boffset,
										  data.buffers[data.selector ^ 1] + boffset);
				mergeSort(block, blockSize);
			}
	}
	updateMergeSelcetor(&data.selector, blockSize);
}

void hybrid_sort(float *data, size_t dataLen)
{
	float* dataIn = (float*)_mm_malloc(dataLen * sizeof(float), 16);
	float* dataOut= (float*)_mm_malloc(dataLen * sizeof(float), 16);
	std::copy(data, data + dataLen, dataIn);
	DoubleBuffer<float> hdata(dataIn, dataOut);
	const size_t chunkSize = dataLen / chunkFactor;
	const size_t blockSize = cacheSizeInByte() / (cacheFactor * sizeof(float));
	std::cout << "selected block size on cpu: " << blockSize << std::endl;
	//TODO: if blockSize == 0, give it a default value.
	mergeStage(hdata, dataLen, chunkSize, blockSize);
	multiWayStage(hdata, dataLen, chunkSize, blockSize);
	resultTest(hdata.Current(), dataLen);
	_mm_free(dataIn);
	_mm_free(dataOut);
}

