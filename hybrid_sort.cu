#include <iostream>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <xmmintrin.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <boost/timer/timer.hpp>
#include <boost/format.hpp>
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <test/test_util.h>
#include "util.h"
#include "cpu_sort.h"

//const size_t chunkFactor = 2;
//const size_t dlFactor = 27;

template<typename T>
struct hybridDispatchParams
{
	size_t gpuChunkSize;
	size_t cpuChunkSize;
	size_t mergeBlockSize;
	size_t multiwayBlockSize;
	size_t gpuMergeLen;
	size_t gpuMultiwayLen;
	
	hybridDispatchParams(size_t dataLen)
	{
		int mergeFactor = 3;
		int multiwayFactor = 1;
		int cacheFactor = 2; //what is the most suitable cache size?
		gpuChunkSize = dataLen / (mergeFactor + 1);
		cpuChunkSize = gpuChunkSize / omp_get_max_threads();
		mergeBlockSize = cacheSizeInByte() / (cacheFactor * sizeof(T));
		multiwayBlockSize = 512;
		gpuMergeLen = gpuChunkSize * mergeFactor;
		gpuMultiwayLen = dataLen * multiwayFactor / (multiwayFactor + 1);
	}
};

template<typename T>
struct hybridDispatchParams3
{
	size_t gpuChunkLen;
	size_t cpuChunkLen;
	size_t cpuBlockLen;
	size_t gpuPart;
	size_t cpuPart;
	int threads;
	/*size_t multiwayBlockSize;
	  size_t gpuMergeLen;
	  size_t gpuMultiwayLen;*/
	
	//now, dataLen must be power of 2.
	hybridDispatchParams3(size_t dataLen)
	{
		threads = omp_get_max_threads();
		size_t baseChunkLen = lastPower2(cacheSizeInByte3() / (2 * sizeof(T)));
		//TODO: when GPU global memory fewer than the half of dataLen?
		//TODO: other way to cut data lists to more fit in capacity of GPU
		//TODO: to find a more portable solution
		if (dataLen < baseChunkLen)
		{
			gpuChunkLen = 0;
			gpuPart = 0;
		}
		else if (dataLen < 1 << 26)
		{
			gpuChunkLen = dataLen >> 1;
			gpuPart = dataLen >> 1;
		}
		else if (dataLen < 1 << 28)
		{
			gpuChunkLen = (dataLen >> 2) * 3;
			gpuPart = gpuChunkLen;
		}
		else if (dataLen == 1 << 28)
		{
			gpuChunkLen = dataLen >> 2;
			gpuPart = gpuChunkLen * 3;
		}
		else
		{
			gpuChunkLen = 1 << 27;
			gpuPart = (dataLen >> 2) * 3;
		}
		/*gpuChunkLen =
		  dataLen < baseChunkLen ? 0 : std::min(dataLen >> 1, 1 << 27);*/
		cpuChunkLen =
			gpuChunkLen > 0 ? std::min(gpuChunkLen, baseChunkLen) : dataLen;
		cpuPart = dataLen - gpuPart;
		cpuBlockLen = cpuChunkLen / threads;
	}
	
	hybridDispatchParams3(size_t dataLen, size_t gpuPartLen)
	{
		threads = omp_get_max_threads();
		size_t baseChunkLen = lastPower2(cacheSizeInByte3() / (2 * sizeof(T)));
		gpuPart = gpuPartLen;
		gpuChunkLen = std::min(gpuPart, size_t(1 << 27));
		cpuPart = dataLen - gpuPart;
		cpuChunkLen = std::min(baseChunkLen, cpuPart);
		cpuBlockLen = cpuChunkLen / threads;
	}
};

float gpu_sort(DoubleBuffer<float> &data, size_t dataLen, size_t blockLen,
			   int sSelector, int dSelector);
void gpu_sort_serial(float *data, size_t dataLen, size_t blockLen);
void gpu_sort_test(float *data, rsize_t dataLen);
float *cpu_sort_sse_parallel(DoubleBuffer<float> &data, rsize_t dataLen);
void hybrid_sort(float *data, size_t dataLen);
void hybrid_sort3(float *data, size_t dataLen, double (&results)[2]);
void mergeTest(size_t minLen, size_t maxLen, int seed);
void multiWayTest(size_t minLen, size_t maxLen, int seed);
void multiWayTestMedian(size_t maxLen, int seed);

int main(int argc, char **argv)
{
	rsize_t dataLen = 1 << 27; //default length of sorted data
	int seed = 1979090303;  //default seed for generate random data sequence
	//std::cout << omp_get_max_threads() << std::endl;
	CommandLineArgs args(argc, argv);
	args.GetCmdLineArgument("l", dataLen);
	args.GetCmdLineArgument("s", seed);
	std::cout << dataLen << " " << seed << "\n";
	args.DeviceInit();
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	//mergeTest(1<<16, 1<<30, seed);
	//multiWayTest(1<<16, 1<<28, seed);
	multiWayTestMedian(1<<19, seed);
	/*float *data = new float[dataLen];
	  GenerateData(seed, data, dataLen);
	  gpu_sort_test(data, dataLen);*/
	//gpu_sort_serial(data, dataLen, dataLen);
	//delete [] data;
	/*for (int dlf = 20; dlf < 26; ++dlf)
	  {
	  dataLen = 1 << dlf;
	  std::cout << "data length: " << dataLen << std::endl;
	  float *data = new float[dataLen];
	  GenerateData(seed, data, dataLen);
	  hybrid_sort(data, dataLen);
	  delete [] data;
	  //std::cout << "loop time: " << dlf << std::endl;
	  }*/
	/*dataLen = 1 << 23;
	  float *data = new float[dataLen];
	  GenerateData(seed, data, dataLen);
	  hybrid_sort3(data, dataLen);
	  delete [] data;*/
	std::cout << "test complete." << std::endl;
	//resultTest(cpu_sort_sse_parallel(hdata, dataLen), dataLen);
	//resultTest(mergeSortInBlockParallel(dataIn, dataOut, dataLen), dataLen);
	//gpu_sort(dataIn, dataLen, dataLen >> 2);
	//gpu_sort_serial(dataIn, dataLen, dataLen >>2);
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
float gpu_sort(DoubleBuffer<float> &data, size_t dataLen, size_t blockLen,
			   int sSelector, int dSelector)
{
	int blockNum = dataLen / blockLen;
	size_t blockBytes = sizeof(float) * blockLen;
	cudaStream_t *streams = new cudaStream_t[blockNum];
	for (int i = 0; i < blockNum; ++i)
		cudaStreamCreate(&streams[i]);
    cub::DoubleBuffer<float> d_keys;
	//int gSelector = 1;
    cub::CachingDeviceAllocator cda;
    cda.DeviceAllocate((void**) &d_keys.d_buffers[0], sizeof(float) * dataLen);
    cda.DeviceAllocate((void**) &d_keys.d_buffers[1], sizeof(float) * dataLen);
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, blockLen);
	cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaMemcpyAsync(d_keys.d_buffers[0], data.buffers[sSelector], blockBytes, cudaMemcpyHostToDevice, streams[0]);
	int remain_to_upload = blockNum - 1;
	int upload_loop = std::max(1, remain_to_upload >> 1);
	size_t offset = 0;
	size_t up_offset = blockLen;
	for (int i = 0; i < upload_loop; ++i)
	{
		cub::DoubleBuffer<float> chunk(d_keys.d_buffers[0] + offset, d_keys.d_buffers[1] + offset);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, chunk, blockLen, 0, sizeof(float) * 8, streams[i]);
		int upload_blocks = ((remain_to_upload < 2) ? 0 : 2) + (remain_to_upload % 2);
		cudaMemcpyAsync(d_keys.d_buffers[0] + up_offset, data.buffers[sSelector] + up_offset, upload_blocks * blockBytes, cudaMemcpyHostToDevice, streams[i + 1]);
		remain_to_upload -= upload_blocks;
		up_offset += upload_blocks * blockLen;
		offset += blockLen;
	}
	int remain_to_donwload = upload_loop;
	size_t down_offset = 0;
	for (int i = upload_loop; i < blockNum; ++i)
	{
		cub::DoubleBuffer<float> chunk(d_keys.d_buffers[0] + offset, d_keys.d_buffers[1] + offset);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, chunk, blockLen, 0, sizeof(float) * 8, streams[i]);
		int dowload_blocks = 1 + (remain_to_donwload > 1);
		cudaMemcpyAsync(data.buffers[dSelector] + down_offset, d_keys.d_buffers[1] + down_offset, dowload_blocks * blockBytes, cudaMemcpyDeviceToHost, streams[i - 1]);
		remain_to_donwload -= (dowload_blocks - 1);
		down_offset += dowload_blocks * blockLen;
		offset += blockLen;
	}
	cudaMemcpyAsync(data.buffers[dSelector] + dataLen - blockLen, d_keys.d_buffers[1] + dataLen - blockLen, blockBytes, cudaMemcpyDeviceToHost, streams[blockNum - 1]);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float sort_time;
	cudaEventElapsedTime(&sort_time, start, stop);
	std::cout << "time used on gpu sort loop: " << sort_time << std::endl;
	for (int i = 0; i < blockNum; ++i)
		cudaStreamDestroy(streams[i]);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cda.DeviceFree(d_temp_storage);
	cda.DeviceFree(d_keys.d_buffers[0]);
	cda.DeviceFree(d_keys.d_buffers[1]);
	return sort_time;
}

void gpu_sort_serial(float *data, size_t dataLen, size_t blockLen)
{
	boost::timer::auto_cpu_timer t;
    cub::DoubleBuffer<float> d_keys;
    cub::CachingDeviceAllocator cda(true);
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
	//cudaMemcpy(d_keys.d_buffers[d_keys.selector], data, sizeof(float) * dataLen, cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&transfer_time, start, stop);
	std::cout << "time used for host to device transfer: " << transfer_time << std::endl;
	cudaMemsetAsync(d_keys.d_buffers[d_keys.selector ^ 1], 0, sizeof(float) * dataLen);
	//cudaMemset(d_keys.d_buffers[d_keys.selector ^ 1], 0, sizeof(float) * dataLen);
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
	//cudaMemcpy(data, d_keys.Current(), sizeof(float) * dataLen, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&transfer_time, start, stop);
	std::cout << "time used for device to host transfer: " << transfer_time << std::endl;
	for (size_t offset = 0; offset < dataLen; offset += blockLen)
		resultTest(data + offset, blockLen);
	cda.DeviceFree(d_keys.d_buffers[0]);
	cda.DeviceFree(d_keys.d_buffers[1]);
	cda.DeviceFree(d_temp_storage);
}

void gpu_sort_test(float *data, rsize_t dataLen)
{
	if (dataLen < (1 << 20) || dataLen > (1 << 28))
	{
		std::cout << "data length too short or too long!" << std::endl;
		return;
	}
	std::ofstream rFile("/home/aloneranger/source_code/Hybrid_Sort/result.txt",
						std::ios::app);
	rFile << "gpu kernel and transfer test" << std::endl
		  << boost::format("%1%%|15t|") % "data length"
		  << boost::format("%1%%|15t|") % "transfer time"
		  << boost::format("%1%%|15t|") % "kernel time"
		  << std::endl;
	cub::DoubleBuffer<float> d_keys;
    cub::CachingDeviceAllocator cda(true);
    cda.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(float) * dataLen);
    cda.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(float) * dataLen);
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
								   d_keys, dataLen);
	cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
	cudaMemcpy(d_keys.d_buffers[0], data, sizeof(float) * dataLen,
			   cudaMemcpyHostToDevice);
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
								   d_keys, dataLen);
	float *temp = new float[dataLen];
	cudaMemcpy(temp, d_keys.Current(), sizeof(float) * dataLen,
			   cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	resultTest(temp, dataLen);
	std::cout << "warm up complete. " << temp[0] << " " << temp[dataLen - 1]
			  << std::endl;
	delete [] temp;
	cda.DeviceFree(d_temp_storage);
	cudaMemset(d_keys.d_buffers[0], 0, sizeof(float) * dataLen);
	cudaMemset(d_keys.d_buffers[1], 0, sizeof(float) * dataLen);
	d_keys.selector = 0;
	int test_time = 50;
	for (size_t chunk_size = 1 << 17; chunk_size <= dataLen; chunk_size *= 2)
	{
		std::cout << chunk_size << std::endl;
		d_temp_storage = NULL;
		temp_storage_bytes = 0;
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
									   d_keys, chunk_size);
		cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
		float transfer_time = 0.0, kernel_time = 0.0;
		size_t offset = 0;
		for (int i = 0; i < test_time; ++i)
		{
			cudaEvent_t tStart, tStop, sStart, sStop;
			cudaEventCreate(&tStart);
			cudaEventCreate(&tStop);
			cudaEventCreate(&sStart);
			cudaEventCreate(&sStop);
			if (offset == dataLen) {
				offset = 0;
				cudaMemset(d_keys.d_buffers[0], 0, sizeof(float) * dataLen);
				cudaMemset(d_keys.d_buffers[1], 0, sizeof(float) * dataLen);
			}
			cudaEventRecord(tStart, 0);
			cudaMemcpyAsync(d_keys.d_buffers[0] + offset, data + offset,
							sizeof(float) * chunk_size, cudaMemcpyHostToDevice, 0);
			cudaEventRecord(tStop, 0);
			cub::DoubleBuffer<float> chunk(d_keys.d_buffers[0] + offset,
										   d_keys.d_buffers[1] + offset);
			cudaEventRecord(sStart, 0);
			cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
										   chunk, chunk_size);
			cudaEventRecord(sStop, 0);
			cudaDeviceSynchronize();
			float ttime;
			cudaEventElapsedTime(&ttime, tStart, tStop);
			transfer_time += ttime;
			float ktime;
			cudaEventElapsedTime(&ktime, sStart, sStop);
			kernel_time += ktime;
			offset += chunk_size;
			cudaEventDestroy(tStart);
			cudaEventDestroy(tStop);
			cudaEventDestroy(sStart);
			cudaEventDestroy(sStop);
		}
		rFile << boost::format("%1%%|15t|") % chunk_size
			  << boost::format("%1%%|15t|") % (transfer_time / test_time)
			  << boost::format("%1%%|15t|") % (kernel_time / test_time)
			  << std::endl;
		cda.DeviceFree(d_temp_storage);
	}
	
    /*cudaMemcpy(data, d_keys.Current(), sizeof(float) * dataLen,
	  cudaMemcpyDeviceToHost);*/
	cda.DeviceFree(d_keys.d_buffers[0]);
	cda.DeviceFree(d_keys.d_buffers[1]);
	rFile << std::endl << std::endl;
	rFile.close();
}

float *cpu_sort_sse_parallel(DoubleBuffer<float> &data, rsize_t dataLen)
{
	boost::timer::auto_cpu_timer t;
	const rsize_t blockSize = cacheSizeInByte() / (2 * sizeof(float));
    std::cout << "selected block size: " << blockSize << std::endl;
	if (blockSize)
	{
		int w, selector = 0;
		updateMergeSelector(selector, blockSize);
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
			registerSortIteration(chunk, blockSize * 2, chunkSize);
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
		updateMergeSelector(data.selector, blockSize);
		data.selector ^= 1;
	}
	return data.Current();
}

void mergeStage(DoubleBuffer<float> &data, size_t dataLen,
				hybridDispatchParams<float> &params)
{
	//boost::timer::auto_cpu_timer t;
	size_t bsize = params.mergeBlockSize, csize = params.cpuChunkSize;
	int mSelector = 0;
	updateMergeSelector(mSelector, bsize);
	int iSelector = mSelector;
	updateSelectorGeneral(iSelector, bsize, csize);
#pragma omp parallel 
	{
		omp_set_nested(1);
#pragma omp sections
		{
#pragma omp section
			{
				gpu_sort(data, params.gpuMergeLen, params.gpuChunkSize,
						 0, iSelector);
			}
#pragma omp section
			{
#pragma omp parallel for
				for (size_t i = params.gpuMergeLen; i < dataLen; i += csize)
				{
					for (size_t j = i; j < i + csize; j += bsize)
					{
						DoubleBuffer<float> block(data.buffers[0] + j,
												  data.buffers[1] + j);
						mergeSort(block, bsize);
					}
					DoubleBuffer<float> chunk(data.buffers[mSelector] + i,
											  data.buffers[mSelector ^ 1] + i);
					registerSortIteration(chunk, bsize * 2, csize);
				}
			}
		}
	}
	cudaDeviceSynchronize();
	data.selector = iSelector;
}

void multiWayStage(DoubleBuffer<float> &data, size_t dataLen,
				   hybridDispatchParams<float> &params)
{
	//boost::timer::auto_cpu_timer t;
	int gpuChunkNum = params.gpuMergeLen / params.gpuChunkSize;
	int cpuChunkNum = (dataLen - params.gpuMergeLen) / params.cpuChunkSize;
	int boundLen = gpuChunkNum + cpuChunkNum;
	size_t *upperBound = new size_t[boundLen];
	std::fill(upperBound, upperBound + gpuChunkNum, params.gpuChunkSize);
	std::fill(upperBound + gpuChunkNum, upperBound + boundLen, params.cpuChunkSize);
	std::partial_sum(upperBound, upperBound + boundLen, upperBound);
#pragma omp parallel
	{
		omp_set_nested(1);
#pragma omp sections
		{
#pragma omp section
			{
				multiWayMergeHybrid(data, dataLen, upperBound, boundLen, params.gpuChunkSize, 0, params.gpuMultiwayLen);
				int selector = data.selector ^ 1;
				updateMergeSelector(selector, params.multiwayBlockSize);
				gpu_sort(data, params.gpuMultiwayLen, params.gpuChunkSize, data.selector ^ 1, selector);
			}
#pragma omp section
			{
#pragma omp parallel for
				for (size_t i = params.gpuMultiwayLen; i < dataLen; i += params.cpuChunkSize)
					multiWayMergeHybrid(data, dataLen, upperBound, boundLen,
										params.multiwayBlockSize, i, i + params.cpuChunkSize);
				int selector = data.selector ^ 1;
#pragma omp parallel for
				for (size_t i = params.gpuMultiwayLen; i < dataLen; i += params.cpuChunkSize)
					for (size_t j = i; j < i + params.cpuChunkSize;
						 j += params.multiwayBlockSize)
					{
						DoubleBuffer<float> block(data.buffers[selector] + j,
												  data.buffers[selector ^ 1] + j);
						mergeSort(block, params.multiwayBlockSize);
					}
			}
		}
	}
	cudaDeviceSynchronize();
	data.selector ^= 1;
	updateMergeSelector(data.selector, params.multiwayBlockSize);
	delete [] upperBound;
}

void chunkMerge(DoubleBuffer<float> &data, size_t dataLen,
				hybridDispatchParams3<float> &params)
{
	for (size_t i = 0; i < dataLen; i += params.cpuChunkLen)
	{
#pragma omp parallel for
		for (size_t j = i; j < i + params.cpuChunkLen; j += params.cpuBlockLen)
		{
			DoubleBuffer<float> block(data.buffers[data.selector] + j,
									  data.buffers[data.selector ^ 1] + j);
			singleThreadMerge(block, params.cpuBlockLen);
		}
		DoubleBuffer<float> chunk(data.buffers[data.selector] + i,
								  data.buffers[data.selector ^ 1] + i);
		updateSelectorGeneral(chunk.selector, 8, params.cpuBlockLen);
		multiThreadMerge(chunk, params.cpuChunkLen, params.threads,
						 params.cpuBlockLen);
	}
	updateSelectorGeneral(data.selector, 8, params.cpuChunkLen);
}

void medianMerge(DoubleBuffer<float> &data, size_t dataLen,
				 hybridDispatchParams3<float> &params)
{
	int chunkNum = dataLen / params.cpuChunkLen;
	size_t stride = params.cpuChunkLen << 1;
	while (chunkNum > 1)
	{
		for (size_t j = 0; j < dataLen; j += stride)
		{
			DoubleBuffer<float> chunk(data.buffers[data.selector] + j,
									  data.buffers[data.selector ^ 1] + j);
			multiThreadMerge(chunk, stride, 2, params.cpuBlockLen);
		}
		chunkNum >>= 1;
		stride <<= 1;
		data.selector ^= 1;
	}
}

void medianMergeLongList(DoubleBuffer<float> &data, size_t dataLen,
						 hybridDispatchParams3<float> &params, size_t partialLen)
{
	for (size_t i = 0; i < dataLen; i += partialLen)
	{
		DoubleBuffer<float> partial(data.buffers[data.selector] + i,
									data.buffers[data.selector ^ 1] + i);
		medianMerge(partial, partialLen, params);
	}
	updateSelectorGeneral(data.selector, params.cpuChunkLen, partialLen);
}

//if use same buffer store partial sorted data and run multi-way merge, then
//multi-way merge may overwrite data that is not merged yet, result to a wrong
//data list.
void multiWayStage3(DoubleBuffer<float> &data, size_t dataLen,
					hybridDispatchParams3<float> &params, size_t multiBlockLen,
					size_t multiChunkLen)
{
	size_t boundLen = dataLen / multiChunkLen;
	size_t *upperBound = new size_t[boundLen];
	std::fill(upperBound, upperBound + boundLen, multiChunkLen);
	std::partial_sum(upperBound, upperBound + boundLen, upperBound);
	float *tempList=(float *)_mm_malloc(params.cpuChunkLen * sizeof(float), 16);
	int w;
	
	//synchronize problem is the reason that parallel for loop cannot be
	//used. otherwise multi-thread may sort data in same position. this version
	//use static temp buffer for each thread to solve the problem, which may
	//not be best performance.
	//TODO: try circular buffer and/or parallel task to get the best perfomance
	//solution.
#pragma omp parallel private(w)
	{
		w = omp_get_thread_num();
		size_t baseOffset = w * params.cpuBlockLen;
		for (size_t i = baseOffset; i < dataLen; i += params.cpuChunkLen)
		{
			multiWayMergeHybrid(data, dataLen, upperBound, boundLen,
								multiBlockLen, i, i + params.cpuBlockLen);
			for (size_t j = 0; j < params.cpuBlockLen; j += multiBlockLen)
			{
				DoubleBuffer<float>
					block(data.buffers[data.selector ^ 1] + i + j,
						  tempList + baseOffset + j);
				singleThreadMerge(block, multiBlockLen);
			}
		}
	}
	data.selector ^= 1;
	delete [] upperBound;
	delete [] tempList;
}

void hybrid_sort(float *data, size_t dataLen)
{
	std::ofstream rFile("/home/aloneranger/source_code/Hybrid_Sort/result.txt",
						std::ios::app);
	if (rFile.is_open()) rFile << "tested data length: " << dataLen << std::endl;
	float* dataIn = (float*)_mm_malloc(dataLen * sizeof(float), 16);
	float* dataOut= (float*)_mm_malloc(dataLen * sizeof(float), 16);
	std::copy(data, data + dataLen, dataIn);
	DoubleBuffer<float> hdata(dataIn, dataOut);
	
	hybridDispatchParams<float> params(dataLen);
	mergeStage(hdata, dataLen, params);
	multiWayStage(hdata, dataLen, params);
	//resultTest(hdata.Current() + params.gpuMultiwayLen, dataLen - params.gpuMultiwayLen);
	resultTest(hdata.Current(), dataLen);
	//resultTest(hdata.buffers[hdata.selector] + params.gpuMultiwayLen, dataLen - params.gpuMultiwayLen);
	//std::cout << params.gpuChunkSize << std::endl;
	/*std::copy(data, data + dataLen, dataIn);
	  gpu_sort(dataIn, dataLen, dataLen / 8);
	  cudaDeviceSynchronize();*/
	
	const int test_time = 0;
	rFile << boost::format("%1%%|15t|") % "cache factor"
		  << boost::format("%1%%|15t|") % "block length"
		  << boost::format("%1%%|15t|") % "chunk size"
		  << boost::format("%1%%|15t|") % "merge time"
		  << boost::format("%1%%|15t|") % "multi way"
		//<< boost::format("%1%%|15t|") % "gpu omp"
		//<< boost::format("%1%%|15t|") % "gpu cuda"
		  << std::endl;
	for (int j = 1; j <= 64; j *= 2)
	{
		size_t block_size = cacheSizeInByte() / (j * sizeof(float));
		for (int m = 8; m <= 64; m *= 2)
		{
			//double merge_time = 0.0, multiway_time = 0.0; //gpu_time = 0.0;
			//float cuda_time = 0.0;
			size_t chunk_size = dataLen / m;
			if (chunk_size < block_size) continue;
			for (int i = 0; i < test_time; ++i)
			{
				//double start, end;
				/*std::copy(data, data + dataLen, dataIn);
				  start = omp_get_wtime();
				  cuda_time += gpu_sort(dataIn, dataLen, chunk_size);
				  cudaDeviceSynchronize();
				  end = omp_get_wtime();
				  gpu_time += (end - start);*/
				/*std::copy(data, data + dataLen, dataIn);
				  hdata.selector = 0;
				  start = omp_get_wtime();
				  mergeStage(hdata, dataLen, chunk_size, block_size);
				  end = omp_get_wtime();
				  merge_time += (end - start);
				  start = omp_get_wtime();
				  multiWayStage(hdata, dataLen, chunk_size, block_size);
				  end = omp_get_wtime();
				  multiway_time += (end - start);*/
			}
			// rFile << boost::format("%1%%|15t|") % j
			// 	  << boost::format("%1%%|15t|") % block_size
			// 	  << boost::format("%1%%|15t|") % chunk_size
			// 	  << boost::format("%1%%|15t|") % (merge_time / test_time)
			// 	  << boost::format("%1%%|15t|") % (multiway_time / test_time)
			// 	<< boost::format("%1%%|15t|") % (gpu_time / test_time)
			// 	<< boost::format("%1%%|15t|") % (cuda_time / test_time)
			// 	  << std::endl;
		}
	}
	_mm_free(dataIn);
	_mm_free(dataOut);
	rFile << std::endl << std::endl;
	rFile.close();
}

void hybrid_sort3(float *data, size_t dataLen, double (&results)[2])
{
	float* dataIn = (float*)_mm_malloc(dataLen * sizeof(float), 16);
	float* dataOut= (float*)_mm_malloc(dataLen * sizeof(float), 16);
	std::copy(data, data + dataLen, dataIn);
	DoubleBuffer<float> hdata(dataIn, dataOut);
	
	hybridDispatchParams3<float> params(dataLen);
	chunkMerge(hdata, dataLen, params);
	medianMerge(hdata, dataLen, params);
	resultTest(hdata.Current(), dataLen);
	
	const int test_time = 30;
	double cmerge = 0.0, mmerge = 0.0;
	for (int i = 0; i < test_time; ++i)
	{
		std::copy(data, data + dataLen, dataIn);
		std::fill(dataOut, dataOut + dataLen, 0);
		hdata.selector = 0;
		double start, end;
		start = omp_get_wtime();
		chunkMerge(hdata, dataLen, params);
		end = omp_get_wtime();
		cmerge += (end - start);
		start = omp_get_wtime();
		medianMerge(hdata, dataLen, params);
		end = omp_get_wtime();
		mmerge += (end - start);
	}
	results[0] = cmerge / test_time, results[1] = mmerge / test_time;
	_mm_free(dataIn);
	_mm_free(dataOut);
}

void mergeTest(size_t minLen, size_t maxLen, int seed)
{
	std::ofstream rFile("/home/aloneranger/source_code/Hybrid_Sort/result.txt",
						std::ios::app);
	if (rFile.is_open()) 
		rFile << boost::format("%1%%|15t|") % "data length"
			  << boost::format("%1%%|15t|") % "chunk merge"
			  << boost::format("%1%%|15t|") % "median merge"
			  << std::endl;
	
	float *data = new float[maxLen];
	GenerateData(seed, data, maxLen);
	//Now, all length of data lists must be power of 2.
	for (size_t dataLen = minLen; dataLen <= maxLen; dataLen <<= 1)
	{
		double results[2];
		hybrid_sort3(data, dataLen, results);
		rFile << boost::format("%1%%|15t|") % dataLen
			  << boost::format("%1%%|15t|") % results[0]
			  << boost::format("%1%%|15t|") % results[1]
			  << std::endl;
	}
	delete [] data;
	rFile << std::endl << std::endl;
	rFile.close();
}

void multiWayTest(size_t minLen, size_t maxLen, int seed)
{
	std::ofstream rFile("/home/aloneranger/source_code/Hybrid_Sort/result.txt",
						std::ios::app);
	if (rFile.is_open()) 
		rFile << boost::format("%1%%|15t|") % "data length"
			  << boost::format("%1%%|15t|") % "multiBlockLen"
			  << boost::format("%1%%|15t|") % "multiChunkLen"
			  << boost::format("%1%%|15t|") % "merge time"
			  << boost::format("%1%%|15t|") % "multiway time"
			  << std::endl;
	
	float *data = new float[maxLen];
	GenerateData(seed, data, maxLen);
	float* dataIn = (float*)_mm_malloc(maxLen * sizeof(float), 16);
	float* dataOut= (float*)_mm_malloc(maxLen * sizeof(float), 16);
	std::copy(data, data + maxLen, dataIn);
	DoubleBuffer<float> hdata(dataIn, dataOut);
	
	hybridDispatchParams3<float> params(maxLen);
	chunkMerge(hdata, maxLen, params);
	size_t partialLen = params.cpuChunkLen * 4;
	medianMergeLongList(hdata, maxLen, params, partialLen);
	multiWayStage3(hdata, maxLen, params, 512, partialLen);
	resultTest(hdata.buffers[hdata.selector], maxLen);
	
	const int test_time = 30;
	for (size_t dataLen = minLen; dataLen < maxLen; dataLen <<= 1)
	{
		//all these length of multiblockLen can garuantee that the last position
		//of sorted data is one list of hdata.
		hybridDispatchParams3<float> paramsTest(dataLen);
		for (size_t multiBlockLen = 32; multiBlockLen <= paramsTest.cpuBlockLen;
			 multiBlockLen <<= 2)
		{
			double merge = 0.0, multiway = 0.0;
			for (int i = 0; i < test_time; ++i)
			{
				std::copy(data, data + dataLen, dataIn);
				std::fill(dataOut, dataOut + dataLen, 0);
				hdata.selector = 0;
				double start, end;
				start = omp_get_wtime();
				chunkMerge(hdata, dataLen, paramsTest);
				end = omp_get_wtime();
				merge += (end - start);
				start = omp_get_wtime();
				multiWayStage3(hdata, dataLen, paramsTest, multiBlockLen,
							   paramsTest.cpuChunkLen);
				end = omp_get_wtime();
				multiway += (end - start);
				//resultTest(hdata.Current(), dataLen);
			}
			rFile << boost::format("%1%%|15t|") % dataLen
				  << boost::format("%1%%|15t|") % multiBlockLen
				  << boost::format("%1%%|15t|") % paramsTest.cpuChunkLen
				  << boost::format("%1%%|15t|") % (merge / test_time)
				  << boost::format("%1%%|15t|") % (multiway / test_time)
				  << std::endl;
		}
	}
	_mm_free(hdata.buffers[0]);
	_mm_free(hdata.buffers[1]);
}


/*void hybrid_sort31(float *data, size_t dataLen, double (&results)[2])
{
	float* dataIn = (float*)_mm_malloc(dataLen * sizeof(float), 16);
	float* dataOut= (float*)_mm_malloc(dataLen * sizeof(float), 16);
	std::copy(data, data + dataLen, dataIn);
	DoubleBuffer<float> hdata(dataIn, dataOut);
	
	hybridDispatchParams3<float> params(dataLen);
#pragma omp parallel 
	{
		omp_set_nested(1);
#pragma omp sections
		{
#pragma omp section
			{
				if (params.gpuPart)
				{
				}
			}
#pragma omp section
			{
#pragma omp parallel for
			}
		}
	}
	
#pragma omp parallel 
	{
		omp_set_nested(1);
#pragma omp sections
		{
#pragma omp section
			{
			}
#pragma omp section
			{
#pragma omp parallel for
			}
		}
	}
	}*/

void hybrid_sort_test(size_t minLen, size_t maxLen, int seed)
{
	float *data = new float[maxLen];
	GenerateData(seed, data, maxLen);
	float* dataIn = (float*)_mm_malloc(maxLen * sizeof(float), 16);
	float* dataOut= (float*)_mm_malloc(maxLen * sizeof(float), 16);
	std::copy(data, data + maxLen, dataIn);
	DoubleBuffer<float> hdata(dataIn, dataOut);
	
	hybridDispatchParams3<float> params(maxLen);
	chunkMerge(hdata, maxLen, params);
}

void multiWayTestMedian(size_t maxLen, int seed)
{
	float *data = new float[maxLen];
	GenerateData(seed, data, maxLen);
	float* dataIn = (float*)_mm_malloc(maxLen * sizeof(float), 16);
	float* dataOut= (float*)_mm_malloc(maxLen * sizeof(float), 16);
	std::copy(data, data + maxLen, dataIn);
	DoubleBuffer<float> hdata(dataIn, dataOut);
	
	hybridDispatchParams3<float> params(maxLen, 0);
	size_t dataLen = maxLen;
	for (size_t i = 0; i < dataLen; i += params.cpuChunkLen)
	{
#pragma omp parallel for
		for (size_t j = i; j < i + params.cpuChunkLen; j += params.cpuBlockLen)
		{
			DoubleBuffer<float> block(hdata.buffers[hdata.selector] + j,
									  hdata.buffers[hdata.selector ^ 1] + j);
			singleThreadMerge(block, params.cpuBlockLen);
		}
		DoubleBuffer<float> chunk(hdata.buffers[hdata.selector] + i,
								  hdata.buffers[hdata.selector ^ 1] + i);
		updateSelectorGeneral(chunk.selector, 8, params.cpuBlockLen);
		size_t *upperBound = new size_t[params.threads];
		std::fill(upperBound, upperBound + params.threads, params.cpuBlockLen);
		std::partial_sum(upperBound, upperBound + params.threads, upperBound);
		multiWayMergeMedian(chunk, dataLen, upperBound, params.threads,
							params.cpuBlockLen, i, i + params.cpuChunkLen);
		delete [] upperBound;
	}
	_mm_free(dataIn);
	_mm_free(dataOut);
	delete [] data;
}
