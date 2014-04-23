#include <iostream>
#include <vector>
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
	//TODO:more portable and flexible partition method
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
		else //if (dataLen < 1 << 30)
		{
			gpuChunkLen = 1 << 27;
			gpuPart = (dataLen >> 2) * 3;
		}
		/*else
		{
			gpuChunkLen = 1 << 27;
			gpuPart = gpuChunkLen * 7;
			}*/
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
		//std::cout << "baseChunkLen = " << baseChunkLen << std::endl;
		gpuPart = gpuPartLen;
		gpuChunkLen = std::min(gpuPart, size_t(1 << 27));
		cpuPart = dataLen - gpuPartLen;
		cpuChunkLen = std::min(baseChunkLen, cpuPart);
		cpuBlockLen = cpuChunkLen / threads;
	}
};

float gpu_sort(DoubleBuffer<float> &data, size_t dataLen, size_t blockLen,
			   int sSelector, int dSelector);
void gpu_sort_test(float *data, rsize_t dataLen);
void hybrid_sort3(float *data, size_t dataLen, double (&results)[2]);
void mergeTest(size_t minLen, size_t maxLen, int seed);
void hybrid_sort31(float *data, size_t dataLen, double (&results)[2]);

int main(int argc, char **argv)
{
	rsize_t dataLen = 1 << 30; //default length of sorted data
	int seed = 1979090303;  //default seed for generate random data sequence
	CommandLineArgs args(argc, argv);
	args.GetCmdLineArgument("l", dataLen);
	args.GetCmdLineArgument("s", seed);
	args.DeviceInit();
	//cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	//mergeTest(1<<16, 1<<30, seed);
	//multiWayTest(1<<16, 1<<28, seed);
	//multiWayTestMedian(1<<20, 1<<23, seed);
	float *data = new float[dataLen];
	GenerateData(seed, data, dataLen);
	double times[2];
	hybrid_sort31(data, dataLen, times);
	  /*gpu_sort_test(data, dataLen);*/
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
	  hybrid_sort3(data, dataLen);*/
	delete [] data;
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


//tSelector is target seclector, specify which buffer the result should be
//copied to.
void gpu_sort(DoubleBuffer<float> &data, size_t dataLen, size_t startOffset, 
			  size_t chunkLen, int tSelector)
{
    cub::DoubleBuffer<float> d_keys;
    cub::CachingDeviceAllocator cda(true);
    cda.DeviceAllocate((void**) &d_keys.d_buffers[0], sizeof(float) * chunkLen);
    cda.DeviceAllocate((void**) &d_keys.d_buffers[1], sizeof(float) * chunkLen);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys,
								   chunkLen);
	cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
	for(size_t i = startOffset; i < dataLen; i += chunkLen)
	{
		cudaMemcpyAsync(d_keys.Current(), data.Current() + i,
						sizeof(float) * chunkLen, cudaMemcpyHostToDevice);
		cudaMemsetAsync(d_keys.d_buffers[d_keys.selector ^ 1], 0,
						sizeof(float) * chunkLen);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
									   d_keys, chunkLen);
		cudaMemcpyAsync(data.buffers[tSelector] + i, d_keys.Current(),
						sizeof(float) * chunkLen, cudaMemcpyDeviceToHost);
	}
	cda.DeviceFree(d_temp_storage);
	cda.DeviceFree(d_keys.d_buffers[0]);
	cda.DeviceFree(d_keys.d_buffers[1]);
}

void multiWayMergeGPU(DoubleBuffer<float> &data,size_t dataLen,
					  size_t *upperBound, int chunkNum, size_t chunkLen,
					  size_t startOffset)
{
	size_t *quantileStart = new size_t[chunkNum];
	size_t *quantileEnd = new size_t[chunkNum];
	size_t *loopUBound = new size_t[chunkNum];
	size_t *loopLBound = new size_t[chunkNum];
	DoubleBuffer<size_t> quantile(quantileStart, quantileEnd);
	DoubleBuffer<size_t> bound(loopLBound, loopUBound);
	quantile.buffers[0][0] = 0;
	std::copy(upperBound, upperBound + chunkNum - 1, quantile.buffers[0] + 1);
	if (startOffset)
	{
		quantileCompute(data.buffers[data.selector], quantile, bound, upperBound,
						chunkNum, startOffset, true);
		std::copy(quantile.buffers[1], quantile.buffers[1] + chunkNum,
				  quantile.buffers[0]);
	}
	else
	{
		std::copy(quantile.buffers[0], quantile.buffers[0] + chunkNum,
				  quantile.buffers[1]);
	}
    cub::DoubleBuffer<float> d_keys;
    cub::CachingDeviceAllocator cda(true);
    cda.DeviceAllocate((void**) &d_keys.d_buffers[0], sizeof(float) * chunkLen);
    cda.DeviceAllocate((void**) &d_keys.d_buffers[1], sizeof(float) * chunkLen);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys,
								   chunkLen);
	cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
	for (size_t i = startOffset; i < dataLen; i += chunkLen)
	{
		quantileCompute(data.Current(), quantile, bound, upperBound, chunkNum,
						chunkLen);
		size_t tempLen = 0;
		for (int j = 0; j < chunkNum; ++j)
		{
			size_t len = quantile.buffers[1][j] - quantile.buffers[0][j];
			cudaMemcpyAsync(d_keys.d_buffers[d_keys.selector] + tempLen,
							data.buffers[data.selector] + quantile.buffers[0][j],
							sizeof(float) * len, cudaMemcpyHostToDevice);
			tempLen += len;
		}
		cudaMemsetAsync(d_keys.d_buffers[d_keys.selector ^ 1], 0,
						sizeof(float) * chunkLen);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
									   d_keys, chunkLen);
		cudaMemcpyAsync(data.buffers[data.selector ^ 1] + i, d_keys.Current(),
						sizeof(float) * chunkLen, cudaMemcpyDeviceToHost);
		std::copy(quantile.buffers[1], quantile.buffers[1] + chunkNum,
				  quantile.buffers[0]);
	}
	cda.DeviceFree(d_temp_storage);
	cda.DeviceFree(d_keys.d_buffers[0]);
	cda.DeviceFree(d_keys.d_buffers[1]);
	delete [] quantileStart;
	delete [] quantileEnd;
	delete [] loopUBound;
	delete [] loopLBound;
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

//if use same buffer store partial sorted data and run multi-way merge, then
//multi-way merge may overwrite data that is not merged yet, result to a wrong
//data list.
//TODO:does use task generation process can improve performance?
void multiWayMergeCPU(DoubleBuffer<float> &data, size_t dataLen,
					  size_t *upperBound, size_t chunkNum, 
					  hybridDispatchParams3<float> params)
{
	if (chunkNum == 1) return;
    rsize_t *loopUBound = new rsize_t[chunkNum];
    rsize_t *loopLBound = new rsize_t[chunkNum];
	DoubleBuffer<rsize_t> bound(loopLBound, loopUBound);
	size_t *quantileSet = new size_t[chunkNum * (params.threads + 1)];
	//TODO: initial of first array of quantile must all move into quantile
	//compute functions.
	quantileSet[0] = 0;
	std::copy(upperBound, upperBound + chunkNum - 1, quantileSet + 1);
	float *mwBuffer = (float*)_mm_malloc(params.cpuChunkLen * sizeof(float), 16);
	float **start = new float*[chunkNum * params.threads];
	float **end   = new float*[chunkNum * params.threads];
	for(size_t i  = 0; i < params.cpuPart; i+= params.cpuChunkLen)
	{
		quantileSetCompute(data, quantileSet, bound, upperBound, chunkNum,
						   params.cpuBlockLen, params.threads);
		std::cout << "quantile set compute complete." << std::endl;
	//synchronize problem is the reason that parallel for loop cannot be
	//used. otherwise multi-thread may sort data in same position. this version
	//use static temp buffer for each thread to solve the problem, which may
	//not be best performance.
	//TODO: try circular buffer and/or parallel task to get the best perfomance
	//solution.
		for(size_t x = 0; x < params.threads; ++x)
		{
			size_t y = 0;
			size_t *st = quantileSet + x * chunkNum;
			size_t *ed = st + chunkNum;
			for(size_t z = 0; z < chunkNum; ++z)
			{
				y += (ed[z] - st[z]);
			}
			std::cout << y << " ";
		}
		std::cout << std::endl;
		//#pragma omp parallel for
		for(size_t j = 0; j < params.threads; ++j)
		{
			std::vector<float> unalignVec;
			size_t bOffset = j * params.cpuBlockLen;
			size_t cOffset = j * chunkNum;
			DoubleBuffer<size_t> quantile(quantileSet + cOffset,
										  quantileSet + cOffset + chunkNum);
			multiWayMergeBitonic(data, chunkNum, mwBuffer + bOffset, i + bOffset,
								 quantile, unalignVec, start + cOffset,
								 end + cOffset);
		}
		std::copy(quantileSet + chunkNum * params.threads,
				  quantileSet + chunkNum * (params.threads + 1), quantileSet);
	}
	std::cout << "parallel execution complete." << std::endl;
	//data.selector ^= 1;
	delete [] loopUBound;
	delete [] loopLBound;
	delete [] quantileSet;
	_mm_free(mwBuffer);
	delete [] start;
	delete [] end;
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

void hybrid_sort31(float *data, size_t dataLen, double (&results)[2])
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
					std::cout << "gpu sort beginning..." << std::endl;
					//TODO: can std::copy or partial_sum function used with
					//0 elements?
					int selector = 0;
					updateSelectorGeneral(selector, 8, params.cpuPart);
					gpu_sort(hdata, dataLen, params.cpuPart, params.gpuChunkLen,
							 selector);
				}
			}
#pragma omp section
			{
				chunkMerge(hdata, params.cpuPart, params);
				medianMerge(hdata, params.cpuPart, params);
				std::cout << "merge stage complete." << std::endl;
			}
		}
	}
	cudaDeviceSynchronize();
	/*for(size_t i = params.cpuPart; i < dataLen; i += params.gpuChunkLen)
	    resultTest(hdata.Current() + i, params.gpuChunkLen);
		resultTest(hdata.Current(), params.cpuPart);*/
	//TODO: cpu may not sort to one part, it may have several small parts.
	//this can be decided by test GPU and CPU perfomance. how to guarantee
	//portable?
	//or is there a method to notify CPU, let it terminate sort work, though
	//it may produce a more irregular upperbound, it does not matter to
	//multiwaymerge.
	//int chunkNum = params.gpuPart / params.gpuChunkLen + 1;
	int chunkNum = params.gpuPart ? (params.gpuPart / params.gpuChunkLen + 1) : 1;
	size_t *upperBound = new size_t[chunkNum];
	upperBound[0] = params.cpuPart;
	std::fill(upperBound + 1, upperBound + chunkNum, params.gpuChunkLen);
	std::partial_sum(upperBound, upperBound + chunkNum, upperBound);
	std::cout << "upper bound initialized. " << chunkNum << std::endl;
#pragma omp parallel 
	{
		omp_set_nested(1);
#pragma omp sections
		{
#pragma omp section
			{
				if(params.gpuPart)
				{
					multiWayMergeGPU(hdata, dataLen, upperBound, chunkNum,
									 params.gpuChunkLen, params.cpuPart);
				}
			}
#pragma omp section
			{
				multiWayMergeCPU(hdata, dataLen, upperBound, chunkNum, params);
			}
		}
	}
	multiWayMergeCPU(hdata, dataLen, upperBound, chunkNum, params);
	cudaDeviceSynchronize();
	hdata.selector ^= 1;
	std::cout << "hybrid sort complete, test begin..." << std::endl;
	resultTest(hdata.Current(), dataLen);
	/*std::sort(data, data + dataLen);
	for (size_t i = 0; i < dataLen; ++i)
	{
		if(data[i] != hdata.buffers[hdata.selector][i])
			std::cout << "inconsistent at " << i << " " << data[i] << " " << hdata.buffers[hdata.selector][i] << std::endl;
			}*/
	std::cout << "test complete in function." << std::endl;
	
	delete [] upperBound;
	_mm_free(dataIn);
	_mm_free(dataOut);
}

