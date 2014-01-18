#include <iostream>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <xmmintrin.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <boost/timer/timer.hpp>
#include <boost/format.hpp>
#include <test/test_util.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include "util.h"
#include "cpu_sort.h"

const size_t chunkFactor = 2;
const size_t dlFactor = 27;

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

float gpu_sort(DoubleBuffer<float> &data, size_t dataLen, size_t blockLen,
			   int sSelector, int dSelector);
void gpu_sort_serial(float *data, size_t dataLen, size_t blockLen);
void gpu_sort_test(float *data, rsize_t dataLen);
float *cpu_sort_sse_parallel(DoubleBuffer<float> &data, rsize_t dataLen);
void hybrid_sort(float *data, size_t dataLen);

int main(int argc, char **argv)
{
	rsize_t dataLen = (1 << dlFactor) * chunkFactor; //default length of sorted data
	int seed = 1023;  //default seed for generate random data sequence
	//std::cout << omp_get_max_threads() << std::endl;
	CommandLineArgs args(argc, argv);
	args.GetCmdLineArgument("l", dataLen);
	args.GetCmdLineArgument("s", seed);
	//std::cout << dataLen << " " << seed << "\n";
	args.DeviceInit();
	/*float *data = new float[dataLen];
	GenerateData(seed, data, dataLen);
	gpu_sort_test(data, dataLen);
	gpu_sort_serial(data, dataLen, dataLen);
	delete [] data;*/
	//cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	for (int dlf = 20; dlf < 26; ++dlf)
	{
		dataLen = 1 << dlf;
		std::cout << "data length: " << dataLen << std::endl;
		float *data = new float[dataLen];
		GenerateData(seed, data, dataLen);
		hybrid_sort(data, dataLen);
		float *gData = new float[dataLen];
		std::copy(data, data + dataLen, gData);
		cub::DoubleBuffer<float> d_keys;
		cub::CachingDeviceAllocator cda;
		cda.DeviceAllocate((void**) &d_keys.d_buffers[0], sizeof(float) * dataLen);
		cda.DeviceAllocate((void**) &d_keys.d_buffers[1], sizeof(float) * dataLen);
		void *d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, dataLen);
		cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
		cudaMemcpyAsync(d_keys.d_buffers[d_keys.selector], gData, sizeof(float) * dataLen, cudaMemcpyHostToDevice);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, dataLen, 0, sizeof(float) * 8, 0);
		cudaMemcpyAsync(gData, d_keys.d_buffers[d_keys.selector], sizeof(float) * dataLen, cudaMemcpyDeviceToHost);
		std::sort(data, data + dataLen);
		//cudaDeviceSynchronize();
		for (size_t i = 0; i < dataLen; ++i)
			if (data[i] != gData[i])
			{
				std::cout << "inequivalent at " << i << " " << data[i] << " " << gData[i] << std::endl;
				break;
			}
		std::cout << d_keys.selector << std::endl;
		cda.DeviceFree(d_keys.d_buffers[0]);
		cda.DeviceFree(d_keys.d_buffers[1]);
		cda.DeviceFree(d_temp_storage);
		delete [] gData;
		delete [] data;
		//std::cout << "loop time: " << dlf << std::endl;
	}
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
	// std::cout << "test before sort" << std::endl;
	// for (size_t offset = 0; offset < dataLen; offset += blockLen)
	// 	resultTest(data.buffers[dSelector] + offset, blockLen);
	//boost::timer::auto_cpu_timer t;
	int blockNum = dataLen / blockLen;
	size_t blockBytes = sizeof(float) * blockLen;
	cudaStream_t *streams = new cudaStream_t[blockNum];
	for (int i = 0; i < blockNum; ++i)
		cudaStreamCreate(&streams[i]);
    cub::DoubleBuffer<float> d_keys;
	int gSelector = 1;
    cub::CachingDeviceAllocator cda;
    cda.DeviceAllocate((void**) &d_keys.d_buffers[0], sizeof(float) * dataLen);
    cda.DeviceAllocate((void**) &d_keys.d_buffers[1], sizeof(float) * dataLen);
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
		void *d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, chunk, blockLen);
		cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, chunk, blockLen, 0, sizeof(float) * 8, streams[i]);
		int upload_blocks = ((remain_to_upload < 2) ? 0 : 2) + (remain_to_upload % 2);
		cudaMemcpyAsync(d_keys.d_buffers[0] + up_offset, data.buffers[sSelector] + up_offset, upload_blocks * blockBytes, cudaMemcpyHostToDevice, streams[i + 1]);
		remain_to_upload -= upload_blocks;
		up_offset += upload_blocks * blockLen;
		offset += blockLen;
		cda.DeviceFree(d_temp_storage);
	}
	int remain_to_donwload = upload_loop;
	size_t down_offset = 0;
	for (int i = upload_loop; i < blockNum; ++i)
	{
		cub::DoubleBuffer<float> chunk(d_keys.d_buffers[0] + offset, d_keys.d_buffers[1] + offset);
		void *d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, chunk, blockLen);
		cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, chunk, blockLen, 0, sizeof(float) * 8, streams[i]);
		int dowload_blocks = 1 + (remain_to_donwload > 1);
		cudaMemcpyAsync(data.buffers[dSelector] + down_offset, d_keys.d_buffers[gSelector] + down_offset, dowload_blocks * blockBytes, cudaMemcpyDeviceToHost, streams[i - 1]);
		remain_to_donwload -= (dowload_blocks - 1);
		down_offset += dowload_blocks * blockLen;
		offset += blockLen;
		cda.DeviceFree(d_temp_storage);
	}
	cudaMemcpyAsync(data.buffers[dSelector] + dataLen - blockLen, d_keys.d_buffers[gSelector] + dataLen - blockLen, blockBytes, cudaMemcpyDeviceToHost, streams[blockNum - 1]);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float sort_time;
	cudaEventElapsedTime(&sort_time, start, stop);
	std::cout << "time used on gpu sort loop: " << sort_time << std::endl;
	// for (size_t offset = 0; offset < dataLen; offset += blockLen)
	// 	resultTest(data.buffers[dSelector] + offset, blockLen);
	for (int i = 0; i < blockNum; ++i)
		cudaStreamDestroy(streams[i]);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cda.DeviceFree(d_keys.d_buffers[0]);
	cda.DeviceFree(d_keys.d_buffers[1]);
	return sort_time;
}

void gpu_sort_serial(float *data, size_t dataLen, size_t blockLen)
{
	boost::timer::auto_cpu_timer t;
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
    cub::CachingDeviceAllocator cda;
    cda.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(float) * dataLen);
    cda.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(float) * dataLen);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int test_time = 50;
	for (size_t chunk_size = 1 << 17; chunk_size <= dataLen; chunk_size *= 2)
	{
		void *d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
									   d_keys, chunk_size);
		cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
		float transfer_time = 0.0, kernel_time = 0.0;
		size_t offset = 0;
		for (int i = 0; i < test_time; ++i)
		{
			if (offset == dataLen) offset = 0;
			cudaEventRecord(start, 0);
			cudaMemcpy(d_keys.d_buffers[0] + offset, data + offset,
					   sizeof(float) * chunk_size, cudaMemcpyHostToDevice);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			float ttime;
			cudaEventElapsedTime(&ttime, start, stop);
			transfer_time += ttime;
			cub::DoubleBuffer<float> chunk(d_keys.d_buffers[0] + offset,
										   d_keys.d_buffers[1] + offset);
			cudaEventRecord(start, 0);
			cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
										   chunk, chunk_size);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			float ktime;
			cudaEventElapsedTime(&ktime, start, stop);
			kernel_time += ktime;
			offset += chunk_size;
		}
		rFile << boost::format("%1%%|15t|") % chunk_size
			  << boost::format("%1%%|15t|") % (transfer_time / test_time)
			  << boost::format("%1%%|15t|") % (kernel_time / test_time)
			  << std::endl;
		cda.DeviceFree(d_temp_storage);
	}
	
    /*cudaMemset(d_keys.d_buffers[d_keys.selector ^ 1], 0,
	  sizeof(float) * dataLen);*/
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
		updateMergeSelcetor(selector, blockSize);
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
		updateMergeSelcetor(data.selector, blockSize);
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
	updateMergeSelcetor(mSelector, bsize);
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
// #pragma omp parallel
// 	{
// 		omp_set_nested(1);
// #pragma omp sections
// 		{
// #pragma omp section
// 			{
				//#pragma omp parallel for
				//for (size_t i = 0; i < params.gpuMultiwayLen; i += params.gpuChunkSize)
				multiWayMergeHybrid(data, dataLen, upperBound, boundLen, params.gpuChunkSize, 0, params.gpuMultiwayLen);
				int selector = data.selector ^ 1;
								float tempL, tempR;
				tempL = data.buffers[selector][0];
				for (size_t i = 1; i < params.gpuChunkSize; ++i)
					if (tempL < data.buffers[selector][i]) tempL = data.buffers[selector][i];
				tempR = data.buffers[selector][params.gpuChunkSize];
				for (size_t i = params.gpuChunkSize + 1; i < params.gpuMultiwayLen; ++i)
					if (tempR > data.buffers[selector][i]) tempR = data.buffers[selector][i];
				if (tempL > tempR) std::cout << "multiway merge failed!" << std::endl;
				else std::cout << "multiway merge success" << std::endl;
				std::cout << tempL << "  " << tempR << std::endl;
				// DoubleBuffer<float> gpuChunk(data.buffers[data.selector],
				// 							 data.buffers[data.selector ^ 1]);
				// gpuChunk.selector = selector;
				updateMergeSelcetor(selector, params.multiwayBlockSize);
				gpu_sort(data, params.gpuMultiwayLen, params.gpuChunkSize, data.selector ^ 1, selector);
				tempL = data.buffers[selector][0];
				for (size_t i = 1; i < params.gpuChunkSize; ++i)
					if (tempL < data.buffers[selector][i]) tempL = data.buffers[selector][i];
				tempR = data.buffers[selector][params.gpuChunkSize];
				for (size_t i = params.gpuChunkSize + 1; i < params.gpuMultiwayLen; ++i)
					if (tempR > data.buffers[selector][i]) tempR = data.buffers[selector][i];
				if (tempL > tempR) std::cout << "gpu merge failed!" << std::endl;
				else std::cout << "gpu merge success" << std::endl;
				std::cout << tempL << "  " << tempR << std::endl;
				float *tData = new float[params.gpuMultiwayLen];
				std::copy(data.buffers[selector], data.buffers[selector] + params.gpuMultiwayLen, tData);
				for (size_t i = 0; i < params.gpuMultiwayLen; i += params.gpuChunkSize)
					std::sort(data.buffers[selector] + i, data.buffers[selector] + i + params.gpuChunkSize);
				for (size_t i = 0; i < params.gpuMultiwayLen; ++i)
					if (tData[i] != data.buffers[selector][i])
					{
						std::cout << "inequivalent at: " << i << " " << tData[i] << " " << data.buffers[selector][i] << std::endl;
						break;
					}
				delete [] tData;
				// }
// #pragma omp section
// 			{
// #pragma omp parallel for
// 				for (size_t i = params.gpuMultiwayLen; i < dataLen; i += params.cpuChunkSize)
// 					multiWayMergeHybrid(data, dataLen, upperBound, boundLen,
// 										params.multiwayBlockSize, i, i + params.cpuChunkSize);
// 				int selector = data.selector ^ 1;
// #pragma omp parallel for
// 				for (size_t i = params.gpuMultiwayLen; i < dataLen; i += params.cpuChunkSize)
// 					for (size_t j = i; j < i + params.cpuChunkSize;
// 						 j += params.multiwayBlockSize)
// 					{
// 						DoubleBuffer<float> block(data.buffers[selector] + j,
// 												  data.buffers[selector ^ 1] + j);
// 						mergeSort(block, params.multiwayBlockSize);
// 					}
// 			}
// 		}
// 	}
	cudaDeviceSynchronize();
	data.selector ^= 1;
	updateMergeSelcetor(data.selector, params.multiwayBlockSize);
	delete [] upperBound;
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
	std::cout << params.gpuChunkSize << std::endl;
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

