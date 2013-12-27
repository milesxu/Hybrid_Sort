#include <iostream>
#include <xmmintrin.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <test/test_util.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include "util.h"
#include "cpu_sort.h"

const int cacheFactor = 4; //what is the most suitable cache size?

void gpu_sort(float *data, size_t dataLen, size_t blockLen);
void gpu_sort_3(float *data, size_t dataLen, size_t blockLen);
void gpu_sort_2(float *data, size_t dataLen, size_t blockLen);
void transfer_test(float *data, size_t dataLen);
void gpu_sort_serial(float *data, size_t dataLen, size_t blockLen);
void gpu_sort_loop(float *data, size_t dataLen, size_t blockLen);
void gpu_sort(float *data, rsize_t dataLen);
float *cpu_sort_sse_parallel(DoubleBuffer<float> &data, rsize_t dataLen);

int main(int argc, char **argv)
{
	rsize_t dataLen = 1 << 23; //default length of sorted data
	int seed = 1023;  //default seed for generate random data sequence
	std::cout << omp_get_max_threads() << std::endl;
	CommandLineArgs args(argc, argv);
	args.GetCmdLineArgument("l", dataLen);
	args.GetCmdLineArgument("s", seed);
	std::cout << dataLen << " " << seed << "\n";
	float *data = new float[dataLen];
	GenerateData(seed, data, dataLen);
	args.DeviceInit();
	float* dataIn = (float*)_mm_malloc(dataLen * sizeof(float), 16);
	float* dataOut= (float*)_mm_malloc(dataLen * sizeof(float), 16);
	std::copy(data, data + dataLen, dataIn);
	//DoubleBuffer<float> hdata(dataIn, dataOut);
	//resultTest(cpu_sort_sse_parallel(hdata, dataLen), dataLen);
	//resultTest(mergeSortInBlockParallel(dataIn, dataOut, dataLen), dataLen);
	gpu_sort_loop(dataIn, dataLen, dataLen >> 2);
	gpu_sort(dataIn, dataLen, dataLen >> 2);
	//gpu_sort_serial(dataIn, dataLen, dataLen >>2);
	transfer_test(data, dataLen);
	gpu_sort_3(dataIn, dataLen, dataLen >> 2);
	delete [] data;
	_mm_free(dataIn);
	_mm_free(dataOut);
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

void gpu_sort_loop(float *data, size_t dataLen, size_t blockLen)
{
	int blockNum = dataLen / blockLen;
	size_t blockBytes = sizeof(float) * blockLen;
	//size_t dataBytes = sizeof(float) * dataLen;
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
		int next_stream = (i + 1) % blockNum;
		cub::DoubleBuffer<float> chunk(d_keys.d_buffers[d_keys.selector] + offset, d_keys.d_buffers[d_keys.selector ^ 1] + offset);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, chunk, blockLen, 0, 32, streams[i]);
		int upload_blocks = 2 + remain_to_upload % 2;
		cudaMemcpyAsync(d_keys.d_buffers[d_keys.selector] + up_offset, data + up_offset, upload_blocks * blockBytes, cudaMemcpyHostToDevice, streams[next_stream]);
		remain_to_upload -= upload_blocks;
		up_offset += upload_blocks * blockLen;
		offset += blockLen;
	}
	int selector = d_keys.selector ^ 1;
	int remain_to_donwload = upload_loop;
	size_t down_offset = 0;
	for (int i = upload_loop; i < blockNum; ++i)
	{
		//int next_stream = (i + 1) % blockNum;
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
}

void gpu_sort_2(float *data, size_t dataLen, size_t blockLen)
{
	int nstream = dataLen / blockLen;
	cudaStream_t *streams = new cudaStream_t[nstream];
	for (int i = 0; i < nstream; ++i)
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
	for (int i = 0; i < 2; ++i)
	{
		cudaMemcpyAsync(d_keys.d_buffers[d_keys.selector] + i * blockLen * 2, data + i * blockLen * 2, sizeof(float) * blockLen * 2, cudaMemcpyHostToDevice, streams[i]);
		cub::DoubleBuffer<float> chunk(d_keys.d_buffers[d_keys.selector] + i * blockLen, d_keys.d_buffers[d_keys.selector ^ 1] + i * blockLen);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, chunk, blockLen, 0, 32, streams[i]);
	}
	cub::DoubleBuffer<float> chunk2(d_keys.d_buffers[d_keys.selector] + 2 * blockLen, d_keys.d_buffers[d_keys.selector ^ 1] + 2 * blockLen);
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, chunk2, blockLen, 0, 32, streams[2]);
	cudaMemcpyAsync(data, d_keys.d_buffers[d_keys.selector], sizeof(float) * blockLen * 2, cudaMemcpyDeviceToHost, streams[0]);
	cub::DoubleBuffer<float> chunk3(d_keys.d_buffers[d_keys.selector] + dataLen - blockLen, d_keys.d_buffers[d_keys.selector ^ 1] + dataLen - blockLen);
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, chunk3, blockLen, 0, 32, streams[3]);
	cudaMemcpyAsync(data + 2 * blockLen, chunk2.Current(), sizeof(float) * blockLen, cudaMemcpyDeviceToHost, streams[2]);
	cudaMemcpyAsync(data + dataLen - blockLen, chunk3.Current(), sizeof(float) * blockLen, cudaMemcpyDeviceToHost, streams[3]);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float sort_time;
	cudaEventElapsedTime(&sort_time, start, stop);
	std::cout << "time used on gpu sort 2: " << sort_time << std::endl;
	
}

void gpu_sort(float *data, size_t dataLen, size_t blockLen)
{
	int nstream = dataLen / blockLen;
	cudaStream_t *streams = new cudaStream_t[nstream];
	for (int i = 0; i < nstream; ++i)
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
	cudaMemcpyAsync(d_keys.d_buffers[d_keys.selector], data, sizeof(float) * blockLen, cudaMemcpyHostToDevice,streams[0]);
	cudaMemsetAsync(d_keys.d_buffers[d_keys.selector ^ 1], 0, sizeof(float) * dataLen, streams[0]);
	//cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, )
	size_t offset = 0;
	for (int n = 0; n < nstream - 1; ++n, offset += blockLen)
	{
		int next = n + 1;
		size_t noffset = next * blockLen;
	    cub::DoubleBuffer<float> chunk(d_keys.d_buffers[d_keys.selector] + offset, d_keys.d_buffers[d_keys.selector ^ 1] + offset);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, chunk, blockLen, 0, 32, streams[n]);
		cudaMemcpyAsync(d_keys.d_buffers[d_keys.selector] + noffset, data + noffset, sizeof(float) * blockLen, cudaMemcpyHostToDevice, streams[next]);
		cudaMemcpyAsync(data + offset, chunk.Current(), sizeof(float) * blockLen, cudaMemcpyDeviceToHost, streams[n]);
		//d_keys.selector = chunk.selector;
	}
	cub::DoubleBuffer<float> last_chunk(d_keys.d_buffers[d_keys.selector] + offset, d_keys.d_buffers[d_keys.selector ^ 1] + offset);
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, last_chunk, blockLen, 0, 32, streams[nstream - 1]);
	cudaMemcpyAsync(data + offset, last_chunk.Current(), sizeof(float) * blockLen, cudaMemcpyDeviceToHost, streams[nstream - 1]);
	//cudaMemcpyAsync(data, d_keys.Current(), sizeof(float) * dataLen, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float sort_time;
	cudaEventElapsedTime(&sort_time, start, stop);
	std::cout << "time used on gpu: " << sort_time << std::endl;
	for (size_t offset = 0; offset < dataLen; offset += blockLen)
		resultTest(data + offset, blockLen);
	for (int i = 0; i < nstream; ++i)
		cudaStreamDestroy(streams[i]);
}

void gpu_sort_3(float *data, size_t dataLen, size_t blockLen)
{
	int nstream = dataLen / blockLen;
	cudaStream_t *streams = new cudaStream_t[nstream];
	for (int i = 0; i < nstream; ++i)
		cudaStreamCreate(&streams[i]);
    cub::DoubleBuffer<float> d_keys;
    //cub::CachingDeviceAllocator cda;
    //cda.DeviceAllocate((void**) &d_keys.d_buffers[0], sizeof(float) * dataLen);
    //cda.DeviceAllocate((void**) &d_keys.d_buffers[1], sizeof(float) * dataLen);
	cudaMalloc(&d_keys.d_buffers[0], sizeof(float) * dataLen);
	cudaMalloc(&d_keys.d_buffers[1],sizeof(float) * dataLen);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys,
								   blockLen);
	//cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	cudaEvent_t start, stop, cstart, cstop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&cstart);
	cudaEventCreate(&cstop);
	cudaEventRecord(start, 0);
	cudaMemcpyAsync(d_keys.d_buffers[d_keys.selector], data, sizeof(float) * blockLen, cudaMemcpyHostToDevice,streams[0]);
	cudaMemsetAsync(d_keys.d_buffers[d_keys.selector ^ 1], 0, sizeof(float) * dataLen, streams[0]);
	cub::DoubleBuffer<float> first_chunk(d_keys.d_buffers[d_keys.selector], d_keys.d_buffers[d_keys.selector ^ 1]);
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, first_chunk, blockLen, 0, 32, streams[0]);
	cudaMemcpyAsync(d_keys.d_buffers[d_keys.selector] + blockLen, data + blockLen, sizeof(float) * (dataLen - blockLen), cudaMemcpyHostToDevice, streams[1]);
	int selector = first_chunk.selector;
	cudaEventRecord(cstart, 0);
	for (int i = 1; i < nstream; ++i)
	{
		cub::DoubleBuffer<float> chunk(d_keys.d_buffers[d_keys.selector] + i * blockLen, d_keys.d_buffers[d_keys.selector ^ 1] + i * blockLen);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, chunk, blockLen, 0, 32, streams[i]);
		cudaMemcpyAsync(data + (i - 1) * blockLen, d_keys.d_buffers[selector] + (i - 1) * blockLen, sizeof(float) * blockLen, cudaMemcpyDeviceToHost, streams[i - 1]);
	}
	cudaEventRecord(cstop, 0);
	//cudaMemcpyAsync(data, d_keys.d_buffers[selector], sizeof(float) * (dataLen - blockLen), cudaMemcpyDeviceToHost, streams[2]);
	cudaMemcpyAsync(data + dataLen - blockLen, d_keys.d_buffers[selector] + dataLen - blockLen, sizeof(float) * blockLen, cudaMemcpyDeviceToHost, streams[nstream - 1]);
	//cudaMemcpyAsync(data, d_keys.d_buffers[selector], sizeof(float) * dataLen, cudaMemcpyDeviceToHost, streams[3]);
	// size_t offset = 0;
	// for (int n = 0; n < nstream - 1; ++n, offset += blockLen)
	// {
	// 	int next = n + 1;
	// 	size_t noffset = next * blockLen;
	//     cub::DoubleBuffer<float> chunk(d_keys.d_buffers[d_keys.selector] + offset, d_keys.d_buffers[d_keys.selector ^ 1] + offset);
	// 	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, chunk, blockLen, 0, 32, streams[n]);
	// 	cudaMemcpyAsync(d_keys.d_buffers[d_keys.selector] + noffset, data + noffset, sizeof(float) * blockLen, cudaMemcpyHostToDevice, streams[next]);
	// 	cudaMemcpyAsync(data + offset, chunk.Current(), sizeof(float) * blockLen, cudaMemcpyDeviceToHost, streams[n]);
	// 	//d_keys.selector = chunk.selector;
	// }
	// cub::DoubleBuffer<float> last_chunk(d_keys.d_buffers[d_keys.selector] + offset, d_keys.d_buffers[d_keys.selector ^ 1] + offset);
	// cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, last_chunk, blockLen, 0, 32, streams[nstream - 1]);
	// cudaMemcpyAsync(data + offset, last_chunk.Current(), sizeof(float) * blockLen, cudaMemcpyDeviceToHost, streams[nstream - 1]);
	//cudaMemcpyAsync(data, d_keys.Current(), sizeof(float) * dataLen, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(cstop);
	float compute_time;
	cudaEventElapsedTime(&compute_time, cstart, cstop);
	std::cout << "time used for compute:" << compute_time << std::endl;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float sort_time;
	cudaEventElapsedTime(&sort_time, start, stop);
	std::cout << "time used on gpu: " << sort_time << std::endl;
	for (size_t offset = 0; offset < dataLen; offset += blockLen)
		resultTest(data + offset, blockLen);
	for (int i = 0; i < nstream; ++i)
		cudaStreamDestroy(streams[i]);
}

void transfer_test(float *data, size_t dataLen)
{
	size_t size = sizeof(float) * dataLen;
	float *d_data, time;
	cudaMalloc(&d_data, size);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaMemcpyAsync(d_data, data, size, cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	std::cout << "use cuda function malloc:" << std::endl;
	std::cout << "time used host to device bulk transfer:" << time << std::endl;
	size_t blockLen = dataLen >> 2;
	cudaEventRecord(start, 0);
	for (size_t offset = 0; offset < dataLen; offset += blockLen)
	{
		cudaMemcpyAsync(d_data + offset, data + offset, sizeof(float) * blockLen, cudaMemcpyHostToDevice);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	std::cout << "time use host to device block transfer:" << time << std::endl;
	cudaEventRecord(start, 0);
	cudaMemcpyAsync(data, d_data, size, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	std::cout << "time used device to host bulk transfer:" << time << std::endl;
	cudaEventRecord(start, 0);
	for (size_t offset = 0; offset < dataLen; offset += blockLen)
	{
		cudaMemcpyAsync(data + offset, d_data + offset, sizeof(float) * blockLen, cudaMemcpyDeviceToHost);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	std::cout << "time use device to host block transfer:" << time << std::endl;
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
