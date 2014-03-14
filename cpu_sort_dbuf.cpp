/* sort functions using double buffer class */
#include <iostream>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include "cpu_sort.h"
#include "sse_sort.h"

void mergeInRegister(DoubleBuffer<float> &data, size_t dataLen)
{
	const size_t halfDataLen = dataLen >> 1;
	const int halfArrayLen = rArrayLen >> 1;
	//__m128 *rData = new __m128[rArrayLen];
	__m128 rData[rArrayLen];
	float *ptrOut = data.buffers[data.selector ^ 1];
	float *ptrIn[2], *end[2];
	end[0] = data.Current() + halfDataLen;
	end[1] = data.Current() + dataLen;
	ptrIn[0] = data.Current();
	ptrIn[1] = end[0];
	loadData(ptrIn[1], rData + halfArrayLen, halfArrayLen);
	ptrIn[1] += sortUnitLen;
	while ((ptrIn[0] < end[0]) && (ptrIn[1] < end[1]))
	{
		int index = ((*ptrIn[0]) >= (*ptrIn[1]));
		loadData(ptrIn[index], rData, halfArrayLen);
		ptrIn[index] += sortUnitLen;
		
		_mm_prefetch(ptrIn[0], _MM_HINT_T0);
		_mm_prefetch(ptrIn[1], _MM_HINT_T0);
		bitonicSort16232(rData);
		storeData(ptrOut, rData, halfArrayLen);
		ptrOut += sortUnitLen;
	}
	int index = (ptrIn[0] == end[0]);
	for (; ptrIn[index] < end[index]; ptrIn[index] += sortUnitLen)
	{
		loadData(ptrIn[index], rData, halfArrayLen);
		
		_mm_prefetch(ptrIn[index] + sortUnitLen, _MM_HINT_T0);
		bitonicSort16232(rData);
		storeData(ptrOut, rData, halfArrayLen);
		ptrOut += sortUnitLen;
	}
	storeData(ptrOut, rData + halfArrayLen, halfArrayLen);
	//delete [] rData;
}

// blockNum must be power of 2 and cannot great than 8.
// len indicate how many simd register lanes will be use per block.
void inline loadSimdDataInitial(__m128 *rData, float **blocks, int blockNum,
								int len)
{
	int offset = len;
	for (int i = 1; i < blockNum; i += 2)
	{
		loadData(blocks + i, rData + offset, len);
		offset += 2 * len;
	}
}

void inline loadSimdData(__m128 *rData, float **blocks, float **blockBound,
						 int blockNum, int len)
{
	int second, offset = 0;
	for (int i = 0; i < blockNum; i += 2)
	{
		if (blocks[i] != blockBound[i] && blocks[i + 1] != blockBound[i + 1])
			second = (*(blocks[i]) >= *(blocks[i + 1]));
		else
			second = (blocks[i] == blockBound[i]);
		loadData(blocks + i + second, rData + offset, len);
		offset += 2 * len;
	}
}

//TODO: recall the means of blockNum, may be there are several blocks of output?
void inline storeSimdData(__m128 *rData, float **output, int blockNum, int len,
						  int start)
{
	int offset = start;
	//rData += end;
	for (int i = 0; i < blockNum; i += 2)
	{
		storeData(output + (i >> 1), rData + offset, len);
		offset += 2 * len;
	}
}

// data length and block length both must be power of 2.
void mergeInRegister(DoubleBuffer<float> &data, size_t dataLen, size_t blockLen)
{
	size_t blockNum = dataLen / blockLen;
	size_t bLen = blockLen;
	//__m128 *rData = new __m128[rArrayLen];
	__m128 rData[rArrayLen];
	for (; blockNum >= rArrayLen; blockNum >>= 1, bLen <<= 1)
	{
		float *blocks[rArrayLen];
		float *blockBound[rArrayLen];
		float *output[rArrayLen >> 1];
		for (size_t i = 0; i < blockNum; i += rArrayLen)
		{
			size_t startIndex = i * bLen;
			blocks[0] = data.buffers[data.selector] + startIndex;
			//pointers cannot be added.
			for (int j = 1; j < rArrayLen; ++j) blocks[j] = blocks[j-1] + bLen;
			std::copy(blocks + 1, blocks + rArrayLen, blockBound);
			blockBound[rArrayLen - 1] = blocks[rArrayLen - 1] + bLen;
			output[0] = data.buffers[data.selector ^ 1] + startIndex;
			for (int j = 1; j < (rArrayLen >> 1); ++j)
				output[j] = output[j - 1] + (bLen << 1);
			
			//1 is actually rArrayLen / rArrayLen
			loadSimdDataInitial(rData, blocks, rArrayLen, 1);
			//simdLen actually is rArrayLen * simdLen / rArrayLen
			size_t loop = bLen * 2 / simdLen - 1;
			for (size_t i = 0; i < loop; ++i)
			{
				loadSimdData(rData, blocks, blockBound, rArrayLen, 1);
				bitonicSort428<4>(rData, true);
				storeSimdData(rData, output, rArrayLen, 1, 0);
			}
			storeSimdData(rData, output, rArrayLen, 1, 1);
		}
		/*delete [] blocks;
		delete [] blockBound;
		delete [] output;*/
		data.selector ^= 1;
	}
	//delete [] rData;
	for (; blockNum > 1; blockNum >>= 1, bLen <<= 1)
	{
		float **blocks = new float *[blockNum];
		float **blockBound = new float *[blockNum];
		float **output = new float *[blockNum >> 1];
		blocks[0] = data.buffers[data.selector];
		for (int i = 1; i < blockNum; ++i) blocks[i] = blocks[i - 1] + bLen;
		std::copy(blocks + 1, blocks + blockNum, blockBound);
		blockBound[blockNum - 1] = blocks[blockNum - 1] + bLen;
		output[0] = data.buffers[data.selector ^ 1];
		for (int i = 1; i < (blockNum >> 1); ++i)
			output[i] = output[i - 1] + (bLen << 1);
		
		int len = rArrayLen / blockNum;
		loadSimdDataInitial(rData, blocks, blockNum, len);
		size_t loop = bLen * 2 / (len * simdLen) - 1;
		for (size_t i = 0; i < loop; ++i)
		{
			loadSimdData(rData, blocks, blockBound, blockNum, len);
			if (blockNum == 4) bitonicSort8216<2>(rData, true);
			if (blockNum == 2) bitonicSort16232(rData);
			storeSimdData(rData, output, blockNum, len, 0);
		}
		storeSimdData(rData, output, blockNum, len, len);
		delete [] blocks;
		delete [] blockBound;
		delete [] output;
		data.selector ^= 1;
	}
}

bool inline multipleOf(size_t offset, int factor)
{
	return ((offset & ~(factor - 1)) == offset);
}

void inline swapFloat(float &a, float &b)
{
	float temp = a;
	a = b;
	b = temp;
}

//TODO: modify to general mode, namely two array pointer, not only one data.
int inline loadUnalignData(float *data, size_t &offsetA, size_t &offsetB,
							float *unalignData, int factor, bool start)
{
	size_t begin[2], end[2];
	size_t factornot = ~(factor - 1);
	if (start)
	{
		begin[0] = offsetA;
		end[0] = (offsetA + factor) & factornot;
		begin[1] = offsetB;
		end[1] = (offsetB + factor) & factornot;
		offsetA = end[0];
		offsetB = end[1];
	}
	else
	{
		begin[0] = offsetA & factornot;
		end[0] = offsetA;
		begin[1] = offsetB & factornot;
		end[1] = offsetB;
		offsetA = begin[0];
		offsetB = begin[1];
	}
	int n = 0, selector = (data[begin[0]] > data[begin[1]]);
	//remember that the value of unalignData cannot be changed here.
	/*for (float *ptr = data + startA; ptr < data + endA; ++ptr, ++n)
	  unalignData[n] = *ptr;
	  for (float *ptr = data + startB; ptr < data + endB; ++ptr, ++n)
	  unalignData[n] = *ptr;*/
	for (int i = begin[selector]; i < end[selector]; ++i)
		unalignData[n++] = data[i];
	for (int i = begin[selector ^ 1]; i < end[selector ^ 1]; ++i)
		unalignData[n++] = data[i];
	int lenA = end[selector] - begin[selector];
	if (lenA % simdLen)
	{
		int laneIndex = lenA / simdLen;
		float *ptr = unalignData + laneIndex * simdLen;
		if (ptr[0] > ptr[1]) swapFloat(ptr[0], ptr[1]);
		if (ptr[2] > ptr[3]) swapFloat(ptr[2], ptr[3]);
		if (ptr[0] > ptr[2]) swapFloat(ptr[0], ptr[2]);
		if (ptr[1] > ptr[3]) swapFloat(ptr[1], ptr[3]);
		if (ptr[1] > ptr[2]) swapFloat(ptr[1], ptr[2]);
	}
	return selector;
	/*for (int i = 0; i < simdLen; ++i)
	  for (int j = 0; j < (simdLen - 1); ++j)
	  {
	  int boffset = i * simdLen;
	  if (unalignData[boffset + j] > unalignData[boffset + j + 1])
	  std::cout << "unaligndata sort fail at " << boffset + j << std::endl;
	  }*/
}

//If the trail of two lists is unalign because of median computation,
//we need know where to "insert" the unalign data, to sort them correctly.
//comparation is only done in one list, because the uValue comes from it and
//must larger than all the previous keys. offset is the trail of the other
//list, the return value is number of loops that must be done after unaligned
//data loaded.
int getTrailPosition(float *data, size_t bOffset, size_t eOffset, float uValue,
					 int unitLen)
{
	int n = 0;
	size_t i = eOffset - unitLen;
	while (i >= bOffset) {
		if (data[i] > uValue)
		{
			++n;
			i -= unitLen;
		}
		else
			break;
	}
	return n;
}

//only used by 16 to 32 merge loop, namely only two lists merge to one list.
//must be used as a mediate process, rData must be copied a half, and must be
//empty after this process complete.
void simdMergeLoop2(float *dataIn, float *dataOut, float **blocks,
					float **blockBound, __m128 *rData, int lanes, int unitLen)
{
}

//merge two lists into one list. the two list both reside in dataIn, the
//elements that will be merged is bounded by offset arrays.
void simdMergeGeneral(float *dataIn, float *dataOut, size_t offsetA[2],
					  size_t offsetB[2]) //bool stream = false)
{
	int unitLen = (rArrayLen >> 1) * simdLen;
	int halfArrayLen = rArrayLen >> 1;
	__m128 rData[rArrayLen];
	float **output = new float*;
	*output = dataOut;
	if (!multipleOf(offsetA[0], unitLen))
	{
		float *unalignStart = (float*)_mm_malloc(unitLen * sizeof(float), 16);
		loadUnalignData(dataIn, offsetA[0], offsetB[0], unalignStart, unitLen,
						true);
		loadData(unalignStart, rData + halfArrayLen, halfArrayLen);
		bitonicSort428<2>(rData + halfArrayLen, true);
		bitonicSort8216<1>(rData + halfArrayLen, true);
		_mm_free(unalignStart);
	}
	else
	{
		loadData(dataIn + offsetB[0], rData + halfArrayLen,
				 halfArrayLen);
		offsetB[0] += unitLen;
	}
	int tLoop = 0; 
	float *unalignEnd = NULL;
	if (!multipleOf(offsetA[1], unitLen))
	{
		unalignEnd = (float*)_mm_malloc(unitLen * sizeof(float), 16);
		int selector = loadUnalignData(dataIn, offsetA[1], offsetB[1],
									   unalignEnd, unitLen, false);
		if (selector)
			tLoop = getTrailPosition(dataIn, offsetA[0], offsetA[1],
									 unalignEnd[0], unitLen);
		else
			tLoop = getTrailPosition(dataIn, offsetB[0], offsetB[1],
			unalignEnd[0], unitLen);
	}
	size_t loop = (offsetA[1] - offsetA[0] + offsetB[1] - offsetB[0]) / unitLen;
	loop -= tLoop;
	float *blocks[2], *blockBound[2];
	blocks[0] = dataIn + offsetA[0], blocks[1] = dataIn + offsetB[0];
	blockBound[0] = dataIn + offsetA[1], blockBound[1] = dataIn + offsetB[1];
	/*if (stream)
	{
		for (size_t i = 0; i < loop; ++i)
		{
			loadSimdData(rData, blocks, blockBound, 2, halfArrayLen);
			bitonicSort16232(rData);
			streamData(output, rData, halfArrayLen);
		}
		if (unalignEnd != NULL)
		{
			loadData(unalignEnd, rData, halfArrayLen);
			bitonicSort428<2>(rData, true);
			bitonicSort8216<1>(rData, true);
			bitonicSort16232(rData);
			streamData(output, rData, rArrayLen);
			delete [] unalignEnd;
		}
		else
			streamData(output, rData + halfArrayLen, halfArrayLen);
	}*/
	for (size_t i = 0; i < loop; ++i)
	{
		loadSimdData(rData, blocks, blockBound, 2, halfArrayLen);
		bitonicSort16232(rData);
		storeData(output, rData, halfArrayLen);
	}
	bool ua = false;
	if (unalignEnd != NULL)
	{
		loadData(unalignEnd, rData, halfArrayLen);
		bitonicSort428<2>(rData, true);
		bitonicSort8216<1>(rData, true);
		bitonicSort16232(rData);
		storeData(output, rData, halfArrayLen);
		_mm_free(unalignEnd);
		for (int i = 0; i < tLoop; ++i)
		{
			ua = true;
			loadSimdData(rData, blocks, blockBound, 2, halfArrayLen);
			bitonicSort16232(rData);
			storeData(output, rData, halfArrayLen);
		}
	}
	storeData(output, rData + halfArrayLen, halfArrayLen);
	/*size_t length = offsetA[1] + offsetB[1] - offsetA[0] - offsetB[0];
	  std::cout << length << std::endl;*/
	/*for (size_t i = 0; i < 65535; ++i)
		if (dataOut[i] > dataOut[i + 1])
		std::cout << "simd sort fail at: " << i << " " << ua << " " << tLoop << " " << (*output - dataOut) << " " << dataOut[i] - dataOut[i + 1] << std::endl;*/
	//if (tLoop) std::cout << tLoop << std::endl;
}

void mergeInRegisterUnalignBuffer(DoubleBuffer<float> &data,
								  DoubleBuffer<float> &buffer,
								  size_t *offsetA, size_t *offsetB,
								  size_t outputOffset)
{
	simdMergeGeneral(data.buffers[data.selector],
					 buffer.buffers[buffer.selector] + outputOffset,
					 offsetA, offsetB);
}

void mergeInRegisterUnalign(DoubleBuffer<float> &data, size_t offsetA[2],
							size_t offsetB[2], size_t outputOffset)
{
	simdMergeGeneral(data.buffers[data.selector],
					 data.buffers[data.selector ^ 1] + outputOffset,
					 offsetA, offsetB);
	/*int unitLen = (rArrayLen >> 1) * simdLen;
	int halfArrayLen = rArrayLen >> 1;
	__m128 rData[rArrayLen];
	float **output = new float*;
	*output = data.buffers[data.selector ^ 1] + outputOffset;
	if (!multipleOf(offsetA[0], unitLen))
	{
		float *unalignStart = new float[unitLen];
		loadUnalignData(data.Current(), offsetA[0], offsetB[0], unalignStart, unitLen,
						true);
		loadData(unalignStart, rData + halfArrayLen, halfArrayLen);
		bitonicSort428<2>(rData + halfArrayLen, true);
		bitonicSort8216<1>(rData + halfArrayLen, true);
		delete [] unalignStart;
	}
	else
	{
		loadData(data.Current() + offsetB[0], rData + halfArrayLen,
				 halfArrayLen);
		offsetB[0] += unitLen;
	}
	size_t loop = (offsetA[1] - offsetA[0] + offsetB[1] - offsetB[0]) / unitLen;
	float *unalignEnd = NULL;
	if (!multipleOf(offsetA[1], unitLen))
	{
		unalignEnd = new float[unitLen];
		loadUnalignData(data.Current(), offsetA[1], offsetB[1], unalignEnd,
						unitLen, false);
		--loop;
	}
	float *blocks[2], *blockBound[2];
	blocks[0] = data.Current() + offsetA[0];
	blocks[1] = data.Current() + offsetB[0];
	blockBound[0] = data.Current() + offsetA[1];
	blockBound[1] = data.Current() + offsetB[1];
	for (size_t i = 0; i < loop; ++i)
	{
		loadSimdData(rData, blocks, blockBound, 2, halfArrayLen);
		bitonicSort16232(rData);
		//storeSimdData(rData, output, 2, halfArrayLen, 0);
		storeData(output, rData, halfArrayLen);
	}
	if (unalignEnd != NULL)
	{
		loadData(unalignEnd, rData, halfArrayLen);
		bitonicSort428<2>(rData, true);
		bitonicSort8216<1>(rData, true);
		bitonicSort16232(rData);
		//storeSimdData(rData, output, 2, halfArrayLen, 0);
		storeData(output, rData, rArrayLen);
		delete [] unalignEnd;
	}
	else
		storeData(output, rData + halfArrayLen, halfArrayLen);
	//storeSimdData(rData, output, 2, halfArrayLen, halfArrayLen);
	delete output;*/
}

void mergeInRegisterUnalign(DoubleBuffer<float> &data, size_t offsetA[2],
							size_t offsetB[2], float *outputOffset)
{
	int unitLen = (rArrayLen >> 1) * simdLen;
	int halfArrayLen = rArrayLen >> 1;
	__m128 rData[rArrayLen];
	float **output = new float*;
	*output = outputOffset;
	if (!multipleOf(offsetA[0], unitLen))
	{
		float *unalignStart = new float[unitLen];
		loadUnalignData(data.Current(), offsetA[0], offsetB[0], unalignStart,
						unitLen, true);
		loadData(unalignStart, rData + halfArrayLen, halfArrayLen);
		bitonicSort428<2>(rData + halfArrayLen, true);
		bitonicSort8216<1>(rData + halfArrayLen, true);
		delete [] unalignStart;
	}
	else
	{
		loadData(data.Current() + offsetB[0], rData + halfArrayLen,
				 halfArrayLen);
		offsetB[0] += unitLen;
	}
	size_t loop = (offsetA[1] - offsetA[0] + offsetB[1] - offsetB[0]) / unitLen;
	float *unalignEnd = NULL;
	if (!multipleOf(offsetA[1], unitLen))
	{
		unalignEnd = new float[unitLen];
		loadUnalignData(data.Current(), offsetA[1], offsetB[1], unalignEnd,
						unitLen, false);
		--loop;
	}
	float *blocks[2], *blockBound[2];
	blocks[0] = data.Current() + offsetA[0];
	blocks[1] = data.Current() + offsetB[0];
	blockBound[0] = data.Current() + offsetA[1];
	blockBound[1] = data.Current() + offsetB[1];
	for (size_t i = 0; i < loop; ++i)
	{
		loadSimdData(rData, blocks, blockBound, 2, halfArrayLen);
		bitonicSort16232(rData);
		//storeSimdData(rData, output, 2, halfArrayLen, 0);
		storeData(output, rData, halfArrayLen);
	}
	if (unalignEnd != NULL)
	{
		loadData(unalignEnd, rData, halfArrayLen);
		bitonicSort428<2>(rData, true);
		bitonicSort8216<1>(rData, true);
		bitonicSort16232(rData);
		//storeSimdData(rData, output, 2, halfArrayLen, 0);
		storeData(output, rData, rArrayLen);
		delete [] unalignEnd;
	}
	else
		storeData(output, rData + halfArrayLen, halfArrayLen);
	//storeSimdData(rData, output, 2, halfArrayLen, halfArrayLen);
	delete output;
}

void registerSortIteration(DoubleBuffer<float> &data, rsize_t minStride,
						   rsize_t dataLen)
{
	for (size_t i = minStride; i <= dataLen; i *= 2)
	{
		for (rsize_t j = 0; j < dataLen; j += i)
		{
			DoubleBuffer<float> chunk(data.buffers[data.selector] + j,
									  data.buffers[data.selector ^ 1] + j);
			mergeInRegister(chunk, i);
		}
		data.selector ^= 1;
	}
}

void quantileInitial(DoubleBuffer<rsize_t> &quantile, const rsize_t *upperBound,
					 DoubleBuffer<rsize_t> bound, rsize_t chunkNum,
					 rsize_t quantileLen, bool initial)
{
	for (rsize_t j = 0; j < chunkNum; ++j)
		bound.buffers[1][j] =
			std::min(quantile.buffers[0][j] + quantileLen, upperBound[j]);
	// if (initial)
	// {
	// 	//when all sorted chunks are not equivalent size, average can greater
	// 	//than size of a chunk.
	// 	rsize_t average = quantileLen / chunkNum;
	// 	rsize_t residue = quantileLen % chunkNum;
	// 	for (rsize_t j = 0; j < chunkNum; ++j)
	// 		quantile.buffers[1][j] = bound.buffers[0][j] + average +
	// 			(j < residue);
	// }
	// else
	// {
	// 	rsize_t average = quantileLen / chunkNum;
	// 	rsize_t n = quantileLen, row = 0;
	// 	while (n)
	// 	{
	// 		if (row == chunkNum)
	// 			row = 0;
	// 		rsize_t toBeAdd = std::min(std::max(average, (rsize_t)1), n);
	// 		rsize_t canBeAdd =
	// 			bound.buffers[1][row] - quantile.buffers[1][row];
	// 		quantile.buffers[1][row] += std::min(toBeAdd, canBeAdd);
	// 		n -= std::min(toBeAdd, canBeAdd);
	// 		++row;
	// 	}
	// }
	if (initial) std::copy(bound.buffers[0], bound.buffers[0] + chunkNum,
						   quantile.buffers[1]);
	int *remain = new int[chunkNum];
	std::fill(remain, remain + chunkNum, 1);
	size_t n = quantileLen;
	do
	{
		size_t average = n / std::accumulate(remain, remain + chunkNum, 0);
		for (size_t i = 0; i < chunkNum; ++i)
		{
			if (remain[i])
			{
				size_t toBeAdd = std::min(std::max(average, size_t(1)), n);
				size_t canBeAdd = bound.buffers[1][i] - quantile.buffers[1][i];
				if (toBeAdd < canBeAdd)
				{
					quantile.buffers[1][i] += toBeAdd;
					n -= toBeAdd;
				}
				else
				{
					quantile.buffers[1][i] = bound.buffers[1][i];
					n -= canBeAdd;
					remain[i] = 0;
				}
			}
		}
	}while(n);
	delete [] remain;
}

void quantileCompute(float *data, DoubleBuffer<rsize_t> &quantile,
					 DoubleBuffer<rsize_t> &bound, const rsize_t *upperBound,
					 rsize_t chunkNum, rsize_t quantileLen,
					 bool initial = false)
{
	std::copy(quantile.Current(), quantile.Current() + chunkNum,
			  bound.buffers[0]);
	quantileInitial(quantile, upperBound, bound, chunkNum,
						quantileLen, initial);
	while (true)
	{
		const float *lmax = NULL, *rmin = NULL;
		rsize_t lmaxRow, rminRow;
		for (rsize_t j  = 0; j < chunkNum; ++j)
		{
			rsize_t testIndex = quantile.buffers[1][j];
			if (testIndex > bound.buffers[0][j] &&
				(!lmax || *lmax < data[testIndex - 1]))
			{
				lmax = data + testIndex - 1;
				lmaxRow = j;
			}
			if (testIndex < bound.buffers[1][j] &&
				(!rmin || *rmin > data[testIndex]))
			{
				rmin = data + testIndex;
				rminRow = j;
			}
		}
		if (!lmax || !rmin || lmaxRow == rminRow || *lmax < *rmin ||
			(*lmax == *rmin && lmaxRow < rminRow))
            break;
		bound.buffers[1][lmaxRow] = quantile.buffers[1][lmaxRow] - 1;
		bound.buffers[0][rminRow] = quantile.buffers[1][rminRow] + 1;
		rsize_t deltaMin =
			(bound.buffers[1][rminRow] - bound.buffers[0][rminRow]) >> 1;
		rsize_t deltaMax =
			(bound.buffers[1][lmaxRow] - bound.buffers[0][lmaxRow]) >> 1;
		rsize_t delta = std::min(deltaMin, deltaMax);
		quantile.buffers[1][lmaxRow] =
			bound.buffers[1][lmaxRow] - delta;
		quantile.buffers[1][rminRow] =
			bound.buffers[0][rminRow] + delta;
	}
}

void moveBaseQuantile(DoubleBuffer<float> &data, DoubleBuffer<rsize_t> &quantile,
					  DoubleBuffer<rsize_t> bound, const rsize_t *upperBound,
					  rsize_t chunkNum, rsize_t mergeStride, float **ptrOut)
{
	quantileCompute(data.Current(), quantile, bound, upperBound, chunkNum,
					mergeStride);
	for (rsize_t j = 0; j < chunkNum; ++j)
	{
		std::copy(data.buffers[data.selector] + quantile.buffers[0][j],
				  data.buffers[data.selector] + quantile.buffers[1][j],
				  *ptrOut);
		(*ptrOut) += (quantile.buffers[1][j] - quantile.buffers[0][j]);
	}
	std::copy(quantile.buffers[1], quantile.buffers[1] + chunkNum,
			  quantile.buffers[0]);
}

void multiWayMerge(DoubleBuffer<float> &data, rsize_t dataLen,
				   rsize_t sortedBlockLen, rsize_t mergeStride,
				   rsize_t startIndex, rsize_t endIndex)
{
	rsize_t sortedBlockNum = dataLen / sortedBlockLen;
    rsize_t *quantileStart = new rsize_t[sortedBlockNum];
    rsize_t *quantileEnd = new rsize_t[sortedBlockNum];
    rsize_t *upperBound = new rsize_t[sortedBlockNum];
    rsize_t *loopUBound = new rsize_t[sortedBlockNum];
    rsize_t *loopLBound = new rsize_t[sortedBlockNum];
	DoubleBuffer<rsize_t> quantile(quantileStart, quantileEnd);
	DoubleBuffer<rsize_t> bound(loopLBound, loopUBound);
	for (rsize_t j = 0; j < sortedBlockNum; ++j)
		quantile.buffers[0][j] = sortedBlockLen * j;
	for (rsize_t j = 0; j < sortedBlockNum; ++j)
		upperBound[j] = quantile.buffers[0][j] + sortedBlockLen;
	rsize_t i = startIndex;
	float *ptrOut = data.buffers[data.selector ^ 1];
	if (startIndex)
	{
		ptrOut += i * mergeStride;
		quantileCompute(data.buffers[data.selector], quantile, bound, upperBound,
						sortedBlockNum, mergeStride * i, true);
		std::copy(quantile.buffers[1], quantile.buffers[1] + sortedBlockNum,
				  quantile.buffers[0]);
	}
	else
	{
		std::copy(quantile.buffers[0], quantile.buffers[0] + sortedBlockNum,
				  quantile.buffers[1]);
	}
	for (; i < endIndex - 1; ++i)
		moveBaseQuantile(data, quantile, bound, upperBound, sortedBlockNum,
						 mergeStride, &ptrOut);
	if (endIndex < dataLen / mergeStride)
		moveBaseQuantile(data, quantile, bound, upperBound, sortedBlockNum,
						 mergeStride, &ptrOut);
	else
		for (rsize_t j = 0; j < sortedBlockNum; ++j)
			for (rsize_t k = quantile.buffers[0][j]; k < upperBound[j];
				 ++k)
				*ptrOut++ = data.buffers[data.selector][k];
	delete [] quantileStart;
    delete [] quantileEnd;
    delete [] upperBound;
    delete [] loopUBound;
    delete [] loopLBound;
}

void multiWayMergeGeneral(DoubleBuffer<float> &data, size_t dataLen,
						  size_t sortedChunkLen, size_t mergeStride,
						  size_t startOffset, size_t endOffset)
{
	size_t sortedChunkNum = dataLen / sortedChunkLen;
	size_t *quantileStart = new size_t[sortedChunkNum];
	size_t *quantileEnd = new size_t[sortedChunkNum];
	size_t *upperBound = new size_t[sortedChunkNum];
	size_t *loopUBound = new size_t[sortedChunkNum];
	size_t *loopLBound = new size_t[sortedChunkNum];
	DoubleBuffer<size_t> quantile(quantileStart, quantileEnd);
	DoubleBuffer<size_t> bound(loopLBound, loopUBound);
	for (size_t j = 0; j < sortedChunkNum; ++j)
		quantile.buffers[0][j] = sortedChunkLen * j;
	for (size_t j = 0; j < sortedChunkNum; ++j)
		upperBound[j] = quantile.buffers[0][j] + sortedChunkLen;
	float *ptrOut = data.buffers[data.selector ^ 1] + startOffset;
	if (startOffset)
	{
		quantileCompute(data.buffers[data.selector], quantile, bound, upperBound,
						sortedChunkNum, startOffset, true);
		std::copy(quantile.buffers[1], quantile.buffers[1] + sortedChunkNum,
				  quantile.buffers[0]);
	}
	else
	{
		std::copy(quantile.buffers[0], quantile.buffers[0] + sortedChunkNum,
				  quantile.buffers[1]);
	}
	for (size_t offset = startOffset; offset < endOffset - mergeStride;
		 offset += mergeStride)
	{
		moveBaseQuantile(data, quantile, bound, upperBound, sortedChunkNum,
						 mergeStride, &ptrOut);
	}
	if (endOffset < dataLen)
		moveBaseQuantile(data, quantile, bound, upperBound, sortedChunkNum,
						 mergeStride, &ptrOut);
	else
		for (size_t j = 0; j < sortedChunkNum; ++j)
			for (size_t k = quantile.buffers[0][j]; k < upperBound[j]; ++k)
				*ptrOut++ = data.buffers[data.selector][k];
	delete [] quantileStart;
	delete [] quantileEnd;
	delete [] upperBound;
	delete [] loopUBound;
	delete [] loopLBound;
}

void multiWayMergeHybrid(DoubleBuffer<float> &data, size_t dataLen,
						 size_t *upperBound, size_t chunkNum, size_t mergeStride,
						 size_t startOffset, size_t endOffset)
{
	size_t *quantileStart = new size_t[chunkNum];
	size_t *quantileEnd = new size_t[chunkNum];
	size_t *loopUBound = new size_t[chunkNum];
	size_t *loopLBound = new size_t[chunkNum];
	DoubleBuffer<size_t> quantile(quantileStart, quantileEnd);
	DoubleBuffer<size_t> bound(loopLBound, loopUBound);
	quantile.buffers[0][0] = 0;
	std::copy(upperBound, upperBound + chunkNum - 1, quantile.buffers[0] + 1);
	/*for (size_t j = 1; j < chunkNum; ++j)
	  quantile.buffers[0][j] = upperBound[j - 1];*/
	float *ptrOut = data.buffers[data.selector ^ 1] + startOffset;
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
	size_t end = std::min(endOffset, dataLen - mergeStride);
	for (size_t offset = startOffset; offset < end; offset += mergeStride)
	{
		moveBaseQuantile(data, quantile, bound, upperBound, chunkNum,
						 mergeStride, &ptrOut);
	}
	// if (endOffset < dataLen)
	// 	moveBaseQuantile(data, quantile, bound, upperBound, chunkNum,
	// 					 mergeStride, &ptrOut);
	// else
	if (endOffset == dataLen)
		for (size_t j = 0; j < chunkNum; ++j)
			//for (size_t k = quantile.buffers[0][j]; k < upperBound[j]; ++k)
				//*ptrOut++ = data.buffers[data.selector][k];
		{
			std::copy(data.buffers[data.selector] + quantile.buffers[0][j],
					  data.buffers[data.selector] + upperBound[j], ptrOut);
			ptrOut += (upperBound[j] - quantile.buffers[0][j]);
		}
	delete [] quantileStart;
	delete [] quantileEnd;
	delete [] loopUBound;
	delete [] loopLBound;
}

void mergeSort(DoubleBuffer<float> &data, rsize_t dataLen)
{
	for (rsize_t offset = 0; offset < dataLen; offset += blockUnitLen)
		sortInRegister(data.Current() + offset);
	registerSortIteration(data, blockUnitLen * 2, dataLen);
}

void singleThreadMerge(DoubleBuffer<float> &data, size_t dataLen)
{
	__m128 rData[rArrayLen];
	float *ptr = data.buffers[data.selector];
	size_t loop = dataLen >> logBlockUnitLen; 
	//cannot use pointer to pointer, because load data and store data coexist.
	for (size_t i = 0; i < loop; ++i)
	{
		loadData(ptr, rData, rArrayLen);
		simdOddEvenSort(rData);
		bitonicSort428<4>(rData, true);
		storeData(ptr, rData, rArrayLen);
		ptr += blockUnitLen;
	}
	mergeInRegister(data, dataLen, rArrayLen);
}

void updateMergeSelector(int &selector, rsize_t dataLen)
{
	rsize_t blocks = dataLen / blockUnitLen;
	if (_tzcnt_u64(blocks) & 1) selector ^= 1;
}

void updateSelectorGeneral(int &selector, size_t startLen, size_t dataLen)
{
	size_t blocks = dataLen / startLen;
	if (_tzcnt_u64(blocks) & 1) selector ^= 1;
}

size_t lastPower2(size_t a)
{
	return 1 << (64 - _lzcnt_u64(a) - 1);
}

void mergeSortGeneral(DoubleBuffer<float> &data, size_t dataLen)
{
	for (rsize_t offset = 0; offset < dataLen; offset += blockUnitLen)
		sortInRegister(data.Current() + offset);
	// multiWayMergeGeneral(data, dataLen, blockUnitLen, blockUnitLen, 0, dataLen);
	// data.selector ^= 1;
	size_t sortedBlockLen = blockUnitLen;
	size_t sortedBlockNum = dataLen / blockUnitLen;
	do
	{
		size_t stride = std::min(sortedBlockNum, size_t(16));
		size_t strideLen = stride * sortedBlockLen;
		for (size_t j = 0; j < dataLen; j += strideLen)
		{
			multiWayMergeGeneral(data, dataLen, sortedBlockLen, blockUnitLen,
								 j, j + strideLen);
		}
		data.selector ^= 1;
		for (rsize_t offset = 0; offset < dataLen; offset += blockUnitLen)
			sortInRegister(data.Current() + offset);
		sortedBlockLen = strideLen;
		sortedBlockNum = dataLen / sortedBlockLen;
	}while(sortedBlockNum > 1);
}

//medianA and medianB are both relative offset.
void getMedian(float *data, size_t mergeLen, size_t chunkLen,
			   size_t baseOffset, size_t &medianA, size_t &medianB)
{
	bool equalt = ((mergeLen % 65536) == 0);
	//std::cout << mergeLen << " " << equalt << std::endl;
	size_t startA = medianA, startB = medianB;
	size_t minA, maxA;
	//if chunkA is "shorter" then everything is ok, but if chunkA is "longer",
	//then care must be taken to prevent medianB get a value that bigger
	//than the bound of chunkB.
	//the key is: is mergeLen - (chunkLen - medianB) positive or negative?
	/*if (mergeLen > (chunkLen - offsetA))
	{
		maxA = chunkLen;
	}
	else
	{
		maxA = offsetA + mergeLen;
	}
	if (mergeLen > (chunkLen - offsetB))
		minA = offsetA + mergeLen - (chunkLen - offsetB);
	else
		minA = offsetA;*/
	maxA = std::min(medianA + mergeLen, chunkLen);
	minA = medianA + mergeLen - std::min(mergeLen, (chunkLen - medianB));
	float *blockA = data + baseOffset;
	float *blockB = data + baseOffset + chunkLen;
	while(minA + 1 != maxA)
	{
		size_t median = (minA + maxA) >> 1;
		if(blockA[median] <= blockB[mergeLen - median])
			minA = median;
		else
			maxA = median;
	}
	size_t resultA = (blockA[minA] <= blockB[mergeLen - maxA])? maxA : minA;
	medianA = resultA;
	medianB = mergeLen - resultA;
	/*std::cout << mergeLen << std::endl;
	if ((medianB + medianA - startA - startB) == mergeLen)
		std::cout << "true" << std::endl;
	else
	std::cout << "false" << std::endl;*/
}

void multiThreadMergeGeneral(float *dataIn, float* dataOut, size_t dataLen,
							 int chunkNum, size_t blockLen)
{
	int blockNum = dataLen / blockLen;
	size_t chunkLen = dataLen / chunkNum;
	//float *ptr = dataIn;
	int pairNum = chunkNum >> 1;
	int medianNum =  blockNum - pairNum;
	int mdNumPerPair = medianNum / pairNum;
	int thdNumPerPair = blockNum / pairNum;
	size_t *medianA = new size_t[medianNum];
	size_t *medianB = new size_t[medianNum];
	//may read much fewer elements than sort, so can compute all medians.
	//#pragma omp parallel for
	for (int j = 0; j < medianNum; ++j)
	{
		//length can greater than the length of a chunk.
		size_t length = (j % mdNumPerPair + 1) * blockLen;
		int pairIndex = j / mdNumPerPair;
		size_t baseOffset = pairIndex * chunkLen * 2;
		size_t mA = 0, mB = 0;
		getMedian(dataIn, length, chunkLen, baseOffset, mA, mB);
		medianA[j] = baseOffset + mA;
		medianB[j] = baseOffset + chunkLen + mB;
	}
	//for (int k = 0; k < medianNum; ++k)
	//std::cout << medianA[k] << " " << medianB[k] << std::endl;
	//std::cout << std::endl << std::endl;
	//#pragma omp parallel for
	for (int j = 0; j < blockNum; ++j)
	{
		size_t offsetA[2], offsetB[2];
		int pairIndex = j / thdNumPerPair;
		int offset = j % thdNumPerPair;
		int preThreadNum = pairIndex * mdNumPerPair;
		size_t baseOffset = pairIndex * chunkLen * 2;
		int baseIndex = preThreadNum + offset;
		if (offset)
		{
			offsetA[0] = medianA[baseIndex - 1];
			offsetB[0] = medianB[baseIndex - 1];
		}
		else
		{
			offsetA[0] = baseOffset;
			offsetB[0] = baseOffset + chunkLen;
		}
		if (offset == (thdNumPerPair - 1))
		{
			offsetA[1] = baseOffset + chunkLen;
			offsetB[1] = offsetA[1] + chunkLen;
		}
		else
		{
			offsetA[1] = medianA[baseIndex];
			offsetB[1] = medianB[baseIndex];
		}
		//mergeInRegisterUnalign(data, offsetA, offsetB, j * blockLen);
		//std::cout << thdNumPerPair << " " << chunkLen << " " << baseOffset << " " << pairIndex << " " << offset << " " << j << " " << offsetB[1] + offsetA[1] - offsetA[0] - offsetB[0] << "---------" << std::endl;
		simdMergeGeneral(dataIn, dataOut + j * blockLen, offsetA, offsetB);
	}
	delete [] medianA;
	delete [] medianB;
}

//chunkNum should be exponent of 2, all chunks must be equal of length. 
void multiThreadMerge(DoubleBuffer<float> &data, size_t dataLen, int chunkNum,
					  size_t blockLen)
{
	//int blockNum = dataLen / blockLen;
	for (int i = chunkNum; i > 1; i >>= 1)
	{
		// size_t chunkLen = dataLen / i;
		// float *ptr = data.Current();
		// int pairNum = i >> 1;
		// int medianNum =  blockNum - pairNum;
		// int mdNumPerPair = medianNum / pairNum;
		// int thdNumPerPair = blockNum / pairNum;
		// size_t *medianA = new size_t[medianNum];
		// size_t *medianB = new size_t[medianNum];
		//may read much fewer elements than sort, so can compute all medians.
// #pragma omp parallel for
// 		for (int j = 0; j < medianNum; ++j)
// 		{
			//length can greater than the length of a chunk.
			// size_t length = (j % mdNumPerPair + 1) * blockLen;
			// int pairIndex = j / mdNumPerPair;
			// size_t baseOffset = pairIndex * chunkLen * 2;
			//size_t startA, endA;
			/*if (length > chunkLen)
			{
				startA = length - chunkLen;
				endA = chunkLen;
			}
			else
			{
				startA = 0;
				endA = length;
			}*/
			/*endA = std::min(length, chunkLen);
			startA = length - std::min(length, chunkLen);
			float *blockA = ptr + baseOffset, *blockB = blockA + chunkLen;
			while (startA + 1 != endA)
			{
				//two pointers cannot be added directly, so we use offset here.
				size_t median = (startA + endA) >> 1;
				if (*(blockA + median) <= *(blockB + length - median))
					startA = median;
				else
					endA = median;
			}
			size_t resultA =
				(*(blockA+startA) <= *(blockB+length-endA))? endA : startA;
			medianA[j] = baseOffset + resultA;
			medianB[j] = baseOffset + chunkLen + length - resultA;*/
			// size_t mA = 0, mB = 0;
// 			getMedian(data, length, chunkLen, baseOffset, mA, mB);
// 			medianA[j] = baseOffset + mA;
// 			medianB[j] = baseOffset + chunkLen + mB;
// 		}
// #pragma omp parallel for
// 		for (int j = 0; j < blockNum; ++j)
// 		{
// 			size_t offsetA[2], offsetB[2];
// 			int pairIndex = j / thdNumPerPair;
// 			int offset = j % thdNumPerPair;
// 			int preThreadNum = pairIndex * mdNumPerPair;
// 			size_t baseOffset = pairIndex * chunkLen * 2;
// 			int baseIndex = preThreadNum + offset;
// 			if (offset)
// 			{
// 				offsetA[0] = medianA[baseIndex - 1];
// 				offsetB[0] = medianB[baseIndex - 1];
// 			}
// 			else
// 			{
// 				offsetA[0] = baseOffset;
// 				offsetB[0] = baseOffset + chunkLen;
// 			}
// 			if (offset == (thdNumPerPair - 1))
// 			{
// 				offsetA[1] = baseOffset + chunkLen;
// 				offsetB[1] = offsetA[1] + chunkLen;
// 			}
// 			else
// 			{
// 				offsetA[1] = medianA[baseIndex];
// 				offsetB[1] = medianB[baseIndex];
// 			}
// 			mergeInRegisterUnalign(data, offsetA, offsetB, j * blockLen);
// 		}
// 		delete [] medianA;
// 		delete [] medianB;
		multiThreadMergeGeneral(data.buffers[data.selector],
								data.buffers[data.selector ^ 1], dataLen, i,
								blockLen);
		data.selector ^= 1;
		/*for (size_t j = 0; j < dataLen; j += dataLen * 2 / i)
			for (size_t k = j; k < j - 1 + dataLen * 2 / i; ++k)
				if(*(data.Current() + k) > *(data.Current() + k + 1))
				std::cout << "sort fail at " << i <<" " << k << std::endl;*/
	}
}

//solution only for merge several chunks. dataLen, chunkLen, blockLen must all
//be power of 2.
void multiThreadMergeChunk(DoubleBuffer<float> &data, size_t dataLen,
						   int chunkNum, int blockLen)
{
	int unitLen = (rArrayLen >> 1) * simdLen;
	int halfArrayLen = rArrayLen >> 1;
	int threads = omp_get_max_threads();
	size_t bufferLen = blockLen * threads;
	float *bufferA = (float *)_mm_malloc(bufferLen * sizeof(float), 16);
	float *bufferB = (float *)_mm_malloc(bufferLen * sizeof(float), 16);
	DoubleBuffer<float> hBuffer(bufferA, bufferB);
	float *ptrIn = data.Current();
	float *ptrOut = data.buffers[data.selector ^ 1];
	do
	{
		size_t chunkLen = dataLen / chunkNum;
		size_t stride = std::min(dataLen, threads * 2 * chunkLen);
		for (size_t i = 0; i < dataLen; i += stride)
		{
			int pairNum = i >> 1;
			if (pairNum == threads)
			{
				float *startA = new float[threads];
				float *startB = new float[threads];
				//is here need parallel or not?
				//#pragma omp parallel for
				/*for (int k = 0; k < threads; ++k)
				{
					startA[k] = k * chunkLen * 2;
					startB[k] = k * chunkLen * 2 + chunkLen;
					}*/
				std::fill(startA, startA + threads, 0);
				std::fill(startB, startB + threads, 0);
				for (size_t j = 0; j < stride; j += bufferLen)
				{
#pragma omp parallel for
					for (int k = 0; k < threads; ++k)
					{
						__m128 rData[rArrayLen];
						size_t offsetA[2], offsetB[2];
						size_t baseOffset = k * chunkLen * 2;
						offsetA[0] = startA[k] + baseOffset;
						offsetB[0] = startB[k] + baseOffset + chunkLen;
						offsetA[1] = startA[k], offsetB[1] = startB[k];
						getMedian(data.Current(), blockLen, chunkLen,
								  baseOffset, offsetA[1], offsetB[1]);
						startA[k] = offsetA[1], startB[k] = offsetB[1];
						offsetA[1] += baseOffset;
						offsetB[1] += (baseOffset + chunkLen);
						simdMergeGeneral(data.Current(),
										 hBuffer.Current() + k * bufferLen,
										 offsetA, offsetB);
					}
					multiThreadMerge(hBuffer, bufferLen, threads, blockLen);
					multiThreadMergeGeneral(hBuffer.Current(),
											data.buffers[data.selector ^ 1] +
											j * stride,
											bufferLen, 2, blockLen);
				}
				delete [] startA;
				delete [] startB;
			}
			else
			{
				for (int j = stride / chunkLen; j > 1; j >>= 1)
				{
					
				}
			}
			//multi-way merge, the number of chunks is not decreased once by
			//half.
			chunkNum -= (stride / chunkLen - 1);
		}
		
	}while(chunkNum > 1);
	_mm_free(bufferA);
	_mm_free(bufferB);
}

