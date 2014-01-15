/* sort functions using double buffer class */
#include <iostream>
#include <algorithm>
#include <immintrin.h>
#include "cpu_sort.h"
#include "sse_sort.h"

void mergeInRegister(DoubleBuffer<float> &data, size_t dataLen)
{
	const size_t halfDataLen = dataLen >> 1;
	const int halfArrayLen = rArrayLen >> 1;
	__m128 *rData = new __m128[rArrayLen];
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
	delete [] rData;
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
	if (initial)
	{
		rsize_t average = quantileLen / chunkNum;
		rsize_t residue = quantileLen % chunkNum;
		for (rsize_t j = 0; j < chunkNum; ++j)
			quantile.buffers[1][j] = bound.buffers[0][j] + average +
				(j < residue);
	}
	else
	{
		rsize_t average = quantileLen / chunkNum;
		rsize_t n = quantileLen, row = 0;
		while (n)
		{
			if (row == chunkNum)
				row = 0;
			rsize_t toBeAdd = std::min(std::max(average, (rsize_t)1), n);
			rsize_t canBeAdd =
				bound.buffers[1][row] - quantile.buffers[1][row];
			quantile.buffers[1][row] += std::min(toBeAdd, canBeAdd);
			n -= std::min(toBeAdd, canBeAdd);
			++row;
		}
	}
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
	for (size_t offset = startOffset; offset < endOffset - mergeStride;
		 offset += mergeStride)
	{
		moveBaseQuantile(data, quantile, bound, upperBound, chunkNum,
						 mergeStride, &ptrOut);
	}
	if (endOffset < dataLen)
		moveBaseQuantile(data, quantile, bound, upperBound, chunkNum,
						 mergeStride, &ptrOut);
	else
		for (size_t j = 0; j < chunkNum; ++j)
			/*for (size_t k = quantile.buffers[0][j]; k < upperBound[j]; ++k)
			 *ptrOut++ = data.buffers[data.selector][k];*/
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

void updateMergeSelcetor(int &selector, rsize_t dataLen)
{
	rsize_t blocks = dataLen / blockUnitLen;
	if (_tzcnt_u64(blocks) & 1) selector ^= 1;
}

void updateSelectorGeneral(int &selector, size_t startLen, size_t dataLen)
{
	size_t blocks = dataLen / startLen;
	if (_tzcnt_u64(blocks) & 1) selector ^= 1;
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
		

