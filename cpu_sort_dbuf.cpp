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
	float *ptrLeftEnd = data.Current() + halfDataLen;
	float *ptrRightEnd = data.Current() + dataLen;
	size_t lRemain = halfDataLen, rRemain = halfDataLen;
	loadData(ptrRightEnd - rRemain, rData + halfArrayLen, halfArrayLen);
	rRemain -= sortUnitLen;
	while (lRemain || rRemain)
	{
		bool useLeft;
		if (lRemain && rRemain)
            useLeft = *(ptrLeftEnd - lRemain) < *(ptrRightEnd - rRemain);
        else
            useLeft = lRemain > rRemain;
        if (useLeft)
        {
            loadData(ptrLeftEnd - lRemain, rData, halfArrayLen);
            lRemain -= sortUnitLen;
        }
        else
        {
            loadData(ptrRightEnd - rRemain, rData, halfArrayLen);
            rRemain -= sortUnitLen;
        }
		
        bitonicSort16232(rData);
        storeData(ptrOut, rData, halfArrayLen);
        ptrOut += sortUnitLen;
	}
	storeData(ptrOut, rData + halfArrayLen, halfArrayLen);
	delete [] rData;
}

void registerSortIteration(DoubleBuffer<float> &data, rsize_t minStride,
						   rsize_t maxStride, rsize_t dataLen)
{
	rsize_t sortStride = minStride;
	while (sortStride <= maxStride)
	{
		for (rsize_t j = 0; j < dataLen; j += sortStride)
		{
			DoubleBuffer<float> chunk(data.buffers[data.selector] + j,
					data.buffers[data.selector ^ 1] + j);
			mergeInRegister(chunk, sortStride);
		}
		data.selector ^= 1;
		sortStride *= 2;
	}
}

void quantileInitialByPred(DoubleBuffer<rsize_t> &quantile, const rsize_t *upperBound,
						   DoubleBuffer<rsize_t> bound, rsize_t sortedBlockNum,
						   rsize_t mergeStride)
{
	for (rsize_t j = 0; j < sortedBlockNum; ++j)
		bound.buffers[1][j] =
			std::min(quantile.buffers[0][j] + mergeStride, upperBound[j]);
	rsize_t average = mergeStride / sortedBlockNum;
	rsize_t n = mergeStride, row = 0;
	while (n)
	{
		if (row == sortedBlockNum)
			row = 0;
		rsize_t toBeAdd = std::min(std::max(average, (rsize_t)1), n);
		rsize_t canBeAdd =
			bound.buffers[1][row] - quantile.buffers[1][row];
		quantile.buffers[1][row] += std::min(toBeAdd, canBeAdd);
		n -= std::min(toBeAdd, canBeAdd);
		++row;
	}
}

void quantileCompute(float *data, DoubleBuffer<rsize_t> &quantile,
					 DoubleBuffer<rsize_t> &bound, const rsize_t *upperBound,
					 rsize_t sortedBlockNum, rsize_t mergeStride,
					 rsize_t quantileLen, bool initial = false)
{
	std::copy(quantile.Current(), quantile.Current() + sortedBlockNum,
			  bound.buffers[0]);
	if (initial)
	{
		for (rsize_t j = 0; j < sortedBlockNum; ++j)
			bound.buffers[1][j] =
				std::min(quantile.buffers[0][j] + quantileLen,
						 upperBound[j]);
		rsize_t average = quantileLen / sortedBlockNum;
		rsize_t residue = quantileLen % sortedBlockNum;
		for (rsize_t j = 0; j < sortedBlockNum; ++j)
			quantile.buffers[1][j] = bound.buffers[0][j] + average +
				(j < residue);
	}
	else
	{
		quantileInitialByPred(quantile, upperBound, bound, sortedBlockNum,
							  mergeStride);
	}
	while (true)
	{
		const float *lmax = NULL, *rmin = NULL;
		rsize_t lmaxRow, rminRow;
		for (rsize_t j  = 0; j < sortedBlockNum; ++j)
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
					  rsize_t sortedBlockNum, rsize_t mergeStride, float **ptrOut)
{
	quantileCompute(data.Current(), quantile, bound, upperBound, sortedBlockNum,
					mergeStride, mergeStride);
	//float *ptrOut = data.buffers[data.selector ^ 1];
	for (rsize_t j = 0; j < sortedBlockNum; ++j)
		for (rsize_t k = quantile.buffers[0][j];
			 k < quantile.buffers[1][j]; ++k)
			*(*ptrOut)++ = data.buffers[data.selector][k];
	//quantile.selector ^= 1;
	std::copy(quantile.buffers[1], quantile.buffers[1] + sortedBlockNum,
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
						sortedBlockNum, mergeStride, mergeStride * i, true);
		std::copy(quantile.buffers[1], quantile.buffers[1] + sortedBlockNum,
				  quantile.buffers[0]);
		//quantile.selector ^= 1;
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
		//ptrOut += startOffset;
		quantileCompute(data.buffers[data.selector], quantile, bound, upperBound,
						sortedChunkNum, mergeStride, startOffset, true);
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

void mergeSort(DoubleBuffer<float> &data, rsize_t dataLen)
{
	for (rsize_t offset = 0; offset < dataLen; offset += blockUnitLen)
		sortInRegister(data.Current() + offset);
	registerSortIteration(data, blockUnitLen * 2, dataLen, dataLen);
}

void updateMergeSelcetor(int *selector, rsize_t dataLen)
{
	rsize_t blocks = dataLen / blockUnitLen;
	if (_tzcnt_u64(blocks) & 1)
		*selector ^= 1;
}


		

