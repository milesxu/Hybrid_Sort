/*
 * cpu_sort.h
 *
 *  Created on: 2013-11-13
 *      Author: aloneranger
 */

#ifndef CPU_SORT_H_
#define CPU_SORT_H_

#ifndef _Windows
typedef size_t rsize_t;
#endif

template <typename T>
struct DoubleBuffer
{
    T *buffers[2];
	
    int selector;
	
    inline DoubleBuffer()
    {
        selector = 0;
        buffers[0] = NULL;
        buffers[1] = NULL;
    }
	
    inline DoubleBuffer(T *current, T *alternate)
    {
        selector = 0;
        buffers[0] = current;
        buffers[1] = alternate;
    }
	
    inline T* Current()
    {
        return buffers[selector];
    }
};

void updateMergeSelcetor(int &selector, rsize_t dataLen);
void updateSelectorGeneral(int &selector, size_t startLen, size_t dataLen);
void mergeSort(DoubleBuffer<float> &data, rsize_t dataLen);
void mergeSortGeneral(DoubleBuffer<float> &data, rsize_t dataLen);
void registerSortIteration(DoubleBuffer<float> &data, rsize_t minStride,
						   rsize_t dataLen);
void multiWayMerge(DoubleBuffer<float> &data, rsize_t dataLen,
				   rsize_t sortedBlockLen, rsize_t mergeStride,
				   rsize_t startIndex, rsize_t endIndex);
void multiWayMergeGeneral(DoubleBuffer<float> &data, size_t dataLen,
						  size_t sortedChunkLen, size_t mergeStride,
						  size_t startOffset, size_t endOffset);
void multiWayMergeHybrid(DoubleBuffer<float> &data, size_t dataLen,
						 size_t *upperBound, size_t chunkNum, size_t mergeStride,
						 size_t startOffset, size_t endOffset);

#endif /* CPU_SORT_H_ */

