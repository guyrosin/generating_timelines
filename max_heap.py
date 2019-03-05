import heapq


class MaxHeap:
    """
    Max heap
    """

    def __init__(self, capacity):
        self.heap = []
        self.capacity = capacity

    def add(self, score, term):
        item = (score, term)
        if len(self.heap) >= self.capacity:
            heapq.heappushpop(self.heap, item)
        else:
            heapq.heappush(self.heap, item)

    def heapify(self):
        heapq.heapify(self.heap)
