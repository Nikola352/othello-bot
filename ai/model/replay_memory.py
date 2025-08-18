from collections import deque
import random


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add(self, data):
        self.memory.append(data)

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)