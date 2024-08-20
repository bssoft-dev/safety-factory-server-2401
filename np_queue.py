import numpy as np

class CircularQueue:
    def __init__(self, size):
        self.buffer = np.zeros(size, dtype=np.int16)
        self.size = size
        self.start = 0
        self.end = 0
        self.count = 0

    def append(self, data):
        data_len = len(data)
        if data_len > self.size:
            # Save last data if data size is longer than buffer size
            data = data[-self.size:]
            data_len = self.size

        # Copy data to buffer
        if self.end + data_len <= self.size:
            self.buffer[self.end:self.end + data_len] = data
        else:
            split = self.size - self.end
            self.buffer[self.end:] = data[:split]
            self.buffer[:data_len - split] = data[split:]

        self.end = (self.end + data_len) % self.size
        self.count = min(self.count + data_len, self.size)
        self.start = (self.end - self.count) % self.size

    def get_data(self):
        print(self.count)
        if self.count == 0:
            return np.array([], dtype=np.int16)
        if self.start < self.end:
            return self.buffer[self.start:self.end]
        else:
            return np.concatenate((self.buffer[self.start:], self.buffer[:self.end]))

    def clear_processed(self, amount):
        self.start = (self.start + amount) % self.size
        self.count = max(0, self.count - amount)
