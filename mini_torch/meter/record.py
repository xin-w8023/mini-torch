class Record:
    def __init__(self):
        self.value = 0
        self.step = 0

    def append(self, value):
        self.value = self.value * self.step + value
        self.step += 1
        self.value /= self.step

    def report(self):
        return round(self.value, 4)

    def reset(self):
        self.value = 0
        self.step = 0
