import numpy as np

def chronological_split(data, split_ratio=0.8):
    n = len(data)
    split_point = int(n * split_ratio)
    
    train = data[:split_point]
    test = data[split_point:]
    
    return train, test