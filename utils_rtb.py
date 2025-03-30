import numpy as np
import matplotlib.pyplot as plt
import torch

def create_batches(ids, batch_size):
    for i in range(0, len(ids), batch_size):
        yield ids[i:i + batch_size]