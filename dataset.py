import math
import numpy as np

def int_to_bin(number: int) -> int:
    
    # if number is negative or not an integer raise an error
    if number < 0 or type(number) is not int:
        raise ValueError("only positive integers are allowed")
        
    # converts binary number into a list and returns it
    return [int(x) for x in list(bin(number))[2:]]

from typing import Tuple, List

def data_generator(max_int: int, batch_size: int=16) -> Tuple[List[int], List[List[int]]]:
    
    # calculate number of digits required to represent the largest number provided by user
    # i.e. max length = log2(max_int)
    max_length = int(math.log(max_int, 2))
    
    # generate data, i.e. total batch_size numeer of even integers between 0 and max_int/2
    sampled_integers = np.random.randint(0, int(max_int)/2, batch_size)
    
    # generate labels for that, all would be 1 as all of them are even numbers
    labels = [1] * batch_size
    
    
    # generate binary numbers for training
    data = [int_to_bin(int(x*2)) for x in sampled_integers]
    
    # 0 padding to make the length of all binary numbers of same length i.e. equal to max_length
    data = [([0]*(max_length - len(x))) + x for x in data]

    return labels, data