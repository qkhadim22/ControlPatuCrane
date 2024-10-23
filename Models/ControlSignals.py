
import numpy as np


# Control signal 1
def uref_1(t):
    Lifting_Time_Start_1 = 0.5          # Start of lifting mass, m
    Lifting_Time_End_1 = 1.25              # End of lifting mass, m
    Lowering_Time_Start_1 = 2.0         # Start of lowering mass, m
    Lowering_Time_End_1 = 4.25           # End of lowering mass, m
    Lowering_Time_Start_2 = 5.0         # Start of lowering mass, m
    Lowering_Time_End_2 = 6.5            # End of lowering mass, m
    Lowering_Time_Start_3 = 7.5         # Start of lowering mass, m
    Lowering_Time_End_3 = 9.5           # End of lowering mass, m

    if Lifting_Time_Start_1 <= t < Lifting_Time_End_1:
        u = -1
    elif Lowering_Time_Start_1 <= t < Lowering_Time_End_1:
        u = 1
    elif Lowering_Time_Start_2 <= t < Lowering_Time_End_2:
        u = -1
    elif Lowering_Time_Start_3 <= t < Lowering_Time_End_3:
        u = 1
    else:
        u = 0
    
    
   
    return u




# Control signal 2
def uref_2(t):
    Lifting_Time_Start_1 = 0.5          # Start of lifting mass, m
    Lifting_Time_End_1 = 2.0            # End of lifting mass, m
    Lowering_Time_Start_1 = 3.0         # Start of lowering mass, m
    Lowering_Time_End_1 = 5.0           # End of lowering mass, m
    Lowering_Time_Start_2 = 5.5         # Start of lowering mass, m
    Lowering_Time_End_2 = 7.0           # End of lowering mass, m
    Lowering_Time_Start_3 = 8.0         # Start of lowering mass, m
    Lowering_Time_End_3 = 9.25           # End of lowering mass, m

    if Lifting_Time_Start_1 <= t < Lifting_Time_End_1:
        u = 1
    elif Lowering_Time_Start_1 <= t < Lowering_Time_End_1:
        u = -1
    elif Lowering_Time_Start_2 <= t < Lowering_Time_End_2:
        u = 1
    elif Lowering_Time_Start_3 <= t < Lowering_Time_End_3:
        u = -1
    else:
        u = 0

    return u






