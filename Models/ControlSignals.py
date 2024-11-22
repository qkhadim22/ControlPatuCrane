
import numpy as np


#######################
#LOAD Experimetal data--

# If you change data here, then also change in ControlSignals file
fileName = 'ExpData/CaptureData_2022-04-12_14-24-18_vf_110_pp_140_from_th0_15dg_both_booms_moving_processed.txt'
ExpData = np.loadtxt(fileName, delimiter=' ')

###############################

"""
    Time: ExpData[:,0]
    U1: ExpData[:,1]
    s1: ExpData[:,2]
    p1: ExpData[:,3]*1e5
    p2: ExpData[:,4]*1e5

    U2: ExpData[:,5]
    s2: ExpData[:,6]
    p3: ExpData[:,7]*1e5
    p4: ExpData[:,8]*1e5
    
    pP: ExpData[:,9]
    a1x: ExpData[:,10]
    a1y: ExpData[:,11]
    w1z: ExpData[:,12]
    a2x: ExpData[:,13]
    a2y: ExpData[:,14]
    w2z: ExpData[:,15]
    
    theta1z: ExpData[:,16]
    theta2z: ExpData[:,17]
    
"""

# Control signal 1
def uref_1(t):
    
    """
    Lets comment this part and call 
    Lifting_Time_Start_1 = 0.5          # Start of lifting mass, m
    Lifting_Time_End_1 = 0.7           # End of lifting mass, m
    Lowering_Time_Start_1 = 0.8         # Start of lowering mass, m
    Lowering_Time_End_1 = 6.8           # End of lowering mass, m
    Lowering_Time_Start_2 = 7.0         # Start of lowering mass, m
    Lowering_Time_End_2 = 7.2            # End of lowering mass, m
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
    """

    u = ExpData[t,1]
    
   
    return u




# Control signal 2
def uref_2(t):
    
    """
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
        
    """
    u = ExpData[t,5]

    return u


# Control signal 1
def Pump(t):
    
    """
    Lets comment this part and call 
    Lifting_Time_Start_1 = 0.5          # Start of lifting mass, m
    Lifting_Time_End_1 = 0.7           # End of lifting mass, m
    Lowering_Time_Start_1 = 0.8         # Start of lowering mass, m
    Lowering_Time_End_1 = 6.8           # End of lowering mass, m
    Lowering_Time_Start_2 = 7.0         # Start of lowering mass, m
    Lowering_Time_End_2 = 7.2            # End of lowering mass, m
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
    """

    pP = ExpData[t,9]*1e5
    
   
    return pP






