from math import sin, cos, pi


def OscilatorySource(t,params):
    
    value = params["amplitude"]*sin(params["frequency"]*t + params["phase"])
    
    return value

def PulseSource(t,params):
    
    value = 0
    duration = params["duration"]
    if(duration[0]<=t<=duration[1]):
        value = params["amplitude"]
    
    return value

def LagPulseSource(t,params):
    
    value = 0
    duration = params["duration"]
    lag = params["lag"]
    if(duration[0]<=t<=duration[1]):
        value = params["amplitude"]
    elif(duration[1]<t<=duration[1]+lag):
        value = params["amplitude"]*(1-(t-duration[1])/(lag))
    
    return value
