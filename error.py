from math import pi

# pylint: disable-msg=E0611,F0401
from openmdao.main.api import Component
from openmdao.main.datatypes.api import Float, Array

from scipy.interpolate import interp1d

import numpy as np

import rk4

from chassis_RK4 import Chassis



profile = np.genfromtxt('EPA-highway.csv',delimiter=',')
# t = 764


#profile = np.genfromtxt('EPA-city.csv',delimiter=',')
# t = 1875

time = np.hstack(([0],profile[:,0]))
speed = np.hstack(([0],profile[:,1]))
interp_speed = interp1d(time, speed)
new_time = np.arange(time[0],time[-1] ,4)
new_speed = 0.44704 * interp_speed(new_time)

class Error(Component):


    current_speed = Array([], iotype='in', desc='Velocity of current run')
    target_speed = Array(new_speed, iotype='in', desc='target velocity')
     
    norm = Float(0.0, iotype='out', desc='norm')
   


    # def __init__(self, target_speed=None):
    #     super(Error, self).__init__()
    #     if target_speed is not None: 
    #         self.target_speed = target_speed
    #         #self.error = np.zeros(len(target_speed))


    def execute(self):
        
        
        #self.norm = np.linalg.norm([self.current_speed - self.target_speed])
        
        self.norm = np.linalg.norm([self.current_speed - self.target_speed])**2

    def list_deriv_vars(self): 
        return ('current_speed','target_speed'),( 'norm',)

    def provideJ(self): 
        dnorm_dcurrent_speed = 2 * (self.current_speed-self.target_speed)
        dnorm_dtarget_speed = 2 * (self.target_speed-self.current_speed)

        J = array([[dnorm_dcurrent_speed, dnorm_dtarget_speed]])

        return J
    # def apply_deriv(self, arg, result): 
        
    #     if 'current_speed' in arg: 
    #         result['norm']  = 2 * arg['current_speed-target_speed']
    #     if 'target_speed' in arg: 
    #         result['norm'] = 2 * arg['target_speed-current_speed']


    # def apply_derivT(self, arg, result): 
        
    #     if 'current_speed' in result:
    #         result['current_speed'] = arg['(norm/2) + target_speed'] 
    #         pass
    #     if 'target_speed' in result: 
    #         result['target_speed'] = arg['(norm/2) + current_speed']
    #         pass

if __name__ == '__main__':

    norm = Error()

    norm.run()

    print norm.norm

    import matplotlib.pyplot as plt
    plt.plot(norm.chassis.t, norm.current_speed)
    plt.title('speed vs time')
    plt.xlabel('time (s)')
    plt.ylabel('speed (m/s)')



