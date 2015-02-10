from math import pi

# pylint: disable-msg=E0611,F0401
from openmdao.main.api import Component
from openmdao.main.datatypes.api import Float, Array

from scipy.interpolate import interp1d

import numpy as np

import rk4

from chassis_RK4 import Chassis





class Error(Component):


    current_speed = Array([], iotype='in', desc='Velocity of current run')
    target_speed = Array([], iotype='in', desc='target velocity')
     
    norm = Float(0.0, iotype='out', desc='norm')


    def execute(self):
        self.norm = np.linalg.norm([self.current_speed - self.target_speed])**2

    def list_deriv_vars(self): 
        return ('current_speed','target_speed'),( 'norm',)

    def provideJ(self): 
        dnorm_dcurrent_speed = 2 * (self.current_speed-self.target_speed)
        dnorm_dtarget_speed = 2 * (self.target_speed-self.current_speed)

        J = np.hstack((dnorm_dcurrent_speed, dnorm_dtarget_speed)).reshape((1,-1))

        return J


if __name__ == '__main__':

    norm = Error()

    norm.run()

    print norm.norm

    import matplotlib.pyplot as plt
    plt.plot(norm.chassis.t, norm.current_speed)
    plt.title('speed vs time')
    plt.xlabel('time (s)')
    plt.ylabel('speed (m/s)')



