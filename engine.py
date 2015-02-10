from math import pi

# pylint: disable-msg=E0611,F0401
from openmdao.main.api import Component
from openmdao.main.datatypes.api import Float, Array

from scipy.interpolate import interp1d

import numpy as np




class Engine(Component):

    throttle_pos = Array([],iotype='in',desc='throttle position')
    rpm = Array([], iotype='in', desc='rpm')

    torque=Array([], iotype='out',desc='torque')

    def __init__(self, h = 4, end_time = 764):
        super(Engine, self).__init__()
        
        self.t = np.arange(0, end_time+h, h)
        self.throttle_pos = np.zeros(len(self.t))
        self.rpm = np.zeros(len(self.t))
        self.torque = np.ones(len(self.t))

        
    
    def execute(self):

        self.torque= self.throttle_pos * (-(235./(3500.**2.)*(self.rpm-3500.))**2. + 235.)

if __name__ == '__main__':

    torque = Engine()
    torque.run()


        


        
