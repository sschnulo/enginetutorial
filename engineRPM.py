from math import pi
from openmdao.main.api import Component
from openmdao.main.datatypes.api import Float, Array
from scipy.interpolate import interp1d

import numpy as np


class EngRPM(Component):
    gear_ratio = Array([], iotype='in')
    target_speed = Array([], iotype='in', desc='target velocity')
    eng_RPM = Array([], iotype='out', desc='Velocity of current run')

    def __init__(self, h = 4, end_time = 764):
        super(EngRPM, self).__init__()
        self.t = np.arange(0, end_time+h, h)
        self.gear_ratio = np.ones(len(self.t))
        self.eng_RPM = np.ones(len(self.t))
    
    def execute(self):

        self.eng_RPM = self.target_speed * 100

    def list_deriv_vars(self): 
        return ('target_speed',),( 'eng_RPM',)

    def provideJ(self): 
        d_eng_RPM_d_target_speed = 100

        J = (d_eng_RPM_d_target_speed,)

if __name__ == '__main__':

    RPM= EngRPM()

    RPM.run()
    print RPM.eng_RPM