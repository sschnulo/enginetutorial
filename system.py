from math import pi

# pylint: disable-msg=E0611,F0401
from openmdao.main.api import Assembly
from openmdao.main.api import Component
from openmdao.main.datatypes.api import Float, Array
from openmdao.main.api import set_as_top
from scipy.interpolate import interp1d
import numpy as np
import rk4
from openmdao.lib.drivers.api import SLSQPdriver, NewtonSolver

from chassis_RK4 import Chassis
from error import Error
from engine import Engine 

class System(Assembly):

     def configure(self):
        self.add('chassis', Chassis())
        self.add('error', Error())
        self.add('engine', Engine())
        
        self.connect('chassis.state[1]', 'error.current_speed')
        self.connect('engine.fuel_burn', 'chassis.fuel_burn')
        self.connect('engine.torque', 'chassis.engine_torque')
        self.add('driver', SLSQPdriver())
        self.driver.add_objective('error.norm')
        self.driver.add_parameter('engine.RPM', -5000, 5000)
        # self.add('driver', NewtonSolver())
        #self.driver.add_constraint('error.error=0')
        
        self.driver.gradient_options.force_fd = True
        self.driver.workflow.add(['chassis', 'error'])
         

if __name__ == '__main__':

     trial = set_as_top(System()) 

     trial.run()

     print trial.error.norm


     import matplotlib.pyplot as plt
     plt.plot(trial.chassis.t, trial.error.target_speed,c='b')
     plt.plot(trial.chassis.t, trial.error.current_speed,c='r')
     plt.show()



