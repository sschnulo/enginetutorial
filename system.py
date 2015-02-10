from math import pi
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
import time
from engine import Engine
from engineRPM import EngRPM


profile = np.genfromtxt('EPA-highway.csv',delimiter=',')
# t = 764


#profile = np.genfromtxt('EPA-city.csv',delimiter=',')
# t = 1875

t = np.hstack(([0],profile[:,0]))
speed = np.hstack(([0],profile[:,1]))
interp_speed = interp1d(t, speed)
new_time = np.arange(t[0],t[-1] ,4)
new_speed = 0.44704 * interp_speed(new_time)

class System(Assembly):

    target_speed = Array(new_speed, iotype='in')


    def configure(self):

        self.add('eng_rpm', EngRPM())
        self.add('chassis', Chassis())
        self.add('error', Error())
        self.add('engine', Engine())
        
        self.connect('target_speed', ['eng_rpm.target_speed','error.target_speed',])
        self.connect('eng_rpm.eng_RPM', 'engine.rpm')
        self.connect('engine.torque', 'chassis.engine_torque')
        self.connect('chassis.state[1]', 'error.current_speed')

        self.add('driver', SLSQPdriver())

        self.driver.add_objective('error.norm',)
        self.driver.add_parameter('engine.throttle_pos', -1.0, 1.0)

        #print self.chassis.acceleration


        # self.add('driver', NewtonSolver())
        #self.driver.add_constraint('error.error=0')
        
        self.driver.gradient_options.force_fd = True
        #self.driver.gradient_options.derivative_direction = "forward"
        self.driver.workflow.add(['eng_rpm','engine', 'chassis', 'error', ])
         

if __name__ == '__main__':

     trial = set_as_top(System()) 
     t0 = time.clock()
     trial.run()
     print time.clock() - t0, "seconds process time"
     print trial.error.norm


     import matplotlib.pyplot as plt
     plt.plot(trial.chassis.t, trial.error.target_speed,c='b')
     plt.plot(trial.chassis.t, trial.error.current_speed,c='r')
     plt.figure()
     plt.plot(trial.chassis.t, trial.engine.throttle_pos,c='b')
     plt.show()



