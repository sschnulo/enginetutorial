""" RK4 time integration component """

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Float, Array, Str

# Allow non-standard variable names for scientific calc
# pylint: disable-msg=C0103


class RK4(Component):
    """Inherit from this component to use.
    State variable dimension: (num_states, num_time_points)
    External input dimension: (input width, num_time_points)
    """

    h = Float(.01, units="s", iotype="in",
              desc="Time step used for RK4 integration")

    state_var = Str("", iotype="in",
                    desc="Name of the variable to be used for time "
                         "integration")

    init_state_var = Str("", iotype="in",
                         desc="Name of the variable to be used for initial "
                              "conditions")

    external_vars = Array([], iotype="in", dtype=str,
                          desc="List of names of variables that are external "
                               "to the system but DO vary with time.")

    fixed_external_vars = Array([], iotype="in", dtype=str,
                                desc="List of names of variables that are "
                                     "external to the system but DO NOT "
                                     "vary with time.")

    def initialize(self):
        """Set up dimensions and other data structures."""

        self.y = self.get(self.state_var)
        self.y0 = self.get(self.init_state_var)

        self.n_states, self.n = self.y.shape
        self.ny = self.n_states*self.n
        self.nJ = self.n_states*(self.n + self.n_states*(self.n-1))

        ext = []
        self.ext_index_map = {}
        for e in self.external_vars:
            var = self.get(e)
            self.ext_index_map[e] = len(ext)

            #TODO: Check that shape[-1]==self.n
            ext.extend(var.reshape(-1, self.n))


        for e in self.fixed_external_vars:
            var = self.get(e)
            self.ext_index_map[e] = len(ext)

            flat_var = var.flatten()
            #create n copies of the var
            ext.extend(np.tile(flat_var,(self.n, 1)).T)

        self.external = np.array(ext)

        #TODO
        #check that len(y0) = self.n_states

        self.n_external = len(ext)
        self.reverse_name_map = {
            self.state_var:'y',
            self.init_state_var:'y0'
        }
        e_vars = np.hstack((self.external_vars, self.fixed_external_vars))
        for i, var in enumerate(e_vars):
            self.reverse_name_map[var] = i

        self.name_map = dict([(v, k) for k, v in
                              self.reverse_name_map.iteritems()])


        #TODO
        #  check that all ext arrays of of shape (self.n, )

        #TODO
        #check that length of state var and external
        # vars are the same length

    def f_dot(self, external, state):
        """Time rate of change of state variables.
           external: array or external variables for a single time step
	    state: array of state variables for a single time step.
            This must be overridden in derived classes.
            """
        raise NotImplementedError

    def df_dy(self, external, state):
        """Derivatives of states with respect to states.
            external: array or external variables for a single time step
            state: array of state variables for a single time step.
            This must be overridden in derived classes.
            """

        raise NotImplementedError

    def df_dx(self, external, state):
        """derivatives of states with respect to external vars
            external: array or external variables for a single time step
            state: array of state variables for a single time step.
            This must be overridden in derived classes.
            """
        raise NotImplementedError

    def execute(self):
        """Solve for the states at all time integration points."""

        self.initialize()

        n_state = self.n_states
        n_time = self.n
        h = self.h

        # Copy initial state into state array for t=0
        self.y = self.y.reshape((self.ny, ))
        self.y[0:n_state] = self.y0

        # Cache f_dot for use in linearize()
        size = (n_state, self.n)
        self.a = np.zeros(size)
        self.b = np.zeros(size)
        self.c = np.zeros(size)
        self.d = np.zeros(size)

        for k in xrange(0, n_time-1):
            k1 = (k)*n_state
            k2 = (k+1)*n_state

            # Next state a function of current input
            ex = self.external[:, k] if self.external.shape[0] \
                                       else np.array([])

            # Next state a function of previous state
            y = self.y[k1:k2]

            self.a[:, k] = a = self.f_dot(ex, y)
            self.b[:, k] = b = self.f_dot(ex + h/2., y + h/2.*a)
            self.c[:, k] = c = self.f_dot(ex + h/2., y + h/2.*b)
            self.d[:, k] = d = self.f_dot(ex + h, y + h*c)

            self.y[n_state+k1:n_state+k2] = \
                y + h/6.*(a + 2*(b + c) + d)

        state_var_name = self.name_map['y']
        setattr(self, state_var_name,
                self.y.T.reshape((n_time, n_state)).T)

        #print "executed", self.name

    def provideJ(self):
        """Linearize about current point."""

        n_state = self.n_states
        n_time = self.n
        h = self.h
        I = np.eye(n_state)

        # Sparse Jacobian with respect to states
        #self.Ja = np.zeros((self.nJ, ))
        #self.Ji = np.zeros((self.nJ, ))
        #self.Jj = np.zeros((self.nJ, ))

        # Full Jacobian with respect to states
        self.Jy = np.zeros((self.n, self.n_states, self.n_states))

        # Full Jacobian with respect to inputs
        self.Jx = np.zeros((self.n, self.n_external, self.n_states))

        #self.Ja[:self.ny] = np.ones((self.ny, ))
        #self.Ji[:self.ny] = np.arange(self.ny)
        #self.Jj[:self.ny] = np.arange(self.ny)

        for k in xrange(0, n_time-1):

            k1 = k*n_state
            k2 = k1 + n_state

            ex = self.external[:, k] if self.external.shape[0] \
                                        else np.array([])
            y = self.y[k1:k2]

            a = self.a[:, k]
            b = self.b[:, k]
            c = self.c[:, k]

            # State vars
            df_dy = self.df_dy(ex, y)
            dg_dy = self.df_dy(ex + h/2., y + h/2.*a)
            dh_dy = self.df_dy(ex + h/2., y + h/2.*b)
            di_dy = self.df_dy(ex + h, y + h*c)

            da_dy = df_dy
            db_dy = dg_dy + dg_dy.dot(h/2.*da_dy)
            dc_dy = dh_dy + dh_dy.dot(h/2.*db_dy)
            dd_dy = di_dy + di_dy.dot(h*dc_dy)

            dR_dy = -I - self.h/6.*(da_dy + 2*(db_dy + dc_dy) + dd_dy)
            self.Jy[k, :, :] = dR_dy

            #for i in xrange(n_state):
                #for j in xrange(n_state):
                    #iJ = self.ny + i + n_state*(j + k1)
                    #self.Ja[iJ] = dR_dy[i, j]
                    ##self.Ji[iJ] = k2 + i
                    ##self.Jj[iJ] = k1 + j
                    #self.Ji[iJ] = i*n_time + k + 1
                    #self.Jj[iJ] = j*n_time + k

                    ##print self.Ji[iJ], self.Jj[iJ], self.Ja[iJ]

            # External vars (Inputs)
            df_dx = self.df_dx(ex, y)
            dg_dx = self.df_dx(ex + h/2., y + h/2.*a)
            dh_dx = self.df_dx(ex + h/2., y + h/2.*b)
            di_dx = self.df_dx(ex + h, y + h*c)

            da_dx = df_dx
            db_dx = dg_dx + dg_dy.dot(h/2*da_dx)
            dc_dx = dh_dx + dh_dy.dot(h/2*db_dx)
            dd_dx = di_dx + di_dy.dot(h*dc_dx)

            # Input-State Jacobian at each time point.
            # No Jacobian with respect to previous time points.
            self.Jx[k+1, :, :] = h/6*(da_dx + 2*(db_dx + dc_dx) + dd_dx).T

        #self.J = scipy.sparse.csc_matrix((self.Ja, (self.Ji, self.Jj)),
                                         #shape=(self.ny, self.ny))
        #self.JT = self.J.transpose()
        #self.Minv = scipy.sparse.linalg.splu(self.J).solve


    def apply_deriv(self, arg, result):
        """ Matrix-vector product with the Jacobian. """

        #result = self._applyJint(arg, result)
        result_ext = self._applyJext(arg)

        svar = self.state_var
        if svar in result:
            result[svar] += result_ext
        else:
            result[svar] = result_ext

    # TODO - Uncommment this when it is supported in OpenMDAO.
    #def applyMinv(self, arg, result):
        #"""Apply derivatives with respect to state variables."""

        #state = self.state_var

        #if self.state_var in arg:
            #flat_y = arg[state].flatten()
            #result[state] = self.Minv(flat_y).reshape((self.n_states, self.n))

        #return result


    #def _applyMinvT(self, arg, result):
        #"""Apply derivatives with respect to state variables."""

        #state = self.state_var
        #z = result.copy()
        #if self.state_var in arg:
            #flat_y = arg[state].flatten()
            #result[state] = self.Minv(flat_y, 'T').reshape((self.n_states, self.n))

        #return result


    def _applyJint(self, arg, result):
        """Apply derivatives with respect to state variables."""

        res1 = dict([(self.reverse_name_map[k], v)
                     for k, v in result.iteritems()])

        state = self.state_var
        if state in arg:
            flat_y = arg[state].reshape((self.n_states*self.n))
            result["y"] = self.J.dot(flat_y).reshape((self.n_states, self.n))

        res1 = dict([(self.name_map[k],v) for k, v in res1.iteritems()])
        return res1

    def _applyJext(self, arg):
        """Apply derivatives with respect to inputs"""

        #Jx --> (n_times, n_external, n_states)
        n_state = self.n_states
        n_time = self.n
        result = np.zeros((n_state, n_time))

        # Time-varying inputs
        for name in self.external_vars:

            if name not in arg:
                continue

            # take advantage of fact that arg is often pretty sparse
            if len(np.nonzero(arg[name])[0]) == 0:
                continue

            # Collapse incoming a*b*...*c*n down to (ab...c)*n
            var = self.get(name)
            shape = var.shape
            arg[name] = arg[name].reshape((np.prod(shape[:-1]),
                                           shape[-1]))

            i_ext = self.ext_index_map[name]
            ext_length = np.prod(arg[name][:, 0].shape)
            for j in xrange(n_time-1):
                Jsub = self.Jx[j+1, i_ext:i_ext+ext_length, :]
                J_arg = Jsub.T.dot(arg[name][:, j])
                result[:, j+1:n_time] += np.tile(J_arg, (n_time-j-1, 1)).T

        # Time-invariant inputs
        for name in self.fixed_external_vars:

            if name not in arg:
                continue

            # take advantage of fact that arg is often pretty sparse
            if len(np.nonzero(arg[name])[0]) == 0:
                continue

            ext_var = getattr(self, name)
            if len(ext_var) > 1:
                arg[name] = arg[name].flatten()
            i_ext = self.ext_index_map[name]
            ext_length = np.prod(ext_var.shape)
            for j in xrange(n_time-1):
                Jsub = self.Jx[j+1, i_ext:i_ext+ext_length, :]
                J_arg = Jsub.T.dot(arg[name])
                result[:, j+1:n_time] += np.tile(J_arg, (n_time-j-1, 1)).T

        # Initial State
        name = self.init_state_var
        if name in arg:

            # take advantage of fact that arg is often pretty sparse
            if len(np.nonzero(arg[name])[0]) > 0:
                fact = np.eye(self.n_states)
                result[:, 0] = arg[name]
                for j in xrange(1, n_time):
                    fact = fact.dot(-self.Jy[j-1, :, :])
                    result[:, j] += fact.dot(arg[name])

        return result

    def apply_derivT(self, arg, result):
        """ Matrix-vector product with the transpose of the Jacobian. """

        mode = 'Ken'

        if mode == 'Ken':

            r2 = self._applyJextT_limited(arg, result)

            for k, v in r2.iteritems():
                if k in result and result[k] is not None:
                    result[k] += v
                else:
                    result[k] = v

        elif mode == 'John':

            r2 = self._applyJextT(arg, result)
            r1 = self.applyJintT(arg, result)

            for k, v in r2.iteritems():
                if k in result and result[k] is not None:
                    result[k] += v
                else:
                    result[k] = v

            if self.state_var in r1:
                result[self.state_var] = r1[self.state_var]

            if self.init_state_var in r1:
                result[self.init_state_var] = r1[self.init_state_var]

        else:
            raise RuntimeError('Pick Ken or John')


    def applyJintT(self, arg, required_results):
        """Apply derivatives with respect to state variables."""

        result = {}
        state = self.state_var
        init_state = self.init_state_var

        if state in arg:
            if state in required_results:
                flat_y = arg[state].flatten()
                result[state] = -self.JT.dot(flat_y).reshape((self.n_states, self.n))

                if init_state in required_results:
                    result[init_state] = -result[state][:, 0]
                    for j in xrange(1, self.n):
                        result[init_state] -= result[state][:, j]

        #print self.J
        #print 'arg', arg, 'result', result
        return result

    def _applyJextT(self, arg, required_results):
        """Apply derivatives with respect to inputs. Ignore all contributions
        from past time points and let them come in via previous states
        instead."""

        #Jx --> (n_times, n_external, n_states)
        n_time = self.n
        result = {}

        if self.state_var in arg:

            argsv = arg[self.state_var].T

            # Use this when we incorporate state deriv
            # Time-varying inputs
            for name in self.external_vars:

                if name not in required_results:
                    continue

                ext_var = getattr(self, name)
                i_ext = self.ext_index_map[name]
                ext_length = np.prod(ext_var.shape)/n_time
                result[name] = np.zeros((ext_length, n_time))
                for k in xrange(n_time-1):

                    # argsum is often sparse, so check it first
                    if len(np.nonzero(argsv[k+1, :])[0]) > 0:
                        Jsub = self.Jx[k+1, i_ext:i_ext+ext_length, :]
                        result[name][:, k] += Jsub.dot(argsv[k+1, :])

            # Use this when we incorporate state deriv
            # Time-invariant inputs
            for name in self.fixed_external_vars:

                if name not in required_results:
                    continue

                ext_var = getattr(self, name)
                i_ext = self.ext_index_map[name]
                ext_length = np.prod(ext_var.shape)
                result[name] = np.zeros((ext_length))
                for k in xrange(n_time-1):

                    # argsum is often sparse, so check it first
                    if len(np.nonzero(argsv[k+1, :])[0]) > 0:
                        Jsub = self.Jx[k+1, i_ext:i_ext+ext_length, :]
                        result[name] += Jsub.dot(argsv[k+1, :])

        for k, v in result.iteritems():
            ext_var = getattr(self, k)
            result[k] = v.reshape(ext_var.shape)

        return result

    def _applyJextT_limited(self, arg, required_results):
        """Apply derivatives with respect to inputs"""

        # Jx --> (n_times, n_external, n_states)
        n_time = self.n
        result = {}

        if self.state_var in arg:

            argsv = arg[self.state_var].T
            argsum = np.zeros(argsv.shape)

            # Calculate these once, and use for every output
            for k in xrange(n_time - 1):
                argsum[k, :] = np.sum(argsv[k + 1:, :], 0)

            # argsum is often sparse, so save indices.
            nonzero_k = np.unique(argsum.nonzero()[0])

            # Time-varying inputs
            for name in self.external_vars:

                if name not in required_results:
                    continue

                ext_var = getattr(self, name)
                i_ext = self.ext_index_map[name]
                ext_length = np.prod(ext_var.shape) / n_time
                result[name] = np.zeros((ext_length, n_time))

                i_ext_end = i_ext + ext_length
                for k in nonzero_k:
                    Jsub = self.Jx[k + 1, i_ext:i_ext_end, :]
                    result[name][:, k] += Jsub.dot(argsum[k, :])

            # Time-invariant inputs
            for name in self.fixed_external_vars:

                if name not in required_results:
                    continue

                ext_var = getattr(self, name)
                i_ext = self.ext_index_map[name]
                ext_length = np.prod(ext_var.shape)
                result[name] = np.zeros((ext_length))

                i_ext_end = i_ext + ext_length
                for k in nonzero_k:
                    Jsub = self.Jx[k + 1, i_ext:i_ext_end, :]
                    result[name] += Jsub.dot(argsum[k, :])

            # Initial State
            name = self.init_state_var
            if name in required_results:
                fact = -self.Jy[0, :, :].T
                result[name] = argsv[0, :] + fact.dot(argsv[1, :])
                for k in xrange(1, n_time-1):
                    fact = fact.dot(-self.Jy[k, :, :].T)
                    result[name] += fact.dot(argsv[k+1, :])

        for k, v in result.iteritems():
            ext_var = getattr(self, k)
            result[k] = v.reshape(ext_var.shape)

        return result

    def _applyJextT_limited_old(self, arg, required_results):
        """Apply derivatives with respect to inputs"""

        # Jx --> (n_times, n_external, n_states)
        n_time = self.n
        result = {}

        if self.state_var in arg:

            argsv = arg[self.state_var].T
            argsum = np.zeros(argsv.shape)

            # Calculate these once, and use for every output
            for k in xrange(n_time - 1):
                argsum[k, :] = np.sum(argsv[k + 1:, :], 0)

            # argsum is often sparse, so save indices.
            nonzero_k = np.unique(argsum.nonzero()[0])

            # Time-varying inputs
            for name in self.external_vars:

                if name not in required_results:
                    continue

                ext_var = getattr(self, name)
                i_ext = self.ext_index_map[name]
                ext_length = np.prod(ext_var.shape) / n_time
                result[name] = np.zeros((ext_length, n_time))

                i_ext_end = i_ext + ext_length
                for k in nonzero_k:
                    Jsub = self.Jx[k + 1, i_ext:i_ext_end, :]
                    result[name][:, k] += Jsub.dot(argsum[k, :])

            # Time-invariant inputs
            for name in self.fixed_external_vars:

                if name not in required_results:
                    continue

                ext_var = getattr(self, name)
                i_ext = self.ext_index_map[name]
                ext_length = np.prod(ext_var.shape)
                result[name] = np.zeros((ext_length))

                i_ext_end = i_ext + ext_length
                for k in nonzero_k:
                    Jsub = self.Jx[k + 1, i_ext:i_ext_end, :]
                    result[name] += Jsub.dot(argsum[k, :])

            # Initial State
            name = self.init_state_var
            if name in required_results:
                result[name] = argsv[0, :] + argsum[0, :]

        for k, v in result.iteritems():
            ext_var = getattr(self, k)
            result[k] = v.reshape(ext_var.shape)

        return result