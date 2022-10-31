"""A set of classes to implement spectral models.

    $Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like/Models.py,v 1.151 2017/08/02 23:09:51 burnett Exp $

    author: Matthew Kerr, Joshua Lande
"""
import os
import copy
from collections import OrderedDict
import operator
import numpy as np
from abc import abstractmethod

from scipy.special import kv
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy import roots, optimize
from scipy.misc import derivative

from uw.utilities.parmap import LinearMapper, LogMapper, LimitMapper, ParameterMapper
from uw.utilities import path

class ModelException(Exception): pass

class Model(object):
    """ Spectral model giving dN/dE for a point source.  

        Default units are ph/cm^2/s/MeV.
    """
    
    def __init__(self, p=None, free=None, set_default_limits=False, set_default_oomp_limits=False, **kwargs):
        """ 

         Optional keyword arguments:

             =========    =======================================================
             Keyword      Description
             =========    =======================================================
             p              [p1,p2,...] default values of spectral parameters; see docstring individual model classes
             free           [True, True,...] a boolean list the same length as p giving the free (True) and fixed (False) parameters
             =========    =======================================================

         The input to this object should be fairly flexible:
             
             >>> model = PowerLaw(e0=100)
             >>> print model.e0
             100

         Note that pointlike internally stores parameters using a mapping:

             >>> model = Constant(scale=1, mappers=[LogMapper])
             >>> model['scale']
             1.0
             >>> model.getp('scale',internal=True)
             0.0
             >>> model.setp('scale',1,internal=True)
             >>> model.getp('scale',internal=True)
             1.0
             >>> model.getp('scale')
             10.0
             >>> model = Constant(scale=1, mappers=[LinearMapper])
             >>> model['scale']
             1.0
             >>> model.getp('scale',internal=True)
             1.0

	"""        
        if set_default_limits and set_default_oomp_limits:
            raise ModelException("Either set_default_limits or set_default_oomp_limits can be set")
        

        if hasattr(self,'default_oomp_limits'):
            assert set(self.default_oomp_limits).issubset(set(self.param_names))
        if hasattr(self,'gtlike'):
            assert set(self.gtlike['extra_param_names'].keys()).issubset(set(self.default_extra_params))

        self.background = False
        self.name = self.pretty_name = self.__class__.__name__

        self.npar = len(self.param_names)
        self.internal_cov_matrix = np.zeros([self.npar,self.npar]) #default covariance matrix

        if free is None:
            free = np.ones(self.npar)
        self.free = np.asarray(free, dtype=bool)

        if kwargs.has_key('mappers'):
            self.mappers = kwargs.pop('mappers')
        else:
            self.mappers = copy.deepcopy(self.default_mappers)

        try:
            for k,v in self.default_extra_params.items(): 
                setattr(self,k,v)
        except: pass

        for k,v in self.default_extra_attrs.items():
            setattr(self,k,v)

        self._p = np.zeros(self.npar, dtype=float)
        self._external = self._p.copy()
        if p is None:
            p = self.default_p
        self.set_all_parameters(p)

        # deal with other kwargs. If they are in the list of spectral
        # parameters, update the corresponding value. Tag them to 
        # the class.
        for k,v in kwargs.items():
            if k in self:
                # if k is a param name
                self[k]=v
            elif k in self.default_extra_attrs:
                setattr(self,k,v)
            else:
                raise ModelException("Unable to set parameter unknown parameter %s"  % k)

        if set_default_limits:
            self.set_default_limits()
        if set_default_oomp_limits:
            self.set_default_limits(oomp_limits=True)
    
    def len(self): 
        """ Exmaple:
                >>> model = PowerLaw()
                >>> print model.len()
                2

        """
        return self.npar

    def __contains__(self, item):
        """ Example:
                >>> model = PowerLaw()
                >>> print 'Norm' in model
                True
                >>> print 'Normalization' in model
                False

        """
        try:
            self.__getitem__(item)
            return True
        except ModelException, ex:
            return False
        
    def __getitem__(self, index):
        """ Wrapper around getp:
                
        >>> model=PowerLaw()
        >>> print model['index']
        2.0
        """
        return self.getp(index)
        
    def __setitem__(self, index, value):
        self.setp(index,value)
        
    def getp(self, i, internal=False):
        """ Get external value for parameter # i 

                >>> model=PowerLaw()
                >>> print model['index']
                2.0
        
            You can also get the non-fit extra_params using getp:
                
                >>> print model['e0']
                1000.0

            """
        if i in self.default_extra_params.keys():
            return getattr(self,i)
        i=self.name_mapper(i)
        return self._p[i] if internal else self.mappers[i].toexternal(self._p[i])
    
    def get_all_parameters(self, internal=False):
        """ Get a copy of the full set of parameters (external representation)"""
        if internal:
            return self._p.copy() 
        else:
            #return self._external #np.asarray(map(self.getp,np.arange(self.npar)))
            # old, equivalent, but much slower
            return np.asarray(map(self.getp,np.arange(self.npar)))
    
    def set_all_parameters(self, pars, internal=False):
        """ set all parameters (external representation)"""
        assert len(pars)==self.npar, 'called with %s, expected %d pars' % (pars, self.npar)
        pars  = np.asarray(pars, dtype=float)
        if internal:
            self._p = pars
            assert False, 'this should not be called'
        else:
            for i,p in enumerate(pars):
                self.setp(i,p)

    parameters = property(get_all_parameters, set_all_parameters, doc='array of parameters')

    def setp(self, i, par, internal=False):
        """ set internal value, convert unless internal. 
            
                >>> model=PowerLaw()
                >>> model['index'] = 1
                >>> print model['index'] 
                1.0

            You can also set the non-fit extra_params using setp:

                >>> model['e0'] = 100
                >>> print model.e0
                100
        
        """
        if i in self.default_extra_params:
            return setattr(self,i,par)
        i=self.name_mapper(i)
        self._p[i] = par if internal else self.mappers[i].tointernal(par)
        self._external[i] = self.mappers[i].toexternal(par) if internal else par
    
    @property
    def free_parameters(self):
        """ free parameters (external rep)"""
        return self._external[self.free]

    def setp_gtlike(self, i, par):
        """ Just like setp, but return the paramter
            with the gtlike convention. 

                >>> model=PowerLaw()
                >>> model.setp_gtlike('index',-2)
                >>> print model.getp_gtlike('index')
                -2.0
                >>> print model.getp('index')
                2.0
            """
        if i in self.default_extra_params:
            return self.setp(i,par)

        i=self.name_mapper(i)
        m = self.gtlike['topointlike'][i]
        return self.setp(i,m(par))

    def getp_gtlike(self, i):
        """ Just like getp, but return the paramter
            with the gtlike convention. 
            
                >>> model=PowerLaw()
                >>> model.setp_gtlike('index',-2.0)
                >>> print model.getp_gtlike('index')
                -2.0
                >>> print model.getp('index')
                2.0
            
        """
        if i in self.default_extra_params:
            return self.getp(i)

        i=self.name_mapper(i)
        m = self.gtlike['togtlike'][i]
        return m(self.getp(i))

    def get_mapper(self,i):
        """ Get the mapper for a particular parameter. """
        i=self.name_mapper(i)
        return self.mappers[i]

    def set_mapper(self,i,mapper):
        """ Allow changing the mapper for a given model parameter. 

            The default mapper is Linear for index of PowerLaw

                >>> model = PowerLaw(index=1)
                >>> model.set_error('index',0.5)
                >>> print model.getp('index')
                1.0
                >>> print model.getp('index',internal=True)
                1.0
                >>> print model.error('index')
                0.5
                >>> print model.get_mapper('index')
                <class 'uw.utilities.parmap.LinearMapper'>

            We can switch to a log mapper. The external
            values are all the same, but the internal
            representation changes.

                >>> model.set_mapper('index', LogMapper)
                >>> print model.getp('index')
                1.0
                >>> print model.getp('index',internal=True)
                0.0
                >>> print model.error('index')
                0.5
                >>> print model.get_mapper('index')
                <class 'uw.utilities.parmap.LogMapper'>

            Better typchecking:
                >>> model.set_mapper('index','bad_input')
                Traceback (most recent call last):
                    ...
                ModelException: mapper bad_input must be a subclass of uw.utilities.parmap.ParameterMapper
        """
        i=self.name_mapper(i)

        external_cov_matrix = self.get_cov_matrix()

        if not (isinstance(mapper, type) and issubclass(mapper, ParameterMapper)) and \
            not isinstance(mapper, ParameterMapper):
            raise ModelException("mapper %s must be a subclass of uw.utilities.parmap.ParameterMapper" % str(mapper))

        p=self.getp(i)
        perr=self.error(i)

        self.mappers[i] = mapper

        self.setp(i, p)

        self.set_external_cov_matrix(external_cov_matrix)

    def set_limits(self, i, lower, upper, scale=1, strict=False):
        """ Convenience function for setting limits

                >>> model = PowerLaw(index=1)
                >>> model.set_limits('index',-2,2)
                >>> print model.get_mapper('index')
                LimitMapper(-2,2,1)
                >>> print model.get_limits('index')
                [-2, 2]

            Note, just to be tolerant to gtlike limits where
            the paramete ris the negative of the pointlike limit,
            set_limits will automatically flip reversed limits:
            
                >>> model.set_limits('index',2,-2)
                >>> print model.get_limits('index')
                [-2, 2]

            Note, by default, setting a limit outside existing bound will just
            move the parameter inside the bound:
                >>> model.set_limits('index',2,4)
                WARNING: Found Index=1.0 < 2, minimum allowed value,
                    Setting parameter value to minimum.
                >>> model['index']
                2.0

            We can override this behavior with strict=True
                >>> print model.get_limits('index')
                [2, 4]
                >>> model.set_limits('index',-10, -8, strict=True)
                Traceback (most recent call last):
                    ...
                ModelException: Found Index=2.0 > -8, maximum allowed value
                >>> print model.get_limits('index')
                [2, 4]
        """
        i=self.name_mapper(i)
        name=self.param_names[i]
        param = self[i]

        if upper < lower:
            lower,upper = upper, lower

        if strict==False:
            self.set_mapper(i, LinearMapper)

        if param < lower:
            msg = 'Found %s=%s < %s, minimum allowed value' % (name, param, lower)
            if strict: raise ModelException(msg)
            print 'WARNING: %s,\n    Setting parameter value to minimum.' % msg
            self[i]=lower

        if self[i] > upper:
            msg = 'Found %s=%s > %s, maximum allowed value' % (name, param, upper)
            if strict: raise ModelException(msg)
            print 'Warning %s,\n    Setting parameter value to maximum.'% msg
            self[i] = upper

        self.set_mapper(i,LimitMapper(lower,upper,scale))

    def set_limits_gtlike(self, i, lower, upper, scale=-1, strict=False):
        """ like set_limits, but uses the gtlike convention.

                >>> model = PowerLaw(index=1)
                >>> model.set_limits_gtlike('index',-2,3)
                >>> print model.get_limits('index')
                [-3, 2]
                >>> print model.get_limits_gtlike('index')
                [-2, 3]
                >>> print model.get_scale('index')
                1
                >>> print model.get_scale_gtlike('index')
                -1
        """
        i=self.name_mapper(i)
        m = self.gtlike['topointlike'][i]
        lower, upper, scale = map(m,[lower, upper, scale])
        self.set_limits(i, lower=lower, upper=upper, scale=scale, strict=strict)

    def get_limits_gtlike(self, i):
        """ Get limits using a gtlike convention.
        
            Note, limits should always be [small, bigger].
            Test an edge case causing a bug for Francesco Giordano:

                >>> model = PowerLaw(index=2)
                >>> model.set_mapper('index',LimitMapper(-0.75,2.25,-1.0))
                >>> print model.get_limits('index')
                [-0.75, 2.25]
                >>> print model.get_limits_gtlike('index')
                [-2.25, 0.75]
        """
        i=self.name_mapper(i)
        m = self.gtlike['togtlike'][i]
        limits=self.get_limits(i)
        return sorted(map(m,limits))


    def get_limits(self, i):
        mapper = self.get_mapper(i)
        assert type(mapper) == LimitMapper
        return [mapper.lower, mapper.upper]

    def get_scale(self, i):
        """ 
            >>> model = PowerLaw(index=-1)
            >>> model.set_limits('index', -1, 1, scale=-1)
            >>> model.get_scale('index')
            -1
            >>> model.get_scale_gtlike('index')
            1
        """
        mapper = self.get_mapper(i)
        assert type(mapper) == LimitMapper
        return mapper.scale

    def get_scale_gtlike(self, i):
        """ Same as get_scale, but with gtlike convention. 
        """
        i=self.name_mapper(i)
        m = self.gtlike['togtlike'][i]
        return m(self.get_scale(i))
        
    def get_parameters(self):
        """ Return FREE internal parameters ; used for spectral fitting."""
        return self._p[self.free]

    def set_parameters(self,new_vals):
        """ Set FREE internal parameters; new_vals should have length equal to number of free parameters."""
        assert(len(new_vals)==(self.free).sum())
        self._p[self.free] = new_vals.astype(float) # downcast to float needed?
        self._external[self.free] = [self.mappers[i].toexternal(self._p[i]) for i in np.arange(self.npar)[self.free]]

    def tointernal(self, extpars):
        """ convert list of external parameters to internal"""
        return np.array([f.tointernal(x) for f,x in zip(self.mappers, extpars)])
    
    def name_mapper(self,i):
        """ This object takes in a parameter and maps it to an index.
            Currently, this function is not particularly useful, but it
            could be generalized to allow lazier parameter selection. 

            For example:

                >>> model = PowerLaw()
                >>> model.name_mapper('norm')
                0
                >>> model.name_mapper('index')
                1

            """
        if isinstance(i,str):
            if i.lower() not in np.char.lower(self.param_names):
                raise ModelException("Unknown parameter name %s for model %s" % (i,self.name))
            return np.where(np.char.lower(self.param_names)==i.lower())[0][0]
        else:
            return i

    def freeze(self,i,freeze=True):
        """ Freeze one of the spectral parameters from fitting.
        
                i: a parameter name or index.
                freeze    : if True, freeze parameter; if False, free it

            For example:

                >>> model = PowerLaw()
                >>> print model.free
                [ True  True]
                >>> model.freeze('index')
                >>> print model.free
                [ True False]

            """
        i=self.name_mapper(i)
        self.free[i] = not freeze

    def thaw(self,i): self.freeze(i,freeze=False)

    def set_free(self,i,free):
        """
                >>> model = PowerLaw()
                >>> print model.free
                [ True  True]
                >>> model.set_free('index',False)
                >>> print model.free
                [ True False]
                >>> print model.get_free('index')
                False
                >>> model.set_free('index',True)
                >>> print model.free
                [ True  True]
                >>> print model.get_free('index')
                True

            Previously, get_free would return a numpy object:
                >>> type(model.get_free('index')) == bool
                True
        """
        i=self.name_mapper(i)
        self.free[i] = free

    def get_free(self,i):
        i=self.name_mapper(i)
        return bool(self.free[i])

    def set_external_cov_matrix(self,external_cov_matrix):
        dintdext = 1/self.dexternaldinternal()
        dintdext_t = dintdext.reshape((len(dintdext),1))
        self.internal_cov_matrix = dintdext*external_cov_matrix*dintdext_t

    def set_cov_matrix(self,new_cov_matrix):
        """ Set the free submatrix of the internal format covariance matrix """
        t = np.ravel(new_cov_matrix)
        assert len(t)==self.free.sum()**2, \
            'wrong size for new cov matrix: found %d, expected %d' % (len(t), self.free.sum()**2)
        self.internal_cov_matrix[np.outer(self.free,self.free)] = t

    def dexternaldinternal(self):
        """ Return the derivatives of the external parameters with respect to the internal parameters. 
        (The Jacobian of the transformation, assumed diagonal only)
        """
        return np.asarray([m.dexternaldinternal(p) for m,p in zip(self.mappers,self.get_all_parameters())])

    def get_cov_matrix(self):
        """Return covariance matrix transformed out of log space."""
        p = self.dexternaldinternal()
        p_t=p.reshape((p.shape[0],1))
        return p*self.internal_cov_matrix*p_t

    def get_free_errors(self):
        """Return the diagonal elements of the (log space) covariance matrix for free parameters."""
        return np.diag(self.internal_cov_matrix)[self.free]**0.5

    def corr_coef(self):
        """Return the linear correlation coefficients for the estimated covariance matrix."""
        M, F = self.internal_cov_matrix, self.free
        S = (np.diag(M)**0.5)[F]
        return M[F].T[F] / np.outer(S,S)

    def has_errors(self):
        """ Return true if model has errors. """
        return not np.all(self.internal_cov_matrix==0)

    def statistical(self,absolute=False,two_sided=False):
        """ Return the parameter values and fractional statistical errors.
            If no error estimates are present, return 0 for the fractional error.

            two_sided : bool
                if set, return 3 tuples: values, +errors, -errors    
        """
        p = self.get_all_parameters()
        if not self.has_errors():
            # covariance matrix is invalid
            z = np.zeros_like(p)
            return (p,z,z) if two_sided else (p,z) 
        elif not two_sided:
            vars = np.diag(self.get_cov_matrix()).copy()
            vars[(vars<0)]=0
            p[p==0]=1e-6
            errs = vars**0.5 
            return p,errs/(1 if absolute else p)
        else:
            vars = np.diag(self.internal_cov_matrix).copy()
            vars[vars<0]=0
            errs = vars**0.5
            lo_rat = p - np.asarray([self.mappers[i].toexternal(self.getp(i,internal=True) - errs[i]) 
                                     for i in range(self.npar)])
            hi_rat = np.asarray([self.mappers[i].toexternal(self.getp(i,internal=True) + errs[i]) 
                                 for i in range(self.npar)]) - p
            return p,hi_rat/(1 if absolute else p),lo_rat/(1 if absolute else p)

    @abstractmethod
    def external_gradient(self,e):
        """ Return gradient with respect to external parameters. """
        pass

    def gradient(self,e):
        """ Return gradient with respect to internal parameters. """
        dextdint=self.dexternaldinternal()
        if hasattr(e, '__iter__'):
            dextdint=dextdint.reshape((len(dextdint)),1)
        return self.external_gradient(e)*dextdint

    def error(self,i):
        """ Get the EXTERNAL error for parameter i """
        i=self.name_mapper(i)
        variance = self.get_cov_matrix()[i,i]
        return np.sqrt(variance) if variance>0 else 0.

    def set_error(self,i,error):
        """ Set the INTERNAL error for parameter i 

                >>> model = PowerLaw(mappers=[LogMapper,LogMapper])
                >>> print model.error('norm')
                0.0
                >>> model.set_error('norm',1)
                >>> print model.error('norm')
                1.0

                >>> model = PowerLaw(mappers=[LinearMapper,LinearMapper])
                >>> model.set_error('norm',1)
                >>> print model.error('norm')
                1.0

        """
        i=self.name_mapper(i)

        # delete any covariance
        self.internal_cov_matrix[i,:]=self.internal_cov_matrix[:,i]=0
        dextdint=self.mappers[i].dexternaldinternal(self.getp(i))
        dintdext=1/dextdint
        self.internal_cov_matrix[i,i]=(error*dintdext)**2

    def __str__(self,absolute=False, indent=''):
        """ Return a pretty print version of parameter values and errors.
            indent: string to prepend to each line (must be called explicitly)

                >>> model = PowerLaw()
                >>> print model
                Norm      : 1e-11 
                Index     : 2 
                Ph. Flux  : 9.99e-08 (DERIVED)
                En. Flux  : 1.11e-10 (DERIVED)

        """
        try:
            p, hi_p, lo_p = self.statistical(absolute=absolute,two_sided=True)
        except:
            p, hi_p, lo_p = self.get_all_parameters(), [0]*self.npar, [0]*self.npar
        if not self.background and hi_p is not None:
            if not np.all(self.internal_cov_matrix==0):
                f,fhi,flo    = self.i_flux(e_weight=0,emax=3e5,two_sided=True,cgs=True,error=True)
                e,ehi,elo    = self.i_flux(e_weight=1,emax=3e5,two_sided=True,cgs=True,error=True)
                if not absolute:
                    fhi /= f; flo /= f; ehi /= e; elo /= e;
            else:
                f = self.i_flux(e_weight=0,cgs=True,error=False)
                e = self.i_flux(e_weight=1,cgs=True,error=False)
                fhi = flo = ehi = elo = 0
            p              = np.append(p, [f,e])
            hi_p          = np.append(hi_p, np.abs([fhi,ehi]))
            lo_p          = np.append(lo_p, np.abs([flo,elo]))
            pnames        = self.param_names + ['Ph. Flux','En. Flux']
        else: pnames = self.param_names

        l=[]
        if (not self.background and np.any(lo_p[0:-2]!=0)) or \
                 (self.background and np.any(lo_p!=0)): #if statistical errors are present    
            for i in xrange(len(pnames)):
                t_n = '%-10s' % pnames[i]
                if i < self.npar:
                    # if free is empty (shouldn't happen normally) treat as all False
                    frozen = '' if len(self.free)>i and self.free[i] else '(FROZEN)'
                else:
                    frozen = '(DERIVED)'
                if not absolute:
                    low, high = max(0, lo_p[i]), max(0,hi_p[i]) 
                    if low>1e2 or high >1e2: 
                        low=high=0
                        frozen= '(Failed fit)'
                    q = [high,low,(high*low)**0.5]
                    if np.any(np.isnan(q)): q = 3*[0]
                    l+=[t_n+': (1 + %.3f - %.3f) (avg = %.3f) %-10.3g %s'% (tuple(q) + (p[i],frozen))]
                else:
                    l+=[t_n+': %.3g + %.3g - %.3g (avg = %.3g) %s'%(p[i],hi_p[i],lo_p[i],(hi_p[i]*lo_p[i])**0.5,frozen)]
            return indent+ ('\n'+indent).join(l)
        else: #if no errors are present
            for i in xrange(len(pnames)):
                t_n = '%-10s' % pnames[i]
                if i < self.npar:
                    frozen = '' if self.free[i] else '(FROZEN)'
                else:
                    frozen = '(DERIVED)'
                l+=[t_n+': %.3g %s'%(p[i],frozen)]
            return indent+('\n'+indent).join(l)
            
    def __repr__(self): return self.__str__()

    def i_flux(self,emin=100,emax=1e5,e_weight=0,cgs=False,error=False,two_sided=False, quiet=False):
        """ Return the integral flux, \int_{emin}^{emax} dE E^{e_weight} dN/dE.
            e_weight = 0 gives the photon flux (ph cm^-2 s^-1)
            e_weight = 1 gives the energy flux (MeV cm^-2 s^-1) (see kwargs)

            Optional keyword arguments:

                =========    =======================================================
                Keyword      Description
                =========    =======================================================
                emin         [100] lower bound in MeV
                emax         [1e5] upper bound in MeV
                e_weight     [0] energy power by which to scale dN/dE
                cgs          [False] if True, energy in ergs, else in MeV
                error        [False] if True, return value is a tuple with flux and estimated error
                two_sided    [False] if True, return value is a triple with flux, estimated high and low errors
                quiet        [False] set to suppress error message
                =========    =======================================================
        """

        # check for a divergent flux
        # ADW: This is very bad for hard sources...
        #if 100*self(100) <= 1e5*self(1e5): emax = min(5e5,emax)

        try:
            func    = self if e_weight == 0 else lambda e: self(e)*e**e_weight
            units  = 1.60218e-6**(e_weight) if cgs else 1. #extra factor from integral!
            epsabs = min(func(emin),func(emax))*1e-10 # needed since epsrel does not seem to work
            flux    =  units*quad(func,emin,emax,epsabs=epsabs,full_output=True)[0]
            if error:
                # will silently ignore 'free' parameters without errors
                mask = (self.free) * (self.internal_cov_matrix.diagonal()>0)
                args = (emin,emax,e_weight,cgs,False)
                d    = self.__flux_derivs__(*args)[mask]
                dt  = d.reshape( (d.shape[0],1) ) #transpose
                try:
                    err = (d * self.internal_cov_matrix[mask].transpose()[mask] * dt).sum()**0.5
                except:
                    err=np.nan
                if not two_sided:
                    return (flux,err)
                else: #use log transform to estimate two-sided errors
                    log_err  = err/flux
                    log_flux = np.log(flux)
                    return (flux,np.exp(log_flux+log_err)-flux,flux-np.exp(log_flux-log_err))

            return flux
        except ModelException, msg:
            if not quiet:
                print 'Encountered a numerical error, "%s", when attempting to calculate integral flux.'%msg
            return np.nan if not error else ([flux, np.nan,np.nan] if two_sided else [flux, np.nan])

    def save_profile(self, filename, emin, emax, numpoints=200, clip_ends=True, clip_fraction=1e-20):
        """ Create a textfile of the spectral model. Useful for running
            gtobssim or gtlike with weird spectra. Also useful
            for creating a numeric representation of a spectral
            model using the FileFunction object. 

            if clip_ends is True, the file will clip spectral points a fraction clip_fraction less than
            the maximum at either end of the spectrum. This is to avoid having numerically small (or 0)
            values in the spectral file which can cause trouble in gtobssim, which does a powerlaw
            interpolation of the points.
        """
        energies=np.logspace(np.log10(emin),np.log10(emax),numpoints)
        fluxes=self(energies)

        if clip_ends:
            max_flux = max(fluxes)
            clip_flux = clip_fraction*max_flux
            # Clip out 0s on either end of spectral file. Otherwise gtobbsim gets angry.
            while fluxes[0] < clip_flux: energies,fluxes=energies[1:],fluxes[1:]
            while fluxes[-1] < clip_flux: energies,fluxes=energies[:-1],fluxes[:-1]

            # Paranoid check:
            if np.any(fluxes==0): raise ModelException("Error: 0s found in differential flux")

        open(path.expand(filename),'w').write('\n'.join(['%g\t%g' % (i,j) for i,j in zip(energies,fluxes)]))

    def set_flux(self,flux,strict=False,**kwargs):
        """ Set the flux of the source. 
                
            This function ensures that after the function call,
                flux == model.i_flux(**kwargs)
            where args and kwargs is consistently passed into i_flux and set_flux

            For example:

                >>> model = PowerLaw(index=2)
                >>> model.set_flux(1e-7, emin=1e3, emax=1e5)
                >>> print '%g' % model.i_flux(emin=1e3, emax=1e5)
                1e-07

            Note that this implementation is robust even when the source
            has an initial flux of 0:

                >>> model.setp(0, -np.inf, internal=True)
                >>> model.set_flux(1e-7)
                >>> print '%g' % model.i_flux()
                1e-07

            Just a simple example of a unitless powerlaw:

                >>> model = PowerLaw(norm=1, index=2, e0=1)
                >>> energies = np.logspace(0,6,100)
                >>> np.allclose(model(energies), energies**-2)
                True

            Previously, this flux computing formula was buggy 
            when the first parameter had limits, since
            setting the value to 1 would screw up the mapping.
            This is fixed in the new code.

                >>> model = PowerLaw(set_default_limits=True)
                >>> model.set_flux(1e-7)
                >>> np.allclose(model.i_flux(),1e-07)
                True
        
            Good error checking when set_flux goes outside bounds:

                >>> model.set_flux(1e-15)
                WARNING: Setting flux=1e-15 sets the prefactor=1.001001001e-19 which is below minimum bound 1e-17. Setting prefactor to limit.
                >>> print model['norm']
                1e-17
                >>> model.set_flux(1e-15, strict=True)
                Traceback (most recent call last):
                    ...
                ModelException: Setting flux=1e-15 sets the prefactor=1.001001001e-19 which is below minimum bound 1e-17

                >>> model.set_flux(1e10)
                WARNING: Setting flux=10000000000.0 sets the prefactor=1001001.001 which is above maximum bound 0.001. Setting prefactor to limit.
                >>> print model['norm']
                0.001

                >>> model.set_flux(1e10, strict=True)
                Traceback (most recent call last):
                    ...
                ModelException: Setting flux=10000000000.0 sets the prefactor=1001001.001 which is above maximum bound 0.001
        """
        mapper = self.get_mapper(0)
        init_p = self.getp(0)
        self.set_mapper(0,LinearMapper)

        self.setp(0, 1) # First, set prefactor to 1
        new_prefactor = flux/self.i_flux(**kwargs)

        if isinstance(mapper,LimitMapper):
            # deal with pesky case of flux setting prefactor outside limits.
            max_limit=mapper.upper
            if new_prefactor > max_limit:
                msg='Setting flux=%s sets the prefactor=%s which is above maximum bound %s' % (flux,new_prefactor,max_limit)
                if strict:
                    self.setp(0, init_p)
                    self.set_mapper(0, mapper)
                    raise ModelException(msg)
                else:
                    print 'WARNING: %s. Setting prefactor to limit.' % msg
                    new_prefactor=max_limit

            min_limit=mapper.lower
            if new_prefactor < min_limit:
                msg='Setting flux=%s sets the prefactor=%s which is below minimum bound %s' % (flux,new_prefactor,min_limit)
                if strict:
                    self.setp(0, init_p)
                    self.set_mapper(0, mapper)
                    raise ModelException(msg)
                else:
                    print 'WARNING: %s. Setting prefactor to limit.' % msg
                    new_prefactor=min_limit

        self.setp(0,new_prefactor)
        self.set_mapper(0, mapper)

    def copy(self): return copy.deepcopy(self)

    def fast_iflux(self,emin=100,emax=1e5):
        """Return a quick calculation for photon flux for models where it is analytically available."""
        return self.i_flux(emin=emin,emax=emax)

    def expected(self,emin,emax,exposure,skydir,event_class=-1,weighting_function=None):
        """ Calculate the expected counts under a particular model.
            Include an optional weighting_function to calculate, e.g., the average PSF
            parameters under this spectral model."""
        
        from pointlike import DoubleVector
        lemin,lemax = np.log10([emin,emax])
        simpsn = max(16,(int(round((lemax-lemin)/0.1)) >> 1) << 1) #10 per decade
        points = np.logspace(lemin,lemax,simpsn+1)
        simpsf = points*np.log(emax/emin)*np.asarray([1.] + ([4.,2.]*(simpsn/2))[:-1] + [1.])/(3.*simpsn)
        
        if event_class < 0:
            exp     = np.asarray(exposure[0].vector_value(skydir,DoubleVector(points))) +\
                        np.asarray(exposure[1].vector_value(skydir,DoubleVector(points)))
        else:
            exp     = np.asarray(exposure[event_class].vector_value(skydir,DoubleVector(points)))
        
        expec = (self(points)*exp*simpsf).sum()
        if weighting_function is not None:
            return (weighting_function(points)*self(points)*exp*simpsf).sum()/expec
        return expec
        
    def __flux_derivs__(self,*args):
        """ Use finite differences to estimate the gradient of the integral flux wrt the spectral parameters."""

        # note -- since spectral parameters are log transformed, just add/subtract a small amount in log space
        delta = 1e-5
        errs = np.asarray([delta] * len(self._p) )
        hi,lo = self.copy(),self.copy()
        derivs = []
        for i in xrange(len(self._p)):
            hi.setp(i,hi._p[i] + errs[i],internal=True)
            lo.setp(i,lo._p[i] - errs[i],internal=True)
            derivs  += [(hi.i_flux(*args) - lo.i_flux(*args))/(2*errs[i])]
            # reset parameters
            hi.setp(i,hi._p[i] - errs[i],internal=True)
            lo.setp(i,lo._p[i] + errs[i],internal=True)

        return np.asarray(derivs)
        
    def full_name(self):
        return self.pretty_name
        
    def flux_relunc(self, energy):
        """ 
        evaluate the relative flux uncertainty at the given energy 
            
        energy: float or array of floats
            
        returns float or array of floats, or np.nan if no covariance info
        """
        M, F = self.get_cov_matrix(), self.free
        if len(F)<2 or M[0,0]==0: return np.nan #no solution unless 2 or more free, fit info
            

        ##### replace this since matrix being deprecated
        # C = np.matrix(M[F].T[F])
        # def unc(energy):
        #     g = np.matrix(self.external_gradient(energy)[F])
        #     t = ( g * C * g.T ).item() 
        #     return np.sqrt(t)/self(energy) if t>0 else t
        C = (M[F].T[F])
        def unc(energy):
            g = (self.external_gradient(energy)[F])
            t = ( np.dot(g, np.dot( C , g.T ))).item() 
            return np.sqrt(t)/self(energy) if t>0 else t
        return np.array(map(unc,energy)) if hasattr(energy, '__iter__') else unc(energy)

    def pivot_energy(self, emax=1e5, exception=True, **kwargs):
        """ find the pivot energy by minimizing the flux uncertainty (knot of the bow tie)
            
            exception: bool
                if False, pass nan as output if invalid
                
            kwargs : optional parameters for fmin
            
            >>> m = LogParabola(p=[1e-11, 2.3, 1e-3, 1000.])
            >>> m.free[3]=False
            >>> c=0.1;ncm = np.array([[1, c, 0],[c, 0.5, 0], [0,0,0]])
            >>> m.set_cov_matrix(ncm)
            >>> round(m.pivot_energy(),1)
            1584.9
        """
        def f(e):
            return self.flux_relunc(e)
        if np.isnan(f(500.)):
            if exception:
                raise ModelException('Spectral fit with at least two free parameters required before calculating pivot energy')
            else: return np.nan
        try:
            fmin_default=dict(disp=0, ftol=0.01, xtol=0.05, maxiter=50)
            fmin_default.update(**kwargs)
            return min(optimize.fmin(f, [self.e0], **fmin_default )[0], emax)
        except FloatingPointError, msg:
            #raise ModelException('pivot_energy failed: %s' % msg)
            return 200. #default: problems with very soft, this is reasonable

    def zero(self):
        """ Set source's flux to 0:

                >>> model = PowerLaw()
                >>> print model.iszero()
                False
                >>> print model['norm']
                1e-11
                >>> model.zero()
                >>> print model.iszero()
                True
                >>> print model['norm']
                0.0
                >>> model.unzero()
                >>> print model['norm']
                1e-11
                >>> print model.iszero()
                False

            Previously, zeroing a source was kind of buggy if the 
            source parameter had a limit:

                >>> model = PowerLaw(norm=5, e0=100)
                >>> print model(100)
                5.0
                >>> model.set_limits('norm',1,10)
                >>> model.zero()
                >>> model.get_mapper('norm')
                <class 'uw.utilities.parmap.LinearMapper'>
                >>> print model(100)
                0.0
                >>> model.unzero()
                >>> print model(100)
                5.0
                >>> model.get_limits('norm')
                [1, 10]

        """
        if self.iszero(): return #already zeroed
        self.old_flux = self.getp(0) # get the internal value
        self.old_mapper = self.get_mapper(0)
        self.set_mapper(0, LinearMapper)
        self.setp(0, 0)
        self.old_free = self.free.copy()
        self.free[:] = False
    
    def iszero(self):
        return self.getp(0)==0

    def unzero(self):
        self.setp(0, self.old_flux)
        self.set_mapper(0, self.old_mapper)
        self.free = self.old_free.copy()

    def set_prefactor(self,  prefactor, energy):
        """ Set the prefactor of the source (at a given energy).
            Useful if you want to specify dN/dE for a source at a 
            particular energy. 

                >>> model = PowerLaw(e0=1000)
                >>> model.set_prefactor(1e-10, 100)
                >>> print model(100)
                1e-10
        """

        mapper = self.get_mapper(0)
        self.set_mapper(0,LinearMapper)

        self.setp(0,1) #set prefactor to 1
        self.setp(0, prefactor/self(energy))

        self.set_mapper(0, mapper)


    def set_default_limits(self, strict=False, oomp_limits=False, only_unbound_parameters=False):
        """ Apply default limits to parameters
            of spectral model. Only apply 
            limits to parametesr
            to all parameters.

            For example, by default a PowerLaw has no limits

                >>> model=PowerLaw()
                >>> print model.get_mapper('norm')
                <class 'uw.utilities.parmap.LogMapper'>
                >>> print model.get_mapper('index')
                <class 'uw.utilities.parmap.LinearMapper'>

            But we can easily impose default limits:

                >>> model.set_default_limits(oomp_limits=False)
                >>> np.allclose(model.get_limits('norm'),[1e-17,1e-3])
                True
                >>> print model.get_mapper('norm')
                LimitMapper(1e-17,0.001,1e-09)
                >>> print model.get_scale('norm')
                1e-09
                >>> print model.get_limits('index')
                [-5, 5]
                >>> print model.get_scale('index')
                1

            Note, the OOMP limits work a little differently:

                >>> model.set_default_limits(oomp_limits=True)
                >>> np.allclose(model.get_limits('norm'),[model['norm']/100,model['norm']*100])
                True

            Sometimes OOMP limits will fail and be replaced by normal default limits,

                >>> model.set_mapper('norm',LinearMapper)
                >>> model['norm']=0
                >>> model.set_default_limits(oomp_limits=True)
                WARNING: OOMP limit failed for parameter Norm,
                    Using default limits.
                WARNING: Found Norm=0.0 < 1e-17, minimum allowed value,
                    Setting parameter value to minimum.
                >>> print model.get_limits('norm')
                [1e-17, 0.001]



            We can make a model with limits easily:

                >>> model = PowerLaw(set_default_limits=True)
                >>> print model.get_mapper('norm')
                LimitMapper(1e-17,0.001,1e-09)

            Finally, we note the only_unbound_parameters, which will not change
            alredy existing limits:
            
                >>> model.set_limits('index',-10,10)
                >>> print model.get_limits('index')
                [-10, 10]
                >>> model.set_default_limits(only_unbound_parameters=True)
                >>> print model.get_limits('index')
                [-10, 10]
        """

        for name in self.param_names:

            current_mapper = self.get_mapper(name)
            if only_unbound_parameters and isinstance(current_mapper,LimitMapper):
                continue

            param = self[name]
            default_mapper = self.default_limits[name]
            lower = default_mapper.lower 
            upper = default_mapper.upper
            scale = default_mapper.scale

            if name in self.default_oomp_limits and oomp_limits:
                try:
                    self.set_oomp_limit(name)
                except ModelException, ex:
                    # This is kind of an edge case, but oomp limits can fail.
                    msg = 'OOMP limit failed for parameter %s' % name
                    if strict: raise ModelException(es)
                    print 'WARNING: %s,\n    Using default limits.' % msg
                    self.set_limits(name,lower,upper,scale=scale,strict=strict)
            else:
                self.set_limits(name,lower,upper,scale=scale,strict=strict)

    def set_oomp_limit(self, i):
        """ Set the scale for the parameter i to nicely (in log space) be allow
            roughly 2 orders of magnitiude on either side with a scale that
            is a power of 10:
                
                >>> model=PowerLaw(norm=1.5e-10)
                >>> print model.get_mapper('norm')
                <class 'uw.utilities.parmap.LogMapper'>
                >>> model.set_oomp_limit('norm')
                >>> print model.get_mapper('norm')
                LimitMapper(1e-12,1e-08,1e-10)
        """
        i=self.name_mapper(i)
        val = self[i]

        if val <= 0: 
            raise ModelException("Unable to set OOMP limits for parameters less than or equal to 0")

        scale = 10**round(np.log10(val))
        self.set_limits(i, 1e-2*scale, 1e2*scale,scale)

    @classmethod
    def get_pointlike_name(cls, name):
        """ Convert a gtlike name to a pointlike name. 

                >>> print PowerLaw.get_pointlike_name('Prefactor')
                Norm
                >>> print PowerLaw.get_pointlike_name('Scale')
                e0
                >>> print PowerLaw.get_pointlike_name('asdfadsf')
                Traceback (most recent call last):
                    ...
                ModelException: Name asdfadsf is not a gtlike parameter.
        """
        if name not in cls.gtlike_param_names():
            raise ModelException("Name %s is not a gtlike parameter." % name)

        pointlike_pn, gtlike_pn = cls.param_names, cls.gtlike['param_names']
        if name in gtlike_pn:
            return pointlike_pn[gtlike_pn.index(name)]

        epn=cls.gtlike['extra_param_names']
        if name in epn.values():
            return epn.keys()[epn.values().index(name)]

    @classmethod
    def gtlike_param_names(cls):
        """ Get all gtlike parameters:

                >>> print PowerLaw.gtlike_param_names()
                ['Prefactor', 'Index', 'Scale']
        """
        return cls.gtlike['param_names'] + cls.gtlike['extra_param_names'].values()

    @classmethod
    def from_gtlike(cls, **kwargs):
        """ This simple helper function creates a model object
            using parameters and variables with the gtlike
            convention. For example:

                >>> model = PowerLaw.from_gtlike(
                ... Index=-2.2556178468512353, 
                ... Prefactor=3.792956459007707e-13, 
                ... Scale=3162.2776601683795)
                >>> np.allclose(model['Norm'],3.792956459007707e-13)
                True
                >>> np.allclose(model['Index'],2.2556178468512353)
                True
                >>> np.allclose(model['e0'],3162.2776601683795)
                True

                >>> model.get_mapper('Norm')
                <class 'uw.utilities.parmap.LogMapper'>

            Youc an also pass in other Model parameters into this function:

                >>> model = PowerLaw.from_gtlike(Prefactor=1e-10, set_default_limits=True)
                >>> np.allclose(model['Norm'],1e-10)
                True
                >>> model.get_mapper('Norm')
                LimitMapper(1e-17,0.001,1e-09)

            Nicely, this code crashes on bad input:

                >>> model = PowerLaw.from_gtlike(bad_input=True)
                Traceback (most recent call last):
                    ...
                ModelException: Unable to set parameter unknown parameter bad_input
        """
        gtlike_params=dict()
        for k in kwargs.keys():
            if k in cls.gtlike_param_names():
                gtlike_params[k] = kwargs.pop(k)

        model=cls(**kwargs)
        for gn,v in gtlike_params.items():
            pn=cls.get_pointlike_name(gn)
            model.setp_gtlike(pn,v)
        return model

    def curvature(self, e=None):
        """return estimate of the curvature, using numerical derivatives
        This is exactly the parameter beta for a LogParabola 
        e: float
            energy in Mev, default e0
        """
        def dfun(x):
            def fun(x):
                """flux as function of x=ln(e0/e)"""
                e = self.e0 * np.exp(-x)
                return self(e)
            return derivative(fun,x, dx=0.01)/fun(x)
        x=0 if e is None else np.log(self.e0/e)
        
        return -0.5*derivative(lambda x: dfun(x), x, dx=0.01)

    @property
    def enorm(self):
        """return nu f_nu in eV at the reference energy e0"""
        return self(self.e0)*self.e0**2*1e6

    def set_enorm(self, enorm):
        """ set the Norm parameter to the nu f_nu (in eV) value"""
        factor = enorm / (self(self.e0)*self.e0**2*1e6)
        self.setp(0, self.getp(0)*factor)

        
class PowerLaw(Model):
    """ Implement a power law.  See constructor docstring for further keyword arguments.

        Spectral parameters:

            n0            differential flux at e0 MeV
            gamma        (absolute value of) spectral index

        It is easy to create a PowerLaw and access and set its values:

            >>> model=PowerLaw(p=[1e-7,2])
            >>> model['Norm'] == 1e-7 and model['norm'] == 1e-7
            True

        It is easy to set values internally
            
            >>> model['Norm'] = 1e-6
            >>> print model['Norm']
            1e-06

        When you create the object, you can specify the parameters by kwargs:

           >>> model=PowerLaw(norm=1e-11, index=1)
           >>> model['index'] == 1 and model['norm'] == 1e-11
           True

        You can also make deep copies of the spectral models:

            >>> copy_model=model.copy()
            >>> print copy_model['norm'] 
            1e-11
            >>> model['norm'] = 1e-10
            >>> print copy_model['norm']
            1e-11
            >>> print model['norm']
            1e-10

    """
    default_p=[1e-11, 2.0]
    default_extra_params=OrderedDict((('e0',1e3),))
    param_names=['Norm','Index']
    default_mappers=[LogMapper,LinearMapper]
    default_extra_attrs=dict()

    gtlike = dict(
        name='PowerLaw',
        param_names=['Prefactor','Index'],
        extra_param_names=dict(e0='Scale'),
        topointlike=[operator.pos,operator.neg],
        togtlike=[operator.pos,operator.neg])

    default_limits = dict(
        Norm=LimitMapper(1e-17,1e-3,1e-9),
        Index=LimitMapper(-5,5,1))
    default_oomp_limits=['Norm']

    def __call__(self,e):
        #n0,gamma=self['Norm'],self['Index']
        n0,gamma = self.get_all_parameters()
        return n0*(self.e0/e)**gamma

    def fast_iflux(self,emin=100,emax=1e6):
        n0,gamma=self['Norm'],self['Index']
        return n0/(1-gamma)*self.e0**gamma*(emax**(1-gamma)-emin**(1-gamma))

    def external_gradient(self,e):
        #n0,gamma=self['Norm'],self['Index']
        n0,gamma = self.get_all_parameters()
        f = n0*(self.e0/e)**gamma
        return np.asarray([f/n0,f*np.log(self.e0/e)])

    def pivot_energy(self, exception=True):
        """ assuming a fit was done, estimate the pivot energy 
              this is easy for this model, overrides fit in base class
        """
        A  = self['Norm']
        C = self.get_cov_matrix()
        if C[1,1]==0:
            if exception:
                raise ModelException('PowerLaw fit required before calculating pivot energy')
            else: return np.nan
        return self.e0*np.exp( C[0,1]/(A*C[1,1]) )
        
    def set_e0(self, e0p):
        """ Set a new reference energy, adjusting the norm parameter 
        
            >>> model1=PowerLaw(e0=1e2)
            >>> model2=model1.copy()
            >>> model2.set_e0(1e3)
            >>> np.allclose(model1(np.logspace(1,7,8)),model2(np.logspace(1,7,8)))
            True

        """
        gamma = self['Index']
        self['norm'] *= (e0p/self.e0)**-gamma
        self.e0 = e0p
        
    def full_name(self):
        return '%s, e0=%.0f'% (self.pretty_name,self.e0)
    @property
    def eflux(self):
        return self.e0**2*self['Norm']
        
class ScalingPowerLaw(PowerLaw):
    """ ScalingPowerLaw is just like a PowerLaw, but
        is suitable for scaling the Galactic diffuse. 

        The only differences are default parameters, limits, and oomp limits.

            >>> pl = PowerLaw(norm=1.12,index=-0.12)
            >>> spl = ScalingPowerLaw.from_powerlaw(pl)
            >>> print spl['norm']
            1.12
            >>> print spl['index']
            -0.12
            >>> spl.get_mapper('norm')
            <class 'uw.utilities.parmap.LogMapper'>
            >>> spl.get_mapper('index')
            <class 'uw.utilities.parmap.LinearMapper'>
            >>> spl.set_default_limits()
            >>> spl.get_limits('norm')
            [0.1, 10]
            >>> spl.get_limits('index')
            [-1, 1]
            >>> spl.background
            True
    """
    default_p=[1, 0]

    default_limits = dict(
        Norm=LimitMapper(.1,10),
        Index=LimitMapper(-1,1))
    default_oomp_limits=[]

    def __init__(self,*args,**kwargs):
        super(ScalingPowerLaw, self).__init__(*args, **kwargs)
        self.background = True


    @staticmethod
    def from_powerlaw(model):
        spl=ScalingPowerLaw(norm=model['norm'], index=model['index'])
        spl.set_mapper('norm',model.get_mapper('norm'))
        spl.set_mapper('index',model.get_mapper('index'))
        return spl

class PowerLawFlux(Model):
    """ Implement a power law.  See constructor docstring for further keyword arguments.

        Spectral parameters:

            flux         integrated flux from emin to emax MeV
            gamma        (absolute value of) spectral index

        PowerLawFlux allows the flux to be set:

            >>> model = PowerLawFlux(Int_Flux=15, emin=50, emax=1000)
            >>> np.allclose([model.i_flux(50, 1000),model.fast_iflux(50, 1000)], [15,15])
            True

    """
    default_p=[1e-7 , 2.0]
    default_extra_params=OrderedDict((('emin',1e2),('emax',1e6)))
    param_names=['Int_Flux','Index']
    default_mappers=[LogMapper,LinearMapper]
    default_extra_attrs=dict()

    gtlike = dict(
        name='PowerLaw2',
        param_names=['Integral','Index'],
        extra_param_names=dict(emin='LowerLimit', emax='UpperLimit'),
        topointlike=[operator.pos,operator.neg],
        togtlike=[operator.pos,operator.neg])

    default_limits = dict(
        Int_Flux=LimitMapper(1e-16,1e-4,1e-10),
        Index=LimitMapper(-5,5,1))
    default_oomp_limits=['Int_Flux']

    def __call__(self,e):
        flux,gamma=self.get_all_parameters()
        return (flux*(1-gamma)/(self.emax**(1-gamma)-self.emin**(1-gamma)))*e**(-gamma)

    def fast_iflux(self,emin=100,emax=np.inf):
        flux,gamma=self.get_all_parameters()
        return flux*(emax**(1-gamma) - emin**(1-gamma)) / (self.emax**(1-gamma) - self.emin**(1-gamma))

    def external_gradient(self,e):
        flux,gamma=self.get_all_parameters()
        e1 = self.emax; e0 = self.emin
        d1 = e1**(1-gamma)
        d0 = e0**(1-gamma)
        t  = (np.log(e0)*d0 - np.log(e1)*d1)/(d1 - d0)
        f  = (flux*(1-gamma)/(d1-d0))*e**(-gamma)
        return np.asarray([f/flux,-f*(np.log(e) + (1./(1-gamma) + t))])

    def full_name(self):
        return '%s, emin=%.0f emax=%.0f'% (self.pretty_name,self.emin,self.emax)

class BrokenPowerLaw(Model):
    """ Implement a broken power law.  See constructor docstring for further keyword arguments.

        Spectral parameters:

            n0            differential flux at e0 MeV
            gamma1      (absolute value of) spectral index for e < e_break
            gamma2      (absolute value of) spectral index for e > e_break
            e_break     break energy

    """
    default_p=[1e-11, 2.0, 2.0 ,1e3]
    default_extra_params=OrderedDict()
    param_names=['Norm','Index_1','Index_2', 'E_break']
    default_mappers=[LogMapper,LinearMapper,LinearMapper,LogMapper]
    default_extra_attrs=dict()

    gtlike = dict(
        name='BrokenPowerLaw',
        param_names=['Prefactor','Index1','Index2','BreakValue'],
        extra_param_names=dict(),
        topointlike=[operator.pos,operator.neg,operator.neg,operator.pos],
        togtlike=[operator.pos,operator.neg,operator.neg,operator.pos])

    default_limits = dict(
        Norm=LimitMapper(1e-16,1e-3,1e-9),
        Index_1=LimitMapper(-5,5,1),
        Index_2=LimitMapper(-5,5,1),
        E_break=LimitMapper(30,5e5,1))
    default_oomp_limits=['Norm','E_break']


    def __call__(self,e):
        n0,gamma1,gamma2,e_break=self.get_all_parameters()
        return n0*(e_break/e)**np.where(e<e_break,gamma1,gamma2)

    def external_gradient(self,e):
        n0,gamma1,gamma2,e_break=self.get_all_parameters()
        mask = e < e_break
        x = e_break/e
        lx = np.log(x)
        g = np.where(mask,gamma1,gamma2)
        f = n0*x**g
        return np.asarray([f/n0,f*lx*mask,f*lx*(~mask),f/e_break*g])

class BrokenPowerLawFlux(Model):
    """ Similar to PowerLawFlux for BrokenPowerLaw spectrum, the integral 
        flux is the free parameter rather than the Prefactor.

        Spectral parameters:

           flux      integrated flux from emin to emax MeV
           gamma1  (absolute value of) spectral index for e < e_break
           gamma2  (absolute value of) spectral index for e > e_break
           e_break break energy
    """
    default_p=[1e-7, 2.0, 2.0 , 1e3]
    default_extra_params=OrderedDict((('emin',1e2),('emax',1e6)))
    param_names=['Int_Flux','Index_1','Index_2', 'E_break']
    default_mappers=[LogMapper,LinearMapper,LinearMapper,LogMapper]
    default_extra_attrs=dict()

    gtlike = dict(
        name='BrokenPowerLaw2',
        param_names=['Integral','Index1','Index2','BreakValue'],
        extra_param_names=dict(emin='LowerLimit', emax='UpperLimit'),
        topointlike=[operator.pos,operator.neg,operator.neg,operator.pos],
        togtlike=[operator.pos,operator.neg,operator.neg,operator.pos])

    default_limits = dict(
        Int_Flux=LimitMapper(1e-13,1e-1,1e-7),
        Index_1=LimitMapper(-5,5,-1),
        Index_2=LimitMapper(-5,5,-1),
        E_break=LimitMapper(30,5e5,1))
    default_oomp_limits=['Int_Flux','E_break']

    def __call__(self,e):
        flux,gamma1,gamma2,e_break=self.get_all_parameters()
        if self.emax < e_break:
             norm=(1-gamma1)*e_break**(-gamma1)/(self.emax**(1-gamma1)-self.emin**(1-gamma1))
        elif self.emin > e_break:
             norm=(1-gamma2)*e_break**(-gamma2)/(self.emax**(1-gamma2)-self.emin**(1-gamma2))
        else:
             norm=1/(e_break**(gamma1)*(e_break**(1-gamma1)-self.emin**(1-gamma1))/(1-gamma1) + \
                        e_break**(gamma2)*(self.emax**(1-gamma2)-e_break**(1-gamma2))/(1-gamma1))

        return flux*norm*np.where( e < e_break, (e_break/e)**gamma1, (e_break/e)**gamma2 )

    def full_name(self):
        return '%s, emin=%.0f emax=%.0f'% (self.pretty_name,self.emin,self.emax)

class BrokenPowerLawCutoff(Model):
    """ Implement a broken power law.  See constructor docstring for further keyword arguments.

        Spectral parameters:
        
            n0            differential flux at e0 MeV
            gamma1      (absolute value of) spectral index for e < e_break
            gamma2      (absolute value of) spectral index for e > e_break
            e_break     break energy
    """
    default_p=[1e-11,2,2,1e3,3e3]
    default_extra_params=OrderedDict()
    param_names=['Norm','Index_1','Index_2','E_break','Cutoff']
    default_mappers=[LogMapper,LinearMapper,LinearMapper,LogMapper]
    default_extra_attrs=dict()

    def __call__(self,e):
        n0,gamma1,gamma2,e_break,cutoff=self.get_all_parameters()
        return n0*np.where( e < e_break, (e_break/e)**gamma1, (e_break/e)**gamma2 )*np.exp(-e/cutoff)

class SmoothBrokenPowerLaw(Model):
    """ Implement a smoothed broken power law. This is similar to a broken 
        powerlaw but uses the parameter beta to smoothly interpolate between 
        the two powerlaws.

        The implemenation is exactly the same as the gtlike implementation:
            http://glast.stanford.edu/cgi-bin/viewcvs/Likelihood/src/SmoothBrokenPowerLaw.cxx

        but with the indices the negative of the gtlike indices for internal consistency.
            

        Everything is defined exactly the same as gtlike except for the 
        usual replacement that gamma1 and gamma2 are defined to be the 
        negative of the gtlike definition. 

        Note that this function has a somewhat confusing feature that when
        beta>0, the function always becomes softer for energies greater then
        the break. For beta<0, the function always becomes harder for energies
        greater than the break. So you need to pick beta in advance depending
        upon whether you want your function to slope up or down.

        For now, I am not allowing beta to be fit. I am not sure anybody ever fits its.

        Also note that e0 is used in the spectral function and can be set (but 
        not fit) by passing in e0 when initialized. """

    default_p=[1e-11,2.0,2.0,1e3]
    default_extra_params=OrderedDict((('e0',1e3),('beta',0.1),))
    param_names=['Norm','Index_1','Index_2','E_break']
    default_mappers=[LogMapper,LinearMapper,LinearMapper,LogMapper]
    default_extra_attrs=dict()

    gtlike = dict(
        name='SmoothBrokenPowerLaw',
        param_names=['Prefactor','Index1','Index2','BreakValue'],
        extra_param_names=dict(e0='Scale', beta='Beta'),
        topointlike=[operator.pos,operator.neg,operator.neg,operator.pos],
        togtlike=[operator.pos,operator.neg,operator.neg,operator.pos])

    default_limits = dict(
        Norm=LimitMapper(1e-16,1e-3,1e-9),
        Index_1=LimitMapper(-5,5,1),
        Index_2=LimitMapper(-5,5,1),
        E_break=LimitMapper(30,5e5,1))
    default_oomp_limits=['Norm','E_break']

    def __call__(self,e):
        n0,gamma1,gamma2,e_break=self.get_all_parameters()

        # switch from pointlike to gtlike convention
        gamma1=-gamma1
        gamma2=-gamma2

        # This is exactly the gtlike formula
        return n0*(e/self.e0)**gamma1*\
               (1+(e/e_break)**((gamma1-gamma2)/self.beta))**(-self.beta)

    def external_gradient(self,e):
        """ lots of derivatives. You can derive them all by hitting the 
            derivative on log(dN/dE). """

        n0,gamma1,gamma2,e_break=self.get_all_parameters()

        f=self(e)
        df_n0=f/n0

        bottom=1.0+(e_break/e)**(-(gamma1-gamma2)/self.beta)
        df_gamma1=f*(np.log(self.e0/e)-np.log(e_break/e)/bottom)
        df_gamma2=f*np.log(e_break/e)/bottom
        df_break=-f*(gamma1-gamma2)/e_break/bottom

        # for completeness, here is the gradient with respect to beta.
        # For now, no compelling reason to fit it.
        # df_beta=f*(-np.log(1.0+(e_break/e)**((gamma1-gamma2)/self.beta))
        #            +((gamma1-gamma2)/self.beta)*np.log(e_break/e)/bottom)

        return np.asarray([df_n0,df_gamma1,df_gamma2,df_break])

    def full_name(self):
        return '%s, e0=%.0f, beta=%.3g'% (self.pretty_name,self.e0,self.beta)

class LogParabola(Model):
    """ Implement a log parabola (for blazars.)  See constructor docstring for further keyword arguments.

        Spectral parameters:
        
            n0            differential flux at e0 MeV
            alpha        (absolute value of) constant spectral index
            beta         co-efficient for energy-dependent spectral index
            e_break     break energy
    """
    default_p=[1e-11, 2.0, 1e-5,2e3]
    default_extra_params=OrderedDict()
    param_names=['Norm','Index','beta','E_break']
    default_mappers=[LogMapper,LinearMapper,LinearMapper,LogMapper]
    default_extra_attrs=dict()

    gtlike = dict(
        name='LogParabola',
        param_names=['norm','alpha','beta','Eb'],
        extra_param_names=dict(),
        topointlike=[operator.pos,operator.pos,operator.pos,operator.pos],
        togtlike=[operator.pos,operator.pos,operator.pos,operator.pos])

    default_limits = dict(
        Norm=LimitMapper(1e-17,1e-3,1e-9),
        Index=LimitMapper(-5,5,1),
        beta=LimitMapper(-0.2,3,1),
        E_break=LimitMapper(30,5e5,1))
    default_oomp_limits=['Norm','E_break']

    def __call__(self,e):
        n0,alpha,beta,e_break=self.get_all_parameters()
        x = np.log(e_break/e)
        y = (alpha - beta*x)*x
        return n0*np.exp(y)

    def external_gradient(self,e):
        n0,alpha,beta,e_break=self.get_all_parameters()
        x =np.log(e_break/e)
        y = (alpha - beta*x)*x
        f = n0*np.exp(y) # np.clip(y, -10, 100))
        return np.asarray([f/n0, f*x, -f*x**2, f*(alpha-2*beta*x)/e_break])

    # overridden in base class -- leave here for refererence or later check
    #def pivot_energy(self):
    #    """  
    #    Estimate the pivot energy
    #    Warning, there are potentially 3 real solutions to the pivot energy equation, and
    #    the code does not try to choose one : it returns all the real roots.
    #    """
    #    A  = 10**self._p[0]
    #    C = self.get_cov_matrix()
    #    if not self.free[2]:
    #        # the case beta fixed is equivalent to a PWL for the determination of the pivot energy.
    #        if C[1,1]==0:
    #            raise ModelException('Models.LogParabola: Fit required before calculating pivot energy')
    #        ebreak = 10**self._p[3]
    #        return ebreak*np.exp( C[0,1]/(A*C[1,1]) )
    #    #the solution in the general case comes from the resolution of a cubic equation.
    #    #We assume here that Norm, alpha, and beta are the free parameter of the fit.
    #    #Indeed E_break should never be set to free anyway
    #    a= 2.*C[2,2]
    #    b= 3.*C[1,2]
    #    c= C[1,1]-2.*C[0,2]/A
    #    d= -C[0,1]/A
    #    results = roots([a,b,c,d])
    #    results = 10**self._p[3]*np.exp(results)
    #    print "Pivot energy solutions for the LogParabola : ",results
    #    return np.real(results[np.isreal(results)])

    def set_e0(self, e0p):
        """ set a new break energy, adjusting the Norm and Index parameter,
            so that the differential flux remains the same. Beta remains unchanged in this
            transformation.

            >>> m = LogParabola(p=[1e-11, 2.0, 0.5, 2000])
            >>> m2 = m.copy()
            >>> m2.set_e0(1000)
            >>> abs(m.i_flux()-m2.i_flux())<1e-20
            True
        """
        n0, alpha, beta, e_break = self.get_all_parameters()
        x = np.log(e_break/e0p)
        self.setp(0, n0 * np.exp(alpha*x-beta*x*x))
        self.setp(1, alpha-2*beta*x)
        self.setp(3, e0p)
 
    def create_powerlaw(self, beta_max=3e-3):
        """ if beta is small and fixed, return an equivalent PowerLaw, otherwise just return self 

            >>> m = LogParabola(p=[1e-11, 2.5, 1e-3, 2000])
            >>> m.free[2]=False
            >>> m2 = m.create_powerlaw()
            >>> abs(m.i_flux()/m2.i_flux()-1)<1e-2
            True
        """
        if self[2]>beta_max or self.free[2]: return self
        nm = PowerLaw(p=[self[0], self[1]], e0=self[3], mappers=self.mappers[:-2])
        nm.internal_cov_matrix=self.internal_cov_matrix[:-2,:-2]
        return nm
    
    @property
    def e0(self): return self['e_break']
        
    @property
    def eflux(self):
        n0, alpha,beta, ebreak = self.get_all_parameters()
        return n0 * ebreak**2

class ExpCutoff(Model):
    """ Implement a power law with exponential cutoff.  See constructor docstring for further keyword arguments.

        Spectral parameters:
        
            Norm       differential flux at e0 MeV
            Index      (absolute value of) spectral index
            Cutoff      e-folding cutoff energy (MeV)
    """
    default_p=[1e-11, 2.0, 2e3]
    default_extra_params=OrderedDict((('e0',1e3),))
    param_names=['Norm','Index','Cutoff']
    default_mappers=[LogMapper,LinearMapper,LogMapper]
    default_extra_attrs=dict()

    default_limits = dict(
        Norm=LimitMapper(1e-15,1e-3,1e-9),
        Index=LimitMapper(0,5,1),
        Cutoff=LimitMapper(100,3e8,1000),
        )
    default_oomp_limits=['Norm','Cutoff']

    def __call__(self,e):
        n0,gamma,cutoff=self.get_all_parameters()
        return n0* (self.e0/e)**gamma * np.exp(-e/cutoff)

    def external_gradient(self,e):
        n0,gamma,cutoff=self.get_all_parameters()
        f = n0* (self.e0/e)**gamma * np.exp(-e/cutoff)
        return np.asarray([f/n0,f*np.log(self.e0/e),f*e/cutoff**2])


    #def pivot_energy(self):
    #    """ assuming a fit was done, estimate the pivot energy 
    #          
    #    """
    #    A  = self['Norm']
    #    C = self.get_cov_matrix()
    #    if C[1,1]==0:
    #        raise ModelException('%s fit required before calculating pivot energy' %self.name)
    #    return self.e0*np.exp( C[0,1]/(A*C[1,1]) )
    def flux_relunc(self, energy):
        """ Return relative uncertainty, ignoring contribution from exponential, if the parameter is free
        """
        f2 = self.free[2]
        self.free[2]=False
        r = super(ExpCutoff, self).flux_relunc(energy)
        self.free[2]=f2
        return r
        
    def set_e0(self, e0p):
        """ Set a new reference energy, adjusting the norm parameter 

                >>> model1 = ExpCutoff(e0=1e2)
                >>> model2 = model1.copy()
                >>> model2.set_e0(1e3)
                >>> model2.e0
                1000.0
                >>> np.allclose(model1(np.logspace(1,7,8)),model2(np.logspace(1,7,8)))
                True

        """
        gamma = self['Index']
        self['norm'] *= (e0p/self.e0)**-gamma
        self.e0 = e0p

        
    def full_name(self):
        return '%s, e0=%.0f'% (self.pretty_name,self.e0)
 
    @property
    def eflux(self):
        n0 = self['norm']
        return n0 * self.e0**2

    def create_super_cutoff(self, free_b=False):
        """ return an equivalent PLSuperExpCutoff model
        free_b : bool
            if True, unfreeze b

                >>> ec=ExpCutoff()
                >>> sec=ec.create_super_cutoff()
                >>> np.all(ec.get_all_parameters() == sec.get_all_parameters()[:-1])
                True
                >>> ec.mappers == sec.mappers[:-1]
                True
                >>> sec['b'] == 1.0
                True
                >>> ec.e0 == sec.e0
                True
        """
        sup = PLSuperExpCutoff(e0=self.e0)

        for p in ['Norm', 'Index', 'Cutoff']:
            sup[p] = self[p]
            sup.set_free(p,self.get_free(p))
            sup.set_mapper(p,self.get_mapper(p))
            sup.set_error(p,self.error(p))

        sup['b'] = 1
        sup.set_free('b', False)

        # Make a deep copy of everything to be safe
        return sup.copy()

class AllCutoff(Model):
    """ Implement an exponential cutoff.  This for the case when cutoff too low to constrain index.
        See constructor docstring for further keyword arguments.

        Spectral parameters:
        
            n0            differential flux at e0 MeV
            cutoff      e-folding cutoff energy (MeV)

    """
    default_p=[1e-11, 1e3]
    default_extra_params=OrderedDict()
    param_names=['Norm','Cutoff']
    default_mappers=[LogMapper,LogMapper]
    default_extra_attrs=dict()

    def __call__(self,e):
        n0,cutoff=self.get_all_parameters()
        if cutoff < 0: return 0
        return n0*np.exp(-e/cutoff)

class PLSuperExpCutoff(Model):
    """Implement a power law with hyper-exponential cutoff.  See constructor docstring for further keyword arguments.

        Spectral parameters:
        
            n0            differential flux at e0 MeV
            gamma        (absolute value of) spectral index
            cutoff      e-folding cutoff energy (MeV)
            b             additional power in the exponent
        """
    default_p=[1e-11, 2.0, 2e3 ,1.]
    default_extra_params=OrderedDict((('e0',1e3),))
    param_names=['Norm','Index','Cutoff', 'b']
    default_mappers=[LogMapper,LinearMapper,LogMapper,LinearMapper]
    default_extra_attrs=dict()

    gtlike = dict(
        name='PLSuperExpCutoff',
        param_names=['Prefactor','Index1','Cutoff','Index2'],
        extra_param_names=dict(e0='Scale'),
        topointlike=[operator.pos,operator.neg,operator.pos,operator.pos],
        togtlike=[operator.pos,operator.neg,operator.pos,operator.pos])

    default_limits = dict(
        Norm=LimitMapper(1e-15,1e-3,1e-9),
        Index=LimitMapper(0,5,1),
        Cutoff=LimitMapper(10,3e8,1000),
        b=LimitMapper(-5,5,1)
        )
    default_oomp_limits=['Norm','Cutoff']

    def __call__(self,e):
        n0,gamma,cutoff,b=self.get_all_parameters()
        return n0*(self.e0/e)**gamma*np.exp(-(e/cutoff)**b)


    def external_gradient(self,e):
        n0,gamma,cutoff,b=self.get_all_parameters()
        f = self(e)
        return np.asarray([f/n0,f*np.log(self.e0/e),
                           f*(b/cutoff)*(e/cutoff)**b,f*(e/cutoff)**b*np.log(cutoff/e)])


    #def pivot_energy(self):
    #    """  
    #    Assuming a fit was done, estimate the pivot energy. The parameter b is assumed to be fixed at 1.
    #    """
    #    if self._p[3]!=0. : print "WARNING: b is not 1, the pivot energy computation might be inaccurate"
    #    N0, gamma, Ecut, beta = self.get_all_parameters()
    #    E0 = self.e0
    #    C = self.get_cov_matrix()
    #    if C[1,1]==0:
    #        raise ModelException('%s fit required before calculating pivot energy' %self.name)
    #    if not self.free[2]:
    #        # The case b and Ecut fixed is equivalent to a PWL for the determination of the pivot energy.
    #        return E0*np.exp( C[0,1]/(N0*C[1,1]) )
    #    # We assume here that Norm, gamma, and Ecut are the free parameter of the fit.
    #    # To estimate Epivot we need to minimize the relative uncertainty on the flux, i.e. we want that the function "func"=0. 
    #    # x=log(Epivot/E0)
    #    a = C[0,1]/N0
    #    b = C[1,1]
    #    c = E0**2./Ecut**(4.)*C[2,2]
    #    d = E0/N0/Ecut**2.*C[0,2]
    #    f = E0/Ecut**2.*C[1,2]
    #    func = lambda x : a + b*x + c*np.exp(2.*x) + np.exp(x)*(d-f*(1+x))
    #    result = optimize.newton(func, 1)
    #    result = E0*np.exp(result)
    #    print "Pivot energy solution (in MeV) for the PLSuperExpCutoff : ",result
    #    return result
        
    def flux_relunc(self, energy):
        """ Return relative uncertainty, ignoring contribution from exponential, if the parameter is free
        """
        f2 = self.free[2]
        self.free[2]=False
        r = super(PLSuperExpCutoff, self).flux_relunc(energy)
        self.free[2]=f2
        return r

    def set_e0(self, e0p):
        """ set a new reference energy, adjusting the norm parameter:

        >>> model1=PLSuperExpCutoff(e0=1e2)
        >>> model2=model1.copy()
        >>> model2.set_e0(1e3)
        >>> model1['Index'] = model2['Index']
        >>> np.allclose(model1(np.logspace(1,7,8)),model2(np.logspace(1,7,8)))
        True
        """
        gamma = self['Index']
        self['norm'] *= (e0p/self.e0)**-gamma
        self.e0 = e0p

class PLSuperExpCutoff4(Model):
    """ Power-law with hyperexponential cutoff.
    
    Introduced to minimize covariance between parameters.
    
    f(e) = n_0 * (e/e_0)^(d/b-gamma) * exp[d/b^2*(1-(e/e_0)^b]
    
    Spectral parameters:
            n0          differential flux at e0 MeV
            gamma       spectral index: e^(-gamma)
            d           curvature parameter
            b           hyperexponential/asymmetry parameter
        """
    default_p = [1e-11, 2.0, 0.8, 1.]
    default_extra_params=OrderedDict((('e0',1.5e3),))
    param_names=['Norm','Index','Curvature', 'b']
    default_mappers=[LogMapper,LinearMapper,LogMapper,LinearMapper]
    default_extra_attrs=dict()

    gtlike = dict(
        name='PLSuperExpCutoff4',
        param_names=['Prefactor','IndexS','ExpfactorS','Index2'],
        extra_param_names=dict(e0='Scale'),
        topointlike=[operator.pos,operator.neg,operator.pos,operator.pos],
        togtlike=[operator.pos,operator.neg,operator.pos,operator.pos])

    default_limits = dict(
        Norm=LimitMapper(1e-15,1e-3,1e-9),
        Index=LimitMapper(0,5,1),
        Curvature=LimitMapper(1e-3,2,1),
        b=LimitMapper(-5,5,1)
        )
    default_oomp_limits=['Norm','Curvature']

    def __call__(self,e):
        """ Return the differential intensity.
        >>> m = PLSuperExpCutoff4([1e-11,2,0.8,1], e0=1e3)
        >>> expected = np.asarray([5.54576203e-09, 3.25605721e-10, 
        >>>     1.00000000e-11, 4.71063799e-16])
        >>> np.allclose(m(10**np.arange(1,5)),expected)
        >>> True
        """
        n0,gamma,d,b = self.get_all_parameters()
        x = e*(1./self.e0)
        return n0*x**(-gamma+d/b)*np.exp((d/b**2)*(1-x**b))

    def external_gradient(self,e):
        """
        >>> m = PLSuperExpCutoff4([1e-11,2,0.8,1], e0=1e3)
        >>> expected = np.asarray([1,0,0,0])
        >>> np.allclose(m.external_gradient(1000),expected)
        >>> True
        """
        n0,gamma,d,b = self.get_all_parameters()
        f = self(e)
        x = e*(1./self.e0)
        lx = np.log(x)
        xb = x**b
        return np.asarray([f/n0,-f*lx, f*(lx/b + (1-xb)/b**2),
                           f*(-d/b**2)*(lx*(1+xb) + 2./b*(1-xb))])

    def flux_relunc(self, energy):
        """ Return relative uncertainty, ignoring contribution from exponential, if the parameter is free
        """
        f2 = self.free[2:].copy()
        self.free[2:] = False
        r = super().flux_relunc(energy)
        self.free[2:] = f2
        return r

    def set_e0(self, e0p):
        """ set a new reference energy, adjusting params as needed.

        >>> model1=PLSuperExpCutoff4(e0=1e2)
        >>> model2=model1.copy()
        >>> model2.set_e0(1e3)
        >>> np.allclose(model1(np.logspace(1,7,8)),model2(np.logspace(1,7,8)))
        True
        """
        n0,gamma,d,b = self.get_all_parameters()
        a = self.e0/e0p
        self['Curvature'] = d/a**b
        self['Index'] = gamma-d/b*(1-1/a**b)
        norm_scale = a**(d/b-gamma)*np.exp((d/b**2)*(1/a**b-1))
        self['norm'] *= 1/norm_scale
        self.e0 = e0p

class MonoenergeticCurvature(Model):
    """ Implement a curvature spectrum for a monoenergetic electron.               Spectral parameters:
            n0            differential flux at ec
            ec            critical energy/frequency [MeV]
        """
    default_p=[1e-11, 1000]
    #default_extra_params=OrderedDict((('e0',1e3),))
    default_extra_params=dict()
    param_names=['Norm','Cutoff']
    default_mappers=[LogMapper,LogMapper]
    default_extra_attrs=dict()

    default_limits = dict(
        Norm=LimitMapper(1e-15,1e-3,1e-9),
        Cutoff=LimitMapper(10,3e8,1000),
        )
    default_oomp_limits=['Norm','Cutoff']

    def __call__(self,e):
        n0,cutoff=self.get_all_parameters()
        f = lambda x: kv(5./3,x)
        if hasattr(e,'__len__'):
            rvals = np.empty(len(e),dtype=float)
            for i in xrange(len(rvals)):
                rvals[i] = n0*quad(f,e[i]/cutoff,np.inf)[0]
        else:
            rvals = n0*quad(f,float(e)/cutoff,np.inf)[0]
        return rvals

    def external_gradient(self,e):
        raise NotImplementedError


class CompositeModel(Model):
    """ A model which joins other models. Subclasses must
        implement the __call__(e) function which says
        how to join the models. 
        
        CompositeModels should act just like regular models, implementing all
        of the parameters you expect for Models. 
        
        Here, we show how it works for a SumModel, the simplest CompositeModel.
        
        First, we create a SumModel

            >>> m1=PowerLaw(index=2);m1.set_flux(1)
            >>> m2=LogParabola(beta=2); m1.set_flux(1)
            >>> sum_model=SumModel(m1,m2)

        The sum_model has a param_names, mappers

            >>> sum_model.param_names == m1.param_names + m2.param_names
            True
            >>> sum_model.mappers == m1.mappers + m2.mappers
            True

        The CompositeModel.npar is the sum of parameters
            >>> sum_model.npar == m1.npar + m2.npar
            True

        And CompositeModel.npars is an array of the length of each parameter
            >>> np.all(sum_model.npars == [m1.npar,m2.npar])
            True

        And the parameters for the model is just the sum of the parameters for each model individually

            >>> np.all(sum_model.get_all_parameters() == np.append(m1.get_all_parameters(),m2.get_all_parameters()))
            True
            >>> np.all(sum_model._p == np.append(m1._p,m2._p))
            True

            >>> np.all(sum_model.free==np.append(m1.free,m2.free))
            True

        Also, getting and setting parameters individually acts just as one would expect

            >>> sum_model.mappers[1]
            <class 'uw.utilities.parmap.LinearMapper'>
            >>> sum_model[1] 
            2.0
            >>> sum_model.getp(1)
            2.0
            >>> print sum_model._p[1]
            2.0
            >>> sum_model[1] = -1
            >>> sum_model[1] 
            -1.0
            >>> sum_model.getp(1)
            -1.0
            >>> print sum_model._p[1]
            -1.0

            >>> sum_model.setp(1,1.5)
            >>> sum_model.getp(1)
            1.5

        Also, the free parameters should be totally transparent:

            >>> sum_model.free[1]
            True
            >>> sum_model.freeze(1)
            >>> sum_model.free[1]
            False

        So does getting and setting mappers:
            >>> m=sum_model.get_mapper(1)
            >>> print m
            <class 'uw.utilities.parmap.LinearMapper'>
            >>> sum_model.set_mapper(1,LogMapper)
            >>> print sum_model.get_mapper(1)
            <class 'uw.utilities.parmap.LogMapper'>
            >>> sum_model.set_mapper(1,m)
            >>> print sum_model.get_mapper(1)
            <class 'uw.utilities.parmap.LinearMapper'>

        And after the individual parameter modifications, the CompositeModel._p
        and CompositeModel.free arrays are updated consistently in the individual
        Model objects and in the Composite object:

            >>> np.all(sum_model._p == np.append(m1._p,m2._p))
            True

            >>> np.all(sum_model.free==np.append(m1.free,m2.free))
            True

        Note, there are duplicate names in this model:
            
            >>> print sum_model.duplicate_names()
            True
            >>> sum_model.default_limits
            Traceback (most recent call last):
                ...
            ModelException: default_limits not defined since there are duplicate parameter names in models.

            >>> sum_model.default_oomp_limits
            Traceback (most recent call last):
                ...
            ModelException: default_oomp_limits not defined since there are duplicate parameter names in models.

        default_p should be defined for CompositeModels:

            >>> np.all(sum_model.default_p == m1.default_p + m2.default_p)
            True

        And so should sfpl.default_mappers:
            >>> np.all(sum_model.default_mappers == m1.default_mappers + m2.default_mappers)
            True

        Previously there was a bug with having 3 spatial parameters: 
            >>> c = SumModel(PowerLaw(),PowerLaw(),PowerLaw())
            >>> print c._p
            [-11.   2. -11.   2. -11.   2.]
            >>> print c.free
            [ True  True  True  True  True  True]

        Note, previously this was buggy:

            >>> print c._p
            [-11.   2. -11.   2. -11.   2.]
            >>> c._p = [1]*6
            >>> np.all(c._p == [1]*6)
            True

            >>> c.mappers == [LogMapper, LinearMapper]*3
            True
            >>> c.mappers = [LogMapper]*6
            >>> c.mappers == [LogMapper]*6
            True
    """
    default_extra_params=OrderedDict() # Don't allow any of these
    default_extra_attrs=dict()

    def __init__(self, *models):
        if len(models) < 1:
            raise ModelException("CompositeModel must be created with more than one spectral model")
        for m in models:
            if not isinstance(m,Model):
                raise ModelException("CompositeModel must be created with a list of models.")

        self.name = self.__class__.__name__
        self.models = models
        self.internal_cov_matrix = np.zeros([self.npar,self.npar]) #default covariance matrix

    @abstractmethod
    def __call__(self,e): 
        """ Must be implemented by a subclass. """
        pass

    @property
    def param_names(self):
        return reduce(operator.add,[i.param_names for i in self.models])

    def _get_all_mappers(self):
        return reduce(operator.add,[i.mappers for i in self.models])

    def _set_all_mappers(self, value):
        assert(len(self.mappers) == len(value))
        counter=0
        for i,n in enumerate(self.npars):
            self.models[i].mappers = value[counter:counter+n]
            counter += n
    mappers = property(_get_all_mappers, _set_all_mappers, doc='array of mappers')

    @property
    def pretty_name(self):
        return self.operator_str.join([i.pretty_name for i in self.models])

    @property
    def background(self):
        """ Seems like a reasonable test. """
        return np.any([i.background for i in self.models])

    @background.setter
    def background(self,b):
        for m in self.models: m.background = b

    @property
    def npars(self): 
        return np.asarray([i.npar for i in self.models])

    @property
    def npar(self): 
        return sum([i.npar for i in self.models])

    @property
    def _p(self): 
        return np.concatenate([i._p for i in self.models])

    @_p.setter
    def _p(self, value):
        assert(len(self._p) == len(value))
        counter=0
        for i,n in enumerate(self.npars):
            self.models[i]._p = value[counter:counter+n]
            counter += n

    @property
    def free(self): 
        return np.concatenate([i.free for i in self.models])

    @free.setter
    def free(self, value):
        assert(len(self.free) == len(value))
        counter=0
        for i,n in enumerate(self.npars):
            self.models[i].free = np.array(value[counter:counter+n])
            counter += n

    def duplicate_names(self):
        return len(set(self.param_names)) != self.npar

    @property
    def default_p(self):
        return np.concatenate([i.default_p for i in self.models])

    @property
    def default_mappers(self):
        return np.concatenate([i.default_mappers for i in self.models])


    @property
    def default_limits(self):
        if self.duplicate_names(): 
            raise ModelException("default_limits not defined since there are duplicate parameter names in models.")

        return dict(reduce(operator.add,[i.default_limits.items() for i in self.models]))
    
    @property
    def default_oomp_limits(self):
        if self.duplicate_names(): 
            raise ModelException("default_oomp_limits not defined since there are duplicate parameter names in models.")
        return reduce(operator.add,[i.default_oomp_limits for i in self.models])

    @property
    def _external(self):
        return np.concatenate([i._external for i in self.models])
    
    def get_model_and_parameter_index(self,i):
        """ For a given parameter 'i', get the 
            model index (in the list of models)
            and parameter index (in the list of
            parameters for the selected model. """
        i=self.name_mapper(i)
        counter=0
        for k,n in enumerate(self.npars):
            if i < counter+n:
                break
            counter+=n
        model_index=k
        parameter_index=i-counter
        return model_index,parameter_index

    def set_parameters(self,new_vals):
        """ 
            >>> m1=PowerLaw()
            >>> m2=LogParabola()
            >>> sum_model=SumModel(m1,m2)
            >>> sum_model.set_parameters(np.asarray([1,2,3,4,5,6]))
            >>> print sum_model.get_parameters()
            [ 1.  2.  3.  4.  5.  6.]
            >>> m1.get_parameters()
            array([ 1.,  2.])
            >>> m2.get_parameters()
            array([ 3.,  4.,  5.,  6.])

        """
        assert np.all(self.free), 'This version of set_parameters does not handle non-free parameters'
        counter=0
        for n,model in zip(self.npars,self.models):
            model.set_parameters(new_vals[counter:counter+n])
            counter+=n

    def setp(self, i, *args, **kwargs):
        if i in self.default_extra_params.keys() + self.default_extra_attrs.keys():
            setattr(self, i, *args, **kwargs)
        else:
            model_index,parameter_index=self.get_model_and_parameter_index(i)
            self.models[model_index].setp(parameter_index,*args, **kwargs)

    def set_mapper(self,i,*args, **kwargs):
        model_index,parameter_index=self.get_model_and_parameter_index(i)
        self.models[model_index].set_mapper(parameter_index,*args, **kwargs)

    def freeze(self,i,*args,**kwargs):
        model_index,parameter_index=self.get_model_and_parameter_index(i)
        self.models[model_index].freeze(parameter_index,*args, **kwargs)


class FrontBackConstant(CompositeModel):
    """ Composite model that is either/or, for front or back
        select which constant based on value (0 or 1) of ct
    """
    name = 'FrontBackConstant'
    operator_str='+'

    gtlike = dict(
        # This is a bit ugly because there is no Gtlike
        # implementation for this spectral model. Here,
        # we just output with the pointlike naming
        name='FrontBackConstant',
        param_names=['Scale_front','Scale_back'],
        extra_param_names=dict(),
        topointlike=[operator.pos,operator.pos],
        togtlike=[operator.pos,operator.pos])

    default_limits = dict(
        Scale_front=LimitMapper(0,10,1),
        Scale_back=LimitMapper(0,10,1),
        )
    default_oomp_limits=[]

    def __init__(self, f=1, b=1, **kwargs):
        super(FrontBackConstant, self).__init__(
            Constant(name='Scale_front'),
            Constant(name='Scale_back'), 
            **kwargs)
        self.models[0][0]=f
        self.models[1][0]=b
        self.ct = 0
    
    def set_parameters(self,new_vals):
        """ Set FREE internal parameters; new_vals should have length equal to number of free parameters.
        Special version overriding CompositeModel, which does not allow non-free parameters
        """
        assert(len(new_vals)==(self.free).sum()), 'problem with FrontBackConstant: %s'%new_vals
        if len(new_vals)==1: # only setting one
            which = 0 if self.free[0] else 1
            self.models[which].set_parameters(new_vals)
            return
        self.models[0].set_parameters(new_vals[0:1])
        self.models[1].set_parameters(new_vals[1:2])
    
    def __call__(self, e):
        return self.models[self.ct](e)
        
    def external_gradient(self, e):
        return np.hstack([(1-self.ct)*self.models[0].external_gradient(e).T, 
                          self.ct*self.models[1].external_gradient(e).T])
                          
    def freeze(self, i, freeze=True):
        """override base class, since doesn't seem to be reliable? This is simpler
        """
        i = self.name_mapper(i)
        free = self.free
        free[i] = not freeze
        self.free = free

    def setp(self, i, value, **kwargs):
        """ i: [string | integer ]
            value: float
        """
        i = self.name_mapper(i)
        self.models[i].setp(0, value, **kwargs)

class SumModel(CompositeModel):
    """ Model is the sum of other models. 
    
        Easy to create: 
            >>> m1=PowerLaw(index=2);m1.set_flux(1)
            >>> m2=LogParabola(beta=2); m1.set_flux(1)

        The model is just the sum of the simpler modesl
            >>> sum_model=SumModel(m1,m2)
            >>> energy = np.asarray([1e2,1e3,1e4])
            >>> np.allclose(sum_model(energy), m1(energy) + m2(energy))
            True

    """
    operator_str = '+'
    name = 'SumModel'

    def __call__(self,e):
        return np.array([model(e) for model in self.models]).sum(axis=0)

    def external_gradient(self,e):
        """ Assume all models have a gradient! """
        return np.append([i.external_gradient(e) for i in self.models])

    def set_flux(self,flux,*args,**kwargs):
        """ Set the flux of the source. 

            Makes sure relative normalization of each component is unchanged
                
                >>> model = SumModel(PowerLaw(),PowerLaw())
                >>> np.allclose(model.i_flux(),PowerLaw().i_flux()*2)
                True
                >>> model.set_flux(1)
                >>> print model.i_flux()
                1.0

        """
        change = flux/self.i_flux(*args,**kwargs)
        for m in self.models: 
            m.setp(0, m.getp(0)*change)

class ProductModel(CompositeModel):
    """ Model is the product of other Models.

        Note: set_flux function just adjusts normalziation of 
              first parameter, which should be correct! 
              
        These things are Easy to create: 

            >>> m1=PowerLaw(index=2);m1.set_flux(1)
            >>> m2=LogParabola(beta=2); m1.set_flux(1)
            >>> prod_model=ProductModel(m1,m2)

        The model is just the product of the simpler modesl

            >>> energy = np.asarray([1e2,1e3,1e4])
            >>> np.allclose(prod_model(energy), m1(energy) * m2(energy))
            True

    """
    operator_str = '*'
    name = 'ProductModel'

    def __call__(self,e):
        return np.array([model(e) for model in self.models]).prod(axis=0)

    def external_gradient(self,e):
        """ Assume all models have a gradient! """
        return np.append([i.external_gradient(e)/i.__call__(e) for i in self.models])*self.__call__(e)

class Constant(Model):
    default_p=[1.]
    default_extra_params=OrderedDict()
    param_names=['Scale']  # note: this name can be overriden for combined 
    default_mappers=[LogMapper]
    default_extra_attrs=dict()

    gtlike = dict(
        name='ConstantValue',
        param_names=['Value'],
        extra_param_names=dict(),
        topointlike=[operator.pos],
        togtlike=[operator.pos])

    default_limits = dict(
        Scale=LimitMapper(1e-3,10,1))
    default_oomp_limits=[]

    def __init__(self, *args, **kwargs):
        """ When you create Constant objects, you can give
            the parameter a different name. This makes
            them adaptable to different contexts (i.e. 
            FrontBackConstant):
            
            First, create a default constant

                >>> model = Constant(Scale=3.0)
                >>> print model.param_names
                ['Scale']
                >>> print model['Scale']
                3.0
                >>> print model.default_limits.keys()
                ['Scale']

            Now, create a constant with the gtlike naming convention:

                >>> model = Constant(name='Value', Value=2.0)
                >>> print model.param_names
                ['Value']
                >>> print model['Value']
                2.0
                >>> print model.default_limits.keys()
                ['Value']

        """
        if 'name' in kwargs:
            name=kwargs.pop('name')
            self.param_names=[name]
            self.default_limits={name:self.default_limits['Scale']}
        super(Constant,self).__init__(*args, **kwargs)

    def __call__(self,e):
        return np.ones_like(e)*self[0]
    
    def fast_iflux(self,emin=100,emax=1e6):
        return (emax-emin)*self[0]

    def external_gradient(self,e):
        return  np.array([np.ones_like(e)])

class InterpConstants(Model):
    default_p=[1.]*5
    default_extra_params=OrderedDict((('e_breaks',np.log10([100,300,1000,3000,3e5])),))
    param_names=['Scale_Vector']
    default_mappers=[LogMapper,LogMapper,LogMapper,LogMapper,LogMapper]
    default_extra_attrs=dict()

    def __call__(self,e):
        interp = interp1d(self.e_breaks,self.get_all_parameters())
        return interp(np.log10(e))

    def set_flux(self,flux,**kwargs):
        raise NotImplementedError("No way to set flux for InterpConstants spectral model")

class FileFunction(Model):
    """ Defines a spectral model from an ascii file with the same
        convention as gtlike. Each point is connected with a power-law
        interpolation.
        
        For example, we could create a file which defines a PowerLaw,
        but very coarsely:

            >>> model = PowerLaw()

        Save out the model to a file
            
            >>> from tempfile import NamedTemporaryFile
            >>> temp = NamedTemporaryFile()
            >>> filename = temp.name
            >>> model.save_profile(filename, emin=1, emax=1e5)
            >>> temp.seek(0)

        Load the file into the FileFunction object

            >>> file_function = FileFunction(normalization=1, file=filename)

        And the power-law is restored, even between the saved points.

            >>> energies = np.logspace(0.122, 4.83, 42)
            >>> np.allclose(model(energies), file_function(energies))
            True

        Note, you cannot set the flux of a file function:

            >>> file_function.set_flux(1)
            >>> print file_function.i_flux()
            1.0

        Previously, pickling file_function objects failed:
            
            >>> import pickle
            >>> import os
            >>> pickle.dump(file_function, open(os.devnull,'w'))


        FileFunctions should be specifiable with the funny gtlike convention:

            >>> from uw.like.Models import FileFunction
            >>> f = "$(GLAST_EXT)/diffuseModels/v2r0p1/isotrop_2year_P76_source_v1.txt"
            >>> ff1=FileFunction(file=f)

        FileFunction.file should preserve the environment variable:

            >>> ff1.file == f
            True

        Similarly, you can read in files with any of the conventions:

            >>> ff2=FileFunction(file="${GLAST_EXT}/diffuseModels/v2r0p1/isotrop_2year_P76_source_v1.txt")
            >>> ff3=FileFunction(file="$GLAST_EXT/diffuseModels/v2r0p1/isotrop_2year_P76_source_v1.txt")
            >>>
            >>> np.all(ff1.flux == ff2.flux) and np.all(ff1.flux == ff3.flux)
            True
            >>> np.all(ff1.energy == ff2.energy) and np.all(ff1.energy == ff3.energy)
            True
    

    """
    default_p=[1]
    default_extra_params=OrderedDict()
    param_names=['Normalization']
    default_mappers=[LogMapper]
    default_extra_attrs=dict()
    default_extra_attrs=OrderedDict((('file',None),))

    gtlike = dict(
        name='FileFunction',
        param_names=['Normalization','Normalization'],
        extra_param_names=dict(),
        topointlike=[operator.pos],
        togtlike=[operator.pos])

    default_limits = dict(
        Normalization=LimitMapper(1e-4,1e4,1))
    default_oomp_limits=[]

    def __init__(self,**kwargs):

        super(FileFunction,self).__init__(**kwargs)

        if self.file is None:
            raise ModelException("FileFunction must be created with a file.")

        file=np.genfromtxt(path.expand(self.file),unpack=True)
        self.energy,self.flux=file[0],file[1]

        self.__make_interp__()

    def __make_interp__(self):
        self.interp = interp1d(np.log10(self.energy),np.log10(self.flux),
                bounds_error=False,fill_value=-np.inf)

    def __call__(self,e):
        return self['Normalization']*10**self.interp(np.log10(e))

    def external_gradient(self,e):
        return np.asarray([10**self.interp(np.log10(e))])

    def __getstate__(self):
        """ You cannot pickle an interp1d object. """
        d=copy.copy(self.__dict__)
        del d['interp']
        return d

    def __setstate__(self,state):
        """ recreate interp1d object. """
        self.__dict__ = state
        self.__make_interp__()

class SmoothDoubleBrokenPowerLaw(Model):
    """ Spectral Model Taken to be the same as:

            http://glast.stanford.edu/cgi-bin/viewcvs/Likelihood/src/SmoothDoubleBrokenPowerLaw.cxx

        but with the indices the negative of the gtlike indices for internal consistency.
            >>> pointlike_model = SmoothDoubleBrokenPowerLaw(Beta12=0.5, Beta23=0.25)
            >>> import pyLikelihood
            >>> _funcFactory = pyLikelihood.SourceFactory_funcFactory()
            >>> gtlike_model = _funcFactory.create('SmoothDoubleBrokenPowerLaw')
            >>> for n in pointlike_model.param_names:
            ...     gtlike_model.setParam(n,(-1 if 'index' in n.lower() else 1)*pointlike_model[n])
            >>> for n in ['Beta23', 'Beta12', 'Scale']:
            ...     gtlike_model.setParam(n,getattr(pointlike_model,n))
            >>> energies = np.logspace(1, 6, 10000)
            >>> from uw.darkmatter.spectral import DMFitFunction 
            >>> np.allclose(DMFitFunction.call_pylike_spectrum(gtlike_model, energies),
            ...     pointlike_model(energies), rtol=1e-20, atol=1e-20) 
            True

    """
    default_p=[ 1e-11, 1.0, 1e3, 2.0, 1e4, 3.0, ]
    default_extra_params=OrderedDict((('Beta12',0.05),('Beta23',0.05),('Scale',1e3),))
    param_names=[ 'Prefactor', 'Index1', 'BreakValue12', 'Index2', 'BreakValue23', 'Index3' ]
    default_mappers=[LogMapper,LinearMapper,LogMapper,LinearMapper,LogMapper,LinearMapper]
    default_extra_attrs=dict()

    def __call__(self,e):
        prefactor, index1, breakvalue12, index2, breakvalue23, index3 = self.get_all_parameters()

        # convert from pointlike to gtlike definition
        index1, index2, index3 = -index1, -index2, -index3

        # This is exactly the formula from gtlike
        return prefactor*(e/self.Scale)**index1*\
            (1 + (e/breakvalue12)**((index1 - index2)/self.Beta12))**-self.Beta12*\
            (1 + (e/breakvalue23)**((index2 - index3)/self.Beta23))**-self.Beta23

    def full_name(self):
        return '%s, scale=%.0f, beta12=%.3g, beta23=%.3g'% (self.pretty_name,self.Scale,self.Beta12,self.Beta23)


class Gaussian(Model):
    """ Following the defintion in:
            http://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html#Gaussian


        >>> from scipy.stats import norm
        >>> g = Gaussian(prefactor=1, mean=0, sigma=1)
        >>> e = np.logspace(1,10,11)
        >>> np.allclose(g(e),norm.pdf(e))
        True
    """
    default_p = [1e-9, 7e4, 1e3]
    default_extra_params = OrderedDict()
    param_names = [ "Prefactor", "Mean", "Sigma"]
    default_mappers = [LogMapper, LogMapper, LogMapper]
    default_extra_attrs=dict()

    gtlike = dict(
        name='Gaussian',
        param_names=param_names,
        extra_param_names=dict(),
        topointlike=[operator.pos]*3,
        togtlike=[operator.pos]*3)

    default_limits = dict(
        Prefactor=LimitMapper(1e-17,1e-3,1e-9),
        Mean=LimitMapper(1,1e6,1),
        Sigma=LimitMapper(1,1e6,1))
    default_oomp_limits=['Prefactor','Mean','Sigma']

    def __call__(self,e):
        prefactor, mean, sigma = self.get_all_parameters()
        return prefactor/(sigma*2.0*np.pi)*np.exp(-(e-mean)**2/(2.0*sigma**2))

if __name__ == "__main__":
    print __doc__
    import doctest
    doctest.testmod()
