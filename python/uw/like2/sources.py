"""
Source classes
$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/sources.py,v 1.53 2018/01/27 15:37:17 burnett Exp $

"""
import os, copy
import numpy as np
from skymaps import SkyDir
from uw.like import  Models
from . import response

# convenience adapters 
def LogParabola(*pars):
    model = Models.LogParabola(p=pars, free=[True,True,False,False])
    return model
def PowerLaw(*pars):   
    model = Models.PowerLaw(p=pars)
    return model
def ExpCutoff(*pars):  
    model = Models.ExpCutoff(p=pars, free=[True, True, False])
    return model
def PLSuperExpCutoff(*pars):  
    model = Models.PLSuperExpCutoff(p=pars, free=[True,True,False,False])
    return model
def Constant(*pars, **kw):   return Models.Constant(p=pars, **kw)
def FBconstant(f,b, **kw): return Models.FrontBackConstant(f,b, **kw)
def PSR_default(): return Models.PLSuperExpCutoff(p=(1e-13, 1.25, 1500, 0.67), free=[True,True, True, False])
    
def ismodel(model):
    """ check that model is an instance of Models.Model"""
    return isinstance(model, Models.Model)

def set_default_bounds( model, force=False):
    """
    Handy utility to set bounds for a model from like.Models
    force=True to override previously set bounds.
    """
    if not force and hasattr(model, 'bounds'):
        # model has bounds. Were they set? check to see if all are None
        notset =  np.all(np.array([np.all(b ==[None,None]) for b in model.bounds]))
        if not notset: return
    bounds=[]
    def to_internal(fun, values):
        return [fun(value) if value is not None else None for value in values]
    for pname, mp in zip(model.param_names, model.mappers):
        plim = (None,None)
        try:
            plim = dict(
                Index=(0.0, 3.5), 
                Norm=(10**-18, 10**-7),
                Scale=(0.001, 4.0),
                beta=(-0.1, 1.), 
                Cutoff=(100., 2e5),
                )[pname.split('_')[0]]
        except: pass
        bounds.append( to_internal(mp.tointernal, plim) )
    model.bounds = np.array(bounds) # convert to array so can mask with free

class Source(object):
    """ base class for all pointlike/like2 sources
    Subclasses are:
        PointSource
        ExtendedSource
        GlobalSource
        
    All instances have the folloiwng properties:
    * model, a Models.Model object
    * skydir : [skymaps.Skydir  | None]
        
    Subclasses must implement a function response(band), which, given a EnergyBand parameter, 
        returns a Response object appropriate for the source. This provides the angular dependence 
        of the response specific the band energy and event type.
    
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.changed = False # flag for bandlike
        assert self.name is not None, 'bad source name'
        self.name = str(self.name) # force to be a string
        if self.skydir is None:
            # global source: keep original model
            self.free = np.array(self.model.free).copy() if self.model is not None else None  # save copy of initial free array to restore
            return
        elif hasattr(self.skydir, '__iter__'): #allow a tuple of (ra,dec)
            self.skydir = SkyDir(*self.skydir)
        if 'model' not in kwargs or self.model is None:
            self.model=LogParabola(1e-14, 2.2, 0, 1e3)
            self.model.free[2:]=False
        elif type(self.model)==str:
            try:
                t =eval(self.model)
            except Exception, exp:
                print 'Failed to evaluate model expression, %s: %s' %(self.model, exp)
                raise
            self.model=t
                
        if self.model.name=='PowerLaw':
            # convert from PowerLaw to LogParabola
            par,sig = self.model.statistical()
            free = self.model.free[:]
            self.model = LogParabola(*(list(par)+[0, self.model.e0]))
            self.model.free[:2] = free
            self.model.free[2:] = False
 
        elif self.model.name=='ExpCutoff':
            try:
                print 'converting %s to PLSuperExpCutoff' %self.name
                self.model = self.model.create_super_cutoff()
            except FloatingPointError:
                print 'Failed'
                
        elif self.model.name=='PowerLawFlux':
            f, gamma = self.model.get_all_parameters() #10**self.model.p
            emin = self.model.emin
            try:
                self.model=LogParabola(f*(gamma-1)/emin, gamma, 0, emin)
            except Exception, msg:
                print 'Failed to create LogParabola for source %s, pars= %s'% (self.name, (f,gamma,emin))
                raise
            self.model.free[2:]=False
        elif self.model.name=='LogParabola':
            #what was this for?
            #if hasattr(self, 'free') and len(self.free)>3: self.free[3]=False
            if sum(self.model.free)==4:
                # do not allow all parameters to be free: freeze E_break if so
                self.model.free[-1]=False
            elif sum(self.model.free)==2 and not self.model.free[1]:
                # undo freezing
                print'Unfreezing E_break for source %s' % self.name
                self.model.free[-1]=True
        if self.model.name not in ['LogParabola','PLSuperExpCutoff','ExpCutoff', 'Constant','PLSuperExpCutoff4']:
            raise Exception('model %s not supported' % self.model.name)
        #self.free = self.model.free.copy()

        if not hasattr(self.model, 'npar'):
            raise Exception('model %s for source %s was not converted to new format'\
                    % (self.model.name, self.name))
        # finally, add bounds to the models object, ignoring similar capability in Models.
        set_default_bounds( self.model )
           
            
    def get_spectral_model(self):
        return self.model
    def set_spectral_model(self, newmodel):
        t =self.model
        self.model = newmodel
        return t
    spectral_model = property(get_spectral_model, set_spectral_model)

    def freeze(self, parname, value=None):
        self.model.freeze(parname)
        if value is not None: self.model.setp(parname, value)
        self.changed=True
        #assert sum(self.model.free)>0, 'cannot freeze all parameters this way'

    def thaw(self, parname):
        self.model.freeze(parname, freeze=False)
        self.changed = True

    def __str__(self):
        return '\tname  : %s\n\tskydir: %s\n\tmodel : %s\n\t\t%s' %\
    (self.name, self.skydir, self.model.name, self.model.__str__(indent='\t\t'))
    def __repr__(self):
        return '%s.%s: \n%s' % (self.__module__,self.__class__.__name__ , self.__str__())
        
    @property
    def isextended(self):
        return hasattr(self, 'dmodel') and not self.isglobal

    @property
    def isglobal(self):
        return self.skydir is None

class PointSource(Source):
    def __init__(self, **kwargs):
        kwargs.update(spatial_model=None) # allow test for extent (no extent!)
        super(PointSource, self).__init__(**kwargs)
    def near(self, otherdir, distance=10):
        return self.skydir.difference(otherdir) < np.radians(distance)
    def copy(self):
        """ return a new PointSource object, with a copy of the model, others"""
        ret = PointSource(**self.__dict__)
        ret.model = self.model.copy()
        return ret
    def response(self, band, roi=None, **kwargs):
        return response.PointResponse(self, band, roi, **kwargs)


class ExtendedSource(Source):

    #def __str__(self):
    #    return self.name + ' '+ self.model.name \
    #            +  (' (free)' if np.any(self.model.free) else ' (fixed)') 
    def __str__(self):
        return '\tname  : %s\n\tskydir: %s\n\tSpatial : %s\n\tmodel : %s\n\t\t%s' %\
    (self.name, self.skydir, self.dmodel.name, self.model.name, self.model.__str__(indent='\t\t'))
 
  
    def near(self, otherdir, distance=10):
        return self.skydir.difference(otherdir) < np.radians(distance)
        
    def copy(self):
        """ return a new ExtendSource object, with a copy of the model object"""
        ret = ExtendedSource(**self.__dict__)
        ret.model = self.model.copy()
        if ret.model.name=='LogParabola':
            ret.model.free[-1]=False # make sure Ebreak is frozen
        return ret
         
    def response(self, band, roi=None, **kwargs):
        """ return a Respose object, which, given a band, can create a convolved image
        and calculate expected counts
        """
        return response.ExtendedResponse(self, band, roi, **kwargs)

        
class GlobalSource(Source):
    def __init__(self, **kwargs):
        super(GlobalSource, self).__init__(**kwargs)
        self.dmodel= kwargs.get('dmodel', None)
        assert self.skydir is None # used as a flag
        # Special option "free_diffuse" from config['input_model'] to free spectral model for diffuse sources
        free = kwargs.pop('free', None) # expect a list of names
        if free is not None and self.name in free:
            self.model.free[0]=True
            if self.model.name=='PowerLaw': self.model.free[1]=True,
            #print '{}, free={}'.format(self, free)

    def copy(self):
        """ return a new GlobalSource object, with a copy of the model, others"""
        ret = GlobalSource(**self.__dict__)
        ret.model = self.model.copy()
        return ret

    def response(self, band, roi=None, **kwargs):
        """ return a Response class for the band"""
        assert self.dmodel, 'Need DiffuseBase object to determine response'
        try:
            resp_class =  dict(
                Isotropic =response.IsotropicResponse,
                MapCube   =response.DiffuseResponse,
                CachedMapCube=response.CachedDiffuseResponse,
                Healpix   =response.DiffuseResponse,
                HealpixCube = response.DiffuseResponse,
                FitsMapCube = response.DiffuseResponse,
                FitsMapCubeList = response.DiffuseResponse,
                IsotropicSpectralFunction = response.IsotropicResponse,
                AziLimb = response.IsotropicResponse,
                GulliLimb = response.IsotropicResponse,
                )[self.dmodel.type]
        except Exception, msg:
            raise Exception('Could not find a response class for source %s:"%s"' %(self,msg))
        try:
            return resp_class(self,band,roi, **kwargs) 
        except Exception: # assume no overlap
            raise
            return response.NoResponse(self, band, roi, **kwargs)
    

