"""
classes presenting views of the likelihood engine in the module bandlike

Each has a mixin to allow the with ... as ... construction, which should restore the BandLikeList


$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/views.py,v 1.22 2017/11/17 22:50:36 burnett Exp $
Author: T.Burnett <tburnett@uw.edu> (based on pioneering work by M. Kerr)
"""

import sys, types
import numpy as np
from scipy import misc, optimize
from skymaps import SkyDir
from . import (roimodel, bandlike, tools, parameterset)

class FitterSummaryMixin(object):
    """mixin to summarize variables"""
    
    def summary(self, select=None, exclude=None, out=None, title=None, gradient=True):
        """ summary table of free parameters, values uncertainties gradient
        
        Parameters:
        ----------
        select : list of integers or string
            integers are indices of parameters
            string is the wildcarded name of a source
        out : open file or None
        title: None or string
        gradient: bool
            set False to not print gradient
            
        """
        if title is not None:
            print >>out, title

        fmt, tup = '%-21s%6s%10s%10s', tuple('Name index value error(%)'.split())
        if gradient:
            grad = self.gradient()
            fmt +='%10s'; tup += ('gradient',)
        print >>out, fmt %tup
        prev=''
        selected = (select, exclude)
        index_array = np.arange(len(self.mask))[self.mask]
        for index, (name, value, rsig) in enumerate(zip(self.parameter_names, 
                                                        self.model_parameters, 
                                                        self.uncertainties)):
            t = name.split('_')
            pname = t[-1]
            sname = '_'.join(t[:-1])
            if sname==prev: name = len(sname)*' '+'_'+pname
            prev = sname
            fmt = '%-21s%6d%10.4g%10s'
            psig = '%.1f'%(rsig*100) if rsig>0 and not np.isnan(rsig) else '***'
            truncname = name[:20]+'*' if len(name)>20 else name
            tup = (truncname, index_array[index], value,psig)
            if gradient:
                fmt +='%10.1f'; tup += (grad[index],)
            print >>out,  fmt % tup
    
    def delta_loglike(self, quiet=True):
        """ estimate change in log likelihood from current gradient 
        """
        try:
            # Old code assuming now-deprecated numpy.matrix
            # gm = np.matrix(self.gradient())
            # H = self.hessian()
            # return (gm * H.I * gm.T)[0,0]/4
            gv = self.gradient()
            H = np.array(self.hessian()) # in case matrix type
            return np.dot(np.dot(gv, np.linalg.inv(H)), gv)/4
        except Exception, msg:
            print 'Failed log likelihood estimate, returning 99.: %s' % msg
            return 99.


class FitPlotMixin(object):
    """mixin  for likelihood function to generate a plot, or set of all plots"""
    
    def estimate_solution(self):
        """ return a tuple with:
            current parameters, 
            estimated parmeters at maximum
            sigmas
        """
        parz = self.get_parameters()
        hess = self.hessian(parz)
        cov = hess.I if len(parz)>1 else 1./hess
        sigs = np.sqrt(np.asarray(cov.diagonal()).flatten())
        parmax = parz-self.gradient(parz)*sigs**2/2.
        return parz, parmax, sigs

    def plot(self, index, ax=None, nolabels=False , y2lim=(-5,5), estimate=None):
        """make a plot showing the log likelihood and its derivative as a function of
        expected sigma, evaluated from the second derivative at the current point
        
        index : int
            index of the parameter
        """
        import matplotlib.pyplot as plt
        # get current parameters, gradient, and the Hessian for estimate of max liklihood position
        parz, parmax, sigs = self.estimate_solution() if estimate is None else estimate
        pz = parz[index]
        part = parz.copy()
        def func(x):
            part[index]=x
            return -self(part) #(restore sign for minimization)
        def gradf(x):
            part[index]=x
            return self.gradient(part)[index]
        x0, sig = (parmax)[index], sigs[index]
        ref = func(x0)
        if ax is None:
            fig, ax = plt.subplots( figsize=(3,3))
        else: fig = ax.figure
        if not np.isnan(sig):
            xsig = np.linspace(-3, 3, 27)
            x =  x0 + xsig * sig 
            ax.plot(xsig, map(func,x)-ref, '-b')
            ax.plot(xsig, -0.5*((x-x0)/sig)**2, '--b')
            ax.plot((pz-x0)/sig, func(pz)-ref, 'dk')
        plt.setp(ax, ylim=(-5,0.5), xlim=(-4,4))
        if not nolabels: ax.set_ylabel('log likelihood')
        j = np.arange(len(self.mask))[self.mask][index]if hasattr(self,'mask') else index
        ax.set_title('#{}: {}'.format(j,self.parameter_names[index]), size=10)
        ax.text( -2,-4.5, '{:9.3f}\n+/- {:5.3f}'.format(pz,sig), size=10, family='monospace', 
            backgroundcolor='w')
        ax.axvline(0, color='grey', ls = ':')
        ax.set_xticks([-2,0,2])
        if not np.isnan(sig):
            ax2 = ax.twinx()
            gradvals = -sig*np.array(map(gradf, x))
            ax2.plot(xsig, gradvals, '-r')
            ax2.axhline(0, color='r', ls=':')
            ax2.set_ylim( y2lim)
            if not nolabels: 
                ax2.set_ylabel('derivative (sig units)')
                ax.set_xlabel('value (sig units)')
            else: ax2.set_yticklabels([])
        
        self.set_parameters(parz) # restore when done
        return fig
    
    def plot_all(self, perrow=5, figsize=None): 
        """
        """
        import matplotlib.pyplot as plt
        n = len(self.parameters)
        if n==1:
            return self.plot(0)
        estimate = self.estimate_solution()
        rows = (n+perrow-1)/perrow
        if figsize is None:
            figsize = (12, 2.5*rows)
        fig, axx = plt.subplots(rows,perrow, 
            figsize=figsize, sharex=True, sharey=True)
        for i, ax in enumerate(axx.flatten()):
            if i>=n: 
                ax.set_visible(False)
            else:
                self.plot(i, ax = ax, nolabels=True, estimate=estimate)
        return fig


class FitterMixin(object):

    def maximize(self,  **kwargs):
        """Maximize likelihood and estimate errors.
        Uses scipy.optimize.fmin_l_bfgs_b
        
        keyword args
        ------------
        m : int
            The maximum number of variable metric corrections
            used to define the limited memory matrix. (The limited memory BFGS
            method does not store the full hessian but uses this many terms in an
            approximation to it.)
        factr : float
            The iteration stops when
            ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps``,
            where ``eps`` is the machine precision, which is automatically
            generated by the code. Typical values for `factr` are: 1e12 for
            low accuracy; 1e7 for moderate accuracy; 10.0 for extremely
            high accuracy.
        pgtol : float
            The iteration will stop when
            ``max{|proj g_i | i = 1, ..., n} <= pgtol``
            where ``pg_i`` is the i-th component of the projected gradient.
        epsilon : float
            Step size used when `approx_grad` is True, for numerically
            calculating the gradient
        iprint : int
            Controls the frequency of output. ``iprint < 0`` means no output;
            ``iprint == 0`` means write messages to stdout; ``iprint > 1`` in
            addition means write logging information to a file named
            ``iterate.dat`` in the current working directory.
        disp : int, optional
            If zero, then no output.  If a positive number, then this over-rides
            `iprint` (i.e., `iprint` gets the value of `disp`).
        maxfun : int
            Maximum number of function evaluations.
        maxiter : int
            Maximum number of iterations.

        """
        from scipy import optimize
        quiet = kwargs.pop('quiet', True)
        if not kwargs.pop('use_gradient', True):
            print 'Warning: ignoring use_gradient=False'
        estimate_errors = kwargs.pop('estimate_errors', True)
        if not quiet: print 'using optimize.fmin_l_bfgs_b with parameter bounds %s\n, kw= %s'% (
                            self.bounds, kwargs)
        parz = self.get_parameters()
        winit = self.log_like()
        assert len(parz)==len(self.gradient()), 'tracking a bug'
        # list of default from the function statement, mods shown
        fit_args=dict(m=10, 
            factr=1e9,  #1e8
            pgtol=1e-3, #1e-05
            epsilon=1e-08, 
            iprint=-1, maxfun=15000, maxiter=15000)
        fit_args.update(kwargs)
        # run the fit
        ret = optimize.fmin_l_bfgs_b(self, parz, 
                bounds=self.bounds,  fprime=self.gradient, **fit_args)
        self.fmin_ret=ret
        if ret[2]['warnflag']>0: 
            print 'Fit failure: check parameters'
            self.set_parameters(parz) #restore if error 
            self.covariance = None 
            raise Exception( 'Fit failure:\n%s' % ret[2])
        if not quiet:
            print ret[2]
        f = ret 
        if estimate_errors:
            self.covariance = cov = np.linalg.inv(self.hessian(f[0])) # was .I
            diag = np.array(cov.diagonal()).flatten()
            bad = diag<0
            if np.any(bad):
                print 'Minimizer warning: bad errors for values %s'\
                    %np.asarray(self.parameter_names)[bad] 
                diag[bad]=0
            return f[1], f[0], np.sqrt(diag)
        else:
            self.covariance = None
            return f[1], f[0], np.nan
        
        
    def modify(self, fraction):
        """change iniital set to fraction of current change; restore will make it permanent
        """
        if fraction==0 : return
        delta = self.get_parameters()-self.initial_parameters
        self.initial_parameters += fraction * delta
        
    def restore(self):
        self.set_parameters(self.initial_parameters)

    def check_gradient(self, delta=1e-5):
        """compare the analytic gradient with a numerical derivative"""
        
        parz = self.get_parameters()
        fz = self(parz)
        grad = self.gradient(parz)
        fprime=[]
        for i in range(len(parz)):
            parz[i]+=delta
            fplus = self(parz)
            assert abs(fplus-fz)>1e-7, 'Fail consistency: variable %d not changing' % i
            parz[i]-=2*delta
            fminus = self(parz)
            parz[i]+= delta
            fzero = self(parz)
            assert abs(fzero-fz)<1e-2, 'Fail consistency: %e, %e ' % (fzero, fz)
            fprime.append((fplus-fminus)/(2*delta))
        return grad, np.array(fprime) 

        # class FitFunction(FitPlotMixin, FitterMixin, tools.WithMixin): 
        #     blike = self
        #     def __init__(self, blike,  **kwargs):
        #         self.blike = blike
        #         self.parameters = blike.sources.parameters
        #         self.sources = self.parameters.free_sources
        #         self.initial_parameters = self.parameters.get_parameters() #ugly
        #     def restore(self):
        #         self.set_parameters(self.initial_parameters)
        #     def get_parameters(self):
        #         return self.parameters.get_parameters()
        #     def set_parameters(self, pars):
        #         self.parameters.set_parameters(pars)
        #         self.blike.update()
        #     @property 
        #     def bounds(self):
        #         return np.concatenate([s.model.bounds[s.model.free] for s in self.sources]) 
        #     def __call__(self, pars=None):
        #         if pars is not None: self.set_parameters(pars)
        #         return -self.blike.log_like()
        #     def log_like(self, summed=True):
        #         """assume that parameters are set, possibility of individual likelihoods"""
        #         return self.blike.log_like(summed)

        #     def gradient(self,pars=None):
        #         if pars is not None: self.set_parameters(pars)
        #         return self.blike.gradient()
        #     def hessian(self, pars=None):
        #         if pars is not None: self.set_parameters(pars)
        #         return self.blike.hessian()
        #     @property
        #     def parameter_names(self):
        #         return self.parameters.parameter_names
             

class FitterView(FitPlotMixin, FitterMixin, FitterSummaryMixin, tools.WithMixin): 

    def __init__(self, blike,  **kwargs):
        self.blike = blike
        self.parameters = blike.sources.parameters
        self.sources = self.parameters.free_sources
        self.initial_parameters = self.parameters[:]
        self.initial_likelihood = self.log_like()
        self.calls=0
        
    def get_parameters(self):
        return self.parameters.get_parameters()
    def set_parameters(self, pars):
        self.parameters.set_parameters(pars)
        self.blike.update()
        
    def save_covariance(self):
        """ store source submatrices of the fit covariance matrix into the models.
        this loses the correlations between sources
        """
        assert hasattr(self, 'covariance'), 'maximize was not run: no covariance to save'
        self.parameters.set_covariance(self.covariance)
    
    def modify(self, fraction):
        """change iniital set to fraction of current change; restore will make it permanent
        """
        if fraction==0 : return
        delta = self.get_parameters()-self.initial_parameters
        self.initial_parameters += fraction * delta
        
    def restore(self):
        self.set_parameters(self.initial_parameters)
    @property 
    def bounds(self):
        return np.concatenate([s.model.bounds[s.model.free] for s in self.sources]) 
    def __call__(self, pars=None):
        if pars is not None: self.set_parameters(pars)
        self.calls+=1
        return -self.blike.log_like()
    def log_like(self, summed=True):
        """assume that parameters are set, possibility of individual likelihoods"""
        return self.blike.log_like(summed)

    def gradient(self,pars=None):
        if pars is not None: self.set_parameters(pars)
        return self.blike.gradient()
    def hessian(self, pars=None):
        if pars is not None: self.set_parameters(pars)
        return self.blike.hessian()
    @property
    def parameter_names(self):
        return self.parameters.parameter_names
    @property
    def model_parameters(self):
        return self.parameters.model_parameters
    @property
    def uncertainties(self):
        return self.parameters.uncertainties
    @property
    def mask(self):
        return self.parameters.mask
             

class SubsetFitterView(parameterset.ParSubSet, FitPlotMixin, FitterMixin, FitterSummaryMixin,tools.WithMixin):

    def __init__(self, blike, select=None, exclude=None):
        self.blike = blike
        super(SubsetFitterView, self).__init__(blike.sources, select, exclude)
        self.initial_parameters = self.parameters[:]
        self.initial_likelihood = self.log_like()
        self.calls=0

    def __repr__(self):
        return '%s.%s: %s '% (self.__module__, self.__class__.__name__, self.selection_description)
    def restore(self):
        self.set_parameters(self.initial_parameters)

    def save_covariance(self):
        """ store source submatrices of the fit covariance matrix into the models.
        this loses the correlations between sources
        """
        assert hasattr(self, 'covariance'), 'maximize was not run: no covariance to save'
        self.set_covariance(self.covariance)

    @property
    def parameters(self):
        return self.get_parameters()
    def set_parameters(self, pars):
        super(SubsetFitterView,self).set_parameters(pars)
        self.blike.update()
    def __call__(self, pars=None):
        if pars is not None: self.set_parameters(pars)
        self.calls +=1
        return -self.blike.log_like()
        
    def log_like(self, summed=True):
        """assume that parameters are set, possibility of individual likelihoods"""
        return self.blike.log_like(summed)
    def gradient(self, pars=None):
        if pars is not None: self.set_parameters(pars)
        return self.blike.gradient()[self.mask]
    def hessian(self,pars=None):
        if pars is not None: self.set_parameters(pars)
        return self.blike.hessian(self.mask) 
        
    def ts(self):
        """ simple test statistic """
        lnow = self()
        pars = self.parameters
        pars[0] = -20 ##### override to be really small self.bounds[0][0]
        return 2 * (self(pars)-lnow)
        

class TSmapView(tools.WithMixin):

    def __init__(self, blike, func, quiet=True):
        self.quiet = quiet
        self.func = func
        self.blike = blike
        self.source = self.func.source
        self.saved_skydir = self.get_dir()
        self.wzero = func.log_like()
    
    def __repr__(self):
        return '%s.%s: source %s' % (self.__module__, self.__class__.__name__, self.source.name)
        
    def set_dir(self, skydir):
        self.source.skydir = skydir
        self.blike.initialize(None, self.source.name ) #sourcenane=self.source.name)
    
    def get_dir(self):
        return self.source.skydir
    skydir = property(get_dir, set_dir)
    
    def restore(self):
        self.set_dir(self.saved_skydir)

    def __call__(self, skydir=None):
        if skydir is not None:
            if not isinstance(skydir, SkyDir):
                skydir = SkyDir(*skydir)
            self.set_dir(skydir)
        return 2*(self.func.log_like()-self.wzero)


class EnergyFluxView(tools.WithMixin):

    def __init__(self, blike, func, energy, **kw):
        
        self.func = func
        self.blike=blike
        self.source = source = self.func.source
        self.model=model = source.spectral_model
        #assert model[0]==model['norm']
        self.norm = model[0]
        self.tointernal = model.mappers[0].tointernal
        self.bound = kw.get('bound', -20)# !!! model.bounds[0][0])
        self.set_energy(energy)

    def set_energy(self, energy=None):
        if energy is None:
            energy=self.model.e0
        self.model[0]=self.norm # get original norm 
        self.source.changed=True
        self.blike.update()
        self.energy = energy
        self.eflux = self.model(energy) * energy**2 * 1e6
        self.ratio = self.model[0]/self.eflux
    
    def __repr__(self):
        return '%s.%s: func=%s, at %.0f MeV' % (self.__module__, self.__class__.__name__,self.func,self.energy)
    def restore(self):
        self.set_energy()

    @tools.ufunc_decorator # make this behave like a ufunc
    def __call__(self, eflux):
        if eflux<=0:
            par = self.bound
        else:
            par = max(self.bound, self.tointernal(eflux*self.ratio))
        return -self.func([par])
        
class NormalizationView(tools.WithMixin):
    """Manage a view defining a function of the normalization factor for a source
    
    """
    def __init__(self, blike, source_name):
        self.blike = blike
        source = blike.sources.find_source(source_name)
        self.model = model= source.model
        self.par = model[0]
        parname = model.param_names[0]
        if not model.free[0]:
            self.freed = (parname, source_name)
            blike.thaw(parname, source_name)
        else: self.freed=None
        self.func = blike.fitter_view(source_name + '_' + parname)

        self.tointernal = model.mappers[0].tointernal
        self.bound = model.bounds[0][0]
    
    @tools.ufunc_decorator
    def __call__(self, norm):
        if norm <=0:
            par = self.bound
        else:
            par = max(self.bound, self.tointernal(norm*self.par))
        return -self.func([par])
    
    def restore(self):
        """If had to thaw, restore"""
        self.func.restore()
        if self.freed is not None and self.model.free[0]:
            self.blike.freeze(*self.freed)
        

class LikelihoodViews(bandlike.BandLikeList):

    """Subclass of BandLikeList with  methods to return views for specific analyses.
    
    * fits: fitter_view, return a FitterView or SubsetFitterView
    * SED : energy_flux_view, a fitterView with a source selected
    * TSmap : tsmap_view : a FitterView with the source flux selected which can have the position changed.
    """
    
    def fitter_view(self, select=None, setpars=None, **kwargs):
        """ return a object to use with a fitter.
            Two versions, one with full set of parameters, other if a subset is specified
        """
        if setpars is not None: 
            self.sources.parameters.setitems(setpars)

        if select is None:
            return FitterView(self, **kwargs)
        return SubsetFitterView(self, select, **kwargs)

    def energy_flux_view(self, source_name, energy=None, **kw):
        """ a functor for a source, which returns log likelihood as a 
                function of the differential energy flux, in eV units, at the given energy
                
        parameters
        ----------
        source_name : string
        energy : [None | float]
            if None, use the reference energy e0
        """
        try:
            source = self.sources.find_source(source_name)
            model = source.model
            func = self.fitter_view(source_name + '_' + model.param_names[0])
        except Exception, msg:
            raise Exception('could not create energy flux function for source %s;%s' %(source_name, msg))
        return EnergyFluxView(self, func, energy, **kw)
        
    def tsmap_view(self, source_name, **kw):
        """Return TSmap function for the source
        """
        if source_name is None and self.sources.selected_source is not None:
            source_name = self.sources.selected_source.name 
        if source_name is None: 
            raise Exception('No source is selected for a tsmap')
        try:
            func = self.fitter_view(source_name+'_Norm')
        except Exception, msg:
            raise Exception('could not create tsmap function for source %s;%s' %(source_name, msg))
        return TSmapView(self, func, **kw)
        
    def normalization_view(self, source_name):
        return NormalizationView(self, source_name)

def make_views(roi_index, rings=2):
    """convenience function to return a LikelihoodViews object for testing
    """
    from . import (configuration, bands, from_healpix)
    config = configuration.Configuration(quiet=True)
    roi_bands = bands.BandSet(config, roi_index, load=True)
    roi_sources = from_healpix.ROImodelFromHealpix(config, roi_index, load_kw=dict(rings=rings))
    return LikelihoodViews(roi_bands, roi_sources)
