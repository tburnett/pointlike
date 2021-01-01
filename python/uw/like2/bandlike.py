"""
Manage spectral and angular models for an energy band to calculate the likelihood, gradient
   
$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/bandlike.py,v 1.67 2018/01/28 21:27:58 kerrm Exp $
Author: T.Burnett <tburnett@uw.edu> (based on pioneering work by M. Kerr)
"""

import sys, types
import numpy as np
from  uw.utilities import keyword_options
from skymaps import SkyDir, Band

config=None
   
class BandLike(object):
    """ manage the likelihood calculation for a band 
    """
    defaults = (
        ('quiet', True, 'set False for info'),
        )
    @keyword_options.decorate(defaults)
    def __init__(self, band, sources, free, roi, **kwargs):
        """
           band    : bands.EnergyBand object
           sources : list of sources.Source objects
           free    : [array of bool | None]
                to select models with variable parameters
                If None, select all
           roi reference to the ROI
        """
        keyword_options.process(self, kwargs)
        # make a list of the Response objects
        self.bandsources = np.array(
                map(lambda s: s.response(band, quiet=self.quiet, roi=roi), sources)
        )
        self.active_mask = map(lambda s: s.active, self.bandsources)

        self.band = band 
        self.roi = roi
        self.event_type_name = config.event_type_name(band.event_type)
        self.exposure_factor = band.exposure.correction
        self.data = band.pix_counts  if band.has_pixels else []# data from the band
        self.pixels=len(self.data)
        
        self.initialize(free)
        self.update()
        # special code to unweight if galactic diffuse too large
        self.unweight = self.make_unweight()
         
    def make_unweight(self):
        """ return an unweighting factor <=1.0 to use to multiply the log likelihood
        
        systematic : float
            a fraction representing the relative systematic uncertainty in the galactic diffuse
        """
        try:
            if hasattr(self['ring'].source, 'systematic'): 
                systematic = self['ring'].source.systematic
            else:  return 1.0
        except: return 1.0
        if systematic==0: return 1.0 
        n = 1/systematic**2
        # m is the number of counts from the galactic diffuse in the footprint of a point source
        psf0 = self.band.psf([0])[0] # to deal with possibilty that psf is not ufunc
        m = self['ring'].counts / (psf0 * self.band.solid_angle)
        u = min(1., n/m)
        return u

    def __str__(self):
        b = self.band
        return '%s.%s: %d models (%d free) applied to band %.0f-%.0f, %s with %d pixels, %d photons'\
                % (self.__module__,self.__class__.__name__,sum(self.active_mask), sum(self.free), b.emin, b.emax, 
                 self.event_type_name, self.pixels, sum(self.data), )
                 
    def __repr__(self): return self.__str__()
    
    def __getitem__(self, i): 
        """ return a BandSource object refererence, either by index, or by source name"""
        if type(i)==types.StringType:
            t = list(self.bandsources)
            for bs in t:
                if i==bs.source.name:
                    return bs
            raise Exception('Source "%s" not found in band sources' %i)
        return self.bandsources[i]
        
    @property
    def pixel_dirs(self):
        """ the list of SkyDirs for the pixels with data"""
        return [SkyDir(w.dir()) for w in self.band.wsdl]
        
    @property
    def inside(self):
        """a list of bools for pixels with data that are inside the active part of the ROI
        Needed to make a mask
        """
        b12index = Band(12).index
        roi_id =b12index(self.roi.roi_dir)
        id12 = np.array(map(b12index, self.pixel_dirs),int) #ids of data pixels for nside=12
        return id12==roi_id

    def initialize(self, free):
        """ should only call if free array changes.
            Saves the combined prediction from the models with fixed parameters
        """
        assert free is not None, 'bad call?'
        self.free = free
        self.free_sources = self.bandsources[self.free]
        self.active_mask = map(lambda s: s.active, self.bandsources)
        
        #compile counts and count masks only for pointsources that are "active", i.e. not too far away or weak
        if len(self.bandsources)>0:

            fixed_sources = self.bandsources[~self.free & self.active_mask]
            self.counts = self.fixed_counts = sum([b.counts for b in fixed_sources])
        else:
            fixed_sources = []
            self.counts = self.fixed_counts = 0

        if not self.band.has_pixels: 
            self.model_pixels=self.fixed_pixels = self.weights= np.array([])
            return
        self.fixed_pixels = np.zeros(self.pixels)
        for m in fixed_sources:
            self.fixed_pixels += m.pix_counts
        self.model_pixels = self.fixed_pixels.copy()
        for m in self.free_sources:
            if m.counts==0:
                print 'Source {} is inactive, but free'.format(m.source.name) 
                continue # should be no inactive free sources?
            self.model_pixels += m.pix_counts
       
    def update(self, reset=False, force=False, **kwargs):
        """ assume that parameters have changed. Update only contributions 
        from models with free parameters, with changed tag. 
        *must* be called before evaluating likelihood.

        reset: bool, default False
            if True, need to reinitialize variable source(s), for change of position or shape
        force: bool, default False
            Force update of response even if source unchanged.
        """
        self.model_pixels[:]=self.fixed_pixels
        self.counts = self.fixed_counts
        for bandsource in self.free_sources:
            if reset: 
                bandsource.initialize()
                bandsource.source.changed=False
            elif bandsource.source.changed or force:
                bandsource.update()
            if self.band.has_pixels: 
                self.model_pixels += bandsource.pix_counts
                self.counts+= bandsource.counts
 
        if self.band.has_pixels: 
            self.weights = self.data / self.model_pixels
 
    def log_like(self):
        """ return the Poisson extended log likelihood """
        try:
            pix = np.sum( self.data * np.log(self.model_pixels) )  if self.pixels>0 else 0
            w = pix - self.counts * self.exposure_factor
            return self.unweight * w
        except FloatingPointError, e:
            print '%s: Floating point error %s evaluating likelihood for band %s' % (e,self,)
            raise
        except Exception, e:
            print '{} Exception evaluating likelihood for band {}'.format(e, self)
    
    def gradient(self):
        """ gradient of the likelihood with resepect to the free parameters
        """
        if len(self.free_sources)==0: return np.array([])
        grads = []
        for m in self.free_sources:
            grads.append(m.grad(self.weights, self.exposure_factor))
            
        return self.unweight * np.concatenate(
                [m.grad(self.weights, self.exposure_factor) for m in self.free_sources]
            )
       
    def model_counts(self, sourcemask=None):
        """ return the model predicted counts for all or a subset of the sources
        sourcemask : array of bool
            select a set of sources. In order of sources from factory
        """
        if sourcemask is not None:
            assert len(sourcemask)==len(self), 'bad input to model_counts'
        t = np.array([s.counts for s in self])
        return sum(t) if sourcemask is None else sum(t[sourcemask])

    def add_source(self, source):
        """ add a new source 
            source: sources.Source object
        """
        # ugly but compact
        t = list(self.bandsources)
        t.append(source.response(self.band))
        self.bandsources = np.array(t)
        
    def del_source(self, source):
        """ remove the source """
        t = list(self.bandsources)
        for bs in t:
            if source.name==bs.source.name:
                t.remove(bs)
                self.bandsources = np.array(t)
                return
        raise Exception('source "%s" not found to delete' % source.name)
       
    def fill_grid(self, sdirs):
        """ fill a grid with values, which are counts/sr, so must be multiplied by the pixel size
        """
        t = np.zeros(len(sdirs))
        for m in self.bandsources:
            t+= m.fill_grid(sdirs)
        return t
        
    def dataframe(self, query=None, sortby='counts', ascending=False):
        """ return a pandas.DataFrame for diagnostics """
        import pandas as pd
        df = pd.DataFrame(
            dict([(s.source.name, 
                dict(counts=round(s.counts,), 
                    overlap=round(s.overlap,3),
                    ts= np.nan if s.source.isglobal else round(s.source.ts), 
                    free=self.free[i],
                    distance=round(np.degrees(s.band.skydir.difference(s.source.skydir)) if s.source.skydir is not None else 0 ,2),
                    extended=s.source.skydir is not None and hasattr(s.source, 'dmodel'),
                    diffuse=s.source.skydir is None,
                    index = s.source.index[0],
                    ring = s.source.index[1],
                    )
                )
                for i,s in enumerate(self[self.active_mask])]
                )
            ).T
        if query is not None:
            df = df.query(query)
        if sortby is not None:
            df = df.sort_values(by=sortby, ascending=ascending)
        return df

    def counts_in_pixel(self, source_index, skydir):
        """ return a tuple of predicted signal and background counts in the pixel corresponding to skydir
        Note that if the pixel has no data, it will not have been computed for the model; instead
        this will return the average background
 
        source_index : int
            the index of the source
        skydir : SkyDir object
        """
        from skymaps import WeightedSkyDir
        band = self.band
        if not hasattr(self, 'pixel_indeces'): 
            self.pixel_indeces = list([band.b.index(x) for x in band.wsdl])
        source = self[source_index]
        hp_index = band.b.index(skydir)
        try:
            pixel_index = self.pixel_indeces.index(hp_index)
            signal = source.counts * source.pixel_values[pixel_index]
            back  = self.model_pixels[pixel_index] - signal
        except:
            # missing pixel; calculate the expected source counts
            # and estimate total by mean of model in ROI
            signal = source.flux_value(skydir) * source.counts
            back = self.model_pixels.mean()
        return signal, back 

    def fluxes(self, skydir):
        """Return an array of the fluxes for all models at the position
        skydir : Skydir object | (ra,dec) tuple
        """
        loc = skydir if isinstance( skydir,SkyDir,) else SkyDir(*skydir)
        t = []
        for bb in self:
            try:
                t.append( bb(loc))
            except Exception, msg:
                print 'Fail flux calculation for {}:{}'.format(bb, msg)
                t.append(0)
        return np.array(t)

    def __call__(self, skydir):
        """ return the total counts/sr for the given direction
        skydir : Skydir object | (ra,dec) tuple
        
        Note: to get counts, multiply by self.band.pixel_area
        """
        ret= np.sum(self.fluxes(skydir)) 
        assert not np.isnan(ret), 'NaN value detected at skydir {}'.format(skydir)
        return ret
    
         
class BandLikeList(list):
    """Manage a list of BandLike objects
    """
    
    def __init__(self, roi_bands, roi_sources):
        """ create a list, one per band, of BandLike objects for a given ROI 
        Methods to calculate the likelihood, gradient, and hessian
        
        parameters
        ----------        
        roi_bands : list of ROIBand objects
        roi_sources :  sourcelist.SourceList object
            a list of Source objects 
        """
        self.setup(roi_bands, roi_sources)
        
    def setup(self, roi_bands, roi_sources):
        """ initialize from bands and sources. 
        If called again, remove current set of BandLike objects
        """
        global config
        self.sources = roi_sources
        self.bands = roi_bands
        config = roi_bands.config #set global
        while len(self)>0:
            self.pop()
        for band in roi_bands:
            free = self.sources.free if self.sources else []
            bl = BandLike(band, self.sources, free, self) 
            self.append( bl)
            
        self.set_selected(self)# set selected for a subset?
        self.all_energies = self.energies[:]
        self.roi_dir = roi_bands.roi_dir
        
    #  the band selection mechanism, used by log_like, update and gradient   
    def set_selected(self, values):
        """setter for the property selected, which must be a subset of self"""
        if values in self: # single guy
            self._selected = [values]
            return
        assert set(values).issubset(self), 'Improper selection'
        self._selected = values
    def get_selected(self):
        return self._selected
    selected = property(get_selected, set_selected)
        
    def select(self, index=None, event_type=None, elow=None, ehigh=None,):
        """ Select an energy band or bands
        parameters:
        ----------
        index: None or integer
            an index into the list of energies; if None, select all bands
        event_type : None, integer, or name
            if None or 'all', select all
        elow,ehigh : None | float
            If elow[None] is set, override others, select all bands with energy>elow and < ehigh (if set)
        """
        etindex = config.select_event_type(event_type)
        type_select = lambda x : True if etindex is None else x==etindex
        

        if elow is not None:
            esel = lambda b: b.band.energy>elow and (b.band.energy<ehigh if ehigh is not None else True)
            etsel= lambda b: b.band.event_type==etindex if event_type is not None else True
            t = filter(lambda b: esel(b) and etsel(b) , self[:]) 

            assert len(t)>0, 'no bands selected with event_type, elow,ehigh= ({},{},{})'.format(event_type,elow,ehigh)
            self.selected=t
            return
        if index is None and etindex is None: #select all (initially selected) bands
            selected_bands = self[:]
        else:
            energy = self.all_energies[index] if index is not None else None
            energy_select = lambda x : True if energy is None else abs(x-energy)<1
            type_select = lambda x : True if etindex is None else x==etindex
            selected_bands = filter(lambda b: energy_select(b.band.energy) and type_select(b.band.event_type), self)
            if len(selected_bands)==0:
                raise Exception( 'did not find any bands for energy %.1f and event_type %s: %s are available'\
                 %( energy, etindex, self.energies))
        self.selected = selected_bands

    @property
    def free_sources(self):
        """ list of sources with currently free parameters
        """
        return np.array(self.sources)[self.sources.free]
    
    def add_source(self, newsource=None, **kwargs):
        """ add a source to the ROI
        
        parameters
        ----------
        newsource : PointSource object or None
            if None, expect source to be defined by the keywords
        keywords:
            name : string
            model : optional, string or like.Models.Model object
                A string will be evaluated, e.g. 'PowerLaw(1e-14, 2.0)'
            skydir : either an (ra,dec) tuple, or skymaps.SkyDir object
        """
        source = self.sources.add_source(newsource, **kwargs)
        for band in self:
            band.add_source(source)
        self.initialize()
        return source
        
    def del_source(self, source_name):
        """ delete the specific source (which can be expressed with wildcards 
        returns the source object
        """
        source  = self.sources.del_source(source_name)
        for b in self:
            b.del_source(source)
        self.initialize()
        return source
        
    def set_model(self, model, source_name=None):
        """ replace the current model, return reference to previous
        
        model : string, or like.Models.Model object
            if string, evaluate. Note that 'PowerLaw(1e-11,2.0)' will work. Also supported:
            ExpCutoff, PLSuperExpCutoff, LogParabola, each with all parameters required.
        source_name: None or string
            if None, use currently selected source
        """
        source, old_model = self.sources.set_model(model, source_name=source_name)
        self.initialize()
        return old_model

    @property
    def energies(self):
        return  np.sort(list(set([ sm.band.energy for sm in self._selected])))
    @property
    def emin(self):
        return np.array([b.band.emin for b in self._selected]).min()
    @property
    def emax(self):
        return np.array([b.band.emax for b in self._selected]).max()
        
    def __repr__(self):
        sel = '%d bands'%len(self.bands) if len(self.selected)==len(self) else '%d / %d selected bands'\
            %(len(self.selected),len(self))
        return '%s.%s: \n\t%s\n\t%s\n\tParameters: %d in %d/%d free sources' % (
            self.__module__, self.__class__.__name__,
            self.sources, sel, len(self.sources.parameters), 
            len(self.free_sources),  len(self.sources))

    def initialize(self, free=None, sourcename=None):
        """ initialize the specifed source in  all selected bands
            and update the band"""
        for b in self._selected:
            if sourcename is not None:
                b[sourcename].initialize()
            else:
                b.initialize(free if free is not None else self.sources.free)
            b.update()
        
    # the following methods sum over the current set of bands
    def log_like(self, summed=True):
        """log likelihood for current set of bands
        summed : bool, optional
        if false, return the array of likelihods for each band
        """
        r = np.array([b.log_like() for b in self._selected])
        return  sum(r) if summed else r
        
    def update(self, **kwargs):
        for b in self._selected: 
            b.update( **kwargs)
        self.sources.parameters.clear_changed()
        
    def gradient(self):
        return np.array([blike.gradient() for blike in self._selected]).sum(axis=0) 
        
    def hessian(self, mask=None, delta=1e-6):
        """ return a hessian matrix based on the current parameter set
        This makes a numerical derivative of the analytic gradient, so not exactly
        symmetric, but the the result must be (nearly) symmetric.
        
        mask : [None | array of bool]
            If present, must have dimension of the parameters, will generate a sub matrix
        
        For sigmas and correlation coefficients, invert to covariance
                cov =  self.hessian().I
                sigs = np.sqrt(cov.diagonal())
                corr = cov / np.outer(sigs,sigs)
        """
        # get the source parameter management object
        parameters = self.sources.parameters
        parz = parameters.get_parameters()
        if mask is None: mask = np.ones(len(parz),bool)
        else:
            mask = np.asarray(mask)
            assert len(mask)==len(parz)
        # initial values for the likelihood and gradient
        fzero = self.log_like()
        glast = gzero = self.gradient()[mask]
        t = []
        for i in np.arange(len(parz))[mask]:
            # increment current variable and get new gradient
            parameters[i] = parz[i]+delta
            self.update()
            gnow = self.gradient()[mask]
            # numerical derivative of gradient with respect to this parameter
            t.append( (gnow-glast)/delta)
            glast = gnow
        hess = np.array(t)
        # depricated
        #hess = np.matrix(t)
        parameters.set_parameters(parz) #restore all parameters, check that no difference
        self.update()
        assert abs(fzero-self.log_like())<1e-2
        return hess 
       
    def delta_loglike(self):
        """ estimate change in log likelihood from current gradient 
        """
        try:
            # rewrite due to deprication of numpy.matrix
            # gm = np.matrix(self.gradient())
            # H = self.hessian()
            # return (gm * H.I * gm.T)[0,0]/4
            g = self.gradient()
            H = self.hessian()
            return np.dot( np.dot(g, np.linalg.inv(H)) , g)/4
        except Exception, msg:
            print 'Failed log likelihood estimate, returning 99.: %s' % msg
            return 99.

    def parameter_summary(self, out=None):
        """formatted summary of parameters, values, gradient
        out : None or open stream
        """
        print >>out, 'log likelihood: %.1f' % self.log_like()
        print >>out,'\n%-21s %8s %8s' % ('parameter', 'value', 'gradient')
        print >>out,  '%-21s %8s %8s' % ('---------', '-----', '--------')
        for u in zip(self.sources.parameter_names, self.sources.parameters, self.gradient()):
            print >>out, '%-21s %8.2f %8.1f' % u
       
    def freeze(self, parname, source_name=None, value=None):
        """ freeze the parameter, optionally set the value
        
        parname : name or index
            if a string with an underscore, interpret as source_parname
        source_name: None or string
            if None, use currently selected source
        value : float or None
            if float, set the value
        """
        if parname.find('_')>0 and source_name is None:
            source_name, parname = parname.split('_')
        source = self.get_source(source_name)
        source.freeze(parname, value)
        self.reinitialize()
        
    def reinitialize(self):
        """Used to reinitialize after change in number of free parameters
        """
        self.sources.initialize()
        self.initialize(free=self.sources.free)
        self.update()
       
    def thaw(self, parname, source_name=None):
        """ thaw the parameter
        
        parname : name or index
            if a string with an underscore, interpret as source_parname
        source_name: None or string
            if None, use currently selected source
        """
        # sunmoon_free = self.get_source('SunMoon').model.free[:]
        if parname.find('_')>0 and source_name is None:
            source_name, parname = parname.split('_')
        source = self.get_source(source_name)
        source.thaw(parname)
        #print 'Thawed: {}_{}'.format(source_name, parname)
        # make sure SunMoon does not change - don't understand how this seems to happen!
        # self.get_source('SunMoon').model.free=[False]
        self.reinitialize()

    def select_band(self, energy, et):
        """ Find a band, given the energy and event type"""
        for bl in self:
            if energy<=bl.band.emin : continue
            if energy>bl.band.emax : continue
            if et != bl.band.event_type: continue
            return bl
        raise Exception(
            'No band found for energy {:.1f}, event_type {}'.format(energy,ct))

    def source_weight(self, source_index, ra,dec, energy, et):
        """ Return weight for a source

        source_index : integer | string  | None
            if string, look up source (allow wild cards)
            if None, assume already selected
        ra, dec, energy: float
        et : event type, (0,1) for (front,back)
        """
        try:
            source_index = int(source_index)
        except ValueError,TypeError:
            self.get_source(source_index)
            source_index=self.sources.selected_source_index
        # get the BandLike object for energy and event type
        bl = self.select_band(energy,et)
        f =bl.fluxes((ra,dec) ) 
        fs= f[source_index]
        return fs/(sum(f))

    def dataframe(self, query=None, sortby=None, ascending=False):
        """ return a pandas.DataFrame with local source info 
            query: string | None.
            sortby: string | None
        """
        import pandas as pd
        def eflux(model):
            e=model.pivot_energy()
            return model(e)* e**2 * 1e6
        df = pd.DataFrame(
            dict([(s.name, 
                dict( 
                    free=np.any(s.model.free),
                    distance=round(np.degrees(s.skydir.difference(self.roi_dir)),2),
                    extended = hasattr(s, 'dmodel'),
                    modelname = s.model.name,
                    pindex = s.model[1],
                    curvature= s.model.curvature(),
                    ts = s.ts,
                    flux100 = s.model.i_flux(e_weight=2), 
                    #ra = s.skydir.ra(),
                    #dec= s.skydir.dec(),
                    glat = s.skydir.b(),
                    eflux = eflux(s.model), #energy flux at pivot
                    pivot = s.model.pivot_energy(),
                    #f1gev= s.model(1e3) * 1e12, # eflux at 1 GeV
                    )
                )
                for s in self.sources if not s.isglobal])
            ).T
        if query is not None:
            df = df.query(query)
        if sortby is not None:
            df = df.sort_values(by=sortby, ascending=ascending)
        return df