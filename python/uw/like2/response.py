"""
Classes to compute response from various sources
 
$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/response.py,v 1.21 2018/01/27 15:37:17 burnett Exp $
author:  Toby Burnett
"""
import os, pickle
import numpy as np
import pandas as pd
import healpy
import skymaps
from uw.utilities import keyword_options
from . import convolution, diffuse

class ResponseException(Exception): pass

diffuse_grid_defaults = (
        ('pixelsize', 0.25, 'Size of pixels to use for convolution grid'),
        ('npix',      61,   'Number of pixels (must be an odd number'),
        ('npix2',     91,   'Number of pixels for E<1 GeV'),
        ('quiet',     False, ''),
        ('diffuse_normalization', None, 'dict of spectral normalization'),
        )
        
extended_grid_defaults = [
        ['pixelsize', 0.2, 'Size of pixels to use for convolution grid'],
        ('size',      14,  'Size of grid; npix set to size/pixelsize'),
        ('quiet',     False, ''),
        ]

class Response(object):
    """ Base class for classes that manage the response of a source, in total counts 
    or count density for any position within the ROI. Created by the response function of each source.  
    The constructor for the appropriate subclass is invoked by the source's "response" function, with an 
    EnergyBand as argument. 
    
    """
    def __init__(self, source, band, roi=None, **kwargs):
        """
        source : Source object, inherit from sources.Source
            skydir : position of source, or None if global
            model : associated spectral model
        band : EnergyBand object
            psf, exposure functions for the event type
            skydir, radius : location, size of ROI
            emin, emax : energy range
            ----- pixelization, data ---
            wsdl : list of pixel positions, perhaps None
            pixel_size : pixel solid angle
        roi : None or the current ROI. This is for the diffuse correction, in progress

        """
        self.band=band
        self.source=source
        self.roicenter = self.band.skydir
        self.quiet = kwargs.pop('quiet', True)
        self.roi = roi
        self.active=True # so distant PointSource can be inactive
        self.initialize() # can disable the response

        
    def update(self):
        self.evaluate()
        
    def __repr__(self):
        return '%s.%s: ROI at %s source "%s" at %s' %( self.__module__,self.__class__.__name__,
            self.roicenter, self.source.name, self.source.skydir)

    def exposure_integral(self):
        """Integral of the exposure times the flux at the given position"""
        return self.band.integrator(self.source.model)
    
    def __call__(self, skydir):
        """return the counts/sr for the source at the position"""
        raise NotImplemented
    @property
    def spectral_model(self):
        return self.source.model

class NoResponse(Response):
    """Special case for no overlap
    """
    def initialize(self): 
        self.counts=0
        self.pix_counts=0
    def __call__(self, skydir):
        return 0.

class PointResponse(Response):
    """Manage predictions of the response of a point source
    
    Calculates:
        counts : the expected counts in an ROI
        pixel_values : arraay of counts in a set of pixels, presumably corresponding to pixels with data
        grad : gradient
        
    """
    max_overlap = 1e-2 # zero response below this
    def initialize(self):
        self.overlap = self.band.psf.overlap(self.roicenter, self.band.radius, self.source.skydir)

        # declare inactive if not free, and small overlap
        self.active= self.overlap>PointResponse.max_overlap or np.sum(self.source.model.free)>0
        if not self.active:
            self.counts=0
            return
        self._exposure_ratio = self.band.exposure(self.source.skydir)/self.band.exposure(self.roicenter)
        if self.band.has_pixels:
            if hasattr(self.band.psf, 'cpsf'):
                # old PSF class, uses C++ code for speed
                wsdl = self.band.wsdl
                rvals  = np.empty(len(wsdl),dtype=float)
                self.band.psf.cpsf.wsdl_val(rvals, self.source.skydir, wsdl) #from C++: sets rvals
                self.pixel_values = rvals * self.band.pixel_area
            else:
                #  new psf class: cpsf is internal
                # psf_weights =self.band.psf(
                #     [self.source.skydir.difference(sd) for sd in self.band.wsdl])
                psf_weights = self.band.psf.wsdl_value(self.source.skydir, self.band.wsdl)
                self.pixel_values = psf_weights * self.band.pixel_area
        self.evaluate()

        
    def evaluate(self): 
        """ update values of counts, pix_counts used for likelihood calculation, derivatives
        Called when source parameters change
        """
        if not self.active: return
        model = self.spectral_model
        self.expected = self.band.integrator(model)
        assert not np.isinf(self.expected), 'model integration failure'
        self.counts =  self.expected * self.overlap
        self.model_grad = self.band.integrator( model.gradient)[model.free] #* self.exposure_ratio
        if self.band.has_pixels:
            self.pix_counts = self.pixel_values * self.expected
        
    def grad(self, weights, exposure_factor=1): 
        """ contribution to the overall gradient
        weights : arrary of float
            
            weights = self.data / self.model_pixels
        Assume that evaluate has set model_grad
        """
        if not self.active: return None # don't expect to be called
        model = self.spectral_model
        if np.sum(model.free)==0 : return []
        # Calculate the gradient of a spectral model (wrt its parameters) integrated over the exposure.
        #g = self.band.integrator( model.gradient)[model.free] #* self.exposure_ratio
        g = self.model_grad
        apterm = exposure_factor* self.overlap
        pixterm = (weights*self.pixel_values).sum() if self.band.has_pixels else 0
        return g * (apterm - pixterm)

    def __call__(self, skydir):
        if not self.active: return 0
        return self.band.psf(skydir.difference(self.source.skydir))[0]  * self.expected
     

    
class DiffuseResponse(Response):
        
    defaults = diffuse_grid_defaults
    @keyword_options.decorate(defaults)
    def __init__(self, source, band, roi, **kwargs):
        keyword_options.process(self, kwargs)
        self.setup=False
        self.overlap=1
        super(DiffuseResponse, self).__init__(source, band, roi,  **kwargs)
        self.quiet=kwargs.get('quiet', True)
        
    def initialize(self, force=False):
        if self.setup and not force: return
        self.setup=True
        #set up the spatial model 
        self.dmodel = self.source.dmodel[self.band.event_type]
        self.dmodel.load()
        self.energy = self.band.energy
        self.dmodel.setEnergy(self.band.energy)
        
        roi_index = skymaps.Band(12).index(self.roicenter)
        self._keyword_check(roi_index)
        if getattr(self, 'preconvolved', False):
            #print 'Using preconvolved'
            c = self.roicenter
            cv = healpy.dir2vec(c.l(),c.b(),lonlat=True)
            hplist = healpy.query_disc(self.dmodel.nside,cv, self.band.radius_in_rad)
            smband = skymaps.Band(self.dmodel.nside)
            assert smband.index(self.roicenter) in hplist, 'Problem HEALPix indexing, ROI center not consistent?'

            if self.dmodel.hdulist[1].header.get('TUNIT1',None)=='photons':
                #must rescale from photons/pixel to density: photons/Mev/Sr 
                scale_factor = (self.band.emax-self.band.emin) * smband.pixelArea()
            else: scale_factor=1

            dirs = map(self.dmodel.dirfun, hplist)
            self.evalpoints = lambda dirs : np.array(map(self.dmodel, dirs)) * self.corr / scale_factor
            self.ap_average = self.evalpoints(dirs).mean()
            # **** scale by years ****
            self.ap_average *= self.years/10.
        
        else:
            self.create_grid() # will raise exception if no overlap
            grid = self.grid
            inside = grid.dists< self.band.radius_in_rad
            self.ap_average = grid.cvals[inside].mean()
            self.evalpoints = lambda dirs : grid(dirs, grid.cvals)

        self.delta_e = self.band.emax - self.band.emin
        self.factor = self.ap_average * self.band.solid_angle * self.delta_e
        if self.band.has_pixels:
            self.pixel_values = self.evalpoints(self.band.wsdl) * self.band.pixel_area * self.delta_e

        self.evaluate()
        
    def create_grid(self):
        # create a grid for evaluating counts integral over ROI, individual pixel predictions
        grid = self.grid= convolution.ConvolvableGrid(center=self.roicenter, 
                npix=self.npix if self.energy>1000 else self.npix2, 
                pixelsize=self.pixelsize)
        # this may be overridden
        self.fill_grid()

            
    def fill_grid(self):
        # fill and convolve the grid
        
        self.grid.psf_fill(self.band.psf)
        self.grid.bg_fill(self.band.exposure, self.dmodel)
        #assert False, 'Break point'
        
        #now that the grid is filled with the background model values, perhaps change normalization
        # (should be in model? )
        if hasattr(self,'corr') and self.corr is not None:
            self.grid.bg_vals *= self.corr
        self.grid.convolve()
        
    def evaluate(self):
        norm = self.source.model(self.band.energy)
        self.counts = norm * self.factor
        if self.band.has_pixels:
            self.pix_counts = self.pixel_values * norm

    def grad(self, weights, exposure_factor=1): 
        """ contribution to the overall gradient
        weights : arrary of float
            weights = self.data / self.model_pixels
        """
        model = self.spectral_model
        if np.sum(model.free)==0 : 
            return []
        pixterm = ( self.pixel_values * weights ).sum() if self.band.has_pixels else 0
        return (self.factor*exposure_factor - pixterm) * model.gradient(self.energy)[model.free] 
        
    def __call__(self, skydir):
        self.dmodel.setEnergy(self.band.energy) # needed if convolved
        return self.evalpoints([skydir])[0] * self.delta_e

    def _keyword_check(self, roi_index):
        # check for extra keywords from diffuse spec.
        dfun = self.dmodel
        
        if hasattr(dfun, 'kw') and dfun.kw is not None and len(dfun.kw.keys())>1: 
            # Manage keywords found in the 
            if dfun.kw.get('key', '') == 'gal':
                #print 'Using gal key for galactic correction ', 
                self.corr = DiffuseCorrection(None, diffuse.normalization['gal'])(roi_index,self.energy)
                #print self.corr
            elif dfun.kw.get('key', '') == 'iso':
                et = ['front','back'][self.band.event_type]
                self.corr = DiffuseCorrection(None, diffuse.normalization['iso'][et])(roi_index, self.energy)
                #raise Exception('isotropic key: corr={}'.format(self.corr))
            elif 'correction' in dfun.kw and dfun.kw['correction'] is not None:
                if not self.quiet:
                    print '\t%s loading corrections for source %s from %s.kw:' \
                    % (self.__class__.__name__, self.source.name,  dfun.__class__.__name__)
                corr_file = os.path.expandvars(dfun.kw['correction'])
                self.corr = DiffuseCorrection(corr_file)(roi_index, self.energy) 
            elif 'corr' in dfun.kw:
                # for single roi: the corrections wired into the yaml declaration
                energy_index = np.searchsorted( np.hstack([np.logspace(2,4,9), [1e7]]), self.energy)-1
                self.corr = dfun.kw['corr'][min(energy_index,7)]
            else: 
                self.corr=1.0
            
            if not self.quiet: print '\tcorrections:{}'.format(self.corr)
            self.systematic = dfun.kw.get('systematic', None)
            if self.systematic is not None:
                if not self.quiet:print '\tsystematic: %.3f' % self.systematic
            if 'preconvolved' in dfun.kw:
                self.preconvolved = dfun.kw['preconvolved']
                self.years = dfun.kw.get('years', 10)
        else: 
            self.corr =1.0; self.systematic = None
            
        #cache this result in the diffuse object
        #self.source.corr = self.corr 
        self.source.systematic = self.systematic

    def rescale(self, scale_factor=1):
        """For tests with modified spectrum
        Adjust the model calculation by applying a scale factor
        Return current correction factor
        (Note the default gives access)
        For changes to be taken into account do a "reinitialize()"
        """
        if scale_factor != 1.:
            self.factor *= scale_factor
            self.pixel_values *= scale_factor
            self.corr *= scale_factor
            self.evaluate()
        return self.corr #new total correction

class DiffuseCorrection(object):
    """load, and provide access to a diffuse correction file
    
   Now support diffuse normalization dataframe in the ROI info
    """
    from uw.like2.pub import healpix_map as hm

    def __init__(self, filename, diffuse_normalization=None, elist=np.logspace(2.125,3.875,8).round(), quiet=False):
        """ filename : str
                expect the name of a file in csv format, with 1728 rows and 8 columns
            elist : list of float
        """
        self.elist = list(elist)
        if  filename is None:
            self.correction = None
            self.dn = diffuse_normalization
            return
        
        # new scheme: 
        self.dn= diffuse_normalization
        if self.dn is not None and filename in self.dn:
            try:
                self.correction = self.dn[filename]
                if not quiet:
                    print 'loaded corrections {} for {}'.format(self.correction, filename)
            except:
                print 'No normalization in ROI object: assume none'
                self.correction=None
            return
        corr_file = os.path.expandvars(filename)
        if corr_file[0]!='/' and not os.path.exists(corr_file):
            corr_file = os.path.expandvars(os.path.join('$FERMI','diffuse', corr_file))
        try:
            self.correction = pd.read_csv(corr_file, index_col=0) 
            #print self.correction
        except Exception, msg:
            raise Exception('Error loading correction file %s: %s'% (corr_file,msg))
            
    def plot_ait(self, energy_index= 0, title=None, ax=None, vmin=0.9, vmax=1.1, ait_kw={}, **kwargs):
        """ make an AIT plot, return the figure
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
        t = self.hm.HParray('band%d'%energy_index,self[energy_index])
        ait= t.plot(axes=ax, vmin=vmin, vmax=vmax, ait_kw=ait_kw, cbtext='correction')
        if title is None:
            ait.axes.set_title('corrections for band %d'%energy_index, size=12)
        else:
            ait.axes.set_title(title, size=10)
        return ait.axes.figure
        
    def __getitem__(self,energy_index):
        """return array of correction factors for given energy index"""
        return np.asarray(self.correction[str(energy_index)])
        
    def to_fits(self, filename, nside=256):
        """convert to a FITS file, resampling at the given nside for display in Aladin"""
        t = []
        for i in range(8):
            x = self.hm.HParray('band%d'%i,self[i])
            t.append( self.hm.HPresample(x, nside=nside) )
        self.hm.HEALPixFITS(t).write(filename)
        
    def __call__(self, roi_index, energy):
        if energy > self.elist[-1]: 
            energy_index = len(self.elist)-1 # will apply highest energy correction above table limit
        else:
            try:
                energy_index = self.elist.index(round(energy))
            except:
                raise Exception('energy %f not found in list %s' % (round(energy), self.elist))
        if self.dn is not None:
            if self.correction is None: # new case
                return self.dn[energy_index]
            return self.correction.iloc[energy_index] 
        return self.correction.iloc[roi_index][energy_index] if self.correction is not None else 1.0
        
    def roi_norm(self, roi_index):
        """return a array of correction for the bands for a given ROI"""
        return np.array([self(roi_index, e) for e in self.elist])

        
class CachedDiffuseResponse(DiffuseResponse):
        
    def create_grid(self):
        """ set up the grid from the cached files """
        
        roi_index = skymaps.Band(12).index(self.roicenter)
        dmodel = self.dmodel
        try:
            self.filename = dmodel.files[roi_index]
            self.cached_diffuse = pickle.load(dmodel.opener(self.filename))
        except Exception, msg:
            raise ResponseException( 'Diffuse cache file # %d not found:%s' %(roi_index,msg))
        self.emins = [cd['emin'] for cd in self.cached_diffuse]
        
        self._keyword_check(roi_index)
        
        # find the appropriate cached grid
        energy = self.band.energy
        for index in range(len(self.emins)):
            if energy>self.emins[index] and (index==len(self.emins)-1 or energy<self.emins[index+1])\
                : break
        emin = self.emins[index]    
        assert energy/emin < 1.8 and energy> emin, 'too large a factor: energy, emin=%.0f,%.0f\nemins=%s' % (energy, emin, self.emins)
        cd = self.cached_diffuse[index]
        # create a convolvable grid from it
        self.energy = energy = cd['energy']
        if cd['center'].difference(self.band.sd)>0.1:
            print '%s, WARNING: Logic error? %s not %s' % (self, cd['center'], self.band.sd)
        self.grid = grid = convolution.ConvolvableGrid(self.roicenter,   npix=cd['npix'], pixelsize=cd['pixelsize'])
            
        self.band.set_energy(energy)
        grid.psf_fill(self.band.psf)
        
        # apply correction factor if any; use last value for all higher (depends only on energy)
        vals = cd['vals']
        if hasattr(self, 'corr') and self.corr is not None:
            c = list(self.corr)[index] if index<len(self.corr) else list(self.corr)[-1]
            vals *= c
        else: c = None
        
        # finally do the convolution on the product of exposure and diffuse map, which is passed in 
        grid.bg_fill(self.band.exposure, None, cache=vals)
        
        grid.convolve()

class IsotropicResponse(DiffuseResponse):

    defaults = diffuse_grid_defaults
    @keyword_options.decorate(defaults)    
    def __init__(self, source, band, roi, **kwargs):
        keyword_options.process(self, kwargs)
        super(IsotropicResponse, self).__init__(source, band, roi, **kwargs)
        
    def evaluate(self):
        # deal with FrontBackConstant case, which uses different constants for front/back
        if hasattr(self.source.model, 'ct'): # bit ugly
            self.source.model.ct=self.band.event_type
        super(IsotropicResponse, self).evaluate()
            
    def grad(self, weights, exposure_factor=1): 
        # deal with FrontBackConstant case, which uses different conatants for front/back
        if hasattr(self.source.model, 'ct'): # bit ugly
            self.source.model.ct=self.band.event_type
        return super(IsotropicResponse, self).grad(weights, exposure_factor)


    def fill_grid(self):
        # fill the grid for evaluating counts integral over ROI, individual pixel predictions
        roi_index = skymaps.Band(12).index(self.roicenter)
        dmodel = self.dmodel
        self.corr = 1.0
        # TODO: modify this to use the kw "key" if exists
        haskey = hasattr(dmodel, 'kw') and dmodel.kw is not None
        if haskey and dmodel.kw.get('key', None)=='iso':
            et = ['front','back'][self.band.event_type]
            self.corr = DiffuseCorrection(None, diffuse.normalization['iso'][et])(roi_index, self.energy)
            #print roi_index, et, self.energy, self.corr
        elif haskey and 'correction' in dmodel.kw:
            # Set self.corr according to "correction"
              
            corr_file = dmodel.kw.get('correction', None)
            energy=round(self.band.energy)
            if corr_file is None:
                self.corr=1.0
            else:
                roi_index = skymaps.Band(12).index(self.roicenter)
                self.corr = DiffuseCorrection(corr_file)(roi_index, self.energy)

            #     dn = self.roi.sources.diffuse_normalization
            #     if False: #dn is not None and 'iso' in dn:
            #         # in processsself.corr = dn['iso'][('front','back')[self.]]
            #         assert False, 'need to add if I want'
            #     # load the correction value from the correction file
            #     else:
            #         self.corr = DiffuseCorrection(corr_file)(roi_index, energy)
            # #print 'ISO: {} {} MeV: apply correction {} '.format(corr_file, energy, self.corr)

        grid = self.grid
        grid.cvals = grid.fill(self.band.exposure) * dmodel(self.band.skydir) * self.corr
        

class ExtendedResponse(DiffuseResponse):
    
    def __init__(self, source, band, roi, **kwargs):
        defaults = dict( [x[:2] for x in extended_grid_defaults])
        defaults.update(kwargs)
        if 'npix' not in defaults:
            npix = defaults.pop('size')/defaults['pixelsize']
            defaults['npix'] = int(npix) | 1 # make odd
            #print 'setting npix', defaults
        self.quiet=defaults.get('quiet', True)
        self.initialized = False
        super(ExtendedResponse, self).__init__(source, band, roi, **defaults)
            
    def exposure_integral(self):
        """Perform integral of the exposure times the flux at the source position
        (Override base to specify source position)
        """
        return self.band.exposure.integrator(self.source.skydir, 
                self.band.emin, self.band.emax)(self.source.model)
      
    def initialize(self):
        #set up the spatial model NOTE THIS NEEDS TO BE SAVED
        # that index only pulls up a paramter??? See if this is acutally referenced by clients
        #self.dmodel = self.source.dmodel[self.band.event_type]
  
        self.center = self.source.skydir
         
        self.cvals = dict()
        self.dm_vals = None
        self.create_grid()
        self.convolve()
        self.evaluate()
        self.initialized = True

    def setup_image(self):
        # load the SkyImage as a numpy 2-d array (not used yet?)
        assert hasattr(source.dmodel, 'skyfun'), 'Wrong dmodel? %s' % source.dmodel
        skyfun = source.dmodel.skyfun
        naxis1 = skyfun.naxis1()
        naxis2 = skyfun.naxis2()
        self.image = np.asarray(skyfun.image()).reshape(naxis1,naxis2)
        
    def set_energy(self, energy):
        self.band.set_energy(energy)
    
    @property
    def energy(self):
        return round(self.band.energy)
        
    def create_grid(self):
        """create a grid for all convolutions, evaluate exended model on it
        Note that it is normalized so the integral over solid angle should be 1
        Have the source own the grid; each band has 
        """
        if not hasattr(self.source, 'grid'):
            self.source.grid = convolution.ConvolvableGrid(self.center, pixelsize=self.pixelsize, npix=self.npix)
        self.grid =self.source.grid
        self.dm_vals = self.grid.fill(self.source.dmodel)
        self.dm_vals/= (self.dm_vals.sum() * np.radians(self.pixelsize)**2) 

    def overlap_mask(self):
        """ 
        return a npix x npix array of bools for the part of the grid, which is centered on the source,
        which is inside the ROI circle
        """
        x,y = self.grid.pix(self.roicenter)
        npix = self.grid.npix
        dx2 =((np.arange(npix)-x)**2).reshape(npix,1)
        dy2 =((np.arange(npix)-y)**2)
        d2  = dx2+dy2 # dx2 *dy2 is expanded to a square matrix
        #return d2.max(), d2.min()
        return d2  <= (self.band.radius/self.grid.pixelsize)**2

    def convolve(self, energy=None):
        """ perform the convolution, creating a grid for the specified energy
        
        return the tuple ap_average, overlap, psf_overlap
        """
        assert not self.initialized, 'Already initialized?'
        if energy is not None: self.band.set_energy(energy)
        #if self.grid is None: self.create_grid()
        
        # chedk the PSF size: if less than grid spacing, do not convolve, keep original grid
        #if self.band.psf.inverse_integral(68)< self.pixelsize:
        #    self.cvals[self.energy]=None
        # Fill the psf in the grid
        self.grid.psf_fill(self.band.psf)
        
        # now look at values, decide if want to convolve    
        exp_grid = self.grid.fill(self.band.exposure) # this is expensive, 1.6 s for npix=201
        self.grid.bg_vals = exp_grid * self.dm_vals
        self.grid.convolve()  
        
        # save a copy of cvals for this band
        self.cvals = self.grid.cvals.copy()
        
        # calculations needed which depend on this convolution
        cvals = self.cvals
        badvals = pd.isnull(cvals)
        if np.any(badvals):
            print 'bad vals'
            assert False,'Check strategy here'
        inside = self.overlap_mask() #self.grid.dists< np.radians(self.band.radius)
        if not inside.all():
            #print 'Inside==0 for band {}'.format(self.band) 
            self.ap_average=self.psf_overlap = 0
        else:
            self.ap_average = cvals[inside].mean()
            pvals = self.grid.psf_vals
            self.psf_overlap = pvals[inside].sum() / pvals.sum() 
        self.ap_center = self.grid(self.center, cvals)
        self.overlap = cvals[inside].sum() / cvals.sum()
        if self.overlap<1e-6:
            self.active=False
            self.counts = 0
            return
        self.exposure_at_center = self.band.exposure(self.source.skydir)
        self.exposure_ratio = self.exposure_at_center / self.band.exposure(self.roicenter)
        self.factor = self.overlap * self.exposure_ratio 
        if self.band.has_pixels:
            self.pixel_values = self.grid(self.band.wsdl, self.cvals)\
                /self.exposure_at_center * self.band.pixel_area
        self.expint = self.exposure_integral() # save initial value
        self.initcounts = self.expint * self.factor # "
        self.initmodel = self.source.model.copy()
        
    def evaluate(self):
        if not self.active:
            return
        total_counts = self.exposure_integral()
        self.counts = total_counts * self.factor
        if self.band.has_pixels:
            self.pix_counts = self.pixel_values * total_counts
        
    def grad(self, weights, exposure_factor=1): 
        """ contribution to the overall gradient
        weights : arrary of float
            weights = self.data / self.model_pixels
        """
        model = self.spectral_model
        if np.sum(model.free)==0 : 
            return []
        #pixterm = ( self.pixel_values * weights ).sum() * self.exposure_at_center if self.band.has_pixels else 0
        #return (self.factor*exposure_factor - pixterm) * model.gradient(self.energy)[model.free] 
        g = self.band.integrator( model.gradient)[model.free] #* self.exposure_ratio
        apterm = exposure_factor * self.factor #self.overlap
        pixterm = (weights*self.pixel_values).sum() if self.band.has_pixels else 0
        return g * (apterm - pixterm)

    def __call__(self, skydir, force=False):
        """ return value of perhaps convolved grid for the position
        skydir : SkyDir object | [SkyDir]
        """
        if not self.active: return 0
        return self.grid(skydir, self.cvals) / self.exposure_at_center * self.counts/self.factor
        
    def __repr__(self):
        return '%s.%s: \n\tsource: %s\n\tband  : %s\n\tpixelsize: %.1f, npix: %d' % (
            self.__module__, self.__class__.__name__, self.source,self.band, self.pixelsize, self.npix)
  
          
    def show(self, title, vals, ax=None, logscale=False,
            roi_radius=None, colorbar=True):
        import pylab as plt
        from matplotlib.colors import LogNorm
        if ax is None:
            fig, ax = plt.subplots(figsize=(4,4))
        norm = LogNorm(vmin=0.001, vmax=1.0) if logscale else None
        roi_radius = self.band.radius if roi_radius is None else roi_radius
        self.grid.show_vals(vals/vals.max(), ax=ax, roi_radius=roi_radius,
            roi_dir=self.roicenter, norm=norm, colorbar=colorbar)
        ax.set_title(title, size=10)
        return ax.figure
    
    def show_psf(self, ax=None, colorbar=True):
        return self.show('PSF for %.0f MeV' % self.band.energy, 
            self.grid.psf_vals, roi_radius=0, ax=ax, colorbar=colorbar)
    
    def show_cvals(self, ax=None, colorbar=True):
        return self.show(
            'convolved %s at %.0f MeV' % (self.source.name,self.band.energy), 
            self.cvals[self.energy], logscale=True, ax=ax, colorbar=colorbar)
            
    def show_source(self, ax=None, colorbar=True):
        return self.show(
            'Source %s' % (self.source.name),self.dm_vals, 
                logscale=True, ax=ax, colorbar=colorbar)
            
    def show_all(self):
        import pylab as plt
        fig, axx = plt.subplots(1, 3, figsize=(12,4))
        self.show_psf(ax=axx[0], colorbar=False)
        self.show_source(ax=axx[1], colorbar=False)
        self.show_cvals(ax=axx[2], colorbar=False)
        return fig    
class AllSkyDiffuse(Response):
    """Diffuse response based on HEALPix all sky representation, which uses healpy to 
    convolve with the PSF
    """
        
    # defaults = diffuse_grid_defaults
    # @keyword_options.decorate(defaults)
    def __init__(self, source, band, roi, **kwargs):
        # keyword_options.process(self, kwargs)
        self.setup=False
        self.overlap=1
        super(DiffuseResponse, self).__init__(source, band, roi,  **kwargs)
        self.quiet=kwargs.get('quiet', True)
        
    def initialize(self, force=False):
        if self.setup or force: return
        self.setup=True
        #set up the spatial model 
        self.dmodel = self.source.dmodel[self.band.event_type]
        self.dmodel.load()
        self.energy = self.band.energy
        self.dmodel.setEnergy(self.band.energy)
        
        roi_index = skymaps.Band(12).index(self.roicenter)
        self._keyword_check(roi_index)
        
        # self.create_grid()
        # grid = self.grid
        # inside = grid.dists< self.band.radius_in_rad
        # self.ap_average = grid.cvals[inside].mean()
        self.delta_e = self.band.emax - self.band.emin
        self.factor = self.ap_average * self.band.solid_angle * self.delta_e
        if self.band.has_pixels:
            assert False
            # self.pixel_values = grid(self.band.wsdl, grid.cvals) * self.band.pixel_area * self.delta_e

        self.evaluate()
        
    # def create_grid(self):
    #     # create a grid for evaluating counts integral over ROI, individual pixel predictions
    #     grid = self.grid= convolution.ConvolvableGrid(center=self.roicenter, 
    #             npix=self.npix if self.energy>1000 else self.npix2, 
    #             pixelsize=self.pixelsize)
    #     # this may be overridden
    #     self.fill_grid()

            
    # def fill_grid(self):
    #     # fill and convolve the grid
        
    #     self.grid.psf_fill(self.band.psf)
    #     self.grid.bg_fill(self.band.exposure, self.dmodel)
    #     #assert False, 'Break point'
        
    #     #now that the grid is filled with the background model values, perhaps change normalization
    #     # (should be in model? )
    #     if hasattr(self,'corr') and self.corr is not None:
    #         self.grid.bg_vals *= self.corr
    #     self.grid.convolve()
        
    def evaluate(self):
        norm = self.source.model(self.band.energy)
        self.counts = norm * self.factor
        if self.band.has_pixels:
            self.pix_counts = self.pixel_values * norm

    def grad(self, weights, exposure_factor=1): 
        """ contribution to the overall gradient
        weights : arrary of float
            weights = self.data / self.model_pixels
        """
        model = self.spectral_model
        if np.sum(model.free)==0 : 
            return []
        pixterm = ( self.pixel_values * weights ).sum() if self.band.has_pixels else 0
        return (self.factor*exposure_factor - pixterm) * model.gradient(self.energy)[model.free] 
        
    def __call__(self, skydir):
        assert False
        # return self.grid(skydir, self.grid.cvals)[0] * self.delta_e
    
    def _keyword_check(self, roi_index):
        # check for extra keywords from diffuse spec.
        dfun = self.dmodel
        
        if hasattr(dfun, 'kw') and dfun.kw is not None and len(dfun.kw.keys())>1: 
            # Manage keywords found in the 
            if 'correction' in dfun.kw and dfun.kw['correction'] is not None:
                if not self.quiet:
                    print '\t%s loading corrections for source %s from %s.kw:' \
                    % (self.__class__.__name__, self.source.name,  dfun.__class__.__name__)
                corr_file = os.path.expandvars(dfun.kw['correction'])
                self.corr = DiffuseCorrection(corr_file)(roi_index, self.energy) 
            else: 
                self.corr=None
            
            if not self.quiet: print '\tcorrections:{}'.format(self.corr)
            self.systematic = dfun.kw.get('systematic', None)
            if self.systematic is not None:
                if not self.quiet:print '\tsystematic: %.3f' % self.systematic
        else: 
            self.corr =self.systematic = None
            
        #cache this result in the diffuse object
        #self.source.corr = self.corr 
        self.source.systematic = self.systematic

    def rescale(self, scale_factor=1):
        """For tests with modified spectrum
        Adjust the model calculation by applying a scale factor
        Return current correction factor
        (Note the default gives access)
        For changes to be taken into account do a "reinitialize()"
        """
        if scale_factor != 1.:
            self.factor *= scale_factor
            self.pixel_values *= scale_factor
            self.corr *= scale_factor
            self.evaluate()
        return self.corr #new total correction
