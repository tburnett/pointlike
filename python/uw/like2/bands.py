"""
manage band classes
"""
import os
import numpy as np
import skymaps
import healpy

#energybins = np.logspace(2,5.5,15) # default 100 MeV to 3.16 GeV, 4/decade
energybins = np.logspace(2,6,17) # 100 MeV to 1 TeV, 4/decade
#event_type_min_energy=[100, 100, 1000, 300, 100, 100 ] # minimum energy for given event type
# only FRONT below 300 MeV for now
event_type_min_energy=[100, 300, 1000, 300, 300, 100 ] # minimum energy for given event type

class EnergyBand(object):
    """ Combine three concepts:
    * definition of the binning in solid angle and energy
    * PSF and exposure for the given configuration, appropriate to this band
    * The pixel data extracted from the binned photon data
    """
    def __init__(self, config,  roi_dir, 
            radius=5, event_type=1, emin=10**2, emax=10**2.25):
        """
        
        """
        # define bin boundaries

        self.skydir=roi_dir 
        self.radius =radius
        self.event_type = event_type
        self.emin, self.emax = emin, emax
        self.energy = energy =  np.sqrt(emin*emax)
        
        # save appropriate psf and exposure
        if config.use_old_irf_code: 
            self.psf=config.psfman(event_type, energy)
            self.exposure = config.exposureman(event_type, energy)
        else:
            self.psf = config.irfs.psf(event_type,energy)
            self.exposure = config.irfs.exposure(event_type,energy)
         # used by client to integrate a function of energy over exposure
        self.integrator = self.exposure.integrator(self.skydir, self.emin, self.emax) 
  
        # these changed if data loaded -- see load_data
        self.pixel_area=0
        self.wsdl = None
        self.pix_counts=[]
        #print 'psr and exposure OK for {}'.format(self.title)


    def __repr__(self):
        return '%s.%s: %s' % (self.__module__,self.__class__.__name__, self.title)
    def set_energy(self, energy):
        self.psf.setEnergy(energy)
        self.exposure.setEnergy(energy)
        self.energy=energy
    def load_data(self, cband):
        # cband is a C++ skymaps.Band object
        self.wsdl = skymaps.WeightedSkyDirList(cband, self.skydir, self.radius_in_rad, False)
        self.pix_counts = np.asarray([x.weight() for x in self.wsdl]) if len(self.wsdl)>0 else []
        self.pixel_area = cband.pixelArea()
        self.cband=cband
    @property
    def radius_in_rad(self): return np.radians(self.radius)
    @property #alias, for compatibilty, but deprecated
    def sd(self): return self.skydir
    @property
    def solid_angle(self):
        sd = self.skydir
        try:
            nside = self.cband.nside()
            npix= len(healpy.query_disc(nside, healpy.dir2vec(sd.l(),sd.b(),lonlat=True), self.radius_in_rad))
            return npix*self.pixel_area
        except ValueError:
            return np.pi*self.radius_in_rad**2 
    @property
    def has_pixels(self): return self.wsdl is not None and self.pixels>0
    @property 
    def pixels(self):
        return len(self.wsdl) if self.wsdl is not None else 0
    @property
    def events(self):
        return sum(w.weight() for w in self.wsdl) if self.wsdl is not None else 0
    @property
    def title(self):
        ret = '{:.0}-{:.0} MeV event_type {}'.format(self.emin, self.emax, self.event_type)
        if self.wsdl is not None:
            ret += ', {} pixels, {} events'.format(self.pixels, self.events)
        return ret
        
class BandSet(list):
    """ manage the list of EnergyBand objects
    """
    def __init__(self, config, roi_index, load=False, radius=5):
        """create a list of energy bands
        
        config : configuration.Configuration object
        roi_index : int | None
            specify direction, either as nside=12 index, or from config.roi_spec
        load : bool
            if False, do not load data into the pixels
        """
        global event_type_min_energy
        self.config = config
        emins = self.config['input_model'].get('emin', None)
        if emins is not None:
            assert len(emins)==2 or len(emins)==6, 'expect either 2 or 6'
            # change default 
            event_type_min_energy = emins
        if roi_index is None or roi_index<0:
            self.roi_dir = config.roi_spec.pos
            self.radius = config.roi_spec.radius
            roi_index = None
        else:
            ### HEALPix case
            self.roi_index = roi_index
            self.roi_dir = skymaps.Band(12).dir(roi_index) # could be defined otherwise
            self.radius=radius
        for emin, emax  in zip(energybins[:-1], energybins[1:]):
            for et in config.dataset.event_types:
                if emin<event_type_min_energy[et]: continue
                self.append(EnergyBand(config, self.roi_dir,  event_type=et, radius=self.radius, emin=emin,emax=emax))
        self.has_data = False
        
        if load:
            self.load_data()
    
    def __repr__(self):
        ret = '%s.%s : %d bands %d-%d MeV for ROI %d' % (
            self.__module__,self.__class__.__name__,
            len(self), self[0].emin, self[-1].emax, self.roi_index)
        if self.has_data:
            ret += ', %d pixels, %d events' % (self.pixels, self.events)
        else:
            ret += ', NO DATA!'
        return ret
            
    def find(self, emin, event_type):
        for band in self:
            if round(emin)==round(band.emin) and event_type==band.event_type:
                return band
        raise Exception('Band with emin, event_type = {}, {} not found in {}'.format(round(emin), event_type, self))
    
    def load_data(self):
        """load the current dataset, have the respective bands fill pixels from it
        """
        dset = self.config.dataset
        dset.load()
        found = 0
        for i,cband in enumerate(dset.dmap):
            emin, emax, event_type =  cband.emin(), cband.emax(), cband.event_class()
            if event_type == -2147483648: event_type=0 # previous bug in FITS setup
            try:
                band = self.find(emin, event_type)
                found +=1
            except Exception:
                continue
            band.load_data(cband)
            band.data_index = i # useful to look up associated index
        if found!=len(self):
            print '{}: Loaded subset of bands {} instead of {}'.format( self.__repr__(), found, len(self))
            self[:]= self[:found] 
            self.config.emax=self[-1].emax
        self.has_data = True

    @property
    def pixels(self): return sum( band.pixels for band in self) if self.has_data else 0
    @property
    def events(self): return sum( band.events for band in self) if self.has_data else 0
        
