"""
Manage the diffuse sources

$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/diffuse.py,v 1.57 2018/01/27 15:37:17 burnett Exp $

author:  Toby Burnett
"""
import os, types, collections, zipfile, pickle, glob
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import wcs

import skymaps #from Science Tools: for SkyDir, DiffuseFunction, IsotropicSpectrum
from uw.utilities import healpix_map

class DiffuseException(Exception):pass

normalization = None # global for the dict with normalization factors to apply

class DiffuseBase(object):
    """Base class for global diffuse sources
    expect subclasses to implement SkySpectrum interface
    
    This uses a modified singleton pattern: 
    any subclass will have only one instance associated with the constructor argument, usually a filename.
    see http://stackoverflow.com/questions/42558/python-and-the-singleton-pattern
    """
    _instance_dict = dict()
    #def __init__(self, filename, **kwargs):
    #    assert False, 'base class init called'
    #    
    def __new__(cls, filename, **kwargs):
        if filename not in cls._instance_dict:
            cls._instance_dict[filename] = super(DiffuseBase, cls).__new__(cls)
            #print 'diffuse: loaded ', filename
        return cls._instance_dict[filename]
    
    @classmethod
    def clear(self):
        self._instance_dict = {}
        
    def setupfile(self, filename):
        """ filename: string 
        """
        self.filename=filename
        self.fullfilename = os.path.expandvars(filename)
        if not os.path.lexists(self.fullfilename):
            self.fullfilename = os.path.expandvars(os.path.join('$FERMI','diffuse',self.filename))
        assert os.path.lexists(self.fullfilename), 'Diffuse data file "%s" not found' % self.fullfilename
        self.loaded =  False

    def load(self): 
        assert not hasattr(self, 'filename'), 'Logic error; class %s must implement load' %self.name
        pass
    
    def __repr__(self):
        t= self.__class__.__name__ + ': '+self.filename
        if hasattr(self, 'kw') and self.kw is not None:
            t+= '\n'+ '\n'.join(['\t    %-10s: %s' % item for item in self.kw.items() if item[0]!='filename'])
        return t
        
    @property
    def name(self):
        return self.__class__.__name__
        
    def show(self, title=None, scale='log', **kwargs):
        """make an AIT image for testing
        """
        from uw.utilities import image
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        cmap = kwargs.pop('cmap', None)
        ait = image.AIT(self, **kwargs)
        ait.imshow(title=self.name if title is None else title, scale=scale, vmin=vmin, vmax=vmax, cmap=cmap)
        return ait.axes.figure
       
    def plot_spectra(self, ax=None, glat=0, glon=(0, 2,-2,  30, -30, 90, -90),
                erange=None, title=None, label=None):
        """ show spectral for give glat, various glon values """
        from matplotlib import pylab as plt
        if not self.loaded: self.load()
        ee = np.logspace(1.5,6,101) if erange is None else erange
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(5,5))
        else:
            fig = ax.figure
        for x in glon:
            sd = skymaps.SkyDir(glat,x, skymaps.SkyDir.GALACTIC)
            ax.loglog(ee, map(lambda e: e**2*self(sd,e), ee), 
                        label='b=%.0f'%x if label is None else label)
            # if hasattr(self, 'energies'):
            #     et = self.energies
            #     ax.loglog(et, map(lambda e: e**2*self(sd,e), et), '--')
        
        ax.grid(); 
        if len(glon)>1:
            ax.legend(loc='lower left', prop=dict(size=10))
        plt.setp(ax, xlabel=r'$\mathrm{Energy\ [MeV]}$', ylabel=r'$\mathrm{E^2\ df/dE\ [Mev\ cm^{-2}\ s^{-1}\ sr^{-1}]}$')
        ax.set_title('Galactic diffuse at l=%.0f'%glat if title is None else title, size=12)
        return fig
        
    def plot_map(self, energy=1000, title=None, **kwargs):
        """show a map at the given energy"""
        if not self.loaded: self.load()
        self.setEnergy(energy)
        if not kwargs.pop('log', True):
            kwargs['scale']='linear'
        cbtext = 'flux' if kwargs.get('scale', '')=='linear' else 'log10(flux)' 
        cbtext = kwargs.pop('cbtext', cbtext) # override if set
        fig = self.show(cbtext=cbtext, **kwargs)
        fig.axes[0].set_title('%.0f MeV'%energy if title is None else title, size=12)
        return fig

class Isotropic(DiffuseBase): 
    """Implement the isotropic diffuse
    """
    def __init__(self, filename):
        self.setupfile(filename)
        try:
            t = np.loadtxt(self.fullfilename)
        except Exception,msg:
            raise Exception('Fail to load file %s: %s' % (self.fullfilename, msg))
        self.energies=t[:,0]
        self.spectrum = t[:,1]
        self.loaded=True
        self.kw = None
        self.loge = np.log(self.energies)
        self.setEnergy(1000.)
        global normalization
        if normalization is not None and 'iso' in normalization:
            self.normalization = normalization['iso']
        else:
            self.normalization = None  
        
    def load(self):
        pass

    def setEnergy(self, energy): 
        # set up logarithmic interpolation
        self.energy=energy
        x = np.log(energy)
        j = np.searchsorted(self.loge, x) 
        self.energy_index = i = max(0, min(j-1, len(self.energies)-2))
        a,b = self.loge[i:i+2]
        self.energy_interpolation = (x-a)/(b-a)

    def __call__(self, skydir, energy=None):
        if energy is not None and energy!=self.energy: 
            self.setEnergy(energy)
        a,i = self.energy_interpolation, self.energy_index
        return np.exp(    np.log(self.spectrum[i])   * (1-a) 
                        + np.log(self.spectrum[i+1]) * a     ) 

    def plot_spectrum(self, ax=None,  erange=None, title=None, label=None):
        from matplotlib import pylab as plt
        ee = np.logspace(1.5,6,101) if erange is None else erange
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(5,5))
        else:
            fig = ax.figure

        ax.loglog(ee, map(lambda e: e**2*self(None,e), ee), 
                    label= label)
        ax.set( xlabel=r'$\mathrm{Energy\ [MeV]}$', ylabel=r'$\mathrm{E^2\ df/dE\ [Mev\ cm^{-2}\ s^{-1}\ sr^{-1}]}$')

        ax.set_title('Isotropic diffuse' if title is None else title, size=12)
        return fig


class MapCube(DiffuseBase, skymaps.DiffuseFunction):
    """ wrapper for eventual invocation of skymaps.DiffuseFunction, which interprets a map cube
    load() must be called before use
    """
    def __init__(self, filename):
        """ filename: string or dict
        """
        if not self.__dict__.get('loaded', False): #allows for invokation of singleton
            self.setupfile( filename)
            
    def load(self, interpolate=False):
        if  self.loaded: return
        self.loaded=True
        if not interpolate: 
            pass
            #print 'loading diffuse file %s: warning, not interpolating' %self.filename
        super(MapCube,self).__init__(self.filename, 1000., interpolate)
        # get list of energy planes: a C++ function, replace with local data member
        self.energies = np.array(self.energies(), float)


class HealpixCube(DiffuseBase):
    """ Jean-Marc's vector format, or the column version
    """
    def __init__(self,filename):
        """ filename : string
                Name of a FITS file, perhaps a gz. Either vector or column format
        """
        if not self.__dict__.get('loaded', False):
            self.setupfile( filename)
            
    def load(self):
        if self.loaded: return
        print 'Loading HealpixCube file {}'.format( self.filename )
        assert not self.loaded, 'Should have skipped'
        try:
            try:
                self.hdulist = hdus = fits.open(self.fullfilename)
            except Exception, msg:
                raise DiffuseException('FITS: Failed to open {}: {}'.format(self.fullfilename, msg))
            if hdus[2].columns[0].name=='CHANNEL':
                # binned format: assume next 2 columns are min, max and use geometric mean
                emin,emax = [hdus[2].data.field(i) for i in (1,2)]
                self.energies = np.sqrt(emin*emax)
            else:
                self.energies = hdus[2].data.field(0)
            self.vector_mode = len(hdus[1].columns)==1
            if self.vector_mode:
                # one vector column, expect 2d array with shape (12*nside**2, len(energies))
                self.spectra = hdus[1].data.field(0)
                self.nside = int(np.sqrt(self.spectra.shape[0]/12.))
                assert self.spectra.shape[1]==len(self.energies), 'shape inconsistent with number of energies'
            else:
                # one column per energy: expect len(energies) columns
                hdu1 = hdus[1]
                assert len(hdu1.columns)==len(self.energies) , 'wrong number of columns'
                self.data = hdu1.data
                self.nside = int(np.sqrt(self.data.field(0).flatten().shape[0]/12.))
                
            self.loaded=True
            self.indexfun = skymaps.Band(self.nside).index
            self.dirfun = skymaps.Band(self.nside).dir
        except Exception, msg:
            print 'bad file or unexpected FITS format, file %s: %s' % (self.fullfilename, msg)
            raise
        #self.logeratio = np.log(self.energies[1]/self.energies[0])
        self.loge= np.log(self.energies)
        self.setEnergy(1000.)
           
    def close(self):
        self.hdulist.close()

    def __getitem__(self, index):
        """return a plane as a HParray object"""
        energy = self.energies[index]
        col = self.column(energy)
        return healpix_map.HParray('{:.0f} MeV'.format(energy), col)
    def __len__(self):
        return len(self.energies)
        
    def __iter__(self): # iterate over all keys
        for index in range(len(self)):
            yield self[index]

    def __call__(self, skydir, energy=None):
        if energy is not None and energy!=self.energy: 
            self.setEnergy(energy)
        skyindex = self.indexfun(skydir)
        a = self.energy_interpolation
        u, v = self.eplane1[skyindex], self.eplane2[skyindex]
        if np.abs(a) < 1e-2 or v<=0 or np.isnan(v):
            ret = u
        elif np.abs(1-a)< 1e-2 or u<=0 or np.isnan(u):
            ret = v
        else:
            ret = np.exp( np.log(u) * (1-a) + np.log(v) * a      )
        assert np.isfinite(ret), 'Not finite for {} at {} MeV, {},{},{}'.format(skydir, self.energy, a, u,v)
        if ret<=0:
            #print 'Warning: FLux not positive at {} for {:.0f} MeV a={}'.format(skydir, self.energy,a)
            ret = 0
        return ret

    def setEnergy(self, energy): 
        # set up logarithmic interpolation
        if not self.loaded:
            self.load()
        self.energy=energy
        #r = np.log(energy/self.energies[0])/self.logeratio
        # get the pair of energies
        if energy< self.energies[0]: i=0
        elif energy>self.energies[-1]: i= len(self.energies)-2
        else:
            i = np.where(self.energies>=energy)[0][0]-1
         
        a,b = self.loge[i], self.loge[i+1]
        self.energy_index = i #= max(0, min(int(r), len(self.energies)-2))
        self.energy_interpolation = (np.log(energy)-a)/(b-a)
        if self.vector_mode:
            self.eplane1 = self.spectra[:,i]
            self.eplane2 = self.spectra[:,i+1]
        else:
            self.eplane1 = np.ravel(self.data.field(i))
            self.eplane2 = np.ravel(self.data.field(i+1))
            
    def column(self, energy):
        """ return a full HEALPix-ordered column for the given energy
        """
        self.setEnergy(energy)
        a = self.energy_interpolation
        if a<0.002:
            return self.eplane1
        elif a>0.998:
            return self.eplane2
        return np.exp( np.log(self.eplane1) * (1-a) 
             + np.log(self.eplane2) * a      )
    @property        
    def array(self):
        """return data as a 2-D Numpy array"""
        assert False, 'Better not try this?'
        if not self.loaded: self.load()
        data = self.hdulist[1].data
        return np.array([row[0] for row in data])
        
class FitsMapCube(DiffuseBase):
    """Interpret a FITS layered image, using astropy fits and wcs
    """
    def __init__(self,filename):
        if not self.__dict__.get('loaded', False): #allows for invokation of singleton
            self.setupfile( filename)
            
    def load(self, interpolate=False):
        if  self.loaded: return
        self.loaded=True
        # Load the FITS hdulist using astropy.io.fits
        self.hdulist =hdulist = fits.open(self.fullfilename)
        #print hdulist.info()

        # Parse the WCS keywords in the primary HDU
        self.w = wcs.WCS(hdulist[0].header)
        self.galactic = self.w.axis_type_names[0]=='GLON'
        self.naxis = self.w._naxis[:2]

        # Refer to the layered image and the energy table
        self.img=hdulist[0].data
        self.energies = hdulist[1].data['Energy']
        self.loge = np.log(self.energies)
        self.setEnergy(1000.)

        
    def setEnergy(self, energy): 
        # set up logarithmic interpolation
        if not self.loaded:
            self.load()
        self.energy=energy
        if energy< self.energies[0]: i=0
        elif energy>self.energies[-1]: i= len(self.energies)-2
        else:
            i = np.searchsorted(self.energies,energy)-1
        self.energy_index=i
        a,b = self.loge[i], self.loge[i+1]
        self.energy_interpolation = (np.log(energy)-a)/(b-a)
        self.eplane1=self.img[i]
        self.eplane2=self.img[i+1]
        
    def skydir2pix(self, skydir):
        """ return a tuple for indexing into image plane (note it will be int"""
        ra, dec =(skydir.ra(), skydir.dec()) if not self.galactic else (skydir.l(), skydir.b())
        # note 0-based indexing for array access
        return np.array(self.w.wcs_world2pix([[ra,dec,1]], 0)[0][:2],int)
    
    def pix2skydir(self, pixel):
        """ return a skydir from an image index tuple
        """
        i,j = pixel
        c= self.w.wcs_pix2world(np.array([[i,j,2]]),0)[0][:2];
        if self.galactic:
            return SkyDir(c[0],c[1], SkyDir.GALACTIC)
        return SkyDir(*c)

    def __call__(self, skydir, energy=None):
        if energy is not None and energy!=self.energy: 
            self.setEnergy(energy)
        pix= self.skydir2pix(skydir)
        if np.any(pix<0) or np.any( pix>=self.naxis): return 0
        i,j = pix; skyindex=(j,i)
        a = self.energy_interpolation
        f1, f2=self.eplane1[skyindex], self.eplane2[skyindex]
        if f1==0 or f2==0: return 0
        if np.abs(a)<1e-2:
            ret = f1
        elif np.abs(1-a)< 1e-2:
            ret = f2
        else:
            ret = np.exp( np.log(f1) * (1-a) + np.log(f2) * a  )
        
        return ret

class FitsMapCubeList():
    def __init__(self, filename):
        filenames = open(filename).read().split('\n')
        assert len(filenames)>1, 'Expected more than one filename:\n{}'.format(filenames)
        self.cubelist = map(FitsMapCube, filenames)
        self.loaded=False
    def load(self):
        self.loaded=True
        for cube in self.cubelist:
            cube.load()
    def __call__(self, skydir, energy=None):
        return sum([cube(skydir, energy) for cube in self.cubelist])
    def __repr__(self):
        r =self.__class__.__name__ +': Sum of: '
        for cube in self.cubelist:
            r+='\r\t'+cube.__repr__()
        return r
    def setEnergy(self, energy):
        for cube in self.cubelist:
            cube.setEnergy(energy)

def AllSkyDiffuse(HealpixCube):
    """special version adding convolvability
    """
    def __init__(self,filename):
        super(AllSkyDiffuse,self).__init__(filename)

def make_healpix_spectral_cube(spectra, energies, filename=None):
    """
    Generate a FITS format spectral cube from HEALPix data 
    
        spectra : array with shape (12*nside**2, nenergies) 
        energies : array, length nenergies
        filename : if not None, write the result
    
    return the HDU list
    """
    naxis2 =spectra.shape[0]
    nebins = len(energies)
    assert nebins== spectra.shape[1] , 'Inconsistent length of energy list'
    nside = int(np.sqrt(naxis2/12))
    assert  12*nside**2 == naxis2, 'Size not consistent with HEALPix'
    spec_table = fits.BinTableHDU.from_columns([
            fits.Column(name='Spectra', format='{:2d}D'.format(nebins), unit='Intensity', array=spectra)
        ])
    spec_table.name = 'SKYMAP'
    spec_table.header['PIXTYPE']='HEALPIX'
    spec_table.header['ORDERING']='RING'
    spec_table.header['NSIDE'] = nside
    spec_table.header['FIRSTPIX'] = 0
    spec_table.header['LASTPIX'] = naxis2-1
    spec_table.header['NBRBINS']=(nebins, 'Number of energy bins')
    spec_table.header['EMIN'] = (energies[0],'Minimum energy')
    spec_table.header['DELTAE']= (np.log(energies[1]/energies[0]), 'Step in energy (log)')

    energy_table = fits.BinTableHDU.from_columns([
            fits.Column(name='Energy', format='D', unit='MeV',array=energies)
        ])
    energy_table.name = 'ENERGIES'
    hdus = [fits.PrimaryHDU(header=None), spec_table, energy_table] 
    hdulist = fits.HDUList(hdus)
    if filename is not None:
        if os.path.exists(filename):
            os.remove(filename)
        hdulist.writeto(os.path.expandvars(filename))
    return hdulist

def make_text_spectrum(spectrum, energies, filename):
    """
    Generate a text file table with the energies and spectrum, e.g., for isotropic
    """
    np.savetxt(filename, np.asarray([energies, spectrum]).T, fmt='%.3e')

class Healpix(DiffuseBase):
    """Diffuse map using HEALPix representation.
    Presumes that columns have eneregies (found in hdu#3) which exactly
    correspond to bands

    """
    def __init__(self, filename):
        self.setupfile(filename)
        self.energy=0
        self.eindex=-1
        self.fits = None
        
    def load(self):
        if self.fits is not None: return
        self.fits = fits.open(self.filename)
        table = self.fits[1]
        self.columns = table.data
        self.col_names = [t.name for t in table.columns]
        self.energies = self.fits[2].data.field('MeV')

    def setEnergy(self, energy):
        self.load()
        if energy==self.energy: return
        self.eindex = -1
        for i,e in enumerate(self.energies):
            if abs(e-energy)<0.01*e:
                self.eindex= i
                self.energy=energy
                break
        assert self.eindex>=0, 'Energy %.0f not found' % energy
        cname = self.col_names[self.eindex]
        self.hpm = healpix_map.HParray(cname, self.columns.field(cname))
        
    def __call__(self, skydir, energy=None):
        if energy is not None and energy !=self.energy: self.setEnergy(energy)
        return self.hpm(skydir)
        
    @property
    def name(self):
        return '%s %s' % (self.__class__.__name__, 
            self.col_names[self.eindex] if self.eindex>-1 else 'not loaded')
    
class IsotropicSpectralFunction(DiffuseBase):
    """ wrapper for using a standard spectral function with isotropic
    """
    def __init__(self, expression):
        """
        expression: 
        """
        try:
            self.expression  = expression.split('_')[-1]
            self.spectral_function = eval(self.expression)
        except Exception, msg:
            print 'Failure to evaluate IsotropicSpectralFunction %s : %s' % (self.expression, msg)
            raise
        self.energy=1000
        self.loaded=True
    def __repr__(self):
        return '%s: %s' % (self.__class__.__name__, self.expression )
    def __call__(self, skydir, energy=None):
        return self.spectral_function(self.energy if energy is None else energy)
    def setEnergy(self, energy): self.energy=energy


class PieceWise(object):
    def __init__(self, u):
        a,b = u
        self.a, self.b =a,b
        self.n = len(a)
        self.s = [(b[i+1]-b[i])/(a[i+1]-a[i]) for i in range(self.n-1)]
    def __call__(self, x):
        if x<=self.a[0]: return self.b[0]
        for i in range(self.n-1):
            if x<self.a[i+1]:
                return self.b[i]+(x-self.a[i])*self.s[i]
        return self.b[-1]

class AziLimb(IsotropicSpectralFunction):
    """ Azimuthally symmetric Limb function
    """
    def __init__(self, filename):
        """ file ia a dict like:
        {
            'spectrum' : 'PowerLaw(1e-11, 4.0)',
            'pieces' : [[-1., -0.4, 0.4, 1.0], [0.75, 0, 0, 0.75]],
        }
        
        """
        self.setupfile(filename)
        try:
            txt =open(self.fullfilename).read()
            self.loaded=True
            adict = eval(txt)
            self.loaded=True
            self.energy=100
        except Exception, msg:
            raise Exception('Failure to interpret Limb parameter file "%s": %s'\
                            %(self.fullfilename, msg))
        self.limbfun = PieceWise(adict['pieces'])
        super(AziLimb, self).__init__(adict['spectrum'])

    def load(self): pass
    
    def __repr__(self):
        return '%s: %s, North,South=%.1f,%.1f' % (self.__class__.__name__, self.expression, self.limbfun(1),self.limbfun(-1) )
        
    def __call__(self, skydir, energy=None):
        spec = self.spectral_function(self.energy if energy is None else energy)
        dec = skydir.dec() # only depends on DEC
        return spec * self.limbfun( np.sin(np.radians(dec))  ) 

class GulliLimb(DiffuseBase):
    """Implement the (internal?) text version of Gulli's RA-independent diffuse
    """

    def __init__(self, filename):
        self.setupfile(filename)
        colnames='lower_dec   upper_dec   prefactor   index   beta   prefactor_error  index_error  beta_error'.split()
        try:
            df = pd.read_table(self.fullfilename, sep=r'\s*',skiprows=1, header=None, 
                 names=colnames)
            assert len(df)==90, 'Expect 90 rows, 2-degree increments'
            self.dec_index = lambda dec: max(0, min(89, (int((dec+90.)/2.))))
            
            class Spectrum(object):
                """ log-parabola """
                def __init__(self, i):
                    self.pref,self.index,self.beta = df.prefactor[i], df['index'][i], df.beta[i]
                def __call__(self, e100):
                    return self.pref * e100 ** (self.index + self.beta*np.log(e100))
            self.spec_fun = map(Spectrum, range(90))

        except Exception, msg:
            raise Exception('Failure to interpret Limb parameter file "%s": %s'\
                            %(self.fullfilename, msg))
        self.energy=1000
        self.loaded=True

    def setEnergy(self, energy):
        self.energy = energy
    def load(self): pass
    
    def __call__(self, skydir, energy=None):
        dindex = self.dec_index(skydir.dec())
        if energy is None: energy=self.energy
        spec = self.spec_fun[dindex]( energy /100.)
        return spec 


class CachedMapCube(DiffuseBase):
    """ for compatibility with previous models"""
    def __init__(self, zipfilename):
        self.setupfile(zipfilename)
    
    def load(self):
        z = zipfile.ZipFile(self.filename)
        t = z.namelist()
        if len(t)==1729: # if ziped with foldername
            t = t[1:]
        self.files = sorted(t) 
        assert len(self.files)==1728, 'wrong number of files: expected 1728, found %d' % len(self.files)
        self.opener = z.open

    def __repr__(self):
        return '%s.%s: %s' % (self.__module__, self.__class__.__name__, self.filename)
    #def grid_generator(self, band): #skydir, psf, exposure):
        #"""Return a GridGenerator object
        # """
        #        
        #return convolution.CachedGridGenerator(band, band.sd, self) #psf, exposure, skydir,self, quiet=False, **self.kw)

    # these not implemented, and not needed for this class
    def __call__(self, skydir, energy=None):
        return np.nan
    def setEnergy(self, energy):
        pass

class DiffuseList(list):
    """A list of  event type list of DiffuseBase objects. If only one, applied to all
    """
    def __init__(self, inlist, event_type_names):
        super(DiffuseList,self).__init__(inlist)
        self.gg = None
        self.event_type_names = event_type_names
    def __repr__(self):
        return  '%s.%s\n\t' % (self.__module__, self.__class__.__name__) + '\n\t'.join([t.__repr__() for t in self])
        
    def __getitem__(self, index):
        if len(self)==1: index=0
        try:
            index = ['front','back',
                     'psf0','psf1','psf2','psf3',
                     'edisp0','edisp1','edisp2','edisp3'].index(index)
        except ValueError:
            pass
        if index>1: index -=2 # 
        assert index<4, '{} : bad index {}'.format(self.__class__,index)
        return super(DiffuseList,self).__getitem__(index) 
    def load(self):
        for x in self:
            x.load()
    @property
    def type(self):
        """return the diffuse class name implementing the global source"""
        return self[0].__class__.__name__
        
    def plot_spectra(self, glat=0, glon=(0, 2,-2,  30, -30, 90, -90), title=None):
        import pylab as plt
        fig,ax = plt.subplots(1,1, figsize=(5,5))
        for i,ds in enumerate(self):
            ds.plot_spectra(ax=ax, glat=glat, glon=glon, label=self.event_type_names[i], title=title)
        ax.grid()
        ax.legend(prop=dict(size=10))
        return fig

class IsotropicList(DiffuseList):
    def __init__(self, adict, event_type_names):
        """ Create a list of Isotropic functions, one per entry in event_type_names
            adict: dict
                contains filename, correction, each with wildcards to be expanded
            event_type_names : list of string
               
        """
        try:
            fn, corr, key = adict['filename'], adict.get('correction', None), adict.get('key', None)
        except Exception, msg:
            print "Fail to create IsotropicList: adict=%s, %s" % (adict, msg)
            raise
        for et in event_type_names:
            filename = fn.replace('**', et.upper()).replace('*', et)
            #print fn, filename, et
            correction_data=correction=None
            if corr is not None:
                correction = corr.replace('*', et)
                # if diffuse_normalization is not None and correction in diffuse_normalization:
                #     correction_data = diffuse_normalization[correction]
            ext = os.path.splitext(filename)[-1]
            if ext=='.txt':   iso = Isotropic(filename) 
            elif ext=='.fits': iso = HealpixCube(filename)
            else:
                raise Exception('Unrecognized extension: {}'.format(ext))
            iso.kw = dict(correction =correction, correction_data=correction_data, key=key)
            self.append(iso)
def ConvolvedList(IsotropicList):
    pass
def file_check(files):
    full_files = map( lambda f: os.path.expandvars(os.path.join('$FERMI','diffuse',f)), files)
    check = map(lambda f: os.path.lexists(f) or f[-1]==')', full_files) 
    if not all(check):
        raise DiffuseException('not all diffuse files %s found' % full_files)
    


def expand_file(filename):
    if os.path.lexists(filename):
        return os.path.join(os.getcwd(),filename)
    return os.path.join(os.path.expandvars('$FERMI/diffuse/'),filename)

def etype_insert(value, et):
    v = value.copy()
    for k,x in v.items():
        if type(x)!=str: continue
        x=x.replace('**', et.upper())
        x=x.replace ('*', et)
        v[k]=x
    try:
        g = eval(v['type'])(v['filename'])
    except Exception, msg:
        print 'Fail in instantiate: {} {}'.format(v,msg)
        raise
    g.kw = v
    return g
        
def diffuse_factory(value, diffuse_normalization=None, event_type_names=('front', 'back')):
    """
    Create a DiffuseList object from a text specification. Called from from_healpix to set the 
    value : [string | list | dict ]
        if string: a single filename to apply to all event types; if contains a '*', expand it
            to a list with the list of event type names, e.g., front, back
        if list: a set of filesnames corresponding to the list of event types
        if dict: must have keyword "filename", may have "type" to specify the type, 
            which must be the name of a class in this module inheriting from DiffuseBase
    
        If the keyword "type" is not specified, filenames are examined for extensions:
            txt : Isotropic
            zip : CachedMapCube
            fit for fits : MapCube
        A special case is ')' : IsotropicSpectralFunction
    diffuse_normalization : None | dict
        If dict, a dictionary with keys 'gal', 'iso'
    """
    assert value is not None, 'oops'
    # make the diffuse normalization available to classes 
    global normalization
    normalization = diffuse_normalization

    isdict = issubclass(value.__class__, dict)
    preconvolved = isdict and value.get('preconvolved', False)
    type = None
    if isinstance(value, str):
        if ':' in value:
            type,value = value.split(':')
        if '**' in value:
            value = [value.replace('**', et.upper()) for et in event_type_names]
        elif '*' in value:
            value = [value.replace('*',et) for et in event_type_names]
         
    if not hasattr(value, '__iter__') or isdict and not preconvolved:
        value  = (value,)

 
    if isdict:
        # for new convolved:
        if preconvolved:
           z = [etype_insert(value, et) for et in event_type_names] 
           return DiffuseList(z, event_type_names=event_type_names)

        try:
            files = DiffuseList([val['filename'] for val in value], event_type_names)
        except KeyError:
            raise DiffuseException('expected "filename" key in dict')
        type = value[0].get('type', None)
        kws = value
    else:
        files = DiffuseList(value,event_type_names)
        kws =  None

    # checking only first element, and only upto a comma, if not with '('
    f = files[0].split(',')[0] if files[0].find('(')<0 else files[0]
    ext = os.path.splitext(f)[-1]
    if type is not None:
        try:
            dfun = eval(type)
        except Exception, msg:
            raise DiffuseException('Diffuse type specification "%s" failed: %s'%(type, msg))
    else:
        # inferr class to use from file type
        try:
            dfun = {'.txt': Isotropic, 
                '.fit': MapCube, '.fits': MapCube,
                '.zip': CachedMapCube,
                ')': IsotropicSpectralFunction, 
                }[ext if ext[-1]!=')' else ')']
        except Exception, msg:
            raise DiffuseException('File type, "%s", for diffuse not recognized, from "%s":%s (message :%s)'\
                % (ext, files, ext, msg))
    
    if dfun==IsotropicSpectralFunction or dfun==ConvolvedList:
        diffuse_source = map(dfun,files)
    elif dfun==IsotropicList:
        # special code to handle Isotropic list in a dict with wild cards
        return IsotropicList(value[0], event_type_names=event_type_names)
    else:
        full_files = map( expand_file, files)
        check = map(lambda f: os.path.lexists(f) or f[-1]==')', full_files) 
        if not all(check):
            raise DiffuseException('not all diffuse files %s found' % full_files)
        diffuse_source= map(dfun, full_files) 
    # if a dict, add keywords to the objects
    if kws is not None:
        for x,kw in zip(diffuse_source, kws):
            x.kw = kw
    return DiffuseList(diffuse_source, event_type_names=event_type_names)

class Normalization(object):
    """ Manage diffuse normalization, or correction factors in the ROI
    """
    def __init__(self, roi):

        """ Check noralization correction factors loaded into the Response objects
            To display as a plot:
            diffuse.Normalization(roi).plot()
        """
        self.roi=roi
 
    def __call__(self):
        energies = [int(round(x.band.energy)) for x in self.roi[:16:2]]
        gal = [x[0].corr for x in self.roi[:16:2]]
        assert gal == [x[0].corr for x in self.roi[1:16:2]], 'Expect same gal for front, back'
        t=np.array([
            gal,
            [x[1].corr for x in self.roi[:16:2]],
            [x[1].corr for x in self.roi[1:16:2]],
            ])
        return pd.DataFrame(t, index='gal iso_f iso_b'.split(), columns=energies)

    def plot(self):
        self().T.plot.line(logx=True)

def create_corr_dict(diffuse_dict,  roi_index, event_type_names=('front','back')):
    from uw.like2 import response
    corr_dict = {}

    galf = diffuse_dict['ring'].get('correction',None)
    corr_dict['gal'] = np.ones(8) if galf is None else response.DiffuseCorrection(galf).roi_norm(roi_index)

    isof =  diffuse_dict['isotrop']['correction']
    corr_dict['iso']= dict()
    for x in event_type_names:
        isoc = response.DiffuseCorrection(isof.replace('*',x));
        corr_dict['iso'][x]= isoc.roi_norm(roi_index)
    return corr_dict


# create new diffuse map modified by 


def apply_patch(inpat, patchfile, outpat, event_type_names=('front','back')):
    """Apply a multiplicative patch to a set of event type diffuse cubes

    inpat : string
        File path pattern, with "*" wildcard, to input diffuse cubes
    pathchfile : string
        file name for patch cube
    outpat : string
        File path pattern, with "*" wildcard, to output diffuse cubes

    """
    def modify_by_patch(self, patch, newfile=None):
        modlayer=[]
        for i,layer in enumerate((self)):
            name = layer.name
            a = layer.vec
            if i < len(patch):
                p = patch[i]
            else:
                p = patch[-1]
            b = p.vec
            print layer.name, ',',
            modlayer.append(healpix_map.HParray(layer.name, healpix_map.multiply(a,b)))
        print
        outmap=healpix_map.HEALPixFITS(modlayer, energies=self.energies)
        if newfile is not None: outmap.write(newfile)
        return outmap

    print 'Applying {} to {}'.format(patchfile, inpat)
    patch = HealpixCube(patchfile); patch.load()
    for iet in event_type_names:
        print iet, ':',
        df = HealpixCube(inpat.replace('*', iet)); df.load()
        modify_by_patch(df, patch, outpat.replace('*', iet))    