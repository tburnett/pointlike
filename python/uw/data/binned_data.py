"""
Manage the sparse pointlike data. 
Duplicates the functionality of the C++ class BinnedPhotonData
Implements the new standard data format
http://gamma-astro-data-formats.readthedocs.io/en/latest/skymaps/healpix/index.html#hpx-bands-table
"""
import os, glob, StringIO, pickle
import healpy
from collections import Counter 
import numpy as np
from astropy.io import fits
import pandas as pd
from uw.utilities import keyword_options

def set_dskeys(header, 
           circle=None, #(ra=162.387, dec=24.200,radius=5),
           emin=100,emax=177.82, # must be set
           event_type=1, #the bit: 1 or 2 for front or back
           zmax=100,  thetamax=66.4, #wired in for pointlike
           ):
    """Set the DS keys in the FITS header for Fermi analysis
    """
    
    dsvals = [      
      ('BIT_MASK(EVENT_CLASS,128,P8R3)','DIMENSIONLESS','1:1'),
      ('TIME',        's','TABLE', ':GTI'),
      ('BIT_MASK(EVENT_TYPE,{},P8R3)'.format(event_type),'DIMENSIONLESS','1:1'),
      ('ENERGY',      'MeV' ,'{}:{}'.format(emin,emax), ),
      ('ZENITH_ANGLE','deg','0:{} '.format(zmax) ,),
      ('THETA',       'deg','0:{} '.format(thetamax),),
             ]
    if circle is not None:
        dsvals = dsvals +[ ('POS(RA,DEC)', 'deg ','CIRCLE({},{},{})'.format(*circle)),]
    header.append(('NDSKEYS', 0))
    for n, x in enumerate(dsvals):
        type, unit, value = x[:3]
        fn = '{:1d}'.format(n)
        header.append( ('DSTYP'+fn,  type))
        header.append( ('DSUNI'+fn,  unit))
        header.append( ('DSVAL'+fn , value))
        if len(x)>3:
             header.append(('DSREF'+fn, x[3]))
    assert n>0
    header['NDSKEYS']=n
    return header

def roi_circle(roi_index, galactic=True, radius=5.0):
    """ return (lon,lat,radius) tuple for given nside=12 position
    """
    from skymaps import Band
    sdir = Band(12).dir(roi_index)    
    return (sdir.l(),sdir.b(), radius) if galactic else (sdir.ra(),sdir.dec(), radius)


class GTI(object):
    def __init__(self, gti_hdu):
        self.hdu=gti_hdu
        data = gti_hdu.data 
        self.start=data.START
        self.stop =data.STOP

    def add(self, other):
        self.start=np.hstack([self.start, other.start])
        self.stop =np.hstack([self.stop, other.stop])
        g = self.hdu
        g.columns['START'].array=self.start
        g.columns['STOP'].array = self.stop

    def make_hdu(self):
        cols = [
            fits.Column('START', format='D', unit='s',array=self.start),
            fits.Column('STOP', format='D', unit='s', array=self.stop)
            ]
        return fits.BinTableHDU.from_columns(cols,header=self.hdu.header,name='GTI')
 
    def __repr__(self):
        return '{} intervals from {:.0f} to {:.0f}, {:,.0f} s'.format(len(self.start), 
            self.start[0], self.stop[-1], sum(self.stop-self.start))
    @property
    def livetime(self):
        return sum(self.stop-self.start)


class BandList(object):
    """The list of bands, defined by energy range and event type
    """
    def __init__(self, bands_hdu):
        self.hdu=bands_hdu
        self.bands=np.asarray(bands_hdu.data)
        if 'COUNT' in bands_hdu.columns.names:
            # old format: need this to parse the pixel data
            self.pixelcnt = np.array(map( lambda n: 0 if n<0 else n,
                 bands_hdu.data.field('COUNT')),int)
        else:
            self.pixelcnt=None

     
    def __repr__(self):
        df = self.dataframe()
        return '{} bands from {:.0f} to {:.0f} MeV'.format(
            len(df), df.e_min[0],df.e_max.max())

    def __getitem__(self, index):
        df = self.dataframe()
        return df.iloc[index]
        
    def make_hdu(self, photons=None, version=3):
        """
        """
        # new format
        df = self.dataframe()
        band_cols = [
            fits.Column(name='NSIDE', format='J', array=df.nside),
            fits.Column(name='E_MIN', format='D', array=df.e_min*1e+3, unit='keV'),
            fits.Column(name='E_MAX', format='D', array=df.e_max*1e+3, unit='keV'),
            fits.Column(name='EVENT_TYPE', format='J', array=df.event_type),
        ]
        bands_hdu=fits.BinTableHDU.from_columns(band_cols, name='BANDS')
        bands_hdu.header.update(VERSION=version)
        return bands_hdu

    def dataframe(self):
        data = self.hdu.data
        if self.pixelcnt is None:
            cdata = [data.E_MIN, data.E_MAX, data.EVENT_TYPE, data.NSIDE]
        else:
            cdata = [self.hdu.data.field(cname) for cname in 'emin emax event_class nside'.split()]
        cdata
        df = pd.DataFrame(cdata, index='e_min e_max event_type nside'.split()).T
        df.e_min /=1e3; df.e_max/=1e3 # convert to MeV from keV
        df['event_type']= df.event_type.astype(int)
        df.nside = df.nside.astype(int)
        return df
    
class Pixels(object):
    """The list of pixels
    Each line is a pixel, with a HEALPix index, a channel index, and the number of photons in the pixel 
    """
    def __init__(self, pixel_hdu, pixel_count=None):
        """pixel_hdu : HDU 
           pixel_count : array of int | None
                number of pixels per band, from the band list in old format
        """
        self.hdu = pixel_hdu
        pixeldata = pixel_hdu.data
        if pixel_count is not None:
            # old format: make a list of channel numbers from the bands HDU
            # the old list of pixels was sorted by channel
            chn= []
            for i,c in enumerate(pixel_count):
                if c<0: continue
                chn = chn + [i]*c 
            self.pix = pixeldata.field('INDEX')
            self.cnt = pixeldata.field('COUNT')
            self.chn = np.array(chn,int)
        else:
            # read new format
            self.chn = pixeldata.field('CHANNEL')  # band index
            self.pix = pixeldata.field('PIX')     # pixel index (depends on channel)
            self.cnt = pixeldata.field('VALUE')   # number of photons in bin
        
        self.counter = None
        self._sorted = False

    def _make_counter(self):
        # Make a Counter, with keys combined from channel and pixel id
        if self.counter is None:
            chn = np.array(self.chn,int) # convert to int for shift
            keys = list(np.left_shift(chn,32) + self.pix)
            self.counter = Counter(dict(zip(keys, self.cnt)))
        return self.counter
 
    def add(self, other):
        """combine the current list of pixels with another
        other : Pixels object
        """
        # combine the pixels by updating the Counter dictionary-like objects
        self._make_counter() # create local Counter only if adding another
        self.counter.update(other._make_counter())
    
    def _decode_counter(self):
        # Update the three arrays following adding another set of data
        assert self.counter is not None, 'logic failure'
        items = np.array(self.counter.items(), int)
        keys = items[:,0]
        self.chn = np.right_shift(keys,32)
        self.pix = np.bitwise_and(keys, 2**32-1)
        self.cnt = items[:,1]

    def dataframe(self):
        """return a DataFrame with number of pixels and photons per channel
        """
        if self.counter is not None:
            self._decode_counter()
        channels = sorted(list(set(self.chn))); 
        d = dict()
        for channel in channels:
            c = self.cnt[self.chn==channel]
            d[channel] = {'pixels': len(c), 'photons': sum(c)}
        df = pd.DataFrame(d).T[['pixels', 'photons']]
        df.index.name='chanel'
        return df

    def __getitem__(self, channel):
        """return a list of (pixel, count) pairs for the band 
        """
        if not self._sorted:
            if self.counter is not None:
                self._decode_counter()
            # sort the list of pixels according to channel number (band)
            # create a lookup dictionary with limits for the pixel and count lists
            csort = self.chn.argsort()
            self.chn = self.chn[csort]
            self.pix = self.pix[csort]
            self.cnt = self.cnt[csort]
            channels = sorted(list(set(self.chn)))
            indexchan = list(np.searchsorted(self.chn, channels))+[len(self.chn)]
            self.lookup = dict(zip(channels,zip(indexchan[:-1], indexchan[1:])))
            self._sorted = True
        try:
            a,b = self.lookup[channel]
            return zip(self.pix[a:b], self.cnt[a:b])
        except KeyError:
            return [] # empty list of no entry

    def make_hdu(self):
        """ create a new HDU in new format
            
        """
        if self.counter is not None:
            # needed it result of combining
            self._decode_counter()
        skymap_cols = [
            fits.Column(name='PIX', format='J',    array=self.pix),
            fits.Column(name='CHANNEL', format='I',array=self.chn),
            fits.Column(name='VALUE', format='J',  array=self.cnt),
        ]
        skymap_hdu=fits.BinTableHDU.from_columns(skymap_cols, name='SKYMAP')
        skymap_hdu.header.update(
            PIXTYPE='HEALPIX',
            INDXSCHM='SPARSE',
            ORDERING='RING',
            COORDSYS='GAL',
            BANDSHDU='BANDS',
            AXCOLS='E_MIN,E_MAX',
            )
        return skymap_hdu
    
    def __repr__(self):
        if self.counter is not None:
            npix, nphot = len(self.counter), sum(self.counter.values())
        else:
            npix, nphot = len(self.cnt), sum(self.cnt)
        return '{}: {:,} pixels, {:,} photons'.format(self.__class__, npix, nphot) 


    
class BinFile(object):
    """ A Binned photon data file
    Manages Two FITS tables: 
        * Pixels
        * BandList
    Implements an indexing interface. Now returns a special Band object
    """
    def __init__(self, filenames, outfile=None, adding=False, quiet=True):
        """
        filenames : a FITS file name, or a list
            if a list, combine them
        outfile : filename | Null [default]
            write the corresponding fits file
        """
        if not hasattr(filenames, '__iter__'):
            filenames = [filenames]
        for i,filename in enumerate(filenames):
            if i==0: # first one: will add others, if any to this one
                if not quiet:print '\n"{}" '.format(filename),
                self.hdus=fits.open(filename)
                self.gti=GTI(self.hdus['GTI'])
                if 'PIXELS' in self.hdus: 
                    # old format
                    self.bands=BandList(self.hdus['BANDS'])
                    self.pixels=Pixels(self.hdus['PIXELS'], self.bands.pixelcnt)
                else:
                    # new format
                    self.bands=BandList(self.hdus['BANDS'])
                    self.pixels=Pixels(self.hdus['SKYMAP'])
                if not quiet: print self.pixels.__repr__(),
            else:
                self.add(BinFile(filename, adding=True))
                if not quiet: print self.pixels.__repr__(),
        if not adding: 
            if not quiet: print

        if outfile is not None:
            self.writeto(outfile)

    def fits_info(self):
        output= StringIO.StringIO()
        self.hdus.info(output)
        return output.getvalue()

    def __repr__(self):
        out = self.fits_info()
        out += 'GTI: '+ self.gti.__repr__()  
        out += '\nBands: '+self.bands.__repr__()
        out += '\nPixels: '+self.pixels.__repr__()
        return out

    def __getitem__(self, index):
        """ return a skymaps.Band C++ object corresponding to the band index
        This object implements query_disk for extracting the pixels within a given ROI.
        """
        from skymaps import Band
        b = self.bands[index]
        bb = Band(int(b.nside), int(b.event_type), b.e_min, b.e_max, 0,0)
        # this is an unfortunate loop: need to consider an interface for adding a set of pixels
        for pix, cnt in self.pixels[index]:
            bb.add(int(pix), int(cnt))
        return bb

    def __len__(self): return len(self.bands.bands)

    def add(self, other):
        """ other : BinFile object
        """
        # combine the pixel and GTI arrays
        self.pixels.add(other.pixels)
        self.gti.add(other.gti)

    def dataframe(self):
        """return a DataFrame with Band info and Pixel summary
        """
        dfb = self.bands.dataframe()
        dfp = self.pixels.dataframe()

        df = pd.concat([dfb,dfp], axis=1) 
        df.index.name='band' 
        # this just to replace NaN's for bands missing in pixel list with zeros
        pixcnt = df.pixels.copy()
        gcnt = df.photons.copy()
        missing = pd.isnull(pixcnt)
        pixcnt[missing]=0
        gcnt[missing]=0
        df.pixels=pixcnt.astype(int)
        df.photons=gcnt.astype(int)

        return df

    def writeto(self, filename, overwrite=True):
        """write to a file

        """
        gti_hdu = self.gti.make_hdu()
        bands_hdu=self.bands.make_hdu() 
        pixels_hdu = self.pixels.make_hdu()
        hdus=[self.hdus[0], pixels_hdu, bands_hdu, gti_hdu]
        fits.HDUList(hdus).writeto(filename, overwrite=overwrite)
        print 'wrote file {}'.format(filename)

    def photonCount(self):
        """ method to be consistent with skymaps.BinnedPhotonData
        """
        return sum(self.pixels.cnt)    

    def roi_subset(self,  roi_number, channel, radius=5):
        """Return a tuple:
            (l,b,radius), nside, DataFrame with data values for the HEALPix pixels within the pointlike ROI
        Creates empty pixels if no data in the pixel (input is sparse, output not)
        """
        skymap = self.hdus['SKYMAP'].data
        nside = self.hdus['BANDS'].data.NSIDE[channel]
        
        # select pixels for the given channel
        def select_pixels(channel):
            sel = skymap.CHANNEL==channel
            return pd.DataFrame( np.array(skymap.VALUE[sel], int), index=np.array(skymap.PIX[sel],int), columns=['value'])


        # the set of pixels in an ROI
        def qdisk(nside, glon, glat, radius):
            return healpy.query_disc(nside, healpy.dir2vec(glon,glat,lonlat=True), np.radians(radius))
        pix=qdisk(nside, *roi_circle(roi_number) )
        roi_pix = pd.DataFrame(np.zeros_like(pix), index=pix, columns=['value'])
        data_pix = select_pixels(channel)
        data_in_roi=list(set(data_pix.index).intersection(set(roi_pix.index)))
        print 'Found {} nside={} data pixels for channel {} in ROI {}'.format(len(data_in_roi),nside, channel, roi_number)
        z = data_pix.loc[data_in_roi]
        data_pix_in_roi = data_pix.loc[data_in_roi]
        roi_pix.value = z.value
        roi_pix[pd.isnull(roi_pix.value)]=0
        return roi_circle(roi_number, galactic=False), nside, roi_pix

    def write_roi_fits(self, filename, roi_number, channel, radius=5, overwrite=True):
        """Write a gtlike-format FITS file with the subset of pixels, for a single channel
        """ 
        circle, nside, pixels = self.roi_subset(roi_number, channel, radius); 
        bandsinfo = self.hdus['BANDS'].data 
        emin, emax = bandsinfo.E_MIN[channel], bandsinfo.E_MAX[channel] 

        # PRIMARY
        primary = self.hdus[0].copy()
        header = primary.header
        set_dskeys(header, circle=circle, emin=emin,emax=emax, 
            event_type= 1<< bandsinfo.EVENT_TYPE[channel]) 
        header['FILENAME']=filename.split('/')[-1]
        header['ORIGIN']='UW software team'
        header['CREATOR'] = 'uw/data/binned_data.py'

        # SKYMAP
        skymap_cols = [
            fits.Column(name='PIX', format='J', null=-2147483648, array=pixels.index ),
            fits.Column(name='CHANNEL1', format='D', array=pixels.value) 
        ]
        skymap_hdu=fits.BinTableHDU.from_columns(skymap_cols, name='SKYMAP')
        skymap_hdu.header.update(
            PIXTYPE='HEALPIX',
            COORDSYS='GAL',
            ORDERING='RING',
            ORDER=int(np.log2(nside)),
            NSIDE=nside,
            FIRSTPIX =0,
            LASTPIX = 12*nside**2-1,
            INDXSCHM='EXPLICIT',)

        skymap_hdu.header['HIERARCH HPXREGION'] =\
             'DISK({:.3f},{:.3f},{:.3f})'.format(*roi_circle(roi_number, radius, galactic=True))
          
        # EBOUNDS
        fudge=1 # 1e6 from GeV?
        ebounds_cols = [
            fits.Column(name = 'CHANNEL', format = 'I', array=[1]),
            fits.Column(name= 'E_MIN', format = '1E', unit = 'keV', array=[emin*fudge]),
            fits.Column(name= 'E_MAX', format = '1E', unit = 'keV', array=[emax*fudge])
        ]
        ebounds_hdu = fits.BinTableHDU.from_columns(ebounds_cols, name='EBOUNDS')

        hdus=[primary, skymap_hdu, ebounds_hdu, self.gti.make_hdu()]
        fits.HDUList(hdus).writeto(filename, overwrite=overwrite)
        print 'Wrote file {}'.format(filename)
    
    def make_map(self, channel, nside=64):
        """Make a map with the given nside (which must be the same as the channel, but no check
        """
        m = np.zeros(12*nside**2, int)
        for i, n in self.pixels[channel]:
            m[i] = n
        return m
        
    
    def generate_ccube_files(self, path, roi_index, channels=range(8), overwrite=False):
        """Write out a set of ccube files for Fermi analysis, 
        path: string
            path to the folder with the files. They will be added to a subfolder 
            given by the roi_index
        roi_index: int
            HEALPix index for nside=12 tesselation, range 0-1727
        channels: list of channel numbers, in range 0-31
        overwrite : bool

        TODO: set maximum nside, downsample if needed
        """
        assert os.path.exists(path)
        roi_folder = os.path.join(path, '{:04d}'.format(roi_index))
        if not os.path.exists(roi_folder):
            os.mkdir(roi_folder)
        for cindex in channels:
            fname = roi_folder+'/ccube_{:02d}.fits'.format(cindex)
            if os.path.exists(fname) and not overwrite:
                print 'File {} exists'.format(fname)
            else:
                self.write_roi_fits(fname, roi_index, cindex)

    def summary_plot(self,  title=None, ax=None,):
        from matplotlib import pyplot as plt

        df = self.dataframe()
        # combine event types
        # f= df.query('event_type==0')
        # b= df.query('event_type==1')

        ee = 0.5*(df.e_min+df.e_max)
        pixels = df.pixels.values
        photons =  df.photons.values
        plt.rc('font', size=14)
        if ax is None:
            fig,ax = plt.subplots(figsize=(8,6))
        ax.loglog(ee, pixels , 'xg', ms=10, label='pixels ({:.1f}M total)'.format(sum(pixels)/1e6))
        ax.loglog(ee, photons, '+r', ms=10, label='photons ({:.1f}M total)'.format(sum(photons)/1e6))
        ax.set(xlabel='Energy [MeV]',ylabel='Number per energy band', title=title)
        ax.grid(alpha=0.4)
        ax.legend()


class ConvertFT1(object):

    defaults=(
        # ('ebins', np.hstack([np.logspace(2,4.5, 11), np.logspace(5,6,3)]),'Energy bin array'),
        # ('levels', [[6,7,8,9]+[10]*10, [6,6,7,8,9]+[10]*9 ], 'healpix level values, etype tuple of band values space'),
        ('ebins', np.logspace(2, 6, 17),'Energy bin array'),
        ('orders', [ [6, 6,7,8,9,10]+[12]*10, [6,7, 8,9,10,11]+[12]*10 ], 'healpix order values, etype tuple of band values space'),
        ('etypes', (0,1), 'event type index'),
        ('theta_cut', 66.4, 'Maximum instrument theta'),
        ('z_cut', 100, 'Maximum zenith angle'),
        ('verbose', 0, 'Verbosity'),
    )

    @keyword_options.decorate(defaults)
    def __init__(self, ft1_file,  **kwargs):
        """
        """
        keyword_options.process(self, kwargs)
        self.ft1_hdus=ft1 = fits.open(ft1_file)
        self.tstart = ft1[1].header['TSTART'] # changed from 0 (redundant?) due to new Ballet FT1-generation

        # extract arrays for values of interest
        data =ft1['EVENTS'].data
        self.glon, self.glat, self.energy, self.et, self.z, self.theta =\
             [data[x] for x in 'L B ENERGY EVENT_TYPE ZENITH_ANGLE THETA'.split()]
        #self.front = np.array([x[-1] for x in self.et],bool) # et is arrray of array of bool, last one true if Front
        # generate event_type masks
        self.et_mask={}
        for et in self.etypes:
            self.et_mask[et]= self.et[:,-1-et]
        self.data_cut = np.logical_and(self.theta<self.theta_cut, self.z<self.z_cut)
        if self.verbose>0:
            print 'Found {} events. Removed: {:.2f} %'.format(len(data), 100.- 100*sum(self.data_cut)/float(len(data)));

        # DataFrame with component values for energy and event type, nside
        t = {}
        band=0
        for ie in range(len(self.ebins)-1):
            for et in self.etypes:
                order = self.orders[et if et<2 else 0][ie] # use max if PSF
                nside = 2**order
                t[band]= dict(ie=ie, event_type=et, nside=nside)
                band+=1
        self.df = pd.DataFrame(t).T

    def cuthist(self):
        import matplotlib.pyplot  as plt
        # plot effect of cuts on theta and zenith angle
        ecut = self.energy>100.
        cos = lambda t: np.cos(np.radians(t))
        fig, axx = plt.subplots(1,2, figsize=(10,4))
        ax = axx[0]
        ct = cos(self.theta)
        ax.hist(ct, np.linspace(0,1,51), histtype='step', lw=2)
        ax.hist(ct[ecut], np.linspace(0,1,51), histtype='step', lw=2, label='E>100 MeV')
        ax.axvline(cos(self.theta_cut), color='red', ls=':',
             label='{:.1f} deg'.format(self.theta_cut))
        ax.set(xlabel='cos(theta)',xlim=(1,0.2))
        ax.legend(pos='upper right')
        ax = axx[1]
        cz = cos(self.z)
        ax.hist(cz, np.linspace(-1,1,51), histtype='step', lw=2);
        ax.hist(cz[ecut], np.linspace(-1,1,51), histtype='step', lw=2,label='E>100 MeV');
        ax.axvline(cos(self.z_cut), color='red', ls='--',
            label='{:.0f} deg'.format(self.z_cut))
        ax.set(xlabel='cos(zenith angle)', xlim=(1,-0.5))
        ax.legend(pos='upper right')

    def binner(self, quiet=True):
        # digitize energy: 0 is first bin above 100 MeV, -1 the underflow.
        eindex = np.digitize(self.energy, self.ebins)-1
        self.pix=[]; self.chn=[];  self.cnt=[]

        if not quiet: print ' ie  et  nside  photons     bins'
        for i,band in self.df.iterrows():
            if not quiet:
                print '{:3} {:3} {:6}'.format( band.ie, band.event_type, band.nside), 
            esel = np.logical_and(eindex==band.ie, self.data_cut)
            sel = np.logical_and(esel, self.et_mask[band.event_type])
            #sel = np.logical_and(esel, self.front if band.event_type==0 else np.logical_not(self.front))
            glon_sel = self.glon[sel]
            glat_sel = self.glat[sel]
            hpindex = healpy.ang2pix(int(band.nside), glon_sel, glat_sel, nest=False, lonlat=True)
            a,b = np.array(np.unique(hpindex, return_counts=True))
            self.pix+=list(a)
            self.cnt+=list(b)
            self.chn+=[i]*len(a)
            if not quiet:
                print '{:8} {:8}'.format(sum(sel), len(a))

    def create_fits(self, outfile='test.fits', overwrite=True):
        elow, ehigh = self.ebins[:-1], self.ebins[1:]
        e_min = np.array([elow[i] for i in self.df.ie])
        e_max = np.array([ehigh[i] for i in self.df.ie])

        band_cols = [
            fits.Column(name='NSIDE', format='J', array=self.df.nside),
            fits.Column(name='E_MIN', format='D', array=e_min*1e+3, unit='keV'),
            fits.Column(name='E_MAX', format='D', array=e_max*1e+3, unit='keV'),
            fits.Column(name='EVENT_TYPE', format='J', array=self.df.event_type),
        ]
        bands_hdu=fits.BinTableHDU.from_columns(band_cols, name='BANDS')

        skymap_cols = [
            fits.Column(name='PIX', format='J',    array=self.pix),
            fits.Column(name='CHANNEL', format='I',array=self.chn),
            fits.Column(name='VALUE', format='J',  array=self.cnt),
        ]
        skymap_hdu=fits.BinTableHDU.from_columns(skymap_cols, name='SKYMAP')
        skymap_hdu.header.update(
            PIXTYPE='HEALPIX',
            INDXSCHM='SPARSE',
            ORDERING='RING',
            COORDSYS='GAL',
            BANDSHDU='BANDS',
            AXCOLS='E_MIN,E_MAX',
                    )
        # add the GTI from the FT1 file and write it out
        hdus = [self.ft1_hdus[0],  skymap_hdu, bands_hdu, self.ft1_hdus['GTI']]
        fits.HDUList(hdus).writeto(outfile, overwrite=overwrite)

    def time_record(self, nside=1024):
        """
        For selected events above 100 MeV, Create lists of the times and healpix ids
        (Reducing size from 20 to 9 bytes)
        returns:
            a recarray with dtype [('band', 'i1'), ('hpindex', '<i4'), ('time', '<f4')]
            where
                band:    energy band index*2 + 0,1 for Front/Back 
                hpindex: HEALPIx index for the nside 
                time:    the elapsed time in s from header value TSTART in the FT1 file
        """
        sel = (self.energy>100) & self.data_cut
        glon_sel = self.glon[sel]
        glat_sel = self.glat[sel]
        self.hpindex = healpy.ang2pix(nside, glon_sel, glat_sel, nest=False, lonlat=True).astype(np.int32)
        
        self.times = self.ft1_hdus['EVENTS'].data['TIME']
        ee = self.energy[sel]
        self.band_index = (2*(np.digitize(ee, self.ebins, )-1) + self.et_mask[1][sel]).astype(np.int8)

        return dict(
            tstart=self.tstart,
            ebins = self.ebins,
            timerec=np.rec.fromarrays([
                    self.band_index, 
                    self.hpindex, 
                    (self.times-self.tstart)[sel].astype(np.float32) ], 
                names='band hpindex time'.split())
        )

def run_binner(monthly_ft1_files='/afs/slac/g/glast/groups/catalog/P8_P305/zmax105/*.fits',
        outfolder='$FERMI/data/P8_P305/monthly',
        overwrite=False, verbose=0, **kwargs):

    files=sorted(glob.glob(monthly_ft1_files))
    assert len(files)>0, 'No ft1 files found at {}'.format(monthly_ft1_files)
    gbtotal = np.array([os.stat(filename).st_size for filename in files]).sum()/2**30
    print '{} FT1 files found, {} GB total'.format(len(files), gbtotal)
    
    outfolder = os.path.expandvars(outfolder)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    os.chdir(outfolder) 
    if verbose>0:
        print 'Writing to folder {}'.format(outfolder)

    for ft1_file in files:
        outfile = ft1_file.split('/')[-1].replace('_zmax105.fits', '_zmax100_4bpd.fits')
        if not overwrite and os.path.exists(outfile):
            print 'File {} exists'.format(outfile)
            continue

        bdt = ConvertFT1(ft1_file, verbose=verbose, **kwargs)
        bdt.binner(quiet=verbose<2)
        bdt.create_fits(outfile)
        if verbose>0: 
            print '\twrote {}'.format(outfile)
        else:
            print '.',
    print 'Done.'

def combine_monthly(
        infolder='$FERMI/data/P8_P305/monthly',
        outfolder='$FERMI/data/P8_P305/yearly',
        overwrite=False, test=False):
    infolder = os.path.expandvars(infolder)
    months = sorted(glob.glob(os.path.join(infolder, '*.fits'))) 
    assert len(months)>0, 'No files found at {}'.format(infolder)
    gbtotal = np.array([os.stat(filename).st_size for filename in months]).sum()/float(2**30)
    print '{} monthly binned files found, {:.1f} GB total'.format(len(months), gbtotal)
    
    outfolder=os.path.expandvars(outfolder)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
        print 'created {}'.format(outfolder)
    os.chdir(outfolder) 
    
    for year in range((len(months)+1)/12):
        t = BinFile(months[12*year])
        outfile = 'P305_Source_year{:02d}_zmax100_4bpd.fits'.format(year+1)
        if not overwrite and os.path.exists(outfile):
            print 'File {} exists'.format(outfile)
            continue
        for m in months[12*year+1:12*year+12]:
            t.add(BinFile(m))
        if not test:
            t.writeto(outfile)
        else:
            print 'Testmode, not writing {}'.format(outfile)

def combine_yearly(
        infolder='$FERMI/data/P8_P305/yearly',
        outfolder='$FERMI/data/P8_P305',
        outfilename='{}years_zmax100_4bpd_v2.fits',
        nyears=10,
        overwrite=False, 
        test=False):
    infolder = os.path.expandvars(infolder)
    years = sorted(glob.glob(os.path.join(infolder, '*.fits'))) 
    assert len(years)>0, 'No files found at {}'.format(infolder)
    gbtotal = np.array([os.stat(filename).st_size for filename in years]).sum()/float(2**30)
    print '{} Yearly binned files found, {:.1f} GB total'.format(len(years), gbtotal)
    
    outfolder=os.path.expandvars(outfolder)
    os.chdir(outfolder) 
    print 'loading {}'.format(os.path.split(years[0])[-1])
    t = BinFile(years[0])
    for year in years[1:nyears]:
        print ' adding {}'.format(os.path.split(year)[-1])
        t.add(BinFile(year))
    outfile = outfilename.format(nyears)
    if not test:
        t.writeto(outfile)
    else:
        print 'Testmode, not writing to {}'.format(outfile)
    return t

