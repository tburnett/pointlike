"""
Utilities for managing Healpix arrays
$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/pipeline/pub/healpix_map.py,v 1.7 2011/08/11 21:13:15 burnett Exp $
"""
import os,glob,pickle, types, copy
import pylab as plt
import numpy as np
import pyfits
from skymaps import Band, SkyDir, Hep3Vector, PySkyFunction, SkyImage, Healpix
import skymaps
from pointlike import IntVector
from uw.utilities import image
#from uw.like import pycaldb
#from .. import skymodel # for DisplayMap

class HParray(object):
    """ base class, implement a HEALPix array, provide AIT plot
    """
    def __init__(self, name, vec): 
        self.name=name
        self.vec = vec
        self.nside = int(np.sqrt(len(vec)/12))
        assert len(self.vec)==12*self.nside**2, 'length of %s not consistent with HEALPix' % self.name
    def __getitem__(self, index):
        return self.vec[index]
    def __len__(self): return 12*self.nside**2
    def getcol(self, type=np.float32): return np.asarray(self.vec, type)
    def skyfun(self, skydir):
        return self[Band(self.nside).index(skydir)]
        
    def plot(self, title='', axes=None, fignum=30, ait_kw={}, **kwargs):
        """ make an AIT skyplot from the array 
        title : string
            set the figure title
        ait_kw : dict
            to set kwargs for image.AIT, perhaps pixelsize
        
        Other args passed to imshow
        """
        band = Band(self.nside)
        def skyplotfun(v):
            skydir = SkyDir(Hep3Vector(v[0],v[1],v[2]))
            index = band.index(skydir)
            return self[index]
        if axes is None:
            plt.close(fignum)
            fig = plt.figure(fignum, figsize=(12,6))
        ait=image.AIT(PySkyFunction(skyplotfun) ,axes=axes, **ait_kw)
        ait.imshow(title=title, **kwargs)
        return ait

    def smooth(self, a=0.6):
        """ simple-minded smooth using nearest neighbors 
        """
        newvec = [0]*len(self.vec)
        band = Band(self.nside)
        iv = IntVector() # needed for findNeighbors: first 4 share sides
        b=0.25*(1-a)
        for i in range(len(self.vec)):
            band.findNeighbors(i,iv)
            newvec[i] = a*self.vec[i] + b*sum([self.vec[j] for j in iv[:4]])
            self.vec= newvec
    def average(self, radius=1.0):
        """ average over all pixels within radius"""
        iv = IntVector()
        hp = Healpix(self.nside, Healpix.RING, SkyDir.GALACTIC)
        band = Band(self.nside)
        newvec = [0]*len(self.vec)
        for i in range(len(self.vec)):
            hp.query_disc(band.dir(i), np.radians(radius),iv)
            newvec[i] = np.array([self.vec[j] for j in iv]).mean()
        self.vec = newvec
           
        
def make_index_table(nside, subnside):
    band, subband = Band(nside), Band(subnside)
    npix, nsubpix = 12*nside**2, 12*subnside**2
    t=np.array([band.index(subband.dir(i)) for i in xrange(nsubpix)])
    a = np.arange(nsubpix)
    index_table = [a[t==i] for i in xrange(npix)]
    return index_table

class HPtables(HParray):
    """ assemble an array from tables in a set of ROIs
    """

    def __init__(self, tname, outdir, nside=512, roi_nside=12):
        """ combine the tables generarated at each ROI
        
        tname : name of the table, expect to find folder outdir/tname
        nside : nside parameter that the table was generated with
        """
        self.name = tname
        self.nside = nside
        files =glob.glob(os.path.join(outdir, '%s_table'%tname,'*.pickle'))
        nf = len(files)
        assert nf>0, 'no pickle files found in %s' % os.path.join(outdir, '%s_table'%tname)
        if nf<1728: print 'warning: missing %d files in folder %s' % ((1728-nf), tname)
        files.sort()

        self.vec = np.zeros(12*nside**2)
        self.vec.fill(np.nan)
        pklist = [pickle.load(open(f)) for f in files]
        i12 = [int(f[-11:-7]) for f in files]
        index_table = make_index_table(roi_nside, nside)
        for index, pk in zip(i12,pklist):
            indeces = index_table[index]
            for i,v in enumerate(pk):
                self.vec[indeces[i]]=v
    
class HPskyfun(HParray):
    """generate from a sky function: base class for below
    """
    def __init__(self, name, skyfun, nside):
        """ skyfun : function of position
        """
        self.name = name
        self.skyfun = skyfun
        self.nside=nside
        self.dirfun = Band(self.nside).dir
    def __getitem__(self, index):
        try:
            return self.skyfun(self.dirfun(index))
        except:
            return np.nan
        
    def getcol(self, type=np.float32):
        return np.asarray([self[index] for index in xrange(12*self.nside**2)],type)

class HPfitscube(HPskyfun):
    """ generate from a FITS cube """
    def __init__(self, name,  filename, nside=512,):
        self.skyimage=SkyImage(filename)
        super(HPfitscube,self).__init__(name, self.skyimage, nside)
    def set_layer(self,n):
        # use 1-based indexing for layer umber
        return self.skyimage.setLayer(n-1)+1
    def layers(self): return self.skyimage.layers()

class HPfitscubelayer(HPskyfun):
    """ wrap a HPfitscube instance to define layer number
    """
    def __init__(self, fitscube, layer):
        name = fitscube.name+'#%02d'%layer
        self.layer=layer
        assert layer>0 and layer<=fitscube.layers(), 'layer %d not in range(1,%d)' % (layer, fitscube.layers()+1)
        super(HPfitscubelayer,self).__init__(name, fitscube.skyfun, fitscube.nside)
        self.cube = fitscube
    def getcol(self, type=np.float32):
        last = self.cube.set_layer(self.layer)
        ret = super(HPfitscubelayer,self).getcol(type)
        self.cube.set_layer(last)
        return ret
    def plot(self, *pars, **kwargs):
        last = self.cube.set_layer(self.layer)
        ret=super(HPfitscubelayer,self).plot(*pars, **kwargs)
        self.cube.set_layer(last)
        return ret
        
class HPindex(HPskyfun):
    """ represent healpix index value """
    def __init__(self, name, supernside=12, nside=512):
        hpfun = Band(supernside).index
        super(HPindex,self).__init__(name, hpfun, nside)
    def getcol(self, type=np.float32):
        return np.asarray([self[index] for index in xrange(12*self.nside**2)],type)
    
class HPresample(HParray):
    """ resample from another HParray object with different nside
    """

    def __init__(self, other, nside=256):
        self.name=other.name
        self.vec = other.vec
        self.nside =nside
        self._nside = int(np.sqrt(len(self.vec)/12) ) # internal 
        self.dirfun = Band(self.nside).dir # external direction from index
        self._indexfun = Band(self._nside).index #internal index from direction

    def __getitem__(self, index):
        return self.vec[self._indexfun(self.dirfun(index))]
    def getcol(self, type=np.float32):
        return np.asarray([self[index] for index in xrange(12*self.nside**2)],type)

class HEALPixFITS(list):

    def __init__(self, cols, nside=None):
        """ initialize
        """
        assert len(cols)>0, 'no HParray object list'
        self.nside = nside if nside is not None else cols[0].nside
        for col in cols:
            self.append( col if col.nside==self.nside else HPresample(col, self.nside) )
            print 'appended column %s' %col.name
            
    def write(self, outfile, unit=None):
        makecol = lambda v: pyfits.Column(name=v.name, format='E', unit=unit, array=v.getcol(np.float32))
        cols = map(makecol, self)
        nside = self.nside
        cards = [pyfits.Card(*pars) for pars in [ 
                ('PIXTYPE',  'HEALPIX',          'Pixel algorithm',),
                ('ORDERING', 'RING',             'Ordering scheme'),
                ('NSIDE' ,    nside,        'Resolution Parameter'),
                ('NPIX',     12*nside**2,   '# of pixels'),
                ('FIRSTPIX',  0,                 'First pixel (0 based)'),
                ('LASTPIX',  12*nside**2-1, 'Last pixel (0 based)')]]
        table = pyfits.new_table(cols, header=pyfits.Header(cards))
        table.name = 'healpix' 
        hdus =  [pyfits.PrimaryHDU(header=None),  #primary
                 table,      # this table
                ]
        if os.path.exists(outfile):
            os.remove(outfile)
        pyfits.HDUList(hdus).writeto(outfile)
        print '\nwrote FITS file to %s' % outfile
        
def hpname(index):
    return 'HP12_%04d'%index
        
class Display_map(object):
    """ utility class to handle nice displays of a HParray object
    
    """

    def __init__(self, hptable, outdir,
                map_dir = None,
                nside=12, 
                imshow_kw=dict(interpolation='bilinear', vmin=0, vmax=100, ),
                **kwargs):
        """
        hptable : HParray object
            expect to have vec, nside members
        outdir : sring
            folder containing the skymodel
        map_dir : None, string
            folder name to save the images
        
        """
        from .. import skymodel 
        self.v = hptable.vec
        self.subband = Band(hptable.nside)
        self.band = Band(nside)
        self.n = 12*nside**2
        self.imshow_kw=imshow_kw
        self.scale = kwargs.pop('scale', lambda x: x)
        if type(self.scale) == types.StringTypes:
            if self.scale=='sqrt': self.scale= lambda x: np.sqrt(max(x,0))
            elif self.scale=='log': self.scale=lambda x: np.log10(max(x,0.1))
            else:
                raise Exception, 'unrecognized scale function, %s' %self.scale
        print 'Can generate %d map figures' %(self.n)
        self.outdir = outdir
        self.ZEA_kw = kwargs.pop('ZEA_kw', dict())
        if map_dir is not None:
            self.map_path = os.path.join(outdir,map_dir) 
            if not os.path.exists(self.map_path):
                os.makedirs(self.map_path)
            print 'will save figures in folder %s' % self.map_path
        else: self.map_path = None
        skm = skymodel.SkyModel(outdir)
        self.sources = skm.point_sources+skm.extended_sources
        print 'loaded %d sources from skymodel %s' % (len(self.sources),outdir)
         
    def get_pyskyfun(self):
        return PySkyFunction(self)

    def skyfun(self, v):
        skydir = SkyDir(Hep3Vector(v[0],v[1],v[2]))
        return self.v[self.subband.index(skydir)]
        
    def __call__(self,v):
        skydir = SkyDir(Hep3Vector(v[0],v[1],v[2]))
        t =self.v[self.subband.index(skydir)]
        #if   self.scale=='sqrt': return np.sqrt(max(t,0))
        #elif self.scale=='log':  return np.log10(max(t, 1e-1))
        return self.scale(t) 

    def fill_ait(self, fignum=11, axes=None, show_kw={}, **kwargs):
        if axes is None:
            plt.close(fignum)
            fig=plt.figure(fignum, figsize=(12,8));
            axes=fig.gca()
        pixelsize = kwargs.pop('pixelsize', 0.25)
        ait = image.AIT(self.get_pyskyfun(),axes=axes, pixelsize=pixelsize, **kwargs)
        self.imgplot=ait.imshow(**show_kw)
        return ait

    def fill_zea(self, index, fignum=12, axes=None, **kwargs):
        """ sources is a recarray with name, ra, dec
        """
        if axes is None:
            plt.close(fignum)
            fig = plt.figure(fignum,figsize=(6,6)); 
            axes = fig.gca()
        size = kwargs.pop('size', 10)
        pixelsize = kwargs.pop('pixelsize', 0.1)
        title = kwargs.pop('title', hpname(index))
        label = kwargs.pop('label', '')
        zea = image.ZEA(self.band.dir(index), size=size, pixelsize=pixelsize,**self.ZEA_kw)
        zea.grid()
        zea.fill(self.get_pyskyfun())
        imshow_kw = self.imshow_kw #
        imshow_kw.update(kwargs)
        zea.imshow(**imshow_kw)
        zea.colorbar(label=label)
        axes.set_title(title)
        if self.sources is not None:
            count = 0
            for s in self.sources:
                sdir = s.skydir
                if not zea.inside(sdir):continue
                count += 1
                inside =self.band.index(sdir)==index
                zea.plot_source(s.name, sdir, symbol='*' if inside else 'd', 
                    markersize=14 if inside else 8,
                    color='w')
            print 'found %d sources to plot' %count        
        
        if self.map_path is not None:
            fout = os.path.join(self.map_path,hpname(index)+'.png')
            plt.savefig(fout)
            print 'saved figure to %s' % fout
        plt.draw_if_interactive()
  
class ZEAdisplayTasks(object):
    """ adapt the class that generates a ZEA display for multiprocessing, defining tasks
        
    """
    def __init__(self, tablename, outdir, *pars, **kwargs):
        self.label = kwargs.pop('label', '(nolabel)')
        nside = kwargs.pop('nside', 512)
        self.display = Display_map(HPtables(tablename, outdir, nside=nside), 
            outdir, 
            tablename+'_maps',*pars, **kwargs)
        self.n = self.display.n
        self.title = tablename  
    def names(self):
        return map(hpname,range(self.n))
    def __call__(self, index):
        ait=self.display.fill_zea(index, title=self.title+' ('+hpname(index)+')', label=self.label)
 
 
def tables(outdir='uw40', names=('kde', 'ts'), nside=512):
    """ create list of HParray guys from ROI tables
    """
    return [HPtables(name, outdir, nside=nside) for name in names]
    
    
def diffusefun(gal ='ring_24month_P74_v1.fits', energy=1000, nside=512):
    """ an HParray object wrapping the galactic diffuse """
    f = os.path.expandvars(os.path.join('$FERMI', 'diffuse', gal))
    t = skymaps.DiffuseFunction(f)
    def skyplotfun(skydir):
        return t.value(skydir, energy)
    return HPskyfun('galactic', skyplotfun, nside=nside)
 
#def make_fits(outdir='uw40', outfile='../../pivot/24M7_uw40/aladin512.fits'):
#    t = tables()
#    d = diffusefun()
#    i = HPindex('hp12', 12)
#    HEALPixFITS(t+[d, i]).write(outfile) 
#
#
class Rings(object):
    """ analyze the rings in a fits file with layers """
    
    def __init__(self, nside=256,filename=r'D:\fermi\diffuse\hi\rbands_hi11_qdeg_Ts125.fits.gz'):
        import pyfits
        self.filename = filename
        self.cube = HPfitscube('rings', filename, nside=nside)
        print 'loaded %s, with %d layers' % (filename, self.cube.layers())
        s = pyfits.open(self.filename)
        self.radii = s[1].data.field('Rmax') #outer radii
        assert len(self.radii)==self.cube.layers(), 'layers and Rmax list inconsistent'
        print 'radii: %s' % str(map( lambda x: np.round(x,1), self.radii))
        self.hpf = HEALPixFITS([HPfitscubelayer(self.cube, layer+1) for layer in range(0,self.cube.layers())])
    
    def write_fits(self, outfile):
        """ create a HEALPix fits file """
        self.hpf.write(outfile)
        
    def draw_rings(self, fignum=3,axes=None, scale=1.):
        from matplotlib.patches import Circle
        if axes is None: 
            plt.close(fignum)
            fig = plt.figure(fignum, figsize=(10,10))
            axes = plt.gca()
        axes.axis((-25,25,-30,20))
        for radius in self.radii:
            axes.add_patch( Circle( (0,8.3), radius=radius/scale,lw=2, fc='none'))
        axes.add_patch( Circle( (0, 0), radius = 0.5, fc='blue', ec='none'))
        plt.axhline(0); plt.axvline(0); axes.set_axis_off()
        plt.title('galactic decomposition')

    def _getcols(self):
        """ create columns of values for each pixel"""
        self.cols = [r.getcol() for r in self.hpf]
        #fmean = lambda c: c[-np.isnan(c)].mean()
        #fmax =  lambda c: c[-np.isnan(c)].max()
        #self.means = map(fmean, cols)
        #self.maxes = map(fmax, cols)
        #return means, maxes
    def _plot_setup(self,fignum, axes):
        if self.__dict__.get('cols') is None:
            self._getcols()
        if axes is None:
            plt.close(fignum)
            fig = plt.figure(fignum, figsize=(6,6))
            axes = plt.gca()
        return axes
    def plot_means(self, fignum=4, axes=None):
        axes=self._plot_setup(fignum, axes)
        a = map(lambda c: c[-np.isnan(c)].mean(), self.cols)
        axes.semilogy(np.linspace(1,17,17), a, 'o');
        axes.grid(True)
        axes.set_xlabel('ring number'); 
        axes.set_ylabel('average density')
    def plot_maxes(self, fignum=5, axes=None):
        axes = self._plot_setup(fignum, axes)
        a = map(lambda c: c[-np.isnan(c)].max(), self.cols)
        axes.plot(np.linspace(1,17,17), a, 'o');
        axes.grid(True)
        axes.set_xlabel('ring number'); 
        axes.set_ylabel('maximum density')

def write_hpmaps(prefix='residual', fitsfile='residuals.fits', nside=512, average=1.0, outdir='.'):
    if os.path.exists(fitsfile):
            os.remove(fitsfile)
    t = glob.glob(os.path.join(outdir, prefix+'*_table'))
    assert len(t)>0, 'no tables found'
    names=map(lambda x: os.path.split(x)[1][:-6], t)
    print 'table names:' , names
    cols = [HPtables(name, outdir) for name in names]
    if average>0:
        for col in cols:
            print 'averaging table' , col.name
            col.average(average)
    HEALPixFITS(cols, nside=nside).write(fitsfile)
    print 'wrote file %s with %d columns' %(fitsfile, len(cols))

    
# this crashes now, no idea  
#def exposure(ltcube = 'P7_V4_SOURCE/24M7_lt.fits',
#        irf='P7SOURCE_V4PSF', energy=1000, nside=384):
#    """
#    return and HParray object wrapping the exposure
#    """
#    caldb = pycaldb.CALDBManager(irf)
#    skymaps.Exposure.set_cutoff(0.4)
#    skymaps.EffectiveArea.set_CALDB(caldb.CALDB)
#
#    f = os.path.expandvars(os.path.join('$FERMI','data', ltcube))
#    lc = skymaps.LivetimeCube(f) 
#    #ea  = [skymaps.EffectiveArea('',f) for f in caldb.get_aeff()]
#    aeff = map(lambda f: skymaps.EffectiveArea('',f), caldb.get_aeff())
#    #exposure = [skymaps.Exposure(lc,x) for x in ea]
#    exposure = map(lambda x: skymaps.Exposure(lc,x), aeff)
#    def skyplotfun(skydir):
#        return sum([exp.value(skydir, energy) for exp in exposure])
#    return skyplotfun #HPskyfun('exposure', skyplotfun, nside=nside)
#

#

def make_setup(outdir, title,
        imshow_kw,
        label='', nside=256):
    """
    Use this to setup for mec-generation combining the individual ROI tables to make images centered on the ROIs
    """
    extra='ZEA_kw=dict(galactic=True),imshow_kw=dict(%s),label="%s"' %( imshow_kw,label)
    setup_string =  """\
import os; import numpy as np;os.chdir(r"%(cwd)s")
from uw.like2.pub import healpix_map;
g = healpix_map.ZEAdisplayTasks("%(title)s","%(outdir)s", nside=%(nside)s, %(extra)s)
""" %dict(cwd=os.getcwd(), title=title, outdir=outdir, extra=extra, nside=nside)
    return setup_string
    
from uw.like2.pipeline import pipe

class Setup(pipe.Setup):
    """ subclass that allows parallel processing
    """
    def __init__(self,outdir, title):
        self.outdir = outdir
        if title[:2]=='ts':
            self.setup= make_setup(outdir, title, 
                imshow_kw='interpolation="bilinear", vmin=0,vmax=5,fun=np.sqrt',
                label = r'$\mathrm{\sqrt{TS_{max}-TS}}$',
                )
        elif title=='kde':
                self.setup= make_setup(outdir, title, 
                imshow_kw='interpolation="bilinear",fun=np.log10',
                label='log10(photon density)')
        else:
            assert False, 'title %s not recognized' % title
        self.mecsetup=False #needed from inheritance
     
    def dump(self): pass
    
    def __call__(self):
        return self.setup

## obsolete, but should look at re-implementing
#def get_outdir():
#    """ default outdir from local file"""
#    version= int(open('version.txt').read())
#    return 'uw%02d'%version
#
#def getg(outdir=None, title='ts'):
#    """ for testing locally"""
#    if outdir is None: outdir=get_outdir()
#    setup=Setup(outdir, title)
#    exec(setup(), globals()) # should set g
#    return g
#
# 
#def make_maps(outdir=None, title='all', **kwargs):
#    from uw.like2.pipeline.pipe import pmain
#    if outdir is None: outdir=get_outdir()
#    all = title=='all'
#    if all or title=='kde':
#        pmain(Setup(outdir,'kde'),logpath=None,  **kwargs)
#    if all or title=='ts':
#        pmain(Setup(outdir,'ts'), logpath=None,  **kwargs)
#def main():   
#    from uw.like2.pipeline.pipe import get_mec
#    make_maps(None, 'all', mec=get_mec())
#
#if __name__=='__main__':
#    main()
#     