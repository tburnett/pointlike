"""
Code to generate a set of maps for each ROI
$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/maps.py,v 1.15 2018/01/27 15:37:17 burnett Exp $

"""
import os, sys,  types, glob
import cPickle as pickle
import numpy as np
import pandas as pd
from skymaps import Band, SkyDir, PySkyFunction, Hep3Vector, PythonUtilities 
from uw.like import Models
from . import (sources, sedfuns, ) 
from uw.utilities import image
from uw.like2.pipeline import check_ts
import healpy

# the default nside
nside=512

# convenience adapters for ResidualTS model
def LogParabola(*pars):
    model = Models.LogParabola(p=pars, free=[True,True,False,False])
    sources.set_default_bounds(model)
    return model
def PowerLaw(*pars):   
    model = Models.PowerLaw(p=pars)
    sources.set_default_bounds(model)
    return model
def ExpCutoff(*pars):  
    model = Models.ExpCutoff(p=pars, free=[True, True, False])
    sources.set_default_bounds(model)
    return model
def PLSuperExpCutoff(*pars):  
    model = Models.PLSuperExpCutoff(p=pars, free={True,True,False,False})
    sources.set_default_bounds(model)
    return model

def make_index_table(nside=12, subnside=nside, usefile=True):
    """create, and/or use a table to convert between different nside pixelizations
    The indexing is RING
    """
    filename = os.path.expandvars('$FERMI/misc/index_table_%02d_%03d.pickle' % (nside, subnside) )
    if os.path.exists(filename) and usefile:
        return pickle.load(open(filename))
    print 'generating index table for nside, subnside= %d %d' % (nside, subnside)
    band, subband = Band(nside), Band(subnside)
    npix, nsubpix = 12*nside**2, 12*subnside**2
    t=np.array([band.index(subband.dir(i)) for i in xrange(nsubpix)])
    a = np.arange(nsubpix)
    index_table = [a[t==i] for i in xrange(npix)]
    if usefile:
        pickle.dump(index_table, open(filename,'w'))
    return index_table  

class CountsMap(dict):
    """ A map with counts per HEALPix bin """
    
    def __init__(self, roi, emin=1000., nside=nside):
        """ roi : a region of interest object
            emin : float
                minimum energy for constructing the counts map
            nside : int
                the nside parameter for the HEALPix map
        Note that is is implemented as a dictionary, with the pixel index as key
        """
        self.indexfun = Band(nside).index
        self.nside = nside
        self._fill(roi, emin)
        
    def _fill(self, roi, emin):
        for bandlike in roi.selected:
            band = bandlike.band
            if band.energy < emin: continue
            #t = band.b.event_class() & 3 # note: mask off high bits!
            for wsd in band.wsdl:
                index, wt = self.indexfun(wsd), int(wsd.weight())
                if index in self: self[index]+=wt
                else: self[index] = wt
                
    def __call__(self,v):
        return self.get(self.indexfun(v), 0)

def default_geom():
    """return a DataFrame with indexed by the band, containing emin, emax, event_type, nside
    """
    energies = np.logspace(2, 6, 17)
    elow = energies[:-1]
    ehigh=energies[1:]
    nside_list = ModelCountMaps.nside_array
    geom = dict()
    for ib in range(32):
        ie = ib/2; it=ib%2
        geom[ib]=dict(elow=elow[ie],ehigh=ehigh[ie], 
                      event_type=it, nside=nside_list[ib])
    g = pd.DataFrame(geom).T['elow ehigh event_type nside'.split()]
    g.event_type = g.event_type.astype(int)
    g.nside = g.nside.astype(int)
    return g

class ModelCountMaps(object):
    """ This is not the same as other map-producing classes here.
    It makes, and saves, a list of predicted counts for all the pixels within the central
    HEALPix pixel.
    """
    nside_array = np.ones(32,int) * 1024
    nside_array[:10] = 64,64, 128,64 ,256,128, 256,256, 512,512

    def __init__(self, roi, nbands=None, bandlist=None,  subdir='model_counts'):
        """
        roi : ROI object
            Expect it to behave like an array, indexed by the band index,
            and each element is a SkyFunction, returning the count density.
        bandlist : list of int | None
            bands to process. If None, do them all

        subdir : string | None
            folder name to write results to
        """
        roi_index = Band(12).index(roi.roi_dir)  

        if bandlist is None:
            if nbands is not None: bandlist=range(nbands)
            else: bandlist = range(len(roi))
        print '{:4} {:4} {:6} {:8} {:8} {:8} '.format(*'eb nside cnts mean min max'.split())
        for ebi in bandlist:
            eb = roi[ebi] #EnergyBand object

            # choose nside to be power of 2, override the data
            nside =  ModelCountMaps.nside_array[ebi]
            dirfun = Band(nside).dir
            pixel_area = Band(nside).pixelArea()
            index_table = make_index_table(12, nside)
            pix_ids = index_table[roi_index]
            dirs = map(dirfun, pix_ids)
            cnts = np.array(map(eb, dirs),np.float32) * pixel_area
            assert sum(np.isnan(cnts))==0, 'NaN value(s) found'
            
            print '{:4d} {:4d} {:6d} {:8.2e} {:8.2e} {:8.2e}'.format(
                ebi,nside,len(cnts), cnts.mean(), cnts.min(), cnts.max()),
            if subdir is not None:
                subsubdir = subdir+'/{:02d}'.format(ebi)
                if not os.path.exists(subsubdir): os.makedirs(subsubdir)
                outfile = subsubdir+'/HP12_{:04d}.pickle'.format(roi_index)
                pickle.dump(cnts, open(outfile, 'w'))
                print '\t--> {}'.format(outfile)
            else:
                print '\t (not saved)'


class KdeMap(object):
    """ Implement a SkyFunction that returns KDE data for a given ROI"""
    
    def __init__(self, roi, emin=[500,1000], **kwargs):
        """ roi: an ROIstat object
            emin: list of two minimum energies
        """
        # create local array of references corresponding to bands from ROI 
        self.bands = []
        self.r95 = []
        for  bandlike in roi.selected:
            band = bandlike.band 
            if band.emin < emin[band.event_type]: continue
            self.r95.append( np.radians(band.psf.inverse_integral(95, on_axis=False)) ) #FAIL if True
            self.bands.append(band)
    
    def __call__(self,v,skydir = None):
        """ copied from roi_tsmap.HealpixKDEMap """
        sd = skydir or SkyDir(Hep3Vector(v[0],v[1],v[2]))
        rval = 0
        for i,band in enumerate(self.bands):
            if not band.has_pixels: continue
            rvals = np.empty(len(band.wsdl),dtype=float)
            PythonUtilities.arclength(rvals, band.wsdl, sd)
            mask = rvals < self.r95[i]
            rval +=(band.psf(rvals[mask])*band.pix_counts[mask]).sum()
        return rval

class DiffuseMap(object):

    def __init__(self, roi, **kwargs):
        pass 
        #TDO
    def __call__(self, v, skydir=None):
        sd = skydir or SkyDir(Hep3Vector(v[0],v[1],v[2]))
        # TODO

class ResidualTS(object):
    """ manage a residual TS plot 
    """
    def __init__(self, roi,  **kwargs):
        """
        roi : a ROI_user object

        model : a string
            must evaluate to a Models.Model. e.g. 'LogParabola(6e-14, 1.2, 0, 4500)'

        par_sets : list of list of float or None
            if not None, sets of parameters to apply to the model
        """
        self.roi = roi
        self.model = eval(kwargs.pop('model'))
        sources.set_default_bounds(self.model)
        self.pars = kwargs.pop('par_sets', None)
        self.sourcename='test'
        self.source = roi.add_source(name=self.sourcename, skydir = roi.roi_dir, model=self.model)
        roi.get_source(self.sourcename) # to select
        # the function to miniumze
        self.func = roi.fitter_view(self.sourcename+'_Norm')
        
    def __enter__(self):
        roi.summary()
        roi.fit([self.index]) #check
        print 'TS check:', roi.TS(self.sourcename)

        return self
        
    def __exit__(self,*pars):
        self.reset()
        
    def reset(self):
        self.roi.del_source(self.sourcename)
    
    def trial_fit(self, pars=None):
        if pars is not None:
            self.model.set_all_parameters(pars)
            self.source.changed=True
        else:
            self.model[0]=1e-13 # initial value 
        try:
            self.func.maximize(estimate_errors=False)
            ts = self.roi.TS()
        except:
            ts=0
        return max(0, ts)

    def tsfun(self, skydir):
        # move to new position, reinitialze
        self.source.skydir = skydir
        self.roi.calls =0
        self.roi.initialize(sourcename=self.sourcename)

        if self.pars is None:
            return self.trial_fit()
        return map(self.trial_fit, self.pars)

        
    def __call__(self, v):
        skydir = SkyDir(Hep3Vector(v[0],v[1],v[2]))
        return self.tsfun(skydir)

        

class ResidualUpperLimit(ResidualTS):
    """ save the likelihood function, as the 3-parameter representation of a shifted Poisson, plss the max dev.
    """
    def tsfun(self, skydir):
        self.source.skydir = skydir
        self.roi.calls =0
        self.model[0]=1e-13 # initial value 
        self.roi.initialize(sourcename=self.sourcename)
        #print 'Studying source %s at %s' % (self.sourcename, skydir) ,

        with sedfuns.SED(self.roi, self.sourcename) as sf:
            try: # get a PoissonFItter object 
                pf = sf.select(None, delta=1e-3) 
                poiss = pf.poiss 
                #print 'TS, maxdev: %.2f %.2f' % (poiss.ts, pf.maxdev)
                lim = poiss.percentile()
                if lim==0:
                    print 'Limit is zero at {}'.format(skydir)
                return lim
            except Exception, msg:
                p = 0
                print 'Failed at %s: %s' % (skydir,msg)
                return 0
 

class ROItables(object):
    """ manage one or more tables of values subdividing a HEALpix roi
    
        skyfuns : list of 3-tuples
            the 3-tuples must have: (skyfunction, tablename, dict)
            Implemented now, and default here:
                (ResidualTS,'ts',  dict(photon_index=2.0),) , 
                (KdeMap,    'kde', dict()),
            If skyfunction is a string, evaluate it 

        kwargs options select set of HP indeces.  In priority ordef:
          index_table: table if HP indeces to use
          disc_info: assume (l,b, r), and get all pixels in that disk
          otherwise: use the pixels in the standard nside=12 ROI
    """
    
    def __init__(self, outdir, nside, roi_nside=12, **kwargs):
        self.subdirfun = Band(nside).dir
        disc_info = kwargs.pop('disc_info', ())
        index_table = kwargs.pop('index_table', None)
        self.skyfuns = kwargs.pop('skyfuns', 
              ( (ResidualTS, 'ts', dict(photon_index=2.3),) , 
                (KdeMap,     'kde', dict()),
              ),
            )
        assert len(kwargs)==0, 'Unexpected kwargs: {}'.format(kwargs)

        # allow arbitraty disk, rather than just within a ROI active boundary
        if index_table is not None:
            self.pos_list = [self.subdirfun(int(i)) for i in index_table]
            self.index_table = index_table
        elif disc_info:
            l, b, r = disc_info
            vec = healpy.dir2vec(l,b, lonlat=True)
            index_list = healpy.query_disc(nside=nside, vec=vec, radius=np.radians(r),nest=False)
            self.pos_list = [self.subdirfun(int(i)) for i in index_list]
            self.index_table = index_list
        else:
            self.index_table = make_index_table(roi_nside, nside)
        print 'Will process {} pixels with  spectral models:\n{}'.format(len(self.index_table), self.skyfuns)

 

        if outdir:
            self.subdirs = [os.path.join(outdir, name+'_table_%d' %nside) for s, name, kw in self.skyfuns]
            for subdir in self.subdirs: 
                if not os.path.exists(subdir):  os.makedirs(subdir)
        else:
            self.subdirs=[]
                    
    def process_table(self, skyfun, name, pos_list, outfile=None, **kwargs):
        sys.stdout.flush()
        skytable = np.array([skyfun(p) for p in pos_list])
        print ' min=%6.2e, max=%6.2e, mean=%6.2e ' \
            % (skytable.min(), skytable.max(),skytable.mean()) ,
        if outfile is not None:
            print '--> %s' % outfile
            pickle.dump(skytable, open(outfile,'wb'))
        else: print  
        if hasattr(skyfun,'reset'): skyfun.reset() 
        return skytable
  
    def __call__(self, roi):
        pos_list = getattr(self, 'pos_list', [])
        if not pos_list:
            # not created in ctor: make if from the name
            index = int(roi.name[5:])
            pos_list = [self.subdirfun(int(i)) for i in self.index_table[index]]
        self.skytable=[self.index_table]
          
        for i,fun in enumerate(self.skyfuns):
            skyfun = fun[0] if type(fun[0])!=types.StringType else eval(fun[0])

            stbl =self.process_table(skyfun(roi, **fun[2]), fun[1], pos_list, 
                os.path.join(self.subdirs[i], roi.name+'.pickle') if self.subdirs else None)
            self.skytable.append( stbl)
        return self.skytable


class DisplayTable(object):
    """display the table from an ROI as an image
    """
    def __init__(self, table_name='kde', index=860, nside=nside, **kwargs):
        """
        table_name : 
        index :
        nside :
        
        """
        folder = '%s_table_%d' % (table_name, nside)
        assert os.path.exists(folder), 'Folder %s not found' % folder
        self.hmap = pickle.load(open('%s/HP12_%04d.pickle' % (folder, index)))
        self.index = index
        self.hpindices = list(make_index_table(12,nside)[index])
        self.hpdict = dict((self.hpindices[i],self.hmap[i]) for i in range(len(self.hpindices)));
        self.center = Band(12).dir(index)
        self.indexfun = Band(nside).index
        self.ZEA_kw = kwargs.pop('ZEA_kw', dict(galactic=True, size=10, pixelsize=0.05))
        self.imshow_kw=dict(interpolation='bilinear',  )
        self.scale = kwargs.pop('scale', lambda x: x)
        if type(self.scale) == types.StringTypes:
            if self.scale=='sqrt': self.scale= lambda x: np.sqrt(max(x,0))
            elif self.scale=='log': self.scale=lambda x: np.log10(max(x,0.1))
            else:
                raise Exception, 'unrecognized scale function, %s' %self.scale

    def get_pyskyfun(self):
        return PySkyFunction(self)

    def __call__(self,v):
        skydir = SkyDir(Hep3Vector(v[0],v[1],v[2]))
        i = self.indexfun(skydir)
        try: 
            t = self.hpdict[i]
            self.hit+=1
            return self.scale(t)
        except KeyError: #else:
            self.miss+=1
            return np.nan

    def describe(self):
        """return a description of the map values"""
        return pd.Series(self.hmap).describe()

    def plot(self, axes=None, show_kw=None, **kwargs):
        """ show_kw : dict
                override imshow keywords
            kwargs 
                size
                pixelsize
                galactic
        """
        from uw.utilities import image
        import pylab as plt
        
        fig, ax = plt.subplots(figsize=(6,6))
        title = kwargs.pop('title','HP12_%04d'%self.index)
        kw = self.ZEA_kw
        kw.update(kwargs)
        zea = image.ZEA(self.center, axes=ax, **kw)
        zea.grid()
        self.miss=self.hit=0
        zea.fill(self.get_pyskyfun())
        zea.imshow( **(show_kw if show_kw is not None else self.imshow_kw))
        zea.colorbar()
        if title is not None: ax.set_title(title)
        fig.set_facecolor('white')
        return zea

table_info={'ts':  (ResidualTS, dict(model='LogParabola(1e-13, 2.2, 0, 1000.)')),
            'kde': (KdeMap, dict()),
            'diffuse': (DiffuseMap, dict()),
            'tsx': (ResidualTS, dict(model='LogParabola(1e-12, 2.3, 0, 1000.)')),
            'tsp': (ResidualTS, dict(model='ExpCutoff(1e-13,1.3, 1500.)')),
            'hard': (ResidualTS, dict( model='LogParabola(1e-15, 1.7, 0, 50000.)')),
            'soft': (ResidualTS, dict(model='LogParabola(1e-12, 2.7, 0, 250.)')),
            'peaked': (ResidualTS, dict(model='LogParabola(1e-14, 2.0, 0.5, 2000.)')),
            'mspsens': (ResidualUpperLimit, dict(model='ExpCutoff(1e-13,1.2,2800.)')),
           'mspsens2': (ResidualUpperLimit, dict(model='ExpCutoff(1e-13,1.2,2800.)')),
           'mspts': (ResidualTS, dict(model='ExpCutoff(1e-13,1.2, 2800.)')),
            'mspts2': (ResidualTS, dict(model='ExpCutoff(1e-13,1.2, 2800.)')),
            'all':  (ResidualTS, dict(model='LogParabola(1e-13, 2.2, 0, 1000.)', 
                    par_sets= [[1e-13, 1.7, 0,  50000.], #hard
                            [1e-13, 2.2, 0,   1000.], #flat
                            [1e-13, 2.7, 0,    250.], #soft
                            [1e-13, 2.0, 0.5, 2000.], #peaked
                            [1e-13, 2.0, 0.3, 1000.], #psr
                    ]) )
           }
table_info['flat']=table_info['ts']
table_info['psr']=table_info['tsp']


def residual_maps(roi, folder='residual_maps', nbands=4, test=False):
    
    def residual_map(index):
        b=roi[index] # the BandLike object
        cband = roi.config.dataset.dmap[b.band.data_index] # need to find appropriate one
        nside = cband.nside()
        b=roi[index] # the BandLike object
        assert abs(cband.emin()-b.band.emin)<1.0, '{},{}'.format(cband.emin(), b.band.emin)
        energy = b.band.energy
        event_type = b.band.event_type
        band_label = '{:.0f} Mev {}'.format(energy, ['Front','Back'][event_type])

        # get pixel info: counts, model, positions 
        wsdl = b.band.wsdl # list of WeightedSKyDir objects
        ids = np.array(map( cband.index, wsdl)) # list of ids with nside
        id12 = np.array(map(Band(12).index, wsdl),int) # nside=12 ids
        inside = id12==roi_id
        data = b.data
        model= np.array(b.model_pixels,np.float32)
        galactic = np.array(b[0].pix_counts) #assume!
        isotropic= np.array(b[1].pix_counts)
        
        # combine counts for all local sources
        source_cnts = np.zeros(b.pixels, np.float32)
        for m in b.bandsources:
            if m.source.skydir is None: continue
            if m.counts>0:
                source_cnts += m.pix_counts

        dd = dict(band_index=index, nside=nside,energy=energy, event_type=event_type, 
                  ids=ids, model= model,data= data,inside=inside, source=source_cnts, 
                  galactic=galactic,isotropic=isotropic)
        chi2 = sum((data-model)**2/model)
        print '{:5.0f} {:5d} {:4d}   {:4.0f}/{:4d}   {:+0.1%}'.format(energy, event_type, nside,
                                                                 chi2, len(data), (data/model-1).mean())
        return dd
        
    roi_id = Band(12).index(roi.roi_dir)
    print 'energy type nside  chi2/ ndf  offset'
    maps = [residual_map(index)   for index in range(nbands)]
    if test:
        return maps
    if not os.path.exists(folder):
        os.mkdir(folder)
    filename = '{}/ROI_{:04d}.pickle'.format(folder,roi_id)
    pickle.dump(maps, open(filename,'w'))
    print 'Wrote file {}'.format(filename)


def assemble_tables(table_names, outputfile=None, folder= '.', nside=nside):
    """ assemble one or more healpix tables from individual ROIs into a single FITS file
    parameters
    ----------
    table_names : list of strings
        prefix names, with suffix '_table'. Can be either zip or folders
    outputfile : string
        path, relative to folder, to write output file; default constructed from list of table names
    folder : string
        path name, default current dir.
    """
    tables = [HPtables(name, folder, nside=nside) for name in table_names]
    if outputfile is None:
        outputfile='hptables_'+'_'.join(table_names)+'_%d.fits' %nside
    f = HEALPixFITS(tables)
    f.write(os.path.join(folder,outputfile))
  
def make_index_table(nside=12, subnside=nside, usefile=True):
    """create, and/or use a table to convert between different nside pixelizations
    """
    filename = os.path.expandvars('$FERMI/misc/index_table_%02d_%03d.pickle' % (nside, subnside) )
    if os.path.exists(filename) and usefile:
        return pickle.load(open(filename))
    print 'generating index table for nside, subnside= %d %d' % (nside, subnside)
    band, subband = Band(nside), Band(subnside)
    npix, nsubpix = 12*nside**2, 12*subnside**2
    t=np.array([band.index(subband.dir(i)) for i in xrange(nsubpix)])
    a = np.arange(nsubpix)
    index_table = [a[t==i] for i in xrange(npix)]
    if usefile:
        pickle.dump(index_table, open(filename,'w'))
    return index_table

class MultiMap(object):
    
    def __init__(self, names=['hard','flat','soft', 'peaked', 'psr'], 
                 outdir='.', tname='all', nside=nside, roi_nside=12, fill=np.nan, roi_table_map=None):
        """ combine the tables generarated at each ROI

        names : names to give the columns, 
        nside : nside parameter that the table was generated with, default 512 (or 256)
        tname : name of the table, default 'all'
        fill  : scalar, defaul NaN
            Use to fill missing tables, if any (warning issued)
        
        Direct insertion option:

        roi_table_map : tuple with index_map, rt_table
        """
        self.names=names
        mvec = np.zeros((12*nside**2,len(names)))
        mvec.fill(fill)

        if roi_table_map is None:
            # merege ROI files
            folder = '%s_table_%d'% (tname, nside)
            if os.path.exists('%s.zip' % folder):
                z=zipfile.ZipFile('%s.zip'% folder)
                files = sorted(z.namelist()) # skip  folder?
                opener = z.open
            else:
                if not os.path.exists(folder):
                    raise Exception('Did not find zip file %s.zip or folder  %s'% (folder,folder))
                opener = open
                files = sorted(glob.glob(os.path.join(outdir, folder,'*.pickle')))
            nf = len(files)
            assert nf>0, 'no pickle files found in %s' % os.path.join(outdir, folder)
            if nf<1728: print 'warning: missing %d files in folder %s_table; will fill with %s' % ((1728-nf), tname,fill)
            pklist = [pickle.load(opener(f)) for f in files]
            index_table = make_index_table(roi_nside, nside)
            i12 = [int(f[-11:-7]) for f in files]

            for index, pk in zip(i12,pklist):
                indeces = index_table[index]
                for i,v in enumerate(pk):
                    mvec[indeces[i]] = v 
        else:
            # straigt from  a map
            rtm = roi_table_map
            # special for a single disc -- kluge a special index_table
            index_table, maps = rtm
            assert len(index_table)==maps.shape[0]
            assert len(names)==maps.shape[1]
            for i,m in enumerate(maps.T):
                mvec[index_table,i] = m
    
        bad = np.isnan(mvec)
        
        if np.any(bad): print 'WARNING: pixels not filled in table' 
        else:
            print 'Table "{}" Filled with columns {}'.format(tname, names)
        self.mvec= mvec

        
    def summary(self, tsmin=(10,25,100)):
        stats = lambda n: (self.mvec>n).sum(axis=0)
        return pd.DataFrame([stats(n) for n in tsmin], columns=self.names, index=['TS>{}'.format(x) for x in tsmin]).T
        return (self.mvec>tsmin).sum(axis=0)

    def write_fits(self,  folder='.' ):
        """combine the tables into a FITS file
        """
        from uw.like2.pub import healpix_map as hpm
        tables = [hpm.HParray(name, self.mvec[:,i]) for i,name in enumerate(self.names)]
        self.fitsfile='hptables_'+'_'.join(self.names)+'_%d.fits' % tables[0].nside
        f = hpm.HEALPixFITS(tables)
        f.write(os.path.join(folder,self.fitsfile))


