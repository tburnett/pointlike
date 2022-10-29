"""
Check the residual TS maps for clusters
$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/pipeline/check_ts.py,v 1.14 2018/01/27 15:38:26 burnett Exp $

"""

import os, sys, pickle, argparse, glob
from astropy.io import fits as pyfits
from skymaps import Band, SkyDir
import numpy as np
import pylab as plt
import pandas as pd
from pointlike import IntVector

# this to fix the cluster mean
import healpy
# nside=512
# band = Band(nside)
# def sdir(index):
#     return band.dir(int(index))
band = None
sdir = None


class TSdata(object):
    def __init__(self, outdir, filename, nside=512, fieldname='ts'):
        full_filename = os.path.join(outdir, filename)
        band = Band(nside)
        self.sdir = lambda index: band.dir(index)
        assert os.path.exists(full_filename), 'file, %s, not found' % full_filename
        self.rts = pyfits.open(full_filename)[1].data.field(fieldname)
        assert len(self.rts)==12*(nside)**2, 'wrong nside in file %s: expect %d' %(filename, nside)
        self.glat=None# np.array([sdir(i).b() for i in range(len(self.rts))])
        #self.glon= np.array([sdir(i).l() for i in range(len(self.rts))])
        #print 'read %s ok' %filename
        self.fieldname=fieldname

    def select(self, ts_min=0, b_min=0):
        cut = (self.rts>ts_min)* (abs(self.glat)>b_min)
        return self.rts[cut]
        
    def indices(self, ts_min=10, b_min=0, mask=None):
        """ return an array of indices satisfying the criteria
            mask is nside 512 array to further select
        """
        if b_min>0 and self.glat is None:
            self.glat =np.array([sdir(i).b() for i in range(len(self.rts))])
            cut = (self.rts>ts_min) & (abs(self.glat)>b_min)
        else:
            cut = (self.rts>ts_min)
        if mask is not None:
            cut = cut & mask
        return np.arange(len(self.rts))[cut]
        
def rts_hist(tsdata, fignum=1, bcut=10, bins=np.linspace(0,100,51)):
    plt.close(fignum);
    plt.figure(fignum)
    plt.hist(tsdata.select(),           bins, log=True, label='all')
    plt.hist(tsdata.select(b_min=bcut), bins, log=True, label='|b|>%d'%bcut)
    plt.legend(); plt.grid(True)
    plt.xlabel('TS')

def neighbors(i,j):
    iv = IntVector()
    n = band.findNeighbors(int(i),iv)
    return j in iv

def ajacent_ts(i,rts):
    iv = IntVector()
    n = band.findNeighbors(int(i),iv)
    ats = [rts[i] for i in iv[:4]]
    return sum(ats), max(ats)

def grow(indeces):
    i = 0
    cluster =list(indeces[0:1])
    remainder = indeces[1:]
    while i<len(cluster):
        newremain = []
        for j in remainder:
            if neighbors(cluster[i],j): cluster.append(j)
            else: newremain.append(j)
        i += 1
        remainder = newremain
    return cluster, remainder
        
def cluster(indices, quiet=False):
    if not quiet:
        print 'Clustering %d pixels...' % len(indices)
        sys.stdout.flush()
    ret = []
    rem = indices
    while len(rem)>0:
        clu, rem = grow(rem)
        ret.append(clu)
    if not quiet:
        print 'Found %d clusters' %len(ret)
    return ret

def subclusters(clust, rts, mints=25, quiet=True):
    """split the cluster into multiple clusters with higher thereshold
    """
    cut_cluster = filter(lambda i: rts[i]>mints, clust)
    if len(cut_cluster)==0: return []
    clusters = cluster(cut_cluster, quiet)
    return clusters

def split_clusters(clusters, rts, maxsize=25, split_ts=25):
    # loop over clusters, make a list of split clusters
    splits = []
    for clu in clusters:
        if len(clu)<maxsize: continue
        t = subclusters(clu, rts, split_ts)
        if len(t)<2: continue
        splits += t
    return splits
    

class Cluster(object):
    def __init__(self, rts):
        self.rts = rts
        self.nside = healpy.get_nside(rts)
    
    def group(self, clu):
        """ clu: list of pixel indices"""
        ts = np.array([self.rts[i] for i in clu])
        ### This fails if the cluster straddles ra=0!
        # #print clu, ts
        # dirs = [sdir(i) for i in clu]
        # ra   = np.array([s.ra() for s in dirs])
        # #print ra
        # dec  = np.array([s.dec() for s in dirs])
        # #print dec
        # wra = sum(ts*ra)/sum(ts)
        # wdec = sum(ts*dec)/sum(ts)
        ### new code using healpy
        vec =np.array(healpy.pix2vec(self.nside, clu))
        wvec = (vec*ts).mean(axis=1)/ts.mean()
        l,b = healpy.vec2dir(wvec, lonlat=True)
        self.ts = ts.max()
        self.sdir = SkyDir(l,b, SkyDir.GALACTIC )
 
def monthly_ecliptic_mask( month, elat_max=5):
    """return a nside=512 mask for the given month, an integer starting at 1
    """
    info = pd.read_csv(open(os.path.expandvars(
                '$FERMI/skymodels/P301_monthly/month_info.csv')),index_col=0)
    ecliptic_info= pickle.load(open(os.path.expandvars('$FERMI/misc/ecliptic_512.pickle')))
    elon_vec = ecliptic_info[:,0]
    elat_mask = np.abs(ecliptic_info[:,1])<5
    elon_min = info.sun_elon[month]
    try:
        elon_max = info.sun_elon[month+1]
    except:
        elon_max = elon_min+30
    if elon_min< elon_max:
        elon_mask = (elon_vec>elon_min) & (elon_vec<elon_max)
    else:
        elon_mask = (elon_vec>elon_min) | (elon_vec<elon_max)
                                        
    return elat_mask & elon_mask
        

def make_seeds(tsdata,  filename, fieldname='ts', nside=512 ,rcut=10, bcut=0, 
		out=None, rec=None, seedroot='SEED', minsize=1, max_pixels=30000, mask=None):

    """
    tsdata: object created by TSdata | string | None
        if not a TSdata object, create the TSdata object using filename and fieldname
        
    rec: open file to write tab-delimited file to
    """
    global band, sdir
    band = Band(nside) # replace global
    sdir = lambda index: band.dir(index)
    
    if not isinstance(tsdata, TSdata):
        tsdata = TSdata(outdir='.', filename=filename, fieldname=fieldname, nside=nside)

    # make list of indices of pixels with ts and b above thresholds
    indices  = tsdata.indices(rcut,bcut,mask)
    if len(indices)>max_pixels:
        print 'Too many pixels above TS>{}, {}>{}, to cluster'.format(rcut, len(indices), max_pixels)
        return 0
        
    # create list of the clustered results: each a list of the pixel indeces    
    clusters = cluster(indices)
    
    # split large clusters; add those which have 2 or more sub clusters
    clusters += split_clusters(clusters, tsdata.rts)
    print 'Added split clusters, now %d total' % len(clusters)
    
    # now create list of seeds from the clusters
    cl = Cluster(tsdata.rts)
    if out is not None: print >> out,  '# Region file format: DS9 version 4.0 global color=green'
    if rec is not None:
        print >> rec, 'name\tra\tdec\tts\tsize\tl\tb'
    for i,x in enumerate(clusters):
        if len(x)<minsize: continue
        cl.group(x)
        if out is not None: 
            print >>out,'fk5; point(%8.3f, %8.3f) # point=cross text={%d:%d %.1f}'%\
                ( cl.sdir.ra(), cl.sdir.dec(),i, len(x), cl.ts)
        if rec is not None:
            print >>rec, '%s-%04d\t%8.3f \t%8.3f\t %8.1f\t%8d\t%8.3f \t%8.3f ' %\
                (seedroot, i,cl.sdir.ra(), cl.sdir.dec(),  cl.ts, len(x), cl.sdir.l(),cl.sdir.b())
        
    if rec is not None and rec != sys.stdout: rec.close()
    if out is not None : out.close()
    return len(clusters)
    
def pipe_make_seeds(skymodel, fieldname='ts', prefix_char='S', filenamepattern='seeds/seeds_{}.txt',  minsize=1): # changed from 2!
    """
        # Special check for a month, which should mask out the Sun
    """
    if not os.path.exists('seeds'):
        os.mkdir('seeds')
    if skymodel.startswith('month'):
        month=int(skymodel[5:]);
        try:
            mask = monthly_ecliptic_mask( month)
            print 'created a mask for month %d, with %d pixels set' % (month, sum(mask))
            seedroot = fieldname.upper()+ skymodel[-2:] 
        except:
            mask=None
            print 'No mask found'
            seedroot='TSxx'
    else: 
        mask=None
        seedroot=skymodel.replace('uw', prefix_char)
    fnames = glob.glob('hptables_*_%d.fits' % nside )
    assert len(fnames)>0, 'did not find hptables_*_%d.fits file' %nside
    fname=None
    for fname in fnames:
        t = fname.split('_')
        if fieldname in t:
            print 'Found table %s in file %s' %(fieldname, fname)
            break
    assert fname is not None, 'Table %s not found in files %s' %(fieldname, fnames)

    # Create the seed list by clustering the pixels in the tsmap
    filename = filenamepattern.format( fieldname )
    rec = open(filename, 'w')
    nseeds = make_seeds('test', fname, fieldname=fieldname, rec=rec,
        seedroot=seedroot, minsize=minsize,
        mask=~mask if mask is not None else None)
    print 'Wrote file {} with {} seeds'.format(filename, nseeds)

 
def main(args):
    print args
    global nside
    nside=int(args.nside)
    assert os.path.exists(args.files[0]), 'did not find file %s'%args.files[0]
    tsdata = TSdata('.', args.files[0], args.tsfield)
    rec = open(args.files[1], 'w')
    make_seeds(tsdata, rcut=args.tsmin, bcut=float(args.bmin), rec=rec, seedroot=args.seedroot, minsize=args.minsize)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='examine a TS map file for clusters, create a file of source candidates')
    parser.add_argument('files', nargs=2, help='input FITS file and output text file')
    parser.add_argument('--tsmin',    help='minimum TS for cluster', default=10)
    parser.add_argument('--bmin',     help='minimum |b|',            default=0)
    parser.add_argument('--minsize',  help='minimum cluster size',   default=2)
    parser.add_argument('--seedroot', help='root for seed names',    default='SEED')
    parser.add_argument('--nside',    help='nside for healpix map',  default=nside)
    parser.add_argument('--tsfield',  help='name of field with TS data',  default='ts')
    

    args = parser.parse_args()
    main(args)


    
