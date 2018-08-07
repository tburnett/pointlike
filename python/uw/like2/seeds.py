"""
Seed processing code
$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/seeds.py,v 1.7 2018/01/27 15:37:17 burnett Exp $

"""
import os, sys, time, pickle, glob,  types
import numpy as np
import pandas as pd
from astropy.io import fits
from skymaps import SkyDir, Band
from uw.utilities import keyword_options
from uw.like2 import (tools, sedfuns, maps, sources, localization, roimodel,)
from uw.like2.pipeline import (check_ts,) #oops stagedict) 
#### need to fix!
from uw.like2.pub import healpix_map


def read_seedfile(seedkey,  filename=None, config=None):

    model_name = os.getcwd().split('/')[-1]

    if model_name.startswith('month') and seedkey=='pgw':
        #monthly mode, need to find and load PGW analysis with rouighly equivalent months
        month=int(model_name[5:]); 
        filename='/nfs/farm/g/glast/g/catalog/transients/TBIN_%d_all_pgw.txt'% (month-1)
        assert os.path.exists(filename), 'PGWAVE file %s not found'% filename  
        try:
            seeds = pd.read_table(filename, sep=' ', skipinitialspace=True, index_col=1,
                header=None,
                names='tbin ra dec k_signif pgw_roi fgl_seed fgl_ra fgl_dec fgl_assoc'.split())
        except Exception,msg:
            raise Exception('Failed to read file %s: %s' % (filename, msg))
        names=[]
        for i,s in seeds.iterrows():
            j = int(s.name[4:6]) if s.name[6]=='_' else int(s.name[4:5])
            names.append('PGW_%02d_%03d_%02d' % (month, int(s.pgw_roi), j))
        seeds['name'] = names    
    elif model_name.startswith('month') and seedkey=='PGW':
        # monthly mode, new format PGwave, in a single FITS file
        month=int(model_name[5:]); 

        
        assert os.path.exists(filename), 'PGWAVE file {} not found'.format( filename)  
        t = fits.open(filename)
        df=pd.DataFrame(t[1].data)
        selector = lambda month : (df.run=='1m   ') & (df.TBIN=='TBIN_{:<2d}'.format(month-1))
        cut = selector(month)
        assert sum(cut)>0, 'No seeds found for month {}'.format(month)
        print 'Found {} PGWave seeds'.format(sum(cut))
        ra = np.array(df.Ra[cut],float)
        dec = np.array(df.Dec[cut],float)
        prefix = 'PG{:02d} '.format(int(month))
        # note making it a string type
        name = np.array([prefix + n.split('_')[-1].strip() for n in 'TBIN_{}_'.format(month-1)+df.PGW_name[cut]])
        seeds = pd.DataFrame([name, ra,dec], index='name ra dec'.split()).T

    elif filename is None and config is not None:
        # assume that config[seedkey] is the filename
        if seedkey in config:
            filename = config[seedkey]
        elif os.path.exists('seeds_{}.csv'.format(seedkey)):
            filename='seeds_{}.csv'.format(seedkey)
        else:
            raise Exception('seedkey {} not found in config, or filename'.format(seedkey))
        if os.path.splitext(filename)=='.fits':
            # a standard FITS catalog
            f = fits.open(os.path.expandvars(filename))
            name, ra, dec = [f[1].data.field(x) for x in 'Source_Name RAJ2000 DEJ2000'.split()]
            seeds = pd.DataFrame([name, np.array(ra,float),np.array(dec,float)],
            index='name ra dec'.split()).T
        else:
            seeds = pd.read_csv(filename)

    elif filename is not None:
        # file is cvs
        seeds = pd.read_csv(filename)
    else:
        # reading a TS seeds file
        t = glob.glob('seeds_%s*' % seedkey)
        assert len(t)==1, 'Seed file search, using key {}, failed to find one file\n\t{}'.format( seedkey,t)
        seedfile=t[0]
        try:
            csv_format=seedfile.split('.')[-1]=='csv'
            if csv_format:
                seeds = pd.read_csv(seedfile)
            else:
                seeds = pd.read_table(seedfile)
        except Exception, msg:
            raise Exception('Failed to read file %s, perhaps empty: %s' %(seedfile, msg))
 
    seeds['skydir'] = map(SkyDir, seeds.ra, seeds.dec)
    seeds['hpindex'] = map( Band(12).index, seeds.skydir)
    # check for duplicated names
    dups = seeds.name.duplicated()
    if sum(dups)>0:
        print '\tRemoving {} duplicate entries'.format(sum(dups))
        return seeds[np.logical_not(dups)]
    return seeds

def select_seeds_in_roi(roi, fn='seeds/seeds_all.csv'):
    """ Read seeds from csv file, return those in the given ROI

    roi : int or Process instance
        if the latter, look up index from roi direction. direction
    """
    if type(roi)!=int:
        roi = Band(12).index(roi.roi_dir)
    seeds = pd.read_csv(fn, index_col=0)
    seeds['skydir'] = map(SkyDir, seeds.ra, seeds.dec)
    seeds.index.name = 'name' 
    sel = np.array(map( Band(12).index, seeds.skydir))==roi
    return seeds[sel]

def add_seeds(roi, seedkey='all', config=None,
            model='PowerLaw(1e-14, 2.2)', 
            associator=None, tsmap_dir='tsmap_fail',
            tsmin=5, lqmax=20,
            update_if_exists=False,
            location_tolerance=0.5,
            pair_tolerance=0.25,
            **kwargs):
    """ add "seeds" from a text file the the current ROI
    
        roi : the ROI object
        seedkey : string
            Expect one of 'pgw' or 'ts' for now. Used by read_seedfile to find the list

        associator :
        tsmap_dir
        mints : float
            minimum TS to accept for addition to the model
        lqmax : float
            maximum localization quality for tentative source
    """
    seedfile = kwargs.pop('seedfile', 'seeds/seeds_{}.csv'.format(seedkey))
    seeds = select_seeds_in_roi(roi, seedfile)
    if len(seeds)==0:
        print 'no seeds in ROI'
        return False
    else:
        print 'Found {} seeds in ROI: check positions'.format(len(seeds))

    # check locations: remove too close to sources in ROI, close pairs
    # get free sources to check close
    skydirs = [s.skydir for s  in roi.free_sources]

    nbad = 0
    def mindist(a):
        d = map(a.difference, skydirs)
        n = np.argmin(d)
        return n,  np.degrees(d[n])
    if len(skydirs)>0:
        dd = np.array(map(mindist, seeds.skydir))[:,1]
        seeds['distance']=dd
        ok = dd>location_tolerance
        nbad = len(ok)-sum(ok)
        if nbad>0:
            print 'Found {} too close (<{} deg) to existing source:\n\t{}'.format(
                nbad, location_tolerance, list(seeds[~ok].index))    
    else:
        ok = np.ones(len(seeds))

    # check for close pairs
    zz = []; sd = seeds.skydir.values
    for i, a in enumerate(sd[:-1]):
        for j, b in enumerate(sd[i+1:]):
            d = np.degrees(a.difference(b))
            if d< pair_tolerance:
                zz.append(( i, i+j+1, d ))
    if len(zz)>0:
        print 'Found {} close pairs, within {} deg, keeping one with larger TS:'.format(len(zz),pair_tolerance)
        for z in zz:
            i,j,d = z
            a,b = seeds.iloc[i], seeds.iloc[j] 
            print '\t{:10} {:.1f} {:10}{:.1f} {:.2f}'.format(a.name, a.ts, b.name, b.ts, z[2])
            ok[j if a.ts>b.ts else i]=False
    seeds['ok'] = ok
    seeds = seeds.query('ok==True')


    # add remaining seeds to ROI to allow fitting
    srclist = []
    for i,s in seeds.iterrows():
        if seedkey=='all':
            # use column 'key' to determine the model to use
            model = maps.table_info[s['key']][1]['model']
        try:
            src=roi.add_source(sources.PointSource(name=s.name, skydir=s['skydir'], model=model))
            if src.model.name=='LogParabola':
                roi.freeze('beta',src.name)
            elif src.model.name=='PLSuperExpCutoff':
                roi.freeze('Cutoff', src.name)
            srclist.append(src)
            print '%s: added at %s' % (s.name, s['skydir'])
        except roimodel.ROImodelException, msg:
            if update_if_exists:
                srclist.append(roi.get_source(s.name))
                print '{}: updating existing at {} '.format(s.name, s['skydir'])
            else:
                print '{}: Fail to add "{}"'.format(s.name, msg)
    # Fit only fluxes for each seed first
    seednames = [s.name for s in srclist]
    llbefore = roi.log_like() # for setup?


    # set normalizations with profile fits
    print 'Performing profile fits to set normalization'
    for src in srclist:
        prof= roi.profile(src.name, set_normalization=True)
        src.ts= prof['ts'] if prof is not None else 0
 

    # now fit all norms at once
    seednorms = [s.name+"_Norm" for s in srclist if roi.get_source(s.name).ts>0]
    if len(seednorms)==0:
        print 'Did not find any seeds to fit'
        return False
    try:
        roi.fit(seednorms, tolerance=0.2, ignore_exception=False)
    except Exception, msg:
        print 'Failed to fit seed norms: \n\t{}\nTrying full fit'.format(msg)
        roi.fit(ignore_exception=True)
    #remove those with low TS
    print 'TS values'
    goodseeds = []
    for sname in seednames:
        ts = roi.TS(sname)
        print '%8s %5.1f ' %(sname, ts),
        if ts<tsmin:
            print '<-- remove'
            roi.del_source(sname)
        else:
            goodseeds.append(sname)
            print 'OK'
        

    # fit all parameter for each one, remove if ts< tsmin 
    seednames = goodseeds
    goodseeds = []
    for sname in seednames:
        roi.fit(sname, tolerance=0, ignore_exception=True)
        ts = roi.TS()
        print '  TS = %.1f' % ts
        if ts<tsmin:
            print ' TS<%.1f, removing from ROI' % tsmin
            roi.del_source(sname)
        elif tsmin>0:
            # one iteration of pivot change
            s = roi.get_source(sname)
            roi.repivot([s], min_ts=5)
            # and a localization: remove if fails or poor
            roi.localize(sname, update=True)
            ellipse = s.__dict__.get('ellipse', None) 
            if ellipse is None or ellipse[5]>lqmax:
                if tsmap_dir is not None and os.path.exists(tsmap_dir):
                    roi.plot_tsmap(sname, outdir=tsmap_dir)
                print '--> removing {} \n\t since did not localize'.format(s)
                roi.del_source(sname)
                continue
            roi.get_sed(sname)
            goodseeds.append(sname)
      
    if len(goodseeds)>0:
        # finally, fit full ROI 
        roi.fit(ignore_exception=True, tolerance=0.2)
        return True
    else:
        return False

def create_seeds(keys = ['ts', 'tsp', 'hard', 'soft'], seed_folder='seeds', tsmin=10, 
            merge_tolerance=1.0, update=False, max_pixels=30000,):
    """Process the 
    """
    #keys =stagedict.stagenames[stagename]['pars']['table_keys'] 
    
    modelname = os.getcwd().split('/')[-1]; 
    if modelname.startswith('uw'):
        seedroot=''
    elif modelname.startswith('year'):
        seedroot='y'+modelname[-2:]
    elif modelname.startswith('month'):
        seedroot='m'+modelname[-2:]
    else:
        raise Exception('Unrecognized model name, {}. '.format(modelname))

    # list of prefix characters for each template
    prefix = dict(ts='M', tsp='P', hard='H', soft='L')
    if not os.path.exists(seed_folder):
        os.mkdir(seed_folder)
    table_name = 'hptables_{}_512.fits'.format('_'.join(keys))
    if not (update or os.path.exists(table_name)):
        print "Checking that all ROI map pickles are present..."
        ok = True;
        for key in keys:
            folder = '{}_table_512'.format(key)
            assert os.path.exists(folder), 'folder {} not found'.format(folder) 
            files = sorted(glob.glob(folder+'/*.pickle'))
            print folder, 
            n = files[0].find('HP12_')+5
            roiset = set([int(name[n:n+4]) for name in files])
            missing = sorted(list(set(range(1728)).difference(roiset)))
            if missing==0: ok = False
            print '{} missing: {}'.format(len(missing), missing ) if len(missing)>0 else 'OK' 
        assert ok, 'One or more missing runs'

        print 'Filling tables...'
        healpix_map.assemble_tables(keys)
    assert os.path.exists(table_name)

    # generate txt files with seeds
    print 'Run cluster analysis for each TS table'
    seedfiles = ['{}/seeds_{}.txt'.format(seed_folder, key) for key in keys]

    # make DataFrame tables from seedfiles
    tables=[]
    for key, seedfile in zip(keys, seedfiles):
        print '{}: ...'.format(key),
        if os.path.exists(seedfile) and not update:
            print 'Seedfile {} exists: skipping make_seeds step...'.format(seedfile)
            table = pd.read_table(seedfile, index_col=0)
            print 'found {} seeds'.format(len(table))
        else:
            rec = open(seedfile, 'w')
            nseeds = check_ts.make_seeds('test', table_name, fieldname=key, rec=rec,
                seedroot=seedroot+prefix[key], rcut=tsmin, minsize=1,mask=None, max_pixels=max_pixels,)
            if nseeds>0:
                #read back, set skydir column, add to list of tables
                print '\tWrote file {} with {} seeds'.format(seedfile, nseeds)
                table = pd.read_table(seedfile, index_col=0)
                table['skydir'] = map(SkyDir, table.ra, table.dec)
                table['key'] = key
            else:
                print '\tFailed to find seeds: file {} not processed.'.format(seedfile)
                continue
        tables.append(table)

    if len(tables)<2:
        print 'No files to merge'
        return

    u = merge_seed_files(tables, merge_tolerance);
    print 'Result of merge with tolerance {} deg: {}/{} kept'.format(merge_tolerance,len(u), sum([len(t) for t in tables]))

    outfile ='{}/seeds_all.csv'.format(seed_folder) 
    u.to_csv(outfile)
    print 'Wrote file {} with {} seeds'.format(outfile, len(u))
            
def merge_seed_files(tables, dist_deg=1.0):
    """Merge multiple seed files

        tables : list of data frames
    """
    dist_rad = np.radians(dist_deg)
    for t in tables:
        t['skydir'] = map(SkyDir, t.ra, t.dec)
    
    def find_close(A,B):
        """ helper function: make a DataFrame with A index containg
        columns of the
        name of the closest entry in B, and its distance
        A, B : DataFrame objects each with a skydir column
        """
        def mindist(a):
            d = map(a.difference, B.skydir.values)
            n = np.argmin(d)
            return [B.index[n],  B.ts[n], np.degrees(d[n])]
        df = pd.DataFrame( map(mindist,  A.skydir.values),
               index=A.index,      columns=('id_b', 'ts_b', 'distance'))
        df['ts_a'] = A.ts
        df['id_a'] = A.index
        return df
    
    def merge2(A,B):
        "Merge two tables"
        close_df  = find_close(A,B).query('distance<{}'.format(dist_rad))
        bdups = close_df.query('ts_b<ts_a')
        bdups.index=bdups.id_b
        adups = close_df.query('ts_b>ts_a')
        A['dup'] = adups['id_b']
        B['dup'] = bdups['id_a']
        merged= A[pd.isnull(A.dup)].append( B[pd.isnull(B.dup)])
        return merged.sort_values(by='ra').drop('dup',1)
    
    out = tables[0]
    for t in tables[1:]:
        out = merge2(out, t)
    return out

def create_seedfiles(self, seed_folder='seeds', update=False, max_pixels=30000, merge_tolerance=1.0, 
        tsmin=16):
    """
    Version of create_seeds used for a maps.MultiMap object
    """
    seedfiles = ['{}/seeds_{}.txt'.format(seed_folder, key) for key in self.names];
    prefix = dict(flat='F', peaked='P', hard='H', soft='S')
    if not os.path.exists(seed_folder):
        os.mkdir(seed_folder)
    table_name = self.fitsfile

    # make DataFrame tables from seedfiles
    tables=[]
    modelname = os.getcwd().split('/')[-1]; 
    if modelname.startswith('uw'):
        seedroot='S'+modelname[-2:]
    elif modelname.startswith('year'):
        seedroot='Y'+modelname[-2:]
    elif modelname.startswith('month'):
        seedroot='M'+modelname[-2:]
    else:
        raise Exception('Unrecognized model name, {}. '.format(modelname))

    outfile ='{}/seeds_all.csv'.format(seed_folder) 
    if os.path.exists(outfile) and not update:
        print 'File {} exists'.format(outfile)
        return

    for key, seedfile in zip(self.names, seedfiles):
        print '{}: ...'.format(key),
        if os.path.exists(seedfile) and not update:
            print 'Seedfile {} exists: skipping make_seeds step...'.format(seedfile)
            table = pd.read_table(seedfile, index_col=0)
            print 'found {} seeds'.format(len(table))
        else:
            rec = open(seedfile, 'w')
            nseeds = check_ts.make_seeds('test', table_name, fieldname=key, rec=rec,
                seedroot=seedroot+prefix[key], rcut=tsmin, minsize=1, mask=None, max_pixels=max_pixels,)
            if nseeds>0:
                #read back, set skydir column, add to list of tables
                print '\tWrote file {} with {} seeds'.format(seedfile, nseeds)
                table = pd.read_table(seedfile, index_col=0)
                table['skydir'] = map(SkyDir, table.ra, table.dec)
                table['key'] = key
            else:
                print '\tFailed to find seeds: file {} not processed.'.format(seedfile)
                continue
        tables.append(table)

    u = merge_seed_files(tables, merge_tolerance);
    print 'Result of merge with tolerance {} deg: {}/{} kept'.format(merge_tolerance,len(u), sum([len(t) for t in tables]))

    u.index='name'
    u['name ra dec ts size key'.split()].to_csv(outfile)
    print 'Wrote file {} with {} seeds'.format(outfile, len(u))   