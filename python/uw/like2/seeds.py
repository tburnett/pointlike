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
            tsmin=10, lqmax=20,
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

    def add_seed(s):
        # use column 'key' to determine the model to use
        model = maps.table_info[s['key']][1]['model']
        try:
            src=roi.add_source(sources.PointSource(name=s.name, skydir=s['skydir'], model=model))
            if src.model.name=='LogParabola':
                roi.freeze('beta',src.name)
            elif src.model.name=='PLSuperExpCutoff':
                roi.freeze('Cutoff', src.name)
            print '%s: added at %s' % (s.name, s['skydir'])
        except Exception, msg:
            print '*** fail to add source:', msg
            if update_if_exists:
                src = roi.get_source(s.name)
                print '{}: updating existing source at {} '.format(s.name, s['skydir'])
            else:
                print '{}: Fail to add "{}"'.format(s.name, msg)
                return
        # profile
        prof= roi.profile(src.name, set_normalization=True)
        src.ts= prof['ts'] if prof is not None else 0
        
        # fit Norm
        try:
            roi.fit(s.name+'_Norm', tolerance=0., ignore_exception=False)
        except Exception, msg:
            print '\tFailed to fit seed norm: \n\t{}\nTrying full fit'.format(msg)
            return False
        # fit both parameters
        try:
            roi.fit(s.name, tolerance=0., ignore_exception=False)
        except Exception, msg:
            print '\tFailed to fit seed norm and index:'
            return False
        ts = roi.TS()
        print '\nTS = %.1f' % ts,
        if ts<tsmin:
            print ' <%.1f, Fail to add.' % tsmin
            return False
        else: print ' OK'

        # one iteration of pivot change
        iter = 2
        if iter>0 and roi.repivot([src], min_ts=tsmin,select=src.name ):
            iter -=1

        # and a localization: remove if fails or poor
        roi.localize(s.name, update=True, tolerance=1e-3)
        quality = src.ellipse[5] if hasattr(src, 'ellipse') and src.ellipse is not None else None
        if quality is None or quality>lqmax:
            print '\tFailed localization, quality {}, maximum allowed {}'.format(quality, lqmax)
        
        return True
    
    seedfile = kwargs.pop('seedfile', 'seeds/seeds_{}.csv'.format(seedkey))
    seedlist = select_seeds_in_roi(roi, seedfile)
    if len(seedlist)==0:
        print 'no seeds in ROI'
        return False
    else:
        print 'Found {} seeds from {} in this ROI: check positions'.format(len(seedlist),seedfile)



    good = 0  
    for sname,s in seedlist.iterrows():
        print '='*20, sname, 'Initial TS:{:.1f}'.format(s.ts), '='*20
        if not add_seed( s):
            roi.del_source(sname)
        else: good +=1

    return good>0

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
        bdups = bdups[~bdups.index.duplicated()]
        adups = close_df.query('ts_b>ts_a')
        A['dup'] = adups['id_b']
        B['dup'] = bdups['id_a']
        merged= A[pd.isnull(A.dup)].append( B[pd.isnull(B.dup)])
        return merged.sort_values(by='ra').drop('dup',1)
    
    out = tables[0]
    for t in tables[1:]:
        print 'merging...'
        out = merge2(out, t)
    return out

def create_seedfiles(self, seed_folder='seeds', update=False, max_pixels=30000, merge_tolerance=1.0, 
        nside=512, tsmin=14):
    """
    Version of create_seeds used for a maps.MultiMap object
    """
    seedfiles = ['{}/seeds_{}.txt'.format(seed_folder, key) for key in self.names];
    prefix = dict(flat='F', peaked='P', hard='H', soft='S', psr='N')
    if not os.path.exists(seed_folder):
        os.mkdir(seed_folder)
    table_name = self.fitsfile

    # make DataFrame tables from seedfiles
    tables=[]
    modelname = os.getcwd().split('/')[-1]; 
    if modelname.startswith('uw'):
        seedroot=modelname[-3:]
    elif modelname.startswith('year'):
        seedroot='Y'+modelname[-2:]
    elif modelname.startswith('month'):
        seedroot='M'+modelname[-2:]
    else:
        raise Exception('Unrecognized model name, {}. '.format(modelname))

    outfile ='{}/seeds_all.csv'.format(seed_folder) 
    if os.path.exists(outfile) and not update:
        print 'File {} exists'.format(outfile)
        return pd.read_csv(outfile, index_col=0)

    for key, seedfile in zip(self.names, seedfiles):
        print '{}: ...'.format(key),
        if os.path.exists(seedfile) and not update:
            print 'Seedfile {} exists: skipping make_seeds step...'.format(seedfile)
            table = pd.read_table(seedfile, index_col=0)
            table['key']=key
            print 'found {} seeds'.format(len(table))
        else:
            rec = open(seedfile, 'w')
            nseeds = check_ts.make_seeds('test', table_name, fieldname=key, rec=rec, 
                nside=nside,seedroot=seedroot+prefix[key], rcut=tsmin, minsize=1,
                 mask=None, max_pixels=max_pixels,)
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

    u.index.name='name'
    v = u['ra dec b ts size key'.split()]
    v.to_csv(outfile)
    print 'Wrote file {} with {} seeds'.format(outfile, len(u)) 
    return v  
#############################################################
# code to deal with accidentally removed sources
# def add_old_sources(roi, filename='../uw8608/lost.csv'):
#     """add sources from a csv file
#     """
#     lost = pd.read_csv(filename, index_col=0)
#     print 'read file {} with {} sources'.format(filename, len(lost))
#     sel = lost[lost.roiname==roi.name]
#     if len(sel)>0:
#         print 'adding sources {}'.format(list(sel.index))
#     else:
#         print 'No sources found in this ROI'
#         return False
#     ok = False
#     for name, x in sel.iterrows():
#         psr = name.startswith('PSR')
#         if x.ts<25: continue
#         try:
#             roi.add_source(name=x.name, skydir=(x.ra,x.dec),
#                         model='LogParabola({},{},{},{})'.format(x.flux, x.pindex, x.index2,x.e0))
#             roi.fit(x.name); 
#             ts = roi.TS()
#             roi.freeze('beta')
#             if ts<25:
#                 print 'Not good fit: removing'
#                 roi.del_source(x.name)
#             else:
#                 ok = True
#                 roi.get_sed(update=True)# add SED
#         except Exception, msg:
#             print 'Failed to add : {}'.format(msg)
#     return ok
def add_old_sources(roi, filename = 'missing_dict.pkl'):
    """ Version to add sources from old model, missing in this one
    """
    with open(filename) as inp:
        toaddlist = pickle.load(inp)
        assert len(toaddlist.keys())==1728
    roi_index = int(roi.name[5:])
    toadd = toaddlist[roi_index]
    for name, info in toadd.items():
        print 'adding {}'.format(name)
        try:
            s =roi.add_source(name=name, skydir=info['skydir'], model=info['model'])
            roi.profile(s.name, set_normalization=True)
        except:
            pass
    return True