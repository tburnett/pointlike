"""  
 Setup the ROIband objects for an ROI
 
    $Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/dataset.py,v 1.42 2017/11/22 00:50:43 burnett Exp $

    authors: T Burnett, M Kerr, J. Lande
"""
version='$Revision: 1.42 $'.split()[1]
import os, glob, types, yaml 
import cPickle as pickle
import numpy as np
import skymaps, pointlike #from the science tools
from uw.data import dataman, dssman, binned_data
from uw.utilities import keyword_options
from uw.irfs import caldb # new code

class DataSetError(Exception):pass

class DataSpecification(object):
    """ 
    """
    def __init__(self, folder, quiet=True, **data):
        """
        folder : string
            the path to the folder where the dictionary was found
        data : dict
            dictionary that contains entries sufficient to describe the data, and how to generate it
            
        Note: if interval found, modify the binfile and ltcube name to include
        """
        if folder=='.': folder = os.getcwd()
        def abspath(path):
            if not os.path.isabs(path):
                path = os.path.join(folder,path) 
            return path
            
        for key in 'ft1files ft2files binfile ltcube'.split():
            if key in data and data[key] is not None:
                if hasattr(data[key],'__iter__'):
                    data[key] = [os.path.expandvars(x) for x in data[key]]
                else:
                    #Allow glob patterns for FT1/2 files
                    if key in ('ft1files','ft2files'):
                        data[key] = sorted(glob.glob(os.path.expandvars(data[key])))
                    else:
                        data[key] = (os.path.expandvars(data[key]),)
                data[key] = [abspath(f) for f in data[key]]
                if len(data[key])==1: data[key]=data[key][0]

        interval = data.get('interval', None)
        if interval is not None:
            if interval.startswith('month_'):
                # using 'month_xx' where xx is 01,02,...
                if interval[-1]=='*':
                    interval = os.path.split(os.getcwd())[-1]
                    if interval.find('month')<0 :
                        raise DataSetError('Specified * for month, but path does not contain monthxx: found %s' % interval)
                for name in ('binfile', 'ltcube', 'ft1files'):
                    if data[name] is not None:
                        data[name]=data[name].replace('*', interval[-2:])
            elif interval.startswith('year_'):
               # using 'year_xx' where xx is 01,02,...
                if interval[-1]=='*':
                    interval = os.path.split(os.getcwd())[-1]
                    if interval.find('year')<0 :
                        raise DataSetError('Specified * for year, but path does not contain yearxx: found %s' % interval)
                for name in ('binfile', 'ltcube', 'ft1files'):
                    if data[name] is not None:
                        data[name]=data[name].replace('*', interval[-2:])
                
            else:
               for name in ('binfile', 'ltcube'):
                    data[name]=data[name].replace('.fit', '_%s.fit'%interval)

        if not os.path.exists(data['binfile']):
            print 'binned file "{}" not found: have to create from FT1'.format(data['binfile'])
            ft1s = data['ft1files'] if hasattr(data['ft1files'],'__iter__') else (data['ft1files'],)
            for ft1 in ft1s:
                if ft1 is None or not os.path.exists(ft1):
                    raise DataSetError('FT1 file {} does not exist, needed to create binned photon data file'.format(ft1))
        # check for livetime cube file or files
        nltcube= glob.glob(data['ltcube'])
        if len(nltcube)==0:
            print 'ltcube file {} not found: checking ft2 files'.format(data['ltcube'])
            ft2s = data['ft2files'] if hasattr(data['ft2files'],'__iter__') else (data['ft2files'],)
            if len(ft2s)==0 or ft2s[0]=='none':
                raise DataSetError('No FT2 file(s) specified or found')
            for ft2 in ft2s:
                if ft2 is None or not os.path.exists(ft2):
                    raise DataSetError('FT2 file {} does not exist, needed to create binned photon data file'.format(ft2))

        self.__dict__.update(data)
        if not quiet: print 'data spec:\n', str(self.__dict__)

    def __str__(self):
        return self.data_name
    def __repr__(self):
        return '%s.%s: %s' % (self.__module__, self.__class__.__name__, self.data_name)
        
class Interval(dict):
    """ dictionary for looking up an interval
        Use if no intervals.py file found
        returns an MET tuple, e.g.
        Interval()['day01']
        
    """
    start = 239557417
    day = 24*60*60
    length= [day, 7*day, 365.25*day/12., 365.25*day]
    def __getitem__(self, x):
        for i, kind in enumerate(['day', 'week', 'month', 'year']):
            if not x.startswith(kind): continue
            n = int(x[len(kind):])
            s,w = Interval.start, Interval.length[i]
            return (s+(n-1)*w, s+n*w)
        raise KeyError(x)
        
class DataSet(dataman.DataSpec):
    """
    subclass of data.dataman.DataSpec that uses dicts to specify data, 
        and sets up instrument response and  ROI setup
        
    Note the __call__ function
    """
    defaults = dataman.DataSpec.defaults + (
        ' legacy key words that should be in DSS',
        ('zenithcut',100,'Maximum spacecraft pointing angle with respect to zenith to allow'),
        ('thetacut',66.4,'Cut on photon incidence angle'),
        ('use_weighted_livetime',False,'Use the weighted livetime'),
        ('event_class', 'source', 'either source or clean'),

        ('gti',    None, 'good time interval'),
        ('dss',    None, 'DSS keyword object'),
        ('version', None, ''),
        ('binner', None, ''),
        ('bins',  None,  ''),
        ('interval', None, 'Name for a specific time period: key in a dictionary to (start, stop)'),
        ('binfile', None, 'Replacement name for binfile'),

        ('legacy', False, 'needed to allow no DSS in livetime cubes'),
        
        'keywords controlling instrument response',
        ('irf',None,'Which IRF to use'),
        ('psf_irf',None,'specify a different IRF to use for the PSF; must be in same format/location as typical IRF file!'),
        ('CALDB','$CALDB','override the CALDB specified by $CALDB.'),
        ('custom_irf_dir',None,'override the CUSTOM_IRF_DIR specified by the env. variable'),
        
        'keywords defining actual ROI setup',
        ('emin',100,'Minimum energy for selected bands'),
        ('emax',1e6,'Maximum energy for selected bands'),
        ('minROI', 7, 'minimum ROI radius'),
        ('maxROI', 7, 'maximum ROI radius'),
        #('phase_factor', 1.0, 'phase factor for pulsar phase analysis'),

        ('nocreate', False, 'Set True to suppress file creation'),
        ('postpone', False, 'Set true to postpone opening data until needed'),
        ('quiet', True, 'set False for some output'),
        ('verbose', False, 'more output'),)
        
    
    @keyword_options.decorate(defaults)
    def __init__(self, dataset_spec, **kwargs):
        """
        Create a new DataSet object.

        dataset_spec: str | dict
            an instance of DataSpecification with links to the FT1/FT2,
                            and/or binned data / livetime cube needed for analysis
                            (see docstring for that class) 
        """

        dataspec = self._process_dataset(dataset_spec, quiet=kwargs.get('quiet',False),
            interval=kwargs.pop('interval',None),
            #binfile=kwargs.pop('binfile',None), 
            ).__dict__
        dataspec.update(
                ft1=None, ####### no longer used here?  dataspec.pop('ft1files',None), 
                ft2=dataspec.pop('ft2files',None),
                )

        # replace binfile if requested (for simuation, at least)
 
        new_binfile = kwargs.pop('binfile', None)

        dataspec.update(kwargs)
 
        if new_binfile is not None:
            print 'binfile: {}'.format(new_binfile)
            if not os.path.exists(new_binfile):
                new_binfile = os.path.join(os.path.expandvars('$FERMI/data'),new_binfile)
                if not os.path.exists(new_binfile):
                    raise DataSetError('Replacement binfile not found, {}'.format(new_binfile))
            dataspec['binfile']=new_binfile

        # Now invoke the superclass to actually load the data, which may involve creating the binfile and livetime cube
        super(DataSet,self).__init__(  **dataspec)
        assert self.irf is not None, 'irf was not specifed!'
        
        # new IRF management
        self.CALDBManager = caldb.CALDB(CALDB_dir=self.CALDB, irfname=self.irf)
        if self.exposure_cube is None:
            ltcubes = glob.glob(self.ltcube) #allow for more than one
            self.lt = [skymaps.LivetimeCube(lt,weighted=False) for lt in ltcubes]
            if self.use_weighted_livetime:
                self.weighted_lt = [skymaps.LivetimeCube(lt,weighted=True) for lt in ltcubes]
        else:
            self.lt = None
        if not self.postpone:
            self._load_binfile()
    
    def livetime_cube(self, event_type):
        """return appropriate livetime cube"""
        if not self.psf_event_types or len(self.lt)==1:
            return self.lt[0]
        assert len(self.lt)==len(self.event_types), 'Expected 4 live time cubes'
        return self.lt[event_type-self.event_types[0]]

    def _process_dataset(self, dataset, binfile=None, interval=None, month=None, quiet=False):
        """ Parse the dataset as a string lookup key, or a dict
            interval: string
                name of a 
            month: sub spec.
        """

        if isinstance(dataset, dict):
            dataset['event_class_bit']=dict(source=2, clean=3, extraclean=4)[dataset.get('event_class','source').lower()]
            dataset.setdefault('theta_cut', 66.422)
            self.name = dataset.get('data_name', '(noname)')
            self.dict_file = 'input dict'
            return DataSpecification(os.path.expandvars('$FERMI/data'),  interval=None, gti_mask=None, **dataset)
            
        self.name=dataset
        folders = ['.'] + glob.glob( os.path.join(os.path.expandvars('$FERMI'),'data'))
        for folder in folders :
            self.dict_file = dict_file=os.path.join(folder, 'dataspec.py')
            if not os.path.exists(dict_file):
                continue
            try:
                ldict = eval(open(dict_file).read())
            except Exception, msg:
                raise DataSetError( 'Data dictionary file %s not valid: %s' % (dict_file, msg))
            if interval is not None and not (interval.startswith('month_') or interval.startswith('year_')):
                # note that the underscore is to specify the designated month or year, with their own files
                try:
                    pyfile = os.path.join(folder, 'intervals.py')
                    idict = eval(open(pyfile).read()) if os.path.exists(pyfile) else Interval()
                    gr = idict[interval]
                    gti_mask = skymaps.Gti([gr[0]], [gr[1]])
                    if True: #self.verbose: 
                        if not quiet: print 'apply gti mask %s, %s' %(interval, gti_mask)
                except Exception, msg:
                    raise DataSetError('Interval dictionary file %s, key %s, problem: %s'\
                                % (pyfile, interval, msg))
            else: gti_mask=None
            if dataset in ldict: 
                if not quiet: print 'Opening dataset %s from key in %s' % (dataset, dict_file)
                # translate event class name to appropriate bit
                ddict=ldict[dataset]
                #ddict['event_class_bit']= 4 ######FOR PASS8 now 
                ddict['event_class_bit']=dict(source=2, clean=3, extraclean=4)[ddict.get('event_class','source').lower()]
                return DataSpecification(folder,  interval=interval, gti_mask=gti_mask,  **ddict)
        raise DataSetError('dataset name "%s" not found in %s' % (dataset, folders))

    def load(self):
        if self.postpone:
            self._load_binfile()
            self.postpone=False
 
    def _load_binfile(self):
        if not self.quiet: print 'loading binfile %s ...' % self.binfile ,
        try:
            self.dmap = skymaps.BinnedPhotonData(self.binfile)  
            if not self.quiet: print 'found %d photons in %d bands, energies %.0f-%.0f MeV'\
                    % (self.dmap.photonCount(),len(self.dmap), self.dmap[1].emin(), self.dmap[len(self.dmap)-1].emax())
        except RuntimeError:
            self.dmap = binned_data.BinFile(self.binfile)
            if not self.quiet: print self.dmap
        if self.verbose:
            self.dmap.info()
            print '---------------------'
            
    def info(self, out=None):
        """ formatted table of band contents """
        self.load()
        print >>out, 'File: %s ' %self.binfile
        print >>out, '\n  index    emin      emax  class nside     photons'
        total = 0
        def bignum(n):
            t = '%9d' % n
            return '  '+' '.join([t[0:3],t[3:6],t[6:]])
        for i,band in enumerate(self.dmap):
            fmt = '%5d'+2*'%10d'+2*'%6d'+'%12s'
            print fmt % (i, round(band.emin()), round(band.emax()), 
                    band.event_class()&15, band.nside(), bignum(band.photons()))
            total += band.photons()
        print >>out, 'total%45s'% bignum(total)

    def dss_info(self, indent=''):   
        """ return a formatted table of the DSS keywords
        """
        if self.dss is None:
            return indent+"DSS: no info since FT1 files not specified"
        s = indent+'DSS: %-15s  %-10s%-10s%-10s\n'% tuple('name value units ref'.split() )
        s+= indent
        s+= indent.join(['       %-15s  %-10s%-10s %s\n' %\
            (dss['TYP'],dss['VAL'],dss['UNI'],dss['REF'] ) for dss in self.dss])
        return s
    
    def __str__(self):
        """ Pretty print of cuts/data."""
        s = [] #collections.deque()
        if self.binsperdec is not None:
            s.append('Bins per decade: {0}'.format(self.binsperdec))
        def process_ft(label, files):
            if files is None: 
                #s.append(label + '\tNone')
                return
            s.append(label)
            if len(files) < 10:
                s.append('\t'+ '\n\t'.join(files))
            else:
                s.append('\t'+'\n\t'.join(files[:2]))
                s.append('\t...')
                s.append('\t'+'\n\t'.join(files[-2:]))
        process_ft('FT1 files: ',self.ft1files)
        process_ft('FT2 files: ',self.ft2files)
        s.append('Binned data: {0}'.format(self.binfile))
        if not self.postpone:
            s.append('           :  {:,.0f} photons, {:d} energy bands from {:} to {:} MeV'.format(
                    self.dmap.photonCount(), len(self.dmap), self.dmap[1].emin(), self.dmap[len(self.dmap)-1].emax()))
        else:
            s.append('            (load postponed)')
        s.append('Livetime cube: {0}'.format(self.ltcube))
        s.append(self.gti.__str__())
        s.append(self.dss_info())
        return 'dataset "%s", found in %s:\n  '%(self.name, self.dict_file ) +'\n  '.join(s)
    def __repr__(self):
        return self.__str__()


def main(datadict=dict(dataname='P7_V4_SOURCE_4bpd'), analysis_kw=dict(irf='P7SOURCE_V6')):
    """ for testing: must define a dataset, and and at least an IRF"""
    dataset = DataSet(datadict['dataname'], **analysis_kw)
    return dataset
    
def validate(model_path='.', interval=None, nocreate=False, logfile='dataset.txt', quiet=False,):
    """
    validate the dataset for a model, as defined by the pipeline architecture
    
    model_path: string
        full path to the folder containing the model info, expect to find a file config.txt
        can contain environment variables with $ prefix
    
    interval: [string | None]
        name for an interval to use: override any specified in the config.txt
        
    nocreate: bool
        if False (default), try to create the binned photon data and livetime cubes
        if True, will return False if either file has to be generated
        
    logfile: string
        if not None, name of a file to write info
        
    returns True if the files exist or have been created OK
    """
    actual_path =  os.path.expandvars(model_path)
    assert os.path.exists(actual_path), 'Model path %s not found' % actual_path
    config = '../config.yaml' 
    # should now check for config.txt as well?
    if not os.path.exists(config):
        config = 'config.yaml'
    assert os.path.exists(config), 'Could not find config.txt in current or parent folder'
    try:
        #modelspec = eval(open(os.path.join(actual_path, config)).read())
        modelspec = yaml.load(open(config))
    except Exception, msg:
        raise DataSetError('could not evaluate config.yaml: %s' % msg)
    
    #modelspec = configuration.Configuration(model_path, postpone=True)
    try:
        datadict = modelspec['datadict']
    except Exception,msg:
        print 'Failed to find data set: {}'.format(msg)
        return False
    dataname = datadict['dataname']
    if interval is None:
        interval= datadict.get('interval', None)
        print 'Using interval %s' %interval
    try:
        #dset = DataSet(dataname, irf=modelspec['irf'], nocreate=nocreate, quiet=quiet, **datadict)
        dset = DataSet(dataname,  interval=interval, irf=modelspec['irf'], nocreate=nocreate, quiet=quiet, )
        if logfile is not None: open(logfile, 'w').write(dset.__str__())
    except Exception, msg:
        print 'Failed: %s ' % msg
        raise 
        return False
    return True
    
