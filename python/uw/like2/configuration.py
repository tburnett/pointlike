"""
Manage the analysis configuration

"""
import os, sys, types, StringIO, pprint, yaml
import numpy as np
from uw.irfs import irfman
from . import ( dataset, exposure, psf, from_xml)
import skymaps
from uw.utilities import keyword_options
        
class ROIspec(object):
    """ define an ROI specification
    """
    def __init__(self, healpix_index=None, pos=None, radius=5, radius_factor=0.5):
        """
        parameters
        ---------
        healpix_index : [None | int ]
        pos : [None | tuple of float]
        radius : float
        radius_factor : float
            Factor to apply to define the ROI size for extraction of data to fit.
            This is needed to allow enough size to perform the convolution using a square grid
        """
        if healpix_index is not None:
            self.index = healpix_index
            self.pos = skymaps.Band(12).dir(self.index)
            self.radius=radius
            self.name = 'HP12_%04d' % self.index
        elif pos is not None:
            self.index=None
            self.pos=skymaps.SkyDir(*pos[:2])
            self.radius = pos[2]*radius_factor if len(pos)>2 else radius
            self.name = 'ROI(%.3f,%.3f, %.1f)' % ( self.pos.ra(),self.pos.dec(), self.radius)
        else:
            # no explicit index or position
            self.name = None
            pass

    def __repr__(self):
        s = '%s.%s:' %  (self.__module__, self.__class__.__name__)
        if self.index is None and self.pos is not None:
            return s + ' position, radius= (%.3f,%.3f), %.1f' % ( self.pos.ra(),self.pos.dec(), self.radius)
        elif self.index is not None and self.index>=0:
            return s + 'HEALPix index: %d -> dir(%.3f,%.3f), radius %.1f' % (self.index, self.pos.ra(), self.pos.dec(), self.radius)
        else:
            return s + 'unspecified HEALPix'
        
class Configuration(dict):
    """
    Manage an analysis configuration: the IRF, exposure and data
    This is used by the band class EnergyBand to set up the PSF and exposure for the energy 
    range and sky position.
    """
    defaults =(
        ('bins_per_decade', 4, 'number of bins per decade'),
        ('CALDB', '$CALDB', 'override the CALDB specified by the env var'),
        ('irf', None,  'Set to override value in config.txt : may find in custom_irf_dir'),
        ('extended', None, 'Set to override value in config.txt'),
        ('nocreate', False, 'Set True to prevent creation of a binned photon file'),
        ('quiet', False, 'set to suppress most output'),
        ('postpone', False, 'set true to postpone loading data until requested'),
        ('use_old_irf_code', False, 'allow testing with this switch'),
        )

    @keyword_options.decorate(defaults)
    def __init__(self, configdir='.', **kwargs):
        """ 
        parameters
        ----------
        configdir: folder containing configuration definition, file config.txt. 
            If it is not found, look in the parent directory. If files are in both,
            first load the parent, then update with the configdir version.
        It is a Python dictionary, and must have the keys:
            irf : a text string name for the IRF
            diffuse: a dict defining the diffuse components
                see diffuse.DIffuseDict
            extended: string which is the path to the extended folder
                see extended.ExtendedCatalog
        A data source must be specified: one of two keys must be present:
            datadict: a dict with at least a key 'dataname', its value a key in $FERMI/data/datadict.py
            dataspec: a dict 
        The ROI can be specified with a key
            position: a dict with keys ra, dec, radius.
        The model, or list of sources, is specified in one of the following ways:
            a key input_model
        
        If there is a key 'input_model' and the file 'pickle.zip' does not exist in the directory
        will look in input_model['path'] folder
        """
        keyword_options.process(self, kwargs)

        config = self.load_config(configdir) 
        self.update(config)            
        # check for override from kwargs
        for key in 'extended irf'.split():
            if self.__dict__.get(key, None) is None: 
                if not self.quiet and kwargs.get(key) is not None:
                    print '%s : override config.txt, "%s" => "%s"' %(key, kwargs.get(key,None), config.get(key,None))
                self.__dict__[key]=config.get(key, None)

        # set up IRF, possibly special CALDB
        # ----------
        
        irf = config['irf']
        self.caldb = config.get('CALDB', self.CALDB)
        if 'CUSTOM_IRF_DIR' not in os.environ and os.path.exists(os.path.expandvars('$FERMI/custom_irfs')):
            os.environ['CUSTOM_IRF_DIR'] = os.path.expandvars('$FERMI/custom_irfs')
        custom_irf_dir = os.environ['CUSTOM_IRF_DIR']

        # define diffuse dictionary for constucting spatial models
        # --------------------------------------------------------
        
        self.diffuse = config['diffuse']
        
        # set up dataset -- either datadict with a key to $FERMI/data/dataspec.py, or 'dataspec', with value the dict
        # --------------
        
        datadict = config.get('datadict', None)
        dataspec = config.get('dataspec', None)
        if datadict is not None:
            self.dataset = dataset.DataSet(datadict['dataname'], 
                    interval=datadict.get('interval',None),
                    binfile=datadict.get('binfile', None),
                    nocreate = self.nocreate,
                    CALDB = self.caldb,
                    irf = irf,
                    quiet = self.quiet,
                    postpone=self.postpone,
                    )
        elif dataspec is not None:
            self.dataset = dataset.DataSet(dataspec,
                    irf=irf,
                    CALDB = self.caldb,
                    quiet=self.quiet,
                    nocreate=self.nocreate,
                    postpone = self.postpone,
                                )
        
        else:
            raise Exception('Neither datadict nor dataspec keys in config.txt')
            
        if not self.use_old_irf_code and self.dataset.psf_event_types:
            self.event_type_names = ('PSF0','PSF1','PSF2','PSF3')
        else:
            self.event_type_names = ('front','back')
        if not self.quiet:  print self.dataset
        
        # use dataset to extract psf and exposure, set up respective managers
        
        exposure_correction=datadict.pop('exposure_correction', None) if datadict is not None else None 
        if self.use_old_irf_code:
            self.exposureman = exposure.ExposureManager(self.dataset,
                 exposure_correction=exposure_correction)
            self.psfman = psf.PSFmanager(self.dataset)
            self.irfs=None
        else:       
            self.irfs = irfman.IrfManager(self.dataset, 
                irf_dir=self.caldb, irfname=self.dataset.irf)
        
        # check location of model
        # possibilites are the all-sky pickle.zip, from which any ROI can be extraccted, or a specific set of
        #   sources in an XML file, which is either external or generated from a HEALPix ROI
        # in the latter case, 
        
        input_model = self.input_model = config.get('input_model', None)
            
        self.modeldir = self.configdir
        self.modelname='pickle.zip'
        self.roi_spec = None
        if os.path.exists(os.path.join(self.configdir, self.modelname)):
            self.input_model = None
        elif input_model is not None:
            if not isinstance(input_model, dict):
                raise Exception('input_model must be a dictionary')
            model_keys = ['xml_file','path', 'auxcat', 'free_diffuse', 'tsmin', 'emin']
            if not set(input_model.keys()).issubset(model_keys):
                raise Exception('input model key(s), %s, not recognized: expect %s'
                    % (list(set(input_model.keys()).difference(model_keys)), model_keys))
            input_xml   = input_model.get('xml_file', None)

            # no pickle.zip in config: check path if set
            self.modeldir = input_model.get('path', None)
            if self.modeldir is not None:
                self.modeldir = os.path.expandvars(self.modeldir)
            if self.modeldir is None:
                if input_xml is  None: 
                    raise Exception('Expected either to have access to a all-sky pickle file or a xml_file ')
                if not os.path.exists(os.path.expandvars(input_xml)):
                    raise Exception('Specified XML file, "%s", not found' % input_xml)
                self.input_xml = os.path.expandvars(input_xml)
                self.xml_parser = from_xml.XMLparser(self.input_xml)
                dss_pos = self.dataset.dss.roi_info()
                config_pos = config.get('position', None)
                xml_pos = self.xml_parser.get('position', None)
                if not self.quiet:
                    print 'will load sources from  file %s' % self.input_xml
                    print 'ROI definitions: using the first listed'
                    print '\tconfig: %s' % (list(config_pos) if config_pos is not None else None)
                    print '\tXML:    %s' % xml_pos
                    print '\tDSS:    %s' % dss_pos

                if config_pos is not None:
                    pos = config_pos
                elif xml_pos is not None:
                    pos = xml_pos
                elif dss_pos is not None:
                    pos = dss_pos
                else:
                    raise Exception('No ROI configuration specified, either in config, the XML, or FT1')
                assert len(pos)==3, 'Expected ROI pos to be a list of (ra,dec,radius)'
                self.roi_spec = ROIspec(pos=pos)
                return
                
            elif not os.path.exists(self.modeldir):
                t = os.path.expandvars(os.path.join('$FERMI', self.modeldir))
                if not os.path.exists(t):
                    raise Exception('No source model file found in %s or %s' %(self.modeldir, t) )
                self.modeldir=t
        if not os.path.exists(os.path.join(self.modeldir, self.modelname)) and not self.quiet:
            print 'pickle.zip not found in %s: no model to load' % self.modeldir
        elif not self.quiet:
            print 'Will load healpix sources from %s/%s' % (self.modeldir,self.modelname)
    
    def load_config(self, configdir):
        if configdir.startswith('$'):
            self.configdir = os.path.expandvars(configdir)
        elif not configdir.startswith('/'):
            self.configdir = os.path.join(os.getcwd(), configdir)
        else:
            self.configdir = configdir
        
        if not os.path.exists(self.configdir):
            raise Exception( 'Configuration folder %s not found' % self.configdir)
        
        def get_dict(path):
            ret = None
            for name, reader in [('config.yaml', yaml.load ), 
                                ('config.txt', eval) ]:
                cfile = os.path.join(path, name)
                if os.path.exists(cfile):
                    try:
                        ret = reader(open(cfile).read())
                        if not self.quiet: print 'loaded config file {}'.format(cfile)
                    except Exception, msg:
                        print 'Failed to parse file {}: {}'.format(cfile,msg)
                        raise
                    return ret
            return {}
                
        config = get_dict(os.path.join(self.configdir, '../'))
        config.update(get_dict(self.configdir))

        assert len(config.keys())>0, 'No dicts found'
        return config
        
    def __repr__(self):
        ret= '{}.{}'.format(self.__module__, self.__class__.__name__)
        ret+='\n * configdir {}'.format( self.configdir)
        ret+='\n * irfs      {}'.format(self.irf+' '+(self.irfs.__repr__() if self.irfs is not None else "(old irf code)"))
        out=StringIO.StringIO()
        pprint.pprint(self.diffuse, stream=out, indent=4)
        ret+='\n * diffuse\n   {}'.format(out.getvalue())
        out.close()
        #ret+='\n * diffuse   {}'.format(self.diffuse)
        ret+='\n * {}'.format(self.dataset)
        return ret
        
    def get_bands_for_ROI(self, index):
        return self.get_bands(skymaps.Band(12).dir(index), minROI=5)
    
    def get_bands(self, roi_dir, **kwargs):
        """ return a list of dataset.ROIband objexts for an ROI
        """
        dset = self.dataset
        dset.load()
        band_kwargs = dict(emin=dset.emin, emax=dset.emax, minROI=dset.minROI, maxROI=dset.maxROI)
        band_kwargs.update(kwargs)
        radius = band_kwargs['minROI'] # fixed radius now
        bandlist = []
        for band in dset.dmap:
            emin,emax, event_type =  band.emin(), band.emax(), band.event_class()&5
            if (emin + 1) < band_kwargs['emin'] or (emax - 1) >band_kwargs['emax']: continue
            #print int(emin), event_class
            energy= np.sqrt(emin*emax)
            bandlist.append( bands.BandSet(band, self.psfman(event_type,energy), self.exposureman(event_type,energy), 
                roi_dir, radius))
        return np.asarray(bandlist)
        
    def select_event_type(self, which):
        """convenience function for selecting an event type index
        which : None, int, or string
            if None, return None to indicate all
        
        returns None or the index of the selected event type
        """
        if which is None or which=='all': return None
        etnames = irfman.IrfManager.event_type_names
        try:
            if type(which)==str:
                which = which.lower()
                return etnames.index(which)
            t = etnames[which]
            return which
        except Exception, msg:
            print 'Bad event type, "%s": %s\nMust be one of %s or a valid index' % (which, msg, etnames)
            raise
    def event_type_name(self, event_type):
        """convenience function to access name of an event type index
        """
        return irfman.IrfManager.event_type_names[event_type]

    @property
    def auxcat(self):
        """get the auxillary catalog, if any"""
        if self.input_model is None: return None
        acat = self.input_model.get('auxcat', None) 
        if acat is None: return None
        if os.path.isabs(acat): return acat
        return os.path.join(self.modeldir, acat)
        