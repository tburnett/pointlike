"""
Set up and manage the model for all the sources in an ROI

$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/roimodel.py,v 1.29 2018/01/27 15:37:17 burnett Exp $

"""
import os, pickle
import numpy as np
import pandas as pd
from  uw.utilities import keyword_options
from skymaps import Band, SkyDir
from . import (sources, parameterset, diffuse, extended, to_xml)
from . sources import (PowerLaw, PLSuperExpCutoff, LogParabola, Constant)

class ROImodelException(Exception):pass
        

class ROImodel(list):
    """ Manage the model, or list of sources, for an ROI. Note that it inherits from list.
    
    In particular, provide an interface to serialize the set of free parameters, or define a subset thereof.
    This is delegated to the classes ParameterSet and ParSubSet in the module parameterset
    
    This is a virtual class, needs to be subclassed to provide an interface to a stored version of the sources
    See from_healpix.ROImodelFromHealpix and from_xml.ROImodelFromXML.
    
    Methods are provided to add or remove sources, and change the model associated with a source.
    Can be saved to an XML file, see to_xml.
    """
    defaults = (
        ('quiet', True, 'set False for info'),
        ('ecat',  None, 'If present, use for catalog'),
        ('load_kw',dict(tsmin=[0,25,100], rings=2), 'a dict specific for the loading'),
        )
    @keyword_options.decorate(defaults)
    def __init__(self, config, roi_spec, **kwargs):
        """config : configuration.Configuration object
            used to find model info
        roi_spec : integer or string or ...
            ROI specification, passed to load_sources in a subclass
        """
        keyword_options.process(self, kwargs)
        self.config = config
        self.config.diffuse = config.diffuse.copy() # since something changes??
        
        if self.ecat is None: #speed up if already loaded
            self.ecat = extended.ExtendedCatalog(self.config.extended, quiet=self.quiet)

        # clear if called again
        while len(self)>0:
            self.pop()
            
        # sources loaded by a subclass that must implement this function
        self.load_sources(roi_spec, **self.load_kw)
        
        if config.auxcat is not None:
            self.add_sources(config.auxcat)
        self.initialize()

        print self.summary()
        self.selected_source = None

    def summary(self):
        ns = len(self)
        n_ext = sum([ s.isextended for s in  self])
        n_glob = sum([s.isglobal for s in self])
        n_free = len(self.parameters)
        return '%d total sources: %d extended, %d global, %d free' % ( ns, n_ext, n_glob, n_free )

        
    def initialize(self, **kw):
        """For fast parameter access: must be called if any source changes
        """
        self.parameters = parameterset.ParameterSet(self, **kw)
  
    def parsubset(self, select=None, exclude=None):
        """ return a ParSubSet object with possible initial selection of a subset of the parameters
        """
        return parameterset.ParSubSet(self, select, exclude)
        
    
    # note that the following properties are dynamic, in case sources or their models change interactively
    @property
    def source_names(self): return np.array([s.name for s in self])
    @property
    def models(self): return np.array([s.model for s in self])
    @property
    def free(self): 
        """ mask which defines variable sources: all global and local sources with at least one variable parameter 
        """
        return np.array([ np.any(s.model.free) for s in self])
    @property 
    def bounds(self):
        """ fitter representation of applied bounds """
        return np.concatenate([m.bounds[m.free] for m in self.models])
    
        
    @property
    def parameter_names(self):
        """ array of free parameter names """
        names = []
        for source_name, model in zip(self.source_names, self.models):
            for pname in np.array(model.param_names)[model.free]:
                names.append(source_name.strip()+'_'+pname)
        return np.array(names)
    

    def find_source(self, source_name):
        """ Search for the source with the given name
        
        source_name : [string | None | sources.Source instance ]
            if the first or last character is '*', perform a wild card search, return first match
            if None, and a source has been selected, return it
            if an instance, and in the list, just select it and return it
        """
        if source_name is None:
            if self.selected_source is None:
                raise ROImodelException('No source is selected')
            return self.selected_source
        elif isinstance(source_name, sources.Source):
            if source_name in self.sources:
                self.selected_source = source_name
                return self.selected_source
            not_found()
            
        names = [s.name for s in self]
        def not_found():
            self.selected_source_index =-1
            raise ROImodelException('source %s not found' %source_name)
        def found(s):
            self.selected_source=s
            self.selected_source_index = names.index(s.name)
            return s
        if source_name[-1]=='*':
            for name in names:
                if name.startswith(source_name[:-1]): 
                    return found(self[names.index(name)])
            not_found()
        if source_name[0]=='*':
            for name in names:
                if name.endswith(source_name[1:]): 
                    return found(self[names.index(name)])
            not_found()
        try:
            selected_source = self[names.index(source_name)]
            #if self.selected_source is None or self.selected_source != selected_source:
            #    print 'selected source %s for analysis' % selected_source.name
            return found(selected_source)
        except:
            self.selected_source = None
            not_found()

    def add_source(self, newsource=None, **kw):
        """ add a source to the ROI
        
        parameters
        ----------
        newsource : Source object or None
            if None, expect source to be defined as a PointSource by the keywords
            
        keywords:
            name : string
            model : uw.like.Models object
            skydir : skymaps.SkyDir object | (ra,dec) 
        """
        if newsource is not None:
            assert isinstance(newsource, sources.Source)
        else:
            newsource = sources.PointSource(**kw)
            
        if len(self.source_names)>0 and newsource.name in self.source_names:
            print 'Attempt to add source "{}: a source with that name already exists'.format(
                 newsource.name)
            return None
        self.append(newsource)
        self.initialize()
        return newsource
     
    def del_source(self, source_name):
        """ remove a source from the model for this ROI
        """
        source = self.find_source(source_name) # first get it
        self.remove(source)
        self.initialize()
        return source
        
    def set_model(self, model, source_name=None):
        """ replace the current model, return reference to previous
        
        model : string, or like.Models.Model object
            if string, evaluate. Note that 'PowerLaw(1e-11,2.0)' will work. Also supported:
            ExpCutoff, PLSuperExpCutoff, LogParabola, each with all parameters required.
        source_name: None or string
            if None, use currently selected source
        """
        src = self.find_source(source_name)
        old_model = src.model
        if isinstance(model, str): 
            model = eval(model) 
        assert sources.ismodel(model), 'model must inherit from Model class'
        src.model = model
        src.changed = True
        sources.set_default_bounds(model)
        self.initialize()
        return src, old_model
        
    def to_xml(self, filename,  **kwargs):
        """Create an XML representation
        
        filename : string
            the xml filename
        """
        with open(filename, 'w') as out:
            to_xml.from_roi(self, stream = out, **kwargs)
            
    def add_sources(self, auxcat='plots/seedcheck/good_seeds.csv'):
        """Add new sources from a csv file
        Default assumes analysis by seedcheck
        """
        assert os.path.exists(auxcat), 'auxcat file %s not found' % auxcat
        if os.path.splitext(auxcat)[1] != '.csv':
            raise Exception('Only support csv files, not %s' % auxcat)
        good_seeds = pd.read_csv(auxcat, index_col=0)
        print 'Check %d sources from file %s: ' % (len(good_seeds), auxcat),
        myindex = Band(12).index(self.roi_dir)
        inside = good_seeds['index']==myindex
        ni = sum(inside)
        if ni>0:
            print '%d inside ROI' % ni
        else:
            print 'No sources in ROI %04d' % myindex
        for name, s in good_seeds[inside].iterrows():
            src = pickle.load(open('%s/seedcheck/%s.pickle' % (self.config.modeldir, name)))
            try:
                self.add_source(name=name, skydir=src.skydir, model=src.model)
            finally: pass
        return good_seeds[inside]


