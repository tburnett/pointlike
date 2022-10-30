"""Class for parsing and writing gtlike-style sourceEQUATORIAL libraries.
   Barebones implementation; add additional capabilities as users need.

   $Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/utilities/xml_parsers.py,v 1.93 2018/01/28 19:52:34 burnett Exp $

   author: Matthew Kerr
"""

import xml.sax as x
from os.path import join
import os
from collections import deque

import numpy as np

from skymaps import SkyDir,DiffuseFunction,IsotropicSpectrum,IsotropicPowerLaw,IsotropicConstant

from uw.like.pointspec_helpers import PointSource,get_diffuse_source
from uw.like.roi_diffuse import DiffuseSource
from uw.like.roi_extended import ExtendedSource
from . parmap import LimitMapper, MapperException, LinearMapper
from . import path

# spectral models
from uw.like import Models
import uw.darkmatter.spectral
from uw.like import scalemodels

# spatial models
import uw.darkmatter.spatial
from uw.like import SpatialModels


class XMLException(Exception): pass

def decorate(instring,tablevel=1):
	return ''.join(['\t'*tablevel,instring,'\n'])

class Stack(deque):

    def push(self,element):
        self.append(element)

    def pop(self):
        if len(self) == 0: return
        return super(Stack,self).pop()

    def peek(self):
        if len(self) == 0: return
        return self[-1]

class XMLElement(dict):

    def __init__(self,name,*args,**kwargs):
        self.name = name
        self.children = deque()
        self.parent = None
        super(XMLElement,self).__init__(*args,**kwargs)

    def addChild(self,child):
        child.parent = self
        self.children.append(child)

    def getChild(self,name):
        for child in self.children:
            if child.name == name: return child

    def __str__(self):
        n  = 'Element Name: %s'%(self.name)
        a1 = 'Attributes:\n\tKey:\t\tValue:\n\t---\t\t-----'
        astrings = '\n'.join(['\t%s\t\t%s'%(k,v) for k,v in zip(self.keys(),self.values())])
        c  = 'Child Element Names:'
        cstrings = '\n'.join([x.name for x in self.children])
        return '\n'.join([n,a1,astrings,c,cstrings])

class SourceHandler(x.ContentHandler,Stack):
    
    def __init__(self, pattern = 'source'):
        self.outerElements = deque()
        self.sources       = deque()
        self.pattern       = pattern

    def __call__(self): return self.lastOff

    def startElement(self,name,attrs):
        self.push(XMLElement(name,attrs))

    def endElement(self,name):
        t = self.pop()
        l = self.peek()
        if l is not None:
            l.addChild(t)
        else:
            self.outerElements.append(t)
        if t.name == self.pattern:
            self.sources.append(t)

class XML_to_Model(object):
    """ Map a parsed XML model onto the Model classes.
    
        This class should be extended as more use cases arise.

        Here is a simple example to load in a point source.
        First, we have to setup the xml parser:

            >>> def xml2model(xml):
            ...     from StringIO import StringIO
            ...     parser = x.make_parser()
            ...     xtm = XML_to_Model()
            ...     handler = SourceHandler(pattern='spectrum')
            ...     parser.setContentHandler(handler)
            ...     parser.parse(StringIO(xml))
            ...     return xtm.get_model(handler.sources[0],'source')

        Now, we can parse spectral models:

            >>> model=xml2model('''
            ...     <spectrum  type="PowerLaw">
            ...         <parameter name="Prefactor" value="1.0" free="1" max="100.0" min="0.01" scale="1e-11" />
            ...         <parameter name="Index" value="2.0" free="1" max="5" min="-5" scale="-1" />
            ...         <parameter name="Scale" value="1000.0" free="0" max="1000.0" min="1000.0" scale="1" />
            ...     </spectrum>''')
            >>> real_model=Models.PowerLaw()
            >>> np.allclose(model.get_all_parameters(), real_model.get_all_parameters())
            True
            >>> np.allclose(model['index'],2)
            True

            >>> model=xml2model('''
            ...     <spectrum type="SmoothBrokenPowerLaw">
            ...         <parameter error="1.874888615" free="1" max="100" min="0" name="Prefactor" scale="1e-10" value="14.56863965" />
            ...         <parameter error="0.7738550805" free="1" max="4" min="-2" name="Index1" scale="1" value="1.504042874" />
            ...         <parameter free="0" max="2000" min="30" name="Scale" scale="1" value="200" />
            ...         <parameter error="0.04177558875" free="1" max="-1" min="-5" name="Index2" scale="1" value="-1.891184873" />
            ...         <parameter error="9.888298037" free="1" max="1000" min="80" name="BreakValue" scale="1" value="204.3216401" />
            ...         <parameter free="0" max="10" min="0.01" name="Beta" scale="1" value="0.1" />
            ...     </spectrum>''')
            >>> np.allclose(model['norm'],14.56863965e-10)
            True
            >>> np.allclose(model.get_limits('norm'),[0.0, 1e-08])
            True
            >>> np.allclose(model['index_1'],-1.504042874)
            True
            >>> np.allclose(model.get_limits('index_1'),[-4,2])
            True
            >>> np.allclose(model['index_2'],1.891184873)
            True
            >>> np.allclose(model['e_break'],204.3216401)
            True
            >>> np.allclose(model['beta'],0.1)
            True
            >>> np.allclose(model['e0'],200)
            True

        Model from http://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/xml_model_defs.html#gaussian
            >>> model=xml2model('''
            ... <spectrum type="Gaussian">
            ...     <parameter free="1" max="1000.0" min="0.001" name="Prefactor" scale="1e-09" value="1"/>
            ...     <parameter free="1" max="1e5" min="1e3" name="Mean" scale="1.0" value="7e4"/>
            ...     <parameter free="1" max="30" min="1e4" name="Sigma" scale="1.0" value="1e3"/>
            ... </spectrum>''')
            >>> print model['prefactor']
            1e-09
            >>> print model['mean']
            70000.0
            >>> print model['sigma']
            1000.0

        Test the ScaleFactor models. XML Model taken from
            https://confluence.slac.stanford.edu/display/ST/Science+Tools+Development+Notes?focusedCommentId=103582318#comment-103582318
        but with the ScaleFactor parameter set to 10.0 to be more interesting:

            >>> model=xml2model('''
            ... <spectrum type="ScaleFactor::PowerLaw2">
            ...   <parameter free="1" max="1000.0" min="1e-05" name="Integral" scale="1e-06" value="1.0"/>
            ...   <parameter free="1" max="-1.0" min="-5.0" name="Index" scale="1.0" value="-2.0"/>
            ...   <parameter free="0" max="200000.0" min="20.0" name="LowerLimit" scale="1.0" value="20.0"/>
            ...   <parameter free="0" max="1000000.0" min="20.0" name="UpperLimit" scale="1.0" value="1e6"/>
            ...   <parameter free="1" max="10" min="0." name="ScaleFactor" scale="10." value="1"/>
            ... </spectrum>''')
            >>> type(model) == scalemodels.ScaleFactorPowerLawFlux
            True
            >>> print model['ScaleFactor']
            10.0
            >>> print model['Int_Flux']
            1e-06
            >>> np.allclose(model.get_limits('Int_Flux'),[1e-11,1e-3])
            True
            >>> np.all(model.free == [True]*3)
            True

        ScaleFactor::FileFunction objects are a little bit tricker:
            >>> model=xml2model('''
            ... <spectrum type="ScaleFactor::FileFunction" file="$(GLAST_EXT)/diffuseModels/v2r0p1/isotrop_2year_P76_source_v1.txt" >
            ...   <parameter free="1" max="100" min="1." name="ScaleFactor" scale="1." value="11"/>
            ...   <parameter name="Normalization" value="5.0" free="1" max="10" min="0.1" scale="1" />
            ... </spectrum>''')
            >>> isinstance(model,scalemodels.ScaleFactorFileFunction)
            True
            >>> print model.file
            $(GLAST_EXT)/diffuseModels/v2r0p1/isotrop_2year_P76_source_v1.txt
            >>> print model['ScaleFactor']
            11.0
            >>> print model['Normalization']
            5.0
            >>> np.all(model.free==[True,True])
            True
            >>> print model.mappers
            [LimitMapper(1.0,100.0,1.0), LimitMapper(0.1,10.0,1.0)]

    """

    # List of pointlike models which can be used in gtlike
    savable_models = (
        Models.PowerLaw, Models.PowerLawFlux,
        Models.BrokenPowerLaw, Models.BrokenPowerLawFlux, Models.SmoothBrokenPowerLaw,
        Models.PLSuperExpCutoff,
        Models.PLSuperExpCutoff4,
        Models.Constant, Models.FrontBackConstant,
        Models.LogParabola,
        uw.darkmatter.spectral.DMFitFunction,
        Models.FileFunction,
        Models.Gaussian,
        scalemodels.ScaleFactorPowerLaw,
        scalemodels.ScaleFactorPowerLawFlux,
        scalemodels.ScaleFactorFileFunction,
        scalemodels.ScaleFactorDMFitFunction,
        scalemodels.ScaleFactorPLSuperExpCutoff,
        scalemodels.ScaleFactorPLSuperExpCutoff4,
        scalemodels.ScaleFactorGaussian,
    )

    modict = {model.gtlike['name']:model for model in savable_models}

    def __init__(self):
        pass

    def get_model(self,xml_dict,source_name, scaling=False):
        """ source_name is used for better error message printing. """

        specname = xml_dict['type']
        params   = xml_dict.children

        d = dict()
        for p in params:
            d[p['name']] = p

        if scaling and specname == 'PowerLaw':
            model_class = Models.ScalingPowerLaw
        else:
            if specname not in self.modict:
                raise XMLException('For source %s, spectral model "%s" not implemented' % (source_name,specname))
            model_class = self.modict[specname]

        # certain spectral models require the file to be set
        kwargs = {}
        for key in model_class.default_extra_attrs:
            kwargs[key] = str(xml_dict[key])

        model = model_class(**kwargs)

        for pointlike_name,gtlike_name in zip(model.param_names, model.gtlike['param_names']):
            try:
                pdict = d[gtlike_name]
            except:
                raise XMLException("For source %s, %s parameter %s not found in xml file." % (source_name,specname,gtlike_name))

            for p in ['scale', 'value', 'min', 'max']:
                if not pdict.has_key(p):
                    raise XMLException("For source %s, %s parameter %s must have a %s." % (source_name,specname,gtlike_name,p))

            scale = float(pdict['scale'])

            value = float(pdict['value'])*scale
            min = float(pdict['min'])*scale
            max = float(pdict['max'])*scale

            if np.isnan(value): raise XMLException('For source %s, %s parameter %s is NaN' % (source_name,specname,gtlike_name))

            try:
                model.set_mapper(pointlike_name,LinearMapper)
                model.setp_gtlike(pointlike_name,value)
                model.set_limits_gtlike(pointlike_name,min,max,scale)
            except MapperException:
                import traceback,sys; traceback.print_exc(sys.stdout)
                print 'scale=',scale
                raise XMLException("Error setting parameter %s for source %s. Value is %s and limits are %s to %s" % (gtlike_name,source_name,value,min,max))

            if 'error' in pdict.keys():
                err = np.abs(float(pdict['error'])*scale)
                model.set_error(pointlike_name,err)

            if pdict['free'] not in ['0','1']:
                raise XMLException('For source %s, %s parameter %s must have free="0" or free="1' % (source_name,specname,gtlike_name))
            free = pdict['free'] == '1'
            model.set_free(pointlike_name,free)

        for pointlike_name,gtlike_name in model.gtlike['extra_param_names'].items():
            try:
                pdict = d[gtlike_name]
            except:
                raise XMLException("For source %s, %s parameter %s not found in xml file." % (source_name,specname,gtlike_name))

            if pdict['free'] != '0':
                # Sanity check on validity of xml 
                raise XMLException('For source %s, %s parameter %s cannot be fit (must be free="0")' % (source_name,specname,gtlike_name))

            value=float(pdict['value'])*float(pdict['scale'])
            model.setp(pointlike_name, value)

        return model

class XML_to_SpatialModel(object):

    extended_sources = [SpatialModels.Gaussian, 
                        SpatialModels.PseudoGaussian, 
                        SpatialModels.Disk,
                        SpatialModels.PseudoDisk, 
                        SpatialModels.Ring,
                        uw.darkmatter.spatial.NFW, uw.darkmatter.spatial.PingNFW, uw.darkmatter.spatial.PseudoPingNFW,
                        SpatialModels.EllipticalGaussian, 
                        SpatialModels.EllipticalDisk, 
                        SpatialModels.EllipticalRing, SpatialModels.SpatialMap,
                        SpatialModels.RadialProfile]

    spatialdict = {i.__name__:i for i in extended_sources}
    ## add these for consistency with new convention
    spatialdict['RadialDisk'] = SpatialModels.Disk
    spatialdict['RadialGaussian'] = SpatialModels.Gaussian
    

    @staticmethod
    def get_spatial_model(xml_dict,source_name,diffdir=None):
        """ Kind of like getting a spectral model, but
            instead returns a SpatialModel object. 
            
            Since extended ources don't exist in gtlike, no need
            to keep backwards compatability. Just trying
            to keep a farmilliar interface and make the
            naming convention similar to gtobssim. """

        spatialname = xml_dict['type']
        params      = xml_dict.children


        d = dict()
        for p in params: d[p['name']] = p

        input_kwargs = dict()

        if spatialname == 'SpatialMap' or spatialname == 'RadialProfile':
            # For spatial maps, ignore any of the parameters.
            if diffdir:
                file = join(diffdir,os.path.basename(str(xml_dict['file'])))
            else:
                file = str(xml_dict['file'])
            input_kwargs['file'] = file

        if d.has_key('RA') and d.has_key('DEC'):
            input_kwargs['coordsystem']=SkyDir.EQUATORIAL
        elif d.has_key('L') and d.has_key('B'):
            input_kwargs['coordsystem']=SkyDir.GALACTIC
    
        if not spatialname in XML_to_SpatialModel.spatialdict:                                
            raise XMLExcception("Unrecognized spatial model %s" % spatialname)

        spatial_model = XML_to_SpatialModel.spatialdict[spatialname](**input_kwargs)

        if spatialname == 'SpatialMap':
            return spatial_model

        for i,p in enumerate(spatial_model.param_names):
            if p=='Sigma' and p not in d: 
                d['Sigma'] = d['Radius'] #kluge to adapt to new convention
            pdict =d[p]

            scale = float(pdict['scale'])

            if scale != 1:
                raise Exception("Error reading spatial parameter %s for source %s. Parameter scale must be set to 1" % (p,source_name))

            value = float(pdict['value'])
            spatial_model[p] = value

            if pdict.has_key('min') and pdict.has_key('max'):

                min = float(pdict['min'])
                max = float(pdict['max'])

                if i == 0:
                    if (min != -360 and min !=0) or max != 360:
                        raise Exception("Error reading spatial parameter %s for source %s. Parameter range must be -360 (or 0) to 360" % (p,source_name))
                elif i == 1:
                    if min != -90 or max != 90:
                        raise Exception("Error reading spatial parameter %s for source %s. Parameter range must be -90 to 90" % (p,source_name))
                else:
                    # only read limits for non-position parameters
                    spatial_model.set_limits(p, min, max, absolute=True)

            free = (pdict['free'] == '1')
            spatial_model.set_free(p, free)

        return spatial_model


class Model_to_XML(object):
    """Convert a Model instance to an XML entry.
       
       This class should be extended as more use cases arise.
       
        Here is a simple test making a PowerLaw model:

            >>> def model2xml(model, strict=False, expand_env_vars=False):
            ...     m2x = Model_to_XML(strict=strict)
            ...     m2x.process_model(model, expand_env_vars=expand_env_vars)
            ...     return m2x.getXML(tablevel=0).replace('\\t',' '*4).strip()

        First, create a powerlaw with limits set:

            >>> pl = Models.PowerLaw(norm=1e-11, index=2)
            >>> pl.set_limits('norm', 1e-12, 1e-10, scale=1e-11)
            >>> pl.set_limits('index', -5, 5, scale=1)

            >>> print model2xml(pl)
            <spectrum  type="PowerLaw">
                <parameter name="Prefactor" value="1.0" free="1" max="10.0" min="0.1" scale="1e-11" />
                <parameter name="Index" value="2.0" free="1" max="5" min="-5" scale="-1" />
                <parameter name="Scale" value="1000.0" free="0" max="1000.0" min="1000.0" scale="1" />
            </spectrum>

        Testing edge case of parameters outside limits. The expected behavoir is
        (a) with strict=False, to set the parameter to the edge of the default limitand
        (b) withs trict=True to raise an exception. Note that this will only matter for index
        and not norm since norm is an oomp limit so is always set to the real value:

            >>> pl = Models.PowerLaw(norm=1e-20, index=200)
            >>> print model2xml(pl,strict=True)
            Traceback (most recent call last):
                ...
            ModelException: Found Index=200.0 > 5, maximum allowed value
            >>> xml=model2xml(pl,strict=False)
            Warning Found Index=200.0 > 5, maximum allowed value,
                Setting parameter value to maximum.
            >>> print xml
            <spectrum  type="PowerLaw">
                <parameter name="Prefactor" value="1.0" free="1" max="100.0" min="0.01" scale="1e-20" />
                <parameter name="Index" value="5.0" free="1" max="5" min="-5" scale="-1" />
                <parameter name="Scale" value="1000.0" free="0" max="1000.0" min="1000.0" scale="1" />
            </spectrum>

        Now, create a powerlaw with no limits. Note that it will save out errros correctly:

            >>> pl = Models.PowerLaw(norm=1e-11, index=2)
            >>> pl.set_error('norm',0.5e-11)
            >>> pl.set_error('index',0.25)
            >>> print model2xml(pl)
            <spectrum  type="PowerLaw">
                <parameter name="Prefactor" value="1.0" error="0.5" free="1" max="100.0" min="0.01" scale="1e-11" />
                <parameter name="Index" value="2.0" error="0.25" free="1" max="5" min="-5" scale="-1" />
                <parameter name="Scale" value="1000.0" free="0" max="1000.0" min="1000.0" scale="1" />
            </spectrum>

        We can also save out FileFunction objects:

            >>> from tempfile import NamedTemporaryFile
            >>> temp = NamedTemporaryFile(delete=True)
            >>> pl.save_profile(temp.name, emin=1, emax=1e6)
            >>> fs = Models.FileFunction(Normalization=1,file=temp.name)
            >>> print model2xml(fs).replace(temp.name, '<FILENAME>')
            <spectrum file="<FILENAME>" type="FileFunction">
                <parameter name="Normalization" value="1.0" free="1" max="10000.0" min="0.0001" scale="1" />
            </spectrum>

        Note, writing FileFunctions should preserve environment varaibles:

            >>> filename="$(GLAST_EXT)/diffuseModels/v2r0p1/isotrop_2year_P76_source_v1.txt"
            >>> fs = Models.FileFunction(file=filename)

        First, save it with the env var:

            >>> print model2xml(fs,expand_env_vars=False).replace(filename,'[FILENAME]')
            <spectrum file="[FILENAME]" type="FileFunction">
                <parameter name="Normalization" value="1.0" free="1" max="10000.0" min="0.0001" scale="1" />
            </spectrum>

        Next, save it with the env var expanded:

            >>> print model2xml(fs,expand_env_vars=True).replace(path.expand(filename),'[FILENAME]')
            <spectrum file="[FILENAME]" type="FileFunction">
                <parameter name="Normalization" value="1.0" free="1" max="10000.0" min="0.0001" scale="1" />
            </spectrum>

        Next, save out a PowerLaw with index=-2

            >>> pl = Models.PowerLaw(index=-2)
            >>> print model2xml(pl)
            <spectrum  type="PowerLaw">
                <parameter name="Prefactor" value="1.0" free="1" max="100.0" min="0.01" scale="1e-11" />
                <parameter name="Index" value="-2.0" free="1" max="5" min="-5" scale="-1" />
                <parameter name="Scale" value="1000.0" free="0" max="1000.0" min="1000.0" scale="1" />
            </spectrum>

            >>> sbpl = Models.SmoothBrokenPowerLaw(
            ...     Norm=14.56863965e-10,
            ...     Index_1=-1.504042874,
            ...     Index_2=1.891184873,
            ...     E_break=204.3216401,
            ...     beta=0.1,
            ...     e0=200)

            >>> print model2xml(sbpl)
            <spectrum  type="SmoothBrokenPowerLaw">
                <parameter name="Prefactor" value="1.456863965" free="1" max="100.0" min="0.01" scale="1e-09" />
                <parameter name="Index1" value="-1.504042874" free="1" max="5" min="-5" scale="-1" />
                <parameter name="Index2" value="1.891184873" free="1" max="5" min="-5" scale="-1" />
                <parameter name="BreakValue" value="2.043216401" free="1" max="100.0" min="0.01" scale="100.0" />
                <parameter name="Scale" value="200" free="0" max="200" min="200" scale="1" />
                <parameter name="Beta" value="0.1" free="0" max="0.1" min="0.1" scale="1" />
            </spectrum>

        Model from http://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/xml_model_defs.html#gaussian
            >>> g = Models.Gaussian(prefactor=1e-9, mean=7e4, sigma=1e3)
            >>> print model2xml(g)
            <spectrum  type="Gaussian">
                <parameter name="Prefactor" value="1.0" free="1" max="100.0" min="0.01" scale="1e-09" />
                <parameter name="Mean" value="0.7" free="1" max="100.0" min="0.01" scale="100000.0" />
                <parameter name="Sigma" value="1.0" free="1" max="100.0" min="0.01" scale="1000.0" />
            </spectrum>

            >>> s = scalemodels.ScaleFactorPowerLaw(ScaleFactor=5)
            >>> print model2xml(s)
            <spectrum  type="ScaleFactor::PowerLaw">
                <parameter name="ScaleFactor" value="0.5" free="1" max="100.0" min="0.01" scale="10.0" />
                <parameter name="Prefactor" value="1.0" free="1" max="100.0" min="0.01" scale="1e-11" />
                <parameter name="Index" value="2.0" free="1" max="5" min="-5" scale="-1" />
                <parameter name="Scale" value="1000.0" free="0" max="1000.0" min="1000.0" scale="1" />
            </spectrum>

        The ScaleFactorFileFunction is a bit tricker:

            >>> s = scalemodels.ScaleFactorFileFunction(ScaleFactor=10, Normalization=5,
            ...     file="$(GLAST_EXT)/diffuseModels/v2r0p1/isotrop_2year_P76_source_v1.txt")
            >>> print model2xml(s)
            <spectrum file="$(GLAST_EXT)/diffuseModels/v2r0p1/isotrop_2year_P76_source_v1.txt" type="ScaleFactor::FileFunction">
                <parameter name="ScaleFactor" value="1.0" free="1" max="100.0" min="0.01" scale="10.0" />
                <parameter name="Normalization" value="5.0" free="1" max="10000.0" min="0.0001" scale="1" />
            </spectrum>

   """
    
    def __init__(self,debug=False, strict=False):
        self.x2m = XML_to_Model()
        self.debug = debug
        self.strict = strict

    @staticmethod
    def prepare_model_for_xml(model, scaling, strict):
        """ Make a spectral model suitable for xml parsing.
            * Convert the ExpCutoff model to a PLSuperExpCutoff model.
            * apply default limits to any unbound paramters. """
        name = model.name

        model = model.copy()

        if name == 'ExpCutoff':
            model = model.create_super_cutoff()

        if not isinstance(model,XML_to_Model.savable_models):
            raise XMLException("Unable to save model %s to XML file. Not a savable model" % model.name)

        if scaling and name == 'PowerLaw':
            model = Models.ScalingPowerLaw.from_powerlaw(model)

        model.set_default_limits(strict=strict, oomp_limits=True, only_unbound_parameters=True)

        return model


    def param_strings(self):
        err_strings = ['error="%s" '%(e) if (e>0) else '' for e in self.perr]
        return ['\t<parameter name="%s" value="%s" %sfree="%d" max="%s" min="%s" scale="%s" />'%(e,f,m1,m2,n,s,v)
            for e,f,m1,m2,n,s,v in zip(self.pname,self.pval, err_strings,self.pfree,self.pmax,self.pmin,self.pscale,)]

    def getXML(self,tablevel=1,comment_string=None):
        cstring = [] if comment_string is None else [comment_string]
        strings = ['<spectrum%s type="%s">'%(self.extra_attrs,self.gtlike_name)] + \
                   cstring + self.param_strings() + ['</spectrum>']
        return ''.join([decorate(s,tablevel=tablevel) for s in strings])

    def process_model(self,model,scaling=False, expand_env_vars=False):
        """ Model an instance of Model """

        model = Model_to_XML.prepare_model_for_xml(model, scaling, self.strict)

        self.gtlike_name = model.gtlike['name']

        npar = len(model.param_names + model.default_extra_params.keys())

        self.pname = []
        self.pval = []
        self.pmin, self.pmax = [],[]
        self.pscale = []
        self.perr = []
        self.pfree = []
        self.extra_attrs = ' '

        for pointlike_name,gtlike_name in zip(model.param_names, model.gtlike['param_names']):

            if self.debug: print 'Processing %s'%(param)

            free = model.get_free(pointlike_name)
            val = model.getp_gtlike(pointlike_name)
            err = model.error(pointlike_name)
            min, max = model.get_limits_gtlike(pointlike_name)
            scale = model.get_scale_gtlike(pointlike_name)

            self.pname.append(gtlike_name)
            self.pval.append(val/scale)
            self.perr.append(np.abs(err/scale)) # remember, error is always positive

            scale_min, scale_max = sorted([min/scale, max/scale])
            self.pmin.append(scale_min)
            self.pmax.append(scale_max)

            self.pscale.append(scale)
            self.pfree.append(free)

        for pointlike_name in model.default_extra_params:
            gtlike_name=model.gtlike['extra_param_names'][pointlike_name]
            if self.debug: print 'Processing %s'%(xml_key)

            val = model[pointlike_name]

            self.pname.append(gtlike_name)
            self.pfree.append(False)
            self.pval.append(val)
            self.pscale.append(1)
            self.pmin.append(val)
            self.pmax.append(val)
            self.perr.append(0)

        for key in model.default_extra_attrs:
            attr=getattr(model,key)
            if expand_env_vars: 
                attr=path.expand(attr)
            self.extra_attrs+='%s="%s"' % (key,attr)


def get_skydir(elem):
    """convert a parsed SpatialModel into a SkyDir."""
    d = dict()
    if elem['type'] != 'SkyDirFunction':
        raise XMLException("""The PointSource's SpatialModel must have type="PointSource".""")
    for p in elem.children:
        d[p['name']] = float(p['value'])
    return SkyDir(d['RA'],d['DEC'])

def make_position_params(skydir, ra_free=False, dec_free=False):
    ra = skydir.ra()
    dec = skydir.dec()
    return ['\t<parameter name="RA"  value="%s" free="%s" max="360.0" min="-360.0" scale="1.0" />' %(ra, int(ra_free)),
            '\t<parameter name="DEC" value="%s" free="%s" max="90" min="-90" scale="1.0" />' %(dec, int(ra_free))]

def makePSSpatialModel(skydir, tablevel=1):
    """Encode the spatial model in XML for a point source at ra,dec."""

    strings = [ '<spatialModel type="SkyDirFunction">', ] + \
            make_position_params(skydir) + \
            [ '</spatialModel>' ]
    return ''.join([decorate(st,tablevel=tablevel) for st in strings])

def makeDSConstantSpatialModel(value=1.0, tablevel=1):
    """Encode an isotropic model."""
    
    strings = [
        '<spatialModel type="ConstantValue">',
        '\t<parameter  name="Value" value="%s" free="0" max="10.0" min="0.0" scale="1.0" />' % value,
        '</spatialModel>'
    ]
    return ''.join([decorate(st,tablevel=tablevel) for st in strings])

def makeDSMapcubeSpatialModel(filename='ERROR',tablevel=1, expand_env_vars=False):
    if expand_env_vars:
        filename=path.expand(filename)
    """Encode a mapcube model."""
    strings = [
        '<spatialModel file="%s" type="MapCubeFunction">'%(filename.replace('\\','/')),
        '\t<parameter name="Normalization" value="1.0" free="0" max="1e3" min="1e-3" scale="1.0" />',
        '</spatialModel>'
    ]
    return ''.join([decorate(st,tablevel=tablevel) for st in strings])

def makeExtendedSourceSpatialModel(es,expand_env_vars,tablevel=1):
    """ Create a spatial model.

        Test saving out extended source:

            >>> def make(ds, expand_env_vars=False):
            ...     # remove newlines and convert tabs to spaces.
            ...     m=makeExtendedSourceSpatialModel(ds,expand_env_vars=expand_env_vars, tablevel=0)
            ...     return m.replace('\\t',' '*4).strip()

        Save Disk source:
    
            >>> disk = SpatialModels.Disk(sigma=1, ra=244, dec=57)
                    
            >>> print make(disk)
            <spatialModel type="Disk">
                <parameter name="RA" value="244" free="1" max="360" min="-360" scale="1.0" />
                <parameter name="DEC" value="57" free="1" max="180" min="-180" scale="1.0" />
                <parameter name="Sigma" value="1" free="1" max="3" min="1e-10" scale="1.0" />
            </spatialModel>

            >>> disk.set_cov_matrix(np.asarray([[1,0,0],[0,1,0],[0,0,1]]))
            >>> print disk.error('sigma')
            2.30258509299
            >>> print make(disk)
            <spatialModel type="Disk">
                <parameter name="RA" value="244" error="1.0" free="1" max="360" min="-360" scale="1.0" />
                <parameter name="DEC" value="57" error="1.0" free="1" max="180" min="-180" scale="1.0" />
                <parameter name="Sigma" value="1" error="2.30258509299" free="1" max="3" min="1e-10" scale="1.0" />
            </spatialModel>

        When the coordsystem is GALACTIC, the output will be in L & B:

            >>> disk = SpatialModels.Disk(sigma=1, l=244, b=57)
            >>> print make(disk)
            <spatialModel type="Disk">
                <parameter name="L" value="244" free="1" max="360" min="-360" scale="1.0" />
                <parameter name="B" value="57" free="1" max="180" min="-180" scale="1.0" />
                <parameter name="Sigma" value="1" free="1" max="3" min="1e-10" scale="1.0" />
            </spatialModel>

        Note, saving out spatial models will preserve limits

            >>> disk = SpatialModels.Disk(limits=[[-2,2],[-2,2],[1e-3,30]])
            >>> print make(disk)
            <spatialModel type="Disk">
                <parameter name="RA" value="0" free="1" max="360" min="-360" scale="1.0" />
                <parameter name="DEC" value="0" free="1" max="180" min="-180" scale="1.0" />
                <parameter name="Sigma" value="0.1" free="1" max="30" min="0.001" scale="1.0" />
            </spatialModel>


        Save SpatialModel source:

            >>> from tempfile import NamedTemporaryFile
            >>> temp = NamedTemporaryFile(delete=True)
            >>> disk.save_template(temp.name)
            >>> map=SpatialModels.SpatialMap(file=temp.name)
            >>> print make(map).replace(temp.name, '<FILENAME>')
            <spatialModel type="SpatialMap" file="<FILENAME>" >
                <parameter name="Prefactor" value="1.0" free="0" max="1e3" min="1e-3" scale="1.0" />
            </spatialModel>

        Save RadialProfile source:

            >>> temp = NamedTemporaryFile(delete=True)
            >>> disk.save_profile(temp.name)
            >>> profile=SpatialModels.RadialProfile(file=temp.name)
            >>> print make(profile).replace(temp.name, '<FILENAME>')
            <spatialModel type="RadialProfile" file="<FILENAME>" >
                <parameter name="Normalization" value="1.0" free="0" max="1e3" min="1e-3" scale="1.0" />
                <parameter name="RA" value="0" free="1" max="360" min="-360" scale="1.0" />
                <parameter name="DEC" value="0" free="1" max="180" min="-180" scale="1.0" />
            </spatialModel>

    """
    if es.name == 'SpatialMap' or es.name == 'RadialProfile':
        file=path.expand(es.file) if expand_env_vars else es.file
        strings = [
            '<spatialModel type="%s" file="%s" >' % (es.pretty_name,file),
        ]

    else:
        strings = [ '<spatialModel type="%s">' % es.pretty_name, ]

    if es.name == 'SpatialMap':
        strings += [ '\t<parameter name="Prefactor" value="1.0" free="0" max="1e3" min="1e-3" scale="1.0" />' ]

    if es.name == 'RadialProfile':
        strings += [ '\t<parameter name="Normalization" value="1.0" free="0" max="1e3" min="1e-3" scale="1.0" />' ]

    if es.name != 'SpatialMap':
        # All spatialm models but SpatialMap have RA & DEC

        for name in es.param_names:
            param = es[name]
            err = es.error(name)
            err = 'error="%s" ' % err if err > 0 else ''
            free = es.get_free(name)

            if name == 'L' or name == 'RA':
                min,max=-360,360
            elif name == 'B' or name == 'DEC':
                min,max=-180,180
            else:
                min,max=es.get_limits(absolute=True)[es.mapper(name)]

            strings.append('\t<parameter name="%s" value="%g" %sfree="%d" max="%g" min="%g" scale="1.0" />' % \
                           (name,param,err,free,max,min))

    strings.append('</spatialModel>')
    return ''.join([decorate(st,tablevel=tablevel) for st in strings])

def parse_sourcelib(xml):
    parser = x.make_parser()
    handler = SourceHandler()
    parser.setContentHandler(handler)
    if isinstance(xml,list):
	    [parser.parse(xmlfile) for xmlfile in xml]
    else:
	    parser.parse(xml)
    return handler

def parse_point_sources(handler,roi_dir,max_roi):
    """ Some simple testing. 

        In pointlike, we can load in the example point-like source.

        (XML from http://www.slac.stanford.edu/exp/glast/wb/prod/pages/sciTools_likelihoodTutorial/xmlModelDefinitions.htm)

            >>> from StringIO import StringIO
            >>> def l(xml): 
            ...     ps = parse_point_sources(parse_sourcelib(StringIO(xml)),roi_dir=None,max_roi=None)
            ...     assert len(ps) == 1
            ...     return ps[0]

            >>> ps=l('''<source name="PowerLaw_source" type="PointSource">  
            ...           <spectrum type="PowerLaw">
            ...             <parameter free="1" max="1000.0" min="0.001" name="Prefactor" scale="1e-09" value="1" error="0.5" />
            ...             <parameter free="1" max="-1.0" min="-5." name="Index" scale="1.0" value="-2.1" error="0.25" />
            ...             <parameter free="0" max="2000.0" min="30.0" name="Scale" scale="1.0" value="100.0"/>
            ...           </spectrum>      
            ...           <spatialModel type="SkyDirFunction">
            ...             <parameter free="0" max="360." min="-360." name="RA" scale="1.0" value="83.45"/>
            ...             <parameter free="0" max="90." min="-90." name="DEC" scale="1.0" value="21.72"/>
            ...           </spatialModel>    
            ...         </source>''')
            >>> print ps.name
            PowerLaw_source
            >>> print ps.skydir.ra(), ps.skydir.dec()
            83.45 21.72
            >>> print ps.model.name
            PowerLaw
            >>> print ps.model['Norm']
            1e-09
            >>> print ps.model['Index']
            2.1
            >>> print ps.model.error('Norm')
            5e-10
            >>> print ps.model.error('Index')
            0.25

    """
    point_sources = deque()
    xtm = XML_to_Model()
    for src in handler.sources:
        if src['type'] != 'PointSource': continue
        name = str(src['name'])
        sd = get_skydir(src.getChild('spatialModel'))
        mo = xtm.get_model(src.getChild('spectrum'),name)
        if None in [roi_dir,max_roi] or np.degrees(sd.difference(roi_dir))<max_roi:
            point_sources.append(PointSource(sd,name,mo,leave_parameters=True))
    return list(point_sources)

def parse_diffuse_sources(handler, roi_dir, max_roi, diffdir=None):
    """ 
        Some simple testing:

        First, we define a simpler helper function

            >>> from StringIO import StringIO
            >>> def l(xml): 
            ...     ds = parse_diffuse_sources(parse_sourcelib(StringIO(xml)), roi_dir=None, max_roi=None)
            ...     assert len(ds) == 1
            ...     return ds[0]


        Example loading in the galactic diffuse emission

        (XML from http://www.slac.stanford.edu/exp/glast/wb/prod/pages/sciTools_binnedLikelihoodTutorial/binnedLikelihood_v01archived.htm)
    
            >>> ds=l('''
            ...   <source name="Galpro Diffuse" type="DiffuseSource">
            ...     <spectrum type="ConstantValue">
            ...       <parameter free="0" max="10.0" min="0.0" name="Value" scale="1.0" value="1.0"/>
            ...     </spectrum>
            ...     <spatialModel file="$(GLAST_EXT)/diffuseModels/v2r0p1/ring_2year_P76_v0.fits" type="MapCubeFunction">
            ...       <parameter free="0" max="1000.0" min="0.001" name="Normalization" scale="1.0" value="1.0"/>
            ...     </spatialModel>
            ...   </source>''')
            >>> print ds.smodel.name
            Constant
            >>> print ds.smodel['scale']
            1.0
            >>> print ds.smodel.get_limits('scale')
            [0.0, 10.0]
            >>> type(ds.dmodel) == list and len(ds.dmodel) == 1
            True
            >>> dm=ds.dmodel[0]
            >>> print type(dm)
            <class 'skymaps.DiffuseFunction'>

        Try loading in Galactic Diffuse scaled by a powerlaw:

            >>> ds=l('''<source name="Galactic Diffuse (ring_2year_P76_v0.fits)" type="DiffuseSource">
            ...    <spectrum  type="PowerLaw">
            ...        <parameter name="Prefactor" value="1.0" free="1" max="10" min="0.1" scale="1" />
            ...        <parameter name="Index" value="0.0" free="1" max="1" min="-1" scale="-1" />
            ...        <parameter name="Scale" value="1000.0" free="0" max="1000.0" min="1000.0" scale="1" />
            ...    </spectrum>
            ...    <spatialModel file="$(GLAST_EXT)/diffuseModels/v2r0p1/ring_2year_P76_v0.fits" type="MapCubeFunction">
            ...        <parameter name="Normalization" value="1.0" free="0" max="1e3" min="1e-3" scale="1.0" />
            ...    </spatialModel>
            ... </source>''')
            >>> print ds.name
            Galactic Diffuse (ring_2year_P76_v0.fits)
            >>> model=ds.smodel
            >>> print model.name
            ScalingPowerLaw
            >>> print model['norm']
            1.0
            >>> print model['index']
            0.0
            >>> print model.background
            True
            >>> print model.get_limits('norm')
            [0.1, 10.0]
            >>> print model.get_limits('index')
            [-1.0, 1.0]

        Example loading an isotropic powerLaw source. This code is a klugey, and could be improved

        (XML from http://www.slac.stanford.edu/exp/glast/wb/prod/pages/sciTools_binnedLikelihoodTutorial/binnedLikelihood_v01archived.htm)

            >>> ds=l('''
            ...   <source name="Extragalactic Diffuse" type="DiffuseSource">
            ...     <spectrum type="PowerLaw">
            ...       <parameter free="1" max="100.0" min="1e-05" name="Prefactor" scale="1e-07" value="1.6"/>
            ...       <parameter free="0" max="-1.0" min="-3.5" name="Index" scale="1.0" value="-2.1"/>
            ...       <parameter free="0" max="200.0" min="50.0" name="Scale" scale="1.0" value="100.0"/>
            ...     </spectrum>
            ...     <spatialModel type="ConstantValue">
            ...       <parameter free="0" max="10.0" min="0.0" name="Value" scale="1.0" value="1.0"/>
            ...     </spatialModel>
            ...   </source>''')
            >>> print ds.smodel.name
            PowerLaw
            >>> print ds.smodel['norm']
            1.6e-07
            >>> print ds.smodel['index']
            2.1
            >>> print ds.smodel.e0
            100.0
            >>> type(ds.dmodel) == list and len(ds.dmodel) == 1
            True
            >>> dm=ds.dmodel[0]
            >>> print type(dm)
            <class 'skymaps.IsotropicConstant'>
            >>> print dm.constant()
            1.0

        Example loading an isotropic source from a file
    
        (xml from http://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/likelihood_tutorial.html)

            >>> ds=l('''
            ...   <source name="iso_p7v6source" type="DiffuseSource">
            ...     <spectrum file="$(GLAST_EXT)/diffuseModels/v2r0p1/isotrop_2year_P76_source_v1.txt" type="FileFunction">
            ...       <parameter free="1" max="1000" min="1e-05" name="Normalization" scale="1" value="1" />
            ...     </spectrum>
            ...     <spatialModel type="ConstantValue">
            ...       <parameter free="0" max="10.0" min="0.0" name="Value" scale="1.0" value="1.0"/>
            ...     </spatialModel>
            ...   </source>''')
            >>> print ds.smodel.name
            FileFunction
            >>> print ds.smodel['normalization']
            1.0
            >>> print ds.smodel.file
            $(GLAST_EXT)/diffuseModels/v2r0p1/isotrop_2year_P76_source_v1.txt
            >>> type(ds.dmodel) == list and len(ds.dmodel) == 1
            True
            >>> dm=ds.dmodel[0]
            >>> print type(dm)
            <class 'skymaps.IsotropicConstant'>
            >>> print dm.constant()
            1.0

            
        Now, we are going to test loading extended sources:

        First, create the profile from a simple Gaussian

            >>> from tempfile import NamedTemporaryFile
            >>> gaussian = SpatialModels.Gaussian(center=SkyDir(82.73, 13.38))

        First, create & load a new Disk extended soure

            >>> ds=l('''
            ...   <source name="test_map" type="DiffuseSource">
            ...      <spectrum type="ConstantValue">
            ...        <parameter free="1" max="1000.0" min="0.001" name="Value" scale="1" value="1.0"/>
            ...      </spectrum>
            ...      <spatialModel type="Gaussian">
            ...        <parameter free="0" name="RA" scale="1.0" value="%s"/>
            ...        <parameter free="1" name="DEC" scale="1.0" value="%s"/>
            ...        <parameter free="1" name="Sigma" scale="1.0" value="%s"/>
            ...      </spatialModel>
            ...    </source>''' % (gaussian['ra'], gaussian['dec'],gaussian['sigma']))
            >>> new_gauss =  ds.spatial_model
            >>> np.allclose(new_gauss.p, gaussian.p)
            True
            >>> print new_gauss.free
            [False  True  True]

        Now, create & load an xml file for a SpatialMap

            >>> temp = NamedTemporaryFile(delete=True)
            >>> gaussian.save_template(temp.name)

            >>> ds=l('''
            ...   <source name="test_map" type="DiffuseSource">
            ...      <spectrum type="ConstantValue">
            ...        <parameter free="1" name="Value" scale="1" min="0.1" max="10.0" value="1.0"/>
            ...      </spectrum>
            ...      <spatialModel type="SpatialMap" file="%s">
            ...        <parameter free="0" name="Prefactor" scale="1.0" value="1"/>
            ...      </spatialModel>
            ...    </source>''' % temp.name)
            >>> print ds.model['scale']
            1.0
            >>> map=ds.spatial_model
            >>> print map.name
            SpatialMap
            >>> np.allclose([map.center.ra(), map.center.dec()], [gaussian.center.ra(), gaussian.center.dec()], atol=1e-2, rtol=1e-2)
            True
            >>> np.allclose(map(map.center),gaussian(map.center), atol=1e-2, rtol=1e-2)
            True

        Now, create & load a an xml file for a RadialProfile

            >>> temp = NamedTemporaryFile(delete=True)
            >>> gaussian.save_profile(temp.name, numpoints=1000)

            >>> ds=l('''
            ...   <source name="test_profile" type="DiffuseSource">
            ...      <spectrum type="ConstantValue">
            ...        <parameter free="1" name="Value" scale="1" min="0.1" max="10.0" value="1.0"/>
            ...      </spectrum>
            ...      <spatialModel type="RadialProfile" file="%s">
            ...        <parameter free="0" name="Normalization" scale="1.0" value="1"/>
            ...        <parameter free="0" name="RA" scale="1.0" value="%s"/>
            ...        <parameter free="0" name="DEC" scale="1.0" value="%s"/>
            ...      </spatialModel>
            ...    </source>''' % (temp.name, gaussian.center.ra(), gaussian.center.dec()))
            >>> profile=ds.spatial_model
            >>> print profile.name
            RadialProfile
            >>> np.allclose([profile.center.ra(), profile.center.dec()], [gaussian.center.ra(), gaussian.center.dec()])
            True
            >>> np.allclose(profile(profile.center),gaussian(profile.center), atol=1e-5, rtol=1e-5)
            True

        You should also be able to load in extended soruces in galactic coordinates

            >>> ds=l('''
            ...   <source name="test_map" type="DiffuseSource">
            ...      <spectrum type="ConstantValue">
            ...        <parameter free="1" max="1000.0" min="0.001" name="Value" scale="1" value="1.0"/>
            ...      </spectrum>
            ...      <spatialModel type="Gaussian">
            ...        <parameter free="0" name="L" scale="1.0" value="0"/>
            ...        <parameter free="1" name="B" scale="1.0" value="0"/>
            ...        <parameter free="1" name="Sigma" scale="1.0" value="1"/>
            ...      </spatialModel>
            ...    </source>''')
            >>> ds.spatial_model['l']
            0.0
            >>> ds.spatial_model['b']
            0.0
            >>> ds.spatial_model.coordsystem == SkyDir.GALACTIC
            True

        Note, reading spatial models will preserve limits:

            >>> ds=l('''
            ...   <source name="test_map" type="DiffuseSource">
            ...      <spectrum type="ConstantValue">
            ...        <parameter free="1" max="1000.0" min="0.001" name="Value" scale="1" value="1.0"/>
            ...      </spectrum>
            ...      <spatialModel type="Gaussian">
            ...        <parameter free="0" name="RA" scale="1.0" value="0" min="-360" max="360" />
            ...        <parameter free="1" name="DEC" scale="1.0" value="0" min="-90" max="90" />
            ...        <parameter free="1" name="Sigma" scale="1.0" value="1" min="1e-3" max="1e3" />
            ...      </spatialModel>
            ...    </source>''')
            >>> print ds.spatial_model.get_limits(absolute=True).tolist()
            [[-1.0, 1.0], [-1.0, 1.0], [0.001, 1000.0]]

        When the the position parameters don't the default limits, the code will crash:

            >>> ds=l('''
            ...   <source name="test_map" type="DiffuseSource">
            ...      <spectrum type="ConstantValue">
            ...        <parameter free="1" max="1000.0" min="0.001" name="Value" scale="1" value="1.0"/>
            ...      </spectrum>
            ...      <spatialModel type="Gaussian">
            ...        <parameter free="0" name="RA" scale="1.0" value="0" min="-36" max="36" />
            ...        <parameter free="1" name="DEC" scale="1.0" value="0" min="-9" max="9" />
            ...        <parameter free="1" name="Sigma" scale="1.0" value="1" min="1e-3" max="1e3" />
            ...      </spatialModel>
            ...    </source>''')
            Traceback (most recent call last):
                ...
            Exception: Error reading spatial parameter RA for source test_map. Parameter range must be -360 to 360


        For now, you cannot scale parameters:

            >>> ds=l('''
            ...   <source name="test_map" type="DiffuseSource">
            ...      <spectrum type="ConstantValue">
            ...        <parameter free="1" max="1000.0" min="0.001" name="Value" scale="1" value="1.0"/>
            ...      </spectrum>
            ...      <spatialModel type="Gaussian">
            ...        <parameter free="0" name="RA" scale="2.0" value="0" min="-360" max="360" />
            ...        <parameter free="1" name="DEC" scale="2.0" value="0" min="-90" max="90" />
            ...        <parameter free="1" name="Sigma" scale="2.0" value="1" min="1e-3" max="1e3" />
            ...      </spatialModel>
            ...    </source>''')
            Traceback (most recent call last):
                ...
            Exception: Error reading spatial parameter RA for source test_map. Parameter scale must be set to 1
    """
    gds = get_diffuse_source
    ds = deque()
    xtm = XML_to_Model()
    for src in handler.sources:
        if src['type'] != 'DiffuseSource': 
            continue
        spatial  = src.getChild('spatialModel')
        spectral = src.getChild('spectrum')
        name = str(src['name'])
        if spatial['type'] == 'ConstantValue':
            mo = xtm.get_model(spectral,name, scaling=False)
            ds.append(gds('ConstantValue',None,mo,None,name,diffdir=diffdir))
    
        elif spatial['type'] == 'MapCubeFunction':
            fname = str(path.expand(spatial['file']))
            mo = xtm.get_model(spectral,name, scaling=True)
            ds.append(gds('MapCubeFunction',fname,mo,None,name,diffdir=diffdir))
        elif spatial['type'] in XML_to_SpatialModel.spatialdict:

                spatial_model=XML_to_SpatialModel.get_spatial_model(spatial,name,diffdir=diffdir)
                spectral_model=xtm.get_model(spectral,name)

                sd=spatial_model.center
                if None in [roi_dir,max_roi] or np.degrees(sd.difference(roi_dir))<max_roi:
                    ds.append(ExtendedSource(name=name,
                                             model=spectral_model,
                                             spatial_model=spatial_model))
        else:
            raise XMLException('Diffuse spatial model "{}", source {}, not recognized'.format(
                 spatial['type'], name))
    return list(ds)

def parse_sources(xmlfile,diffdir=None,roi_dir=None,max_roi=None):
    """ Convenience function to parse an entire file into
        point sources and diffuse sources. 
        

            >>> ps,ds=parse_sources(xmlfile='$GLAST_EXT/catalogProducts/v1r2/2FGL/gll_psc_v07.xml',
            ...                     diffdir='/afs/slac/g/glast/groups/catalog/2FGL/gll_psc_v05_templates/')

        The 12 extended 2FGL sources are from Table 2 of 2FGL (http://arxiv.org/pdf/1108.1435v2.pdf)
            >>> set(i.name for i in ds) == set([ '2FGL J0059.0-7242e', '2FGL J0526.6-6825e', '2FGL J0617.2+2234e',
            ...                                  '2FGL J0833.1-4511e', '2FGL J1324.0-4330e', '2FGL J1514.0-5915e',
            ...                                  '2FGL J1801.3-2326e', '2FGL J1805.6-2136e', '2FGL J1824.5-1351e',
            ...                                  '2FGL J1855.9+0121e', '2FGL J1923.2+1408e', '2FGL J2051.0+3040e'])
            True
        
        Prove the roi_dir/max_roi flag work. Get only 2FGL sources within 5 degrees of IC443:

            >>> ps,ds=parse_sources(xmlfile='$GLAST_EXT/catalogProducts/v1r2/2FGL/gll_psc_v07.xml',
            ...                     diffdir='/afs/slac/g/glast/groups/catalog/2FGL/gll_psc_v05_templates/',
            ...                     roi_dir=SkyDir(94.332,22.570), max_roi=5)
            >>> [i.name for i in ps]
            ['2FGL J0608.3+2037', '2FGL J0616.6+2425', '2FGL J0621.2+2508']
            >>> [i.name for i in ds]
            ['2FGL J0617.2+2234e']
    """
    if isinstance(xmlfile, basestring):
        xmlfile=path.expand(xmlfile)
    else:
        xmlfile=map(path.expand,xmlfile)

    handler = parse_sourcelib(xmlfile)
    ps = parse_point_sources(handler,roi_dir,max_roi)
    ds = parse_diffuse_sources(handler,roi_dir,max_roi,diffdir=diffdir)
    return ps,ds

def unparse_point_sources(point_sources, strict=False, expand_env_vars=False, properties=lambda x:''):
    """ Convert a list (or other iterable) of PointSource objects into XML.
        strict : bool
            set True to generate exception, error message identifying offending source, reason
        properties : a function
            the function, if specified, returns a string for the source element with properties, like TS

            Example:

                >>> def ps2xml(ps, expand_env_vars=False):
                ...     ret=unparse_point_sources([ps], expand_env_vars=expand_env_vars)
                ...     return ret[0].strip().replace('\\t', ' '*4)

                >>> ps = PointSource(name='test', model=Models.Constant(), skydir=SkyDir(-30,30))
                >>> print ps2xml(ps)
                <source name="test" type="PointSource"  >
                    <spectrum  type="ConstantValue">
                        <parameter name="Value" value="1.0" free="1" max="10" min="0.001" scale="1" />
                    </spectrum>
                    <spatialModel type="SkyDirFunction">
                        <parameter name="RA"  value="330.0" free="0" max="360.0" min="-360.0" scale="1.0" />
                        <parameter name="DEC" value="30.0" free="0" max="90" min="-90" scale="1.0" />
                    </spatialModel>
                </source>

            Previously, this was buggy. The expand_env_vars flag would case the Prefactor to be nan.
            This doctest protects against that edge case.

                >>> ps = PointSource(name='test', model=Models.PowerLaw(), skydir=SkyDir(20,-88))
                >>> print ps2xml(ps, expand_env_vars=True)
                <source name="test" type="PointSource"  >
                    <spectrum  type="PowerLaw">
                        <parameter name="Prefactor" value="1.0" free="1" max="100.0" min="0.01" scale="1e-11" />
                        <parameter name="Index" value="2.0" free="1" max="5" min="-5" scale="-1" />
                        <parameter name="Scale" value="1000.0" free="0" max="1000.0" min="1000.0" scale="1" />
                    </spectrum>
                    <spatialModel type="SkyDirFunction">
                        <parameter name="RA"  value="20.0" free="0" max="360.0" min="-360.0" scale="1.0" />
                        <parameter name="DEC" value="-88.0" free="0" max="90" min="-90" scale="1.0" />
                    </spatialModel>
                </source>

    """
    xml_blurbs = Stack()
    m2x = Model_to_XML(strict=strict)
    for ps in point_sources:
        skyxml = makePSSpatialModel(ps.skydir)
        try:
            m2x.process_model(ps.model, expand_env_vars=expand_env_vars)
        except Exception, emsg:
            print 'Failed to process source %s: %s' %(ps.name, emsg)
        specxml = m2x.getXML()
        s1 = '\n<source name="%s" type="PointSource" %s >\n'%(ps.name, properties(ps))
        s2 = '</source>'
        xml_blurbs.push(''.join([s1,specxml,skyxml,s2]))
    return xml_blurbs


def process_diffuse_source(ds,strict=False,convert_extended=False,
                           extended_dir_name=None,
                           expand_env_vars=False,filename=None,ctype=None):
    """ Convert an instance of DiffuseSource into an XML blurb.
        
        Some simple testing of saving out diffuse sources:

            >>> from uw.like.pointspec_helpers import get_default_diffuse
            >>> diffdir='$(GLAST_EXT)/diffuseModels/v2r0p1/'
            >>> gfile="ring_2year_P76_v0.fits"
            >>> ifile="isotrop_2year_P76_source_v1.txt"
            >>> gal, iso = get_default_diffuse(diffdir=diffdir,
            ...     gfile=gfile,
            ...     ifile=ifile)
            >>> def model2xml(d, expand_env_vars, strict=False):
            ...     return process_diffuse_source(d, expand_env_vars=expand_env_vars, strict=strict).replace('\\t',' '*4).strip()

            >>> print model2xml(gal, expand_env_vars=False)
            <source name="Galactic Diffuse (ring_2year_P76_v0.fits)" type="DiffuseSource">
                <spectrum  type="PowerLaw">
                    <parameter name="Prefactor" value="1.0" free="1" max="10" min="0.1" scale="1" />
                    <parameter name="Index" value="0.0" free="1" max="1" min="-1" scale="-1" />
                    <parameter name="Scale" value="1000.0" free="0" max="1000.0" min="1000.0" scale="1" />
                </spectrum>
                <spatialModel file="$(GLAST_EXT)/diffuseModels/v2r0p1/ring_2year_P76_v0.fits" type="MapCubeFunction">
                    <parameter name="Normalization" value="1.0" free="0" max="1e3" min="1e-3" scale="1.0" />
                </spatialModel>
            </source>

        Test out the expand_env_vars flag:

            >>> print model2xml(gal, expand_env_vars=True).replace(path.expand(join(diffdir,gfile)),'[FILENAME]')
            <source name="Galactic Diffuse (ring_2year_P76_v0.fits)" type="DiffuseSource">
                <spectrum  type="PowerLaw">
                    <parameter name="Prefactor" value="1.0" free="1" max="10" min="0.1" scale="1" />
                    <parameter name="Index" value="0.0" free="1" max="1" min="-1" scale="-1" />
                    <parameter name="Scale" value="1000.0" free="0" max="1000.0" min="1000.0" scale="1" />
                </spectrum>
                <spatialModel file="[FILENAME]" type="MapCubeFunction">
                    <parameter name="Normalization" value="1.0" free="0" max="1e3" min="1e-3" scale="1.0" />
                </spatialModel>
            </source>

        Previously, this was buggy (having the norm for the ScalingPowerLaw outside the default limit
        for a ScalingPowerLaw. Now, the code issues a warning and sets parameter to lower limit.

            >>> gal.smodel['norm']=1e-2
            >>> xml=model2xml(gal, expand_env_vars=False, strict=True)
            Traceback (most recent call last):
                ...
            ModelException: Found Norm=0.01 < 0.1, minimum allowed value


            >>> xml=model2xml(gal, expand_env_vars=False)
            WARNING: Found Norm=0.01 < 0.1, minimum allowed value,
                Setting parameter value to minimum.
            >>> print xml
            <source name="Galactic Diffuse (ring_2year_P76_v0.fits)" type="DiffuseSource">
                <spectrum  type="PowerLaw">
                    <parameter name="Prefactor" value="0.1" free="1" max="10" min="0.1" scale="1" />
                    <parameter name="Index" value="0.0" free="1" max="1" min="-1" scale="-1" />
                    <parameter name="Scale" value="1000.0" free="0" max="1000.0" min="1000.0" scale="1" />
                </spectrum>
                <spatialModel file="$(GLAST_EXT)/diffuseModels/v2r0p1/ring_2year_P76_v0.fits" type="MapCubeFunction">
                    <parameter name="Normalization" value="1.0" free="0" max="1e3" min="1e-3" scale="1.0" />
                </spatialModel>
            </source>

        Save out the isotropic diffuse:

            >>> print model2xml(iso, expand_env_vars=False)
            <source name="Isotropic Diffuse (isotrop_2year_P76_source_v1.txt)" type="DiffuseSource">
                <spectrum file="$(GLAST_EXT)/diffuseModels/v2r0p1/isotrop_2year_P76_source_v1.txt" ctype="-1" type="FileFunction">
                    <parameter name="Normalization" value="1.0" free="1" max="10000.0" min="0.0001" scale="1" />
                </spectrum>
                <spatialModel type="ConstantValue">
                    <parameter  name="Value" value="1.0" free="0" max="10.0" min="0.0" scale="1.0" />
                </spatialModel>
            </source>

        Test expand_env_vars:

            >>> print model2xml(iso, expand_env_vars=True).replace(path.expand(join(diffdir,ifile)),'[FILENAME]')
            <source name="Isotropic Diffuse (isotrop_2year_P76_source_v1.txt)" type="DiffuseSource">
                <spectrum file="[FILENAME]" ctype="-1" type="FileFunction">
                    <parameter name="Normalization" value="1.0" free="1" max="10000.0" min="0.0001" scale="1" />
                </spectrum>
                <spatialModel type="ConstantValue">
                    <parameter  name="Value" value="1.0" free="0" max="10.0" min="0.0" scale="1.0" />
                </spatialModel>
            </source>


        This was previously a bug found by Francesco Giordano:

            >>> gal, iso = get_default_diffuse(diffdir=diffdir,
            ...     gfile=gfile,
            ...     ifile=ifile)
            >>> gal.smodel.set_mapper('index',LimitMapper(-1,1,-1))
            >>> print model2xml(gal, expand_env_vars=False)
            <source name="Galactic Diffuse (ring_2year_P76_v0.fits)" type="DiffuseSource">
                <spectrum  type="PowerLaw">
                    <parameter name="Prefactor" value="1.0" free="1" max="10" min="0.1" scale="1" />
                    <parameter name="Index" value="-0" free="1" max="1" min="-1" scale="1" />
                    <parameter name="Scale" value="1000.0" free="0" max="1000.0" min="1000.0" scale="1" />
                </spectrum>
                <spatialModel file="$(GLAST_EXT)/diffuseModels/v2r0p1/ring_2year_P76_v0.fits" type="MapCubeFunction">
                    <parameter name="Normalization" value="1.0" free="0" max="1e3" min="1e-3" scale="1.0" />
                </spatialModel>
            </source>
    """
    m2x = Model_to_XML(strict=strict)
    dm = ds.dmodel
    if hasattr(dm,'__len__') and len(dm)==1: dm = dm[0]

    if isinstance(ds,ExtendedSource) or hasattr(ds,'spatial_model'):
        m2x.process_model(ds.smodel,scaling=False)
        specxml  = m2x.getXML()
        spatial  = ds.spatial_model
        spectral = ds.smodel
        if convert_extended and not isinstance(spatial,SpatialModels.SpatialMap): 
            if hasattr(spatial,'original_template') and np.all(spatial.original_parameters == spatial.p):
                # Kludge! this is incase the xml was read in from the 
                # pointspec_helpers.ExtendedSourceArchive and should be saved 
                # out with the original template.
                spatial=SpatialModels.SpatialMap(file=spatial.original_template)
            else:
                if extended_dir_name is not None:
                    folder = extended_dir_name
                else:
                    folder=os.path.dirname(filename or os.getcwd())
                template_name=folder+os.sep if folder != '' else ''
                template_name+='template_%s_%s_%s.fits' % (ds.name.replace(' ','_'),
                                                           spatial.pretty_name, 
                                                           spectral.pretty_name)
                spatial = SpatialModels.convert_spatial_map(spatial,template_name)
                spatial.file = template_name
        skyxml = makeExtendedSourceSpatialModel(spatial,expand_env_vars=expand_env_vars)
        if isinstance(spatial,SpatialModels.SpatialMap) and not np.all(spatial.p==spatial.init_p):
            print 'Warning: When saving out a SpatialModels.SpatialMap object which has been localized, the original unmoved template is saved in the xml model.'

    elif isinstance(dm,DiffuseFunction):
        if hasattr(dm,'filename'):
            # uw.like.pointspec_helpers.get_diffuse_source will
            # store the env varaible without expanding the env variables.
            # If this filename, it is nicer to preserve environment variables.
            # if expand_env_vars=True, the filename will be expanded
            # in makeDSMapcubeSpatialModel
            filename=dm.filename
        else:
            filename = os.path.abspath(dm.name())
        skyxml = makeDSMapcubeSpatialModel(filename=filename, expand_env_vars=expand_env_vars)
        m2x.process_model(ds.smodel,scaling=True)
        specxml = m2x.getXML()
    else:
        skyxml = makeDSConstantSpatialModel()
        #Handle the case where the isotropic is specified by separate front and back components
        if not hasattr(dm,'__len__'): dm = [dm]
        if len(dm)==1:
            ctypes=[-1]
            specxml=''
        elif len(dm)==2:
            ctypes = [0,1]
            specxml='\t<spectrum type="CompositeSpectrum">\n'
        else:
            raise XMLException("Don't know how to handle isotropic model with >2 components")
        for m,ct in zip(dm,ctypes):
            if isinstance(m,IsotropicSpectrum) or isinstance(m,IsotropicConstant):

                if isinstance(m,IsotropicSpectrum):

                    if ds.smodel.name != 'Constant': #not isinstance(ds.smodel,Constant):
                        raise XMLException("Can only save out ConstantValue with FileFunction if it is modulated by a constant value.")

                    filename = os.path.abspath(m.name())
                    model = Models.FileFunction(
                        normalization=ds.smodel['Scale'],
                        file=filename)
                    m2x.process_model(model,scaling=True, expand_env_vars=expand_env_vars)

                elif isinstance(m,IsotropicConstant):
                    model = ds.smodel
                    m2x.process_model(model,scaling=False, expand_env_vars=expand_env_vars)


                m2x.extra_attrs+=' ctype="{0}"'.format(ct)
                specxml += m2x.getXML(tablevel=len(dm))
            elif isinstance(m,IsotropicPowerLaw):
                flux,index=m.flux(),m.index()
                pl=PowerLawFlux(index=index)
                pl.set_flux(flux,emin=100,emax=np.inf)

                if isinstance(ds.smodel,Constant):
                    pl['Int_Flux'] *= ds.smodel['Scale']
                else:
                    raise XMLException("...")
                pl.cov_matrix = ds.smodel.cov_matrix.copy() #correct?
                m2x.process_model(pl,scaling=False, expand_env_vars=expand_env_vars)
                specxml += m2x.getXML(tablevel=len(dm))
            else:
                raise XMLException('Did not recognize %s'%(ds.name))
        if len(dm)==2:
            specxml+='\t</spectrum>\n'
    s1 = '\n<source name="%s" type="DiffuseSource">\n'%(ds.name)
    s2 = '</source>'
    return ''.join([s1,specxml,skyxml,s2])
    
def unparse_diffuse_sources(diffuse_sources,strict=False,
                            convert_extended=False,
                            extended_dir_name=None,
                            expand_env_vars=False,filename=None):
    """Convert a list of DiffuseSources into XML blurbs."""
    xml_blurbs = Stack()
    for ds in diffuse_sources:
        #try:
        xml_blurbs.push(process_diffuse_source(ds,
                                               strict=strict,
                                               convert_extended=convert_extended,
                                               extended_dir_name=extended_dir_name,
                                               expand_env_vars=expand_env_vars,
                                               filename=filename))
        #except Exception, msg:
        #    print 'Failed to create blurb for extended source %s: %s' % (ds.name, msg)
    return xml_blurbs

def writeXML(stacks,filename, title='source_library'):
    """Write XML blurbs to a gtlike-style XML file."""
    f = open(filename,'wb') if type(filename)==str else filename
    f.write('<source_library title="%s">'% title)
    for stack in stacks:
        for elem in stack:
            f.write(elem)
    f.write('\n</source_library>')
    f.close()

def write_sources(point_sources, diffuse_sources, filename, strict=False,
                  convert_extended=False,
                  extended_dir_name=None,
                  expand_env_vars=False):
    source_xml = [unparse_point_sources(point_sources, strict=strict, expand_env_vars=expand_env_vars)]
    if len(diffuse_sources)>0:
        source_xml.append(unparse_diffuse_sources(diffuse_sources,
                                                  strict=strict,
                                                  convert_extended=convert_extended,
                                                  extended_dir_name=extended_dir_name,
                                                  expand_env_vars=expand_env_vars,
                                                  filename=filename))
    writeXML(source_xml,filename)

def writeROI(roi,*args, **kwargs):
    """ Out the contents of an ROIAnalysis source model
        to a gtlike XML file.

        the strict flag raises an exception if any of the spectral
        parameters are outside of the fit range.

        The convert_extended flag converts extended sources to be fully
        compliant with gtlike.  By default, extended sources are saved
        out with a format incompatable with gtlike. This is because
        each diffuse source has its spatial parameters saved out in
        the xml file. This is useful for saving errors on the fit
        spatial parameters as well as for easibly being read back into
        pointlike. OTOH, it is sometimes desirable for extended source
        output to be strictly compatable with gtlike. This can be done
        with the convert_extended flag, which converts all extended
        sources to SpatialMap objects before the xml is created. 

        The extended_dir_name flag specifies the directory to but
        converted extended sources into.
        """
    write_sources(roi.psm.point_sources, roi.dsm.diffuse_sources, *args, **kwargs)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
