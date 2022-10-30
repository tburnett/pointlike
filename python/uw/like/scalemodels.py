""" Code to implement ScaleFactor:: decorator supported
    in gtlike.

    The gtlike feature is documented here:

        https://confluence.slac.stanford.edu/display/ST/Science+Tools+Development+Notes?focusedCommentId=103582318#comment-103582318
    
    Author: Joshua Lande
"""
import operator
from copy import deepcopy

import numpy as np

from uw.like.Models import PowerLaw, PowerLawFlux, FileFunction, PLSuperExpCutoff, PLSuperExpCutoff4,Gaussian, Constant, CompositeModel
from uw.darkmatter.spectral import DMFitFunction

def build_scale_factor(model_class):
    """

        First, create the ScaleFactorPowerLaw and a comparison PowerLaw

            >>> scale = 3.133141
            >>> sfpl=ScaleFactorPowerLaw(ScaleFactor=scale)
            >>> pl = PowerLaw()

            >>> print sfpl.name
            ScaleFactorPowerLaw

            >>> print sfpl.gtlike['name']
            ScaleFactor::PowerLaw

            >>> print sfpl.pretty_name
            ScaleFactor::PowerLaw

            >>> print sfpl.full_name()
            ScaleFactor::PowerLaw, e0=1000, ScaleFactor=3.133141

            >>> print sfpl.e0 == pl.e0
            True

            >>> sfpl.default_extra_params == pl.default_extra_params
            True

            >>> np.all(sfpl.default_p == [1] + pl.default_p)
            True
            >>> print sfpl.param_names == ['ScaleFactor'] + pl.param_names
            True

            >>> print np.all(sfpl.default_mappers == Constant.default_mappers + PowerLaw.default_mappers)
            True
            >>> sfpl.default_extra_params == pl.default_extra_params
            True
            >>> sfpl.default_extra_attrs == sfpl.default_extra_attrs
            True

            >>> print sfpl.default_oomp_limits == ['ScaleFactor'] + PowerLaw.default_oomp_limits
            True

        Make sure that default_limits acts correclty

            >>> dl=sfpl.default_limits
            >>> dl['Norm'] == pl.default_limits['Norm']
            True
            >>> dl['Index'] == pl.default_limits['Index']
            True
            >>> dl['ScaleFactor'] == Constant.default_limits['Scale']
            True

        Make sure the __call__ function is correct
            >>> energies=np.logspace(1,5,100)
            >>> np.all(sfpl(energies) == scale*pl(energies))
            True

        And that the gradient follows the chain rule:

            >>> grad = sfpl.external_gradient(energies)
            >>> np.all(grad[0] == pl(energies))
            True
            >>> np.all(grad[1:] == scale*pl.external_gradient(energies))
            True

        Note, we can set default limits for ScaleFactor objects (necessary for XML creation):
            >>> print sfpl.mappers == Constant.default_mappers + PowerLaw.default_mappers
            True
            >>> print np.all(sfpl.default_mappers == Constant.default_mappers + PowerLaw.default_mappers)
            True
            >>> sfpl.set_default_limits()
            >>> sfpl.mappers == [Constant.default_limits['Scale'],PowerLaw.default_limits['Norm'],PowerLaw.default_limits['Index']]
            True

        Also, you can obtain the unfit parameters either as values of the object or with getp/setp

            >>> sfpl.e0 == pl.e0 and sfpl['e0'] == pl.e0 and sfpl.getp('e0') == pl.e0
            True

        We can create ScaleFactor object for other models. For PowerLawFlux:
            >>> sfpl2=ScaleFactorPowerLawFlux(ScaleFactor=scale)
            >>> pl2 = PowerLawFlux()
            >>> print sfpl2.name
            ScaleFactorPowerLawFlux

            >>> print sfpl2.gtlike['name']
            ScaleFactor::PowerLaw2
            >>> sfpl2.emax == pl2.emax and sfpl2.emax == pl2.emax
            True

        And, of course, the values are just scaled

            >>> np.all(sfpl2(energies) == scale*pl2(energies))
            True

        There is also a ScaleFactorFileFunction object, which acts just like a FileFunction.

            >>> from tempfile import NamedTemporaryFile
            >>> temp = NamedTemporaryFile()
            >>> filename = temp.name
            >>> sfpl2.save_profile(filename, emin=1, emax=1e5)
            >>> temp.seek(0)

            >>> sfff = ScaleFactorFileFunction(ScaleFactor=5.5, normalization=1, file=filename)

            >>> np.allclose(sfff(energies),5.5*sfpl2(energies),rtol=1e-10, atol=1e-10)
            True

        Note, it sets default_extra_attrs correctly:

            >>> sfff.default_extra_attrs == FileFunction.default_extra_attrs
            True
            >>> sfff.file == filename
            True
    """
    # For a description of creating classes on the fly, see:
    #   http://jjinux.blogspot.com/2005/03/python-create-new-class-on-fly.html
    c = type('ScaleFactor' + model_class.__name__, (CompositeModel,), {})

    # Note, default_p, param_names, default_mappers, automatically taken care of by CompositeModel
    c.default_extra_params=model_class.default_extra_params
    c.default_extra_attrs=model_class.default_extra_attrs

    c.gtlike = deepcopy(model_class.gtlike)

    c.gtlike['name']='ScaleFactor::%s' % c.gtlike['name']
    c.gtlike['param_names'].insert(0,'ScaleFactor')
    c.gtlike['topointlike'].insert(0,operator.pos)
    c.gtlike['togtlike'].insert(0,operator.pos)

    def __init__(self, **kwargs):
        scale = Constant(name='ScaleFactor')
        scale.default_oomp_limits=['ScaleFactor']

        if 'ScaleFactor' in kwargs:
            scale['ScaleFactor'] = kwargs.pop('ScaleFactor')
        m=model_class(**kwargs)
        super(c,self).__init__(scale,m)
        self.scale=scale
        self.model=m

    for p in c.default_extra_params.keys() + c.default_extra_attrs.keys():
        # Allow getting and setting the default_extra_params and default_extra_attrs
        # directly through the self.model object.
        get=lambda self: getattr(self.model,p)
        set=lambda self, value: setattr(self.model,p,value)
        setattr(c,p,property(get, set, p))

    c.__init__ = __init__

    c.__call__ = lambda self,e: self.scale.__call__(e)*self.model.__call__(e)

    c.pretty_name = property(lambda self: 'ScaleFactor::%s' % self.model.pretty_name)

    c.full_name = lambda self: 'ScaleFactor::%s, ScaleFactor=%s' % (self.model.full_name(),self['ScaleFactor'])

    def external_gradient(self, e):
        a=self.scale.external_gradient(e)*self.model.__call__(e)
        b=self.scale.__call__(e)*self.model.external_gradient(e)
        return np.concatenate((a,b),axis=0)
    c.external_gradient = external_gradient

    return c


ScaleFactorPowerLaw=build_scale_factor(PowerLaw)
ScaleFactorPowerLawFlux=build_scale_factor(PowerLawFlux)
ScaleFactorFileFunction=build_scale_factor(FileFunction)
ScaleFactorDMFitFunction=build_scale_factor(DMFitFunction)
ScaleFactorPLSuperExpCutoff=build_scale_factor(PLSuperExpCutoff)
ScaleFactorPLSuperExpCutoff4=build_scale_factor(PLSuperExpCutoff4)
ScaleFactorGaussian=build_scale_factor(Gaussian)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
