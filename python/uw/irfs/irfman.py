"""Module for managing instrument response functions.

$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/irfs/irfman.py,v 1.7 2018/01/27 15:35:07 burnett Exp $
Author: Eric Wallace
"""
__version__="$Revision: 1.7 $"

import numpy as np

from . import (caldb, psf, effective_area, exposure, IrfError)
from . import psfman # new version with bug corrected, implements functions with C++ code for speed


class IrfManager(object):
    """Manage loading sets of IRFs.
    """
    event_type_names = ('front','back', 'psf0','psf1','psf2','psf3','edisp0',
                         'edisp1','edisp2','edisp3')
    event_type_partitions = dict(fb = (0,1),
                                 psf = (2,3,4,5),
                                 edisp = (6,7,8,9))

    def __init__(self, dataset, irf_dir="$CALDB", irfname=None, event_types=None,):
        """
        Parameters
        ---------
        dataset : DataSet object or None
            If not None, used to set the irfname, event types, and exposure details
            If None, use parameter values, and define psf without weighting over exposure
        irf_dir : string
            Path to the CALDB folder 
        """
        self.caldb = caldb.CALDB(irf_dir, dataset.irf)
        if dataset is not None:
            self.irfname = dataset.irf
            event_types = 'psf' if dataset.psf_event_types else 'fb'
            self.dataname = dataset.name
            self.ltcube = dataset.ltcube
        else:
            print 'No dataset, loading defaults'
            self.dataname = '(none)'
            self.ltcube=''
            self.irfname = irfname
        irfname_parts = irfname.split('_')
        self.event_class = irfname_parts.pop(1)
        self.irf_version = '_'.join(irfname_parts)
        self.event_types = self._parse_event_type(event_types)
        self._load_irfs(dataset)

    def __repr__(self):
        txt = '{self.__class__}, event_class: {self.event_class}, event_types: {self.event_types}'.format(self=self)
        txt +='\n\tdataset: {self.dataname}'.format(self=self)
        txt +='\n\tCALDB:   {self.caldb}'.format(self=self)
        txt +='\n\tirfname: {self.irfname}'.format(self=self)
        return txt
        
    def _load_irfs(self, dataset):
        psf_info = self.caldb('psf',version=self.irf_version,
                                    event_class = self.event_class,
                                    event_type = self.event_types)
        aeff_info = self.caldb('aeff',version=self.irf_version,
                                      event_class = self.event_class,
                                      event_type = self.event_types)
        self._aeff = {et:effective_area.EffectiveArea(d['filename'],
                                     aeff_extension=d['extensions']['EFF_AREA'],
                                     eff_params_extension=d['extensions']['EFFICIENCY_PARS'])
                            for et,d in aeff_info.items()}
                            
        if dataset is not None:
            if dataset.legacy or hasattr(dataset, 'thetacut'):
                #No DSS keywords, assume dataset has thetacut member
                #TODO: Fix dataman.DataSpec to provide a sensible default
                #      or find something to reference that works for both cases
                cthetamin = np.cos(np.radians(dataset.thetacut))
            else:
                cthetamin = np.cos(np.radians(dataset.theta_cut.get_bounds()[1]))
            # Allows for multiple livetime cubes
            self._exposure = {et:exposure.Exposure(dataset.livetime_cube(et),
                    aeff,cthetamin=cthetamin)
                                for et,aeff in self._aeff.items()}
        else:
            self._exposure = None
        self._psf = {et:psf.PSF(d['filename'],
                                     rpsf_extension=d['extensions']['RPSF'],
                                     psf_scaling_extension=d['extensions']['PSF_SCALING'],
                                     exposure = self._exposure[et] if dataset is not None else None)
                            for et,d in psf_info.items()}

        # laod THB version for PSF management
        self._psfman = psfman.PSFmanager(self.caldb,  livetimefile=self.ltcube)
        
    def psf(self,event_type,energy):
        """Return a BandPSF for the given energy and event_type.
        """
        et = self._parse_event_type(event_type)
        return self._psfman(et, energy) # THB version

        #return self._psf[et].band_psf(energy)

    def exposure(self,event_type,energy):
        """Return a BandExposure obect for the given energy and event_type."""
        assert self._exposure is not None
        et = self._parse_event_type(event_type)
        assert et in self._exposure, 'Event type index, {}, not in {}'.format(et, self._exposure.keys())
        # if et>= len(self._exposure):
        #     k, et =et,  1- (et-2)/2 # 
        #     if getattr(self, '_warned', False):
        #         print 'WARNING: using front or back exposure for a PSF: {}->{}'.format(k, et)
        #         self._warned=True

        return self._exposure[et].band_exposure(energy)

    def _parse_event_type(self,event_type):
        """Find the event type or types for a given event_type selection""" 
        if hasattr(event_type,'__iter__'):
            return [self._parse_event_type(et) for et in event_type]

        try:
            event_type = event_type.lower()
        except AttributeError: #Not a string, assume index
            if event_type in range(10):
                return event_type
            else:
                raise IrfError("Invalid event type index {}. Should be 0-9.".format(event_type))

        try:
            return self.event_type_names.index(event_type)
        except ValueError as exc:
            try: return self.event_type_partitions[event_type]
            except KeyError:
                raise exc

def psf_plots(irfname='P8R2_SOURCE_V6', energies = np.logspace(2,6,21)):
    import matplotlib.pyplot as plt
    fig, axx = plt.subplots(1,2, figsize=(12,5), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.05, wspace=0.075)
    for event_types, ax in zip('fb psf'.split(), axx):
        linestyles=': -. -- -'.split() if event_types=='psf' else ['-', '--']
        iman = IrfManager(None, irfname=irfname, event_types=event_types)
        ets = iman.event_types
        r68s = np.array([[iman.psf(et, energy).r68 for et in ets] for energy in energies])
        for et,ls in zip(ets, linestyles):
            ax.loglog(energies, r68s[:,et-ets[0]], 
                label='{}'.format(iman.event_type_names[et]), ls=ls, lw=2) 
        ax.set_xlabel('Energy (MeV)', fontsize=12)
        if event_types=='fb': ax.set_ylabel('R68 [deg]', fontsize=12)
        ax.grid()
        ax.legend();
    fig.suptitle('PSF R68 plots for IRF {}'.format(irfname),fontsize=14)
    fig.set_facecolor('white')
    return fig

def aeff_plots(irfname='P8R2_SOURCE_V6', x=np.logspace(2,6,21)):
    import matplotlib.pyplot as plt
    fig, axx = plt.subplots(1,2, figsize=(12,5), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.05, wspace=0.075)
    for event_types, ax in zip('fb psf'.split(), axx):
        linestyles=': -. -- -'.split() if event_types=='psf' else ['-', '--']
        irf = IrfManager(None, irfname=irfname, event_types=event_types)
        aeffs = irf._aeff.values() # the functions
        ets=irf.event_type_partitions[event_types]
        #if event_types=='psf': ets = np.flipud(ets) # make psf3 first
        y = np.array([map(aeff,x) for aeff in aeffs])
        for et,ls in zip(ets, linestyles):
            ax.semilogx(x,y[et-ets[0],:], label='{}'.format(irf.event_type_names[et])
                ,lw=2,ls=ls) 
        if event_types=='fb': ax.set_ylabel('Effective Area [cm^2]',fontsize=12)
        ax.set_xlabel('Energy (MeV)', fontsize=12)
        ax.grid(alpha=0.5)
        ax.legend();
    fig.suptitle('Effective Area plots for IRF {}'.format(irfname), fontsize=14)
    fig.set_facecolor('white')
    return fig

