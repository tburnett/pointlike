"""
Code that manages computation and saving tables of the weights for a source, for each band
"""

import sys, os, argparse, glob
import numpy as np
import pandas as pd
import healpy
import pickle
from uw.like2 import main
from  uw.utilities import keyword_options
from astropy.coordinates import SkyCoord

from skymaps import SkyDir, Band

class Weights(object):

    defaults=(
        ('verbose', 1,  'verbosity'),
        ('radius', 7.1, 'Radius to use'),
        ('nside' ,  64, 'nside'),
        ('energy_bins', np.logspace(2,6, 17), 'standard bins'),
        ('energy_count', 8, 'Total number of energy bins to actually use'),      
        ('min_dist',   0.1, 'minimum distance from catalog source to source in model'),
    )
    @keyword_options.decorate(defaults)
    def __init__(self, source_name, **kwargs):
        """
        """
        keyword_options.process(self,kwargs)
        if self.energy_count is not None:
            self.energy_bins = self.energy_bins[:self.energy_count]
        if type(source_name)==str:
            # first check to see if a local soruce
            print 'Searching for name {} ...'.format(source_name),
            fn = glob.glob('sources_*.csv')[0]
            df = pd.read_csv(fn, index_col=0)
            if source_name in df.index:
                print 'Found in model:'
                ra,dec = df.loc[source_name]['ra dec'.split()]
            else: #look up from SkyCoord
                t = SkyCoord.from_name(source_name); 
                ra,dec=( t.fk5.ra.value, t.fk5.dec.value) 
                print 'found by SkyCoord'
        else: # just a tuple
            ra,dec = source_name

        # select an ROI by the ra, dec specified or found from SkyCoord
        if self.verbose>0:
            print 'Selecting ROI at (ra,dec)=({:.2f},{:.2f})'.format(ra,dec)
        self.roi =roi =main.ROI('.', (ra,dec))

        # find the model source by position and select it
        distances = np.degrees(np.array([
            s.skydir.difference(SkyDir(ra,dec)) if s.skydir is not None else 1e9 for s in roi.sources]))
        min_dist = distances.min()
        assert min_dist<self.min_dist, 'Not within {} deg of any source in RoI'.format(self.min_dist)
        si = np.arange(len(distances))[distances==min_dist][0]
        sname = roi.sources.source_names[si] 
        self.source =roi.get_source(sname)
        if self.verbose>0:
            print 'Found model source "{}" within {:.3f} deg of "{}"'.format(self.source.name, min_dist, source_name)

    def make_weights(self, ):
        if self.verbose>0:
            print 'Generating nside={} pixels in radius {} for {} energies'.format(
                self.nside, self.radius, len(self.energy_bins))
        # direection of source

        source_dir = self.source.skydir
        source_index = self.roi.sources.selected_source_index

        def weight(band, skydir):
            f = band.fluxes(skydir) # fluxes for all sources at the position
            return f[source_index]/sum(f)

        # use query_disc to get NEST pixel numbers within given radius of position
        # order them for easier search later
        l,b = source_dir.l(), source_dir.b() 
        center = healpy.dir2vec(l,b, lonlat=True) 
        pix_nest = np.sort(healpy.query_disc(self.nside, center, 
                np.radians(self.radius), nest=True))  

        # convert to skydirs using pointlike code, which assumes RING
        bdir=Band(self.nside).dir
        pix_ring = healpy.nest2ring(self.nside, pix_nest)
        pixel_dirs = map(bdir, pix_ring)

        # and get distances
        pixel_dist = np.array(map(source_dir.difference, pixel_dirs))
        if self.verbose>0:
            print 'Using {} nside={} pixels.  distance range {:.2f} to {:.2f} deg'.format(
             len(pix_nest), self.nside, np.degrees(pixel_dist.min()), np.degrees(pixel_dist.max()))

        # now loop over the bands and all pixels
        wt_dict=dict()
        for band in self.roi: # loop over BandLike objects
            ie = np.searchsorted(self.energy_bins, band.band.energy)-1

            band_id = 2*ie+band.band.event_type
            if self.verbose>1:
                print '{}'.format(band_id),
            wt_dict[band_id] = np.array(
                [ weight(band, pd) for pd in pixel_dirs] ).astype(np.float32)
        if self.verbose>1: print 
        
        return  pix_nest, wt_dict

    def write(self, filename):

        pixels, weights = self.make_weights()
        # note: avoid pointlike objects like SkyDir to unpickle w/ python3
        galactic = lambda x: (x.l(),x.b())
        outdict = dict(
            model_name = '/'.join(os.getcwd().split('/')[-2:]),
            radius=self.radius,
            nside=self.nside,
            order='NEST',
            energy_bins=self.energy_bins,
            source_name= self.source.name,
            source_lb=galactic(self.source.skydir),
            roi_lb  = galactic(self.roi.roi_dir),
            roi_name=self.roi.name,
            pixels= pixels,
            weights = weights,
        )
        pickle.dump(outdict, open(filename, 'wb'))
        if self.verbose>0:
            print 'wrote file {}'.format(filename)

def runit(source_name, filename=None, overwrite=True, **kwargs):
    """ Use model in which this is run to create a weight file
    parameters:
        source_name 
        filename : name of pickled file | None
            if None, create file name
    """

    if filename is None or filename=='(none)':
        filename = 'weight_files/'+source_name.replace(' ','_').replace('+','p')+'_weights.pkl'
    if os.path.exists(filename) and not overwrite:
        print 'File {} exists, not replacing'.format(filename)
    else:
        sw = Weights(source_name, **kwargs)
        sw.write(filename)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Create file with weights for a source')
    parser.add_argument('source_name', nargs=1, help='name for a source')
    parser.add_argument('--file_name' , default=None,   
                        help='file to write to. Default: <source_name>_weights.pkl')
    args = parser.parse_args()

    runit(args.source_name[0], args.file_name)