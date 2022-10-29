"""
Code that manages computation and saving tables of the weights for a source, for each band
"""

from __future__ import print_function
import sys, os, argparse, glob
import numpy as np
import pandas as pd
import healpy
import pickle
from uw.like2 import main, sources
from  uw.utilities import keyword_options
from astropy.coordinates import SkyCoord

from skymaps import SkyDir, Band
from uw.like2.tools import parse_jname, create_jname



class Weights(object):

    defaults=(
        ('verbose', 1,  'verbosity'),
        ('radius', 5.1, 'Radius to use'),
        ('nside' ,  64, 'nside'),
        ('energy_bins', np.logspace(2,6, 17), 'standard bins'),
        ('energy_count', 8, 'Total number of energy bins to actually use'),      
        ('min_dist',   0.1, 'minimum distance from catalog source to source in model'),
        ('new_format', True, 'newer format'),
        ('psf_tol',    2.0, 'factor for r68'),
        ('max_nside',  512, 'maximum nside'),
    )
    @keyword_options.decorate(defaults)
    def __init__(self, source_name, **kwargs):
        """
        source_name: 
        """
        self.roi = kwargs.pop('roi', None)

        keyword_options.process(self,kwargs)
        if self.energy_count is not None:
            self.energy_bins = self.energy_bins[:self.energy_count]
        if self.roi is None:
            if type(source_name)==str:
                if source_name.startswith('J'):
                    # a J-name -- parse it for (ra,dec)
                    ra, dec = parse_jname(source_name)
                else:
                    # first check to see if a local soruce
                    print( 'Searching for name "{}" ...'.format(source_name), end='')
                    fn = glob.glob('sources_*.csv')[0]
                    df = pd.read_csv(fn, index_col=0)
                    if source_name in df.index:
                        print( 'Found in model:')
                        ra,dec = df.loc[source_name]['ra dec'.split()]
                    else: #look up from SkyCoord
                        t = SkyCoord.from_name(source_name); 
                        ra,dec=( t.fk5.ra.value, t.fk5.dec.value) 
                        print( 'found by SkyCoord')
            else: # just a tuple
                ra,dec = source_name
        

            # select an ROI by the ra, dec specified or found from SkyCoord
            if self.verbose>0:
                print( 'Selecting ROI at (ra,dec)=({:.2f},{:.2f})'.format(ra,dec))
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
                print( 'Found model source "{}" within {:.3f} deg of "{}"'.format(self.source.name, min_dist, source_name))
        else:
            self.roi.get_source(source_name)
        self.source_name = source_name



    # def get_weights(self, band, skydir):
    #     f = band.fluxes(skydir) # fluxes for all sources at the position
    #     source_index = self.roi.sources.selected_source_index
    #     sig, tot = f[source_index], sum(f)
    #     return (sig,tot) 

    def make_weights(self, ):
        if self.verbose>0:
            print( 'Generating nside={} pixels in radius {} for {} energies'.format(
                self.nside, self.radius, len(self.energy_bins)) )
        
        
        # direection of source
        source_dir = self.source.skydir
        source_index = self.roi.sources.selected_source_index

        def weight(band, skydir):
            f = band.fluxes(skydir) # fluxes for all sources at the position
            sig, tot = f[source_index], sum(f)
            return sig/tot

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
            print( 'Using {} nside={} pixels.  distance range {:.2f} to {:.2f} deg'.format(
             len(pix_nest), self.nside, np.degrees(pixel_dist.min()), np.degrees(pixel_dist.max())))

        # now loop over the bands and all pixels
        wt_dict=dict()
        for band in self.roi: # loop over BandLike objects
            ie = np.searchsorted(self.energy_bins, band.band.energy)-1
            #if ie>self.energy_count: break  

            band_id = 2*ie+band.band.event_type
            if self.verbose>1:
                print( '{}'.format(band_id), end='')

            wt_dict[band_id] = np.array(
                [ np.atleast_1d(weight(band, pd))[0] for pd in pixel_dirs] ).astype(np.float16)

        if self.verbose>1: print() 
        
        return  pix_nest, wt_dict

    def write(self, filename=None):
        if filename is None:
            filename=self.source_name.replace(' ','_').replace('+','p')

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
            print( 'wrote file {}'.format(filename))

class NewWeights(Weights):
    """Sub class of Weights that implements variable pixel size for 
    weight pixels 
    
    """
        
    def make_weights(self, energy_bins=np.logspace(2,6, 17), test=False):
        
        source_index = self.roi.sources.selected_source_index
        src_model = self.roi.get_source().model
   
        def band_weights(bnd, center,  test=test):
            # weights and pixel ids

            def weight(skydir):
                f = bnd.fluxes(skydir) # fluxes for all sources at the position
                sig, tot = f[source_index], sum(f)
                return (sig/tot)

            # get sorted lost of nest pixels out to e radius given by PSF
            nside = min(self.max_nside, bnd.band.cband.nside())
            psf = bnd.band.psf
            radius = np.float32(max(psf.r68, 0.2) * self.psf_tol)
            pixels = np.sort(healpy.query_disc(nside, center, 
                            np.radians(radius), nest=True))  

            # convert to skydirs using pointlike code, which assumes RING
            bdir=Band(nside).dir
            pixel_dirs = map(bdir, healpy.nest2ring(nside, pixels) )
            # npred, flux for the source
            npred = bnd[source_index].counts
            energy = bnd.band.energy
            flux  = src_model(energy)
         
            # evaluate the weights for these directions
            # now using float16!
            wts = np.array([np.atleast_1d(weight(pd))[0] for pd in pixel_dirs], np.float16)


            return dict(nside=nside, 
                radius=np.float32(radius), npred=np.float32(npred), flux=np.float32(flux),
                pixels=pixels, wts=wts, )

    
        roi=self.roi
        galactic = lambda x: (np.float32(x.l()), np.float32(x.b()))
        source = roi.sources.selected_source
        roi_lb = galactic(roi.roi_dir)
        l,b = source_lb = galactic(source.skydir)
        center = healpy.dir2vec(l,b, lonlat=True)
        #print( 'ROI, source dir', roi_lb, source_lb )

        # create the weight dict
        wt_dict=dict()

        for band in roi: # loop over BandLike objects
            ie = np.searchsorted(roi.energies*1.05, band.band.energy)
            if ie > self.energy_count: break

            band_id = 2*ie+band.band.event_type
            wt_dict[band_id] = band_weights(band, center)
            if self.verbose>0: print( '.', end='')

        
        #  put it into the final dict
        model = source.model

        self.dct = dict(model_name = '/'.join(os.getcwd().split('/')[-2:]),
                  energy_bins=energy_bins,
                  source_name=self.source_name,
                  fitinfo = dict(modelname = model.name,
                            pars = model.get_all_parameters(),
                            errs = model.get_free_errors(),
                            ts   = round(source.ts,1),
                    ),
                  nickname=source.name,
                  roi_lb = roi_lb,
                  source_lb = source_lb, 
                  wt_dict=wt_dict,
                  )    
        if self.verbose: print()
        return self.dct
    
    def write(self, filename=None, overwrite=True):
        import pickle

        if not hasattr(self, 'dct'):
            self.make_weights()
        default_filename =self.source_name.replace(' ','_').replace('+','p')+'_weights.pkl'
        if filename is None or filename=='(none)':
            filename= 'weight_files/'+default_filename    
        elif os.path.isdir(os.path.expanduser(filename)):
            filename = filename+'/'+default_filename
        if os.path.exists(filename) and not overwrite:
            print( 'File {} exists, not replacing'.format(filename) )
            return

        with open(filename, 'wb') as out:
            pickle.dump(self.dct, out )
        if self.verbose>0:
            print( 'wrote file {}'.format(filename) )

    def plot_weight_radius(self, **kwargs):
        """diagnostic plots of weights vs. distance from source
        """
        import matplotlib.pyplot as plt
        if not hasattr(self, 'dct'):
            self.make_weights()

        wt_dict, (l,b) = [self.dct[k] for k in 'wt_dict source_lb'.split()]

        center =healpy.dir2vec(l,b, lonlat=True)

        fig, axx = plt.subplots(2,8, figsize=(15,4), sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0, hspace=0)

        for id, ax in enumerate(axx.flatten()):
            if id not in wt_dict: continue

            pixels, weights, nside = [wt_dict[id][k] for k in 'pixels wts nside'.split()]
            ll,bb = healpy.pix2ang(nside, pixels,  nest=True, lonlat=True)
            cart = lambda l,b: healpy.dir2vec(l,b, lonlat=True)
            radius = np.degrees(np.array(np.sqrt((1.-np.dot(center, cart(ll,bb)))*2), np.float32))

            ax.plot(radius, weights, '.', label='{}'.format(id), color='green' if id%2==0 else 'orange')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(alpha=0.5)

        kw = dict(yscale='log', xlim=(0,4.9), ylim=(4e-4,0.9)) 
        kw.update(kwargs)
        ax.set(**kw)  

        fig.set_facecolor('white');
        fig.suptitle('{}: Weight vs. radius per band'.format(self.source_name), fontsize=14)



def runit(source_name, filename=None, overwrite=True, **kwargs):
    """ Use model in which this is run to create a weight file
    parameters:
        source_name 
        filename : name of pickled file | None | directory
            if None, create file name
            if directory, save filename to it
    """

    sw = NewWeights(source_name, **kwargs)
    sw.write(filename, overwrite)


def multiweights(roi, file_path):
    """Generate weights for all free point souurces in the ROI
    """
    for source in roi.free_sources:

        if not isinstance(source, sources.PointSource): continue
        sname = source.name
        sdir = source.skydir
        jname = create_jname(sdir.ra(), sdir.dec())
        print(sname, '-->', jname, end='')
        wts = NewWeights(sname, roi=roi)
        #wts.source_name = jname
        wts.write(file_path)

    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Create file with weights for a source')
    parser.add_argument('source_name', nargs=1, help='name for a source')
    parser.add_argument('--file_name' , default=None,   
                        help='file to write to. Default: <source_name>_weights.pkl')
    #parser.add_argument('min_dist',  default=0.1, help='minimum dist for detection (deg)')
    args = parser.parse_args()

    runit(args.source_name[0], args.file_name) #, args.min_dist)