"""
Plots involving the 1728 ROIs

$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/analyze/roi_info.py,v 1.12 2018/01/27 15:39:29 burnett Exp $

"""

import os, pickle
import numpy as np
import pylab as plt
import pandas as pd
from skymaps import SkyDir, Band

from . import analysis_base #diagnostics

class ROIinfo(analysis_base.AnalysisBase):
    """ROI-based plots
    <br> %(title)s diffuse component. These are based on the 
            individual ROI fits and examine only the normalization factor. See the spectral information, if present, for more
            information about the consistency of the model for this component.
        """
 
    def setup(self, **kwargs):
        self.plotfolder='rois'
        self.title='ROI summary'
        self.source_name='observed' # default for base class
        self.plots_kw={}
      
        filename = 'rois.pickle'
        pickle_file = 'pickle.zip'
        if not os.path.exists(pickle_file):
            pickle_file =   os.path.join(self.config['input_model']['path'],pickle_file)
        assert os.path.exists(pickle_file), 'No pickle file? {}'.format(pickle_file)
        refresh = kwargs.pop('refresh', not os.path.exists(filename) 
                    or os.path.getmtime(filename)<os.path.getmtime(pickle_file) )
        if refresh:
            files, pkls = self.load_pickles('pickle')
            assert len(files)==1728, 'Expected to find 1728 files, found %d' % len(files)
            rdict= dict()
            exclude = ('sources', 'name')
            for i,pkl in enumerate(pkls):
                skydir = Band(12).dir(i)
                tdict =dict((key,item) for key,item in pkl.items() if key not in exclude )
                tdict.update(sources = pkl['sources'].keys())
                glon = skydir.l() 
                if glon>180: glon-=360.
                glat = skydir.b(); 
                ra = skydir.ra()
                dec= skydir.dec()
                tdict.update(glon = glon, glat=glat, ra=ra, dec=dec )
                rdict[i] = tdict
            self.df = pd.DataFrame(rdict).transpose()
            self.df.to_pickle(filename)
            print 'saved %s' % filename
            # diffuse counts only as a dict
            self.save_diffuse_info()
        else:
            print 'loading %s' % filename
            self.df = pd.read_pickle(filename)
        # move this into refresh?
        rois = self.df
        rx = rois['ra dec glat glon'.split()].copy() 
        try:
            rx['chisq'] = [r['chisq'] for r in rois['counts']]
        except:
            print '***Failed to find counts, skip creating rois.csv'
            return
        rx['npar'] = [len(p) for p in rois.parameters]
        rx.index.name='name'
        ### ??
        #rx['ring'] = [10**p[0] for p in rois.parameters]
        #rx['iso']  = [10**p[1] for p in rois.parameters]
        rx.to_csv('rois.csv')
        print 'saved rois.csv'
        
        self.energy=self.df.ix[0]['counts']['energies']
        self.funcs = []
        self.fnames=[]
                
    def save_diffuse_info(self, outfile='diffuse_info.pkl'):
        """ save a file with counts and normalization per ROI
        """

        dfc = self.df['counts glat glon diffuse_normalization'.split()]
        dfc.to_pickle(outfile)

        print 'Saved file {}'.format(outfile)

    def save_counts(self, outfile='diffuse_counts.pkl'):
        """ save a file with the predicted counts per ROI
        """
        # ff, pp = self.load_pickles()

        # gal_counts = np.array([ pkl['counts']['models'][:2][0][1][:8] for pkl in pp])
        # iso_counts = np.array([ pkl['counts']['models'][:2][1][1][:8] for pkl in pp])
        gal_counts = np.array( [c['models'][0][1][:8] for c in self.df.counts])
        iso_counts = np.array( [c['models'][1][1][:8] for c in self.df.counts])
        with open(outfile, 'wb')  as out:
            pickle.dump( dict(gal=gal_counts, iso= iso_counts), out)

        print 'Saved file {}'.format(outfile)
        
    def default_plots(self):
        # the set of defaults plots to generate: subclasses can add
        self.funcs = [self.counts_map, self.normalization, self.norm_unc, self.norm_vs_dec]
        self.fnames= map(lambda s: self.source_name+'_'+s, ['counts', 'normalization', 'norm_unc',  'norm_vs_dec'])

        
    def skyplot(self, values, ax=None, title='', ecliptic=False,
                    labels=True, colorbar=True, cbtext='', **scatter_kw):
        if ax is None:
            # note that different size might mean different marker size
            fig, ax = plt.subplots( 1,1, figsize=(6,5))
        else: fig = ax.figure
        scat_default = dict(s=60, marker='D', vmin=None, vmax=None, edgecolor='none')
        scat_default.update(scatter_kw)
        
        if ecliptic:
            lon, sinlat = self.ecliptic_coords()
        else:
            lon, sinlat = self.df.glon, np.sin(np.radians(list(self.df.glat)))
        scat =self.basic_skyplot(ax, lon, sinlat, c=values,
            labels=labels, colorbar=colorbar, cbtext=cbtext, **scat_default)
        
        return fig, scat # so can add colorbar later

    def model_counts(self, name, ib=0):
        """ 
        return an array, in ROI order, of the counts corresponding to the name
            name: string
                either 'observed', the name of a diffuse model, or 'sources' for the total
            ib : int or None
                the energy bin number, or return a sum if None
        """
        def select_band(x):
            return x[ib] if ib is not None else x.sum()
        if name=='observed':
            return np.array([ select_band(self.df.ix[i]['counts']['observed']) for i in range(1728)])
        def source_counts(i):
            m = self.df.ix[i]['counts']['models']
            dn = self.df.ix[i]['diffuse_names']
            r=0
            for nm, cnts in m:
                if nm in dn: continue
                r+= select_band(cnts)
            return r 
        if name=='sources':
            return np.array(map(source_counts, range(1728)))
            
        def cts(i):
            m = self.df.ix[i]['counts']['models']
            dn = self.df.ix[i]['diffuse_names']
            k = dn.index(name) if name in dn else -1
            return select_band(m[k][1]) if k>=0 else 0
        return np.array(map(cts, range(len(self.df))))

    def diffuse_models(self,  name):
        """ return list of referernces to the diffuse spectral models
        """
        def mdl(index):
            pkl = self.df.ix[index]
            m =pkl['diffuse_names']
            if name not in m: return None
            return pkl['diffuse'][m.index(name)]
        return map(mdl, range(1728))

    def ecliptic_coords(self):
        """return cartesian arrays for roi positions in ecliptic coords"""
        enp=SkyDir(270,90-23.439281) #ecliptic north pole
        gdir = [SkyDir(l,b, SkyDir.GALACTIC) for l,b in zip(self.df.glon, self.df.glat)]
        edir = np.array([ g.zenithCoords(enp) for g in gdir]); edir[0]
        sinlat = np.sin(np.radians(edir[:,1]))
        lon = edir[:,0]
        lon[lon>180] -= 360
        return lon, sinlat

    def counts_map(self,  hsize= (1.0, 1.7, 1.0, 1.4),  **kwargs):
        """ counts map for %(title)s
        Left:  each ROI, the total counts corresponding to the %(title)s component, 
        for %(energy_selection)s MeV. There are %(total_counts)s total.
        <br>Right: the fraction, %(total_fraction)s percent of the total.
        """
        ib = kwargs.pop('ib', None)
        if 'cb_kw' not in kwargs:
            kwargs['cb_kw'] = dict(shrink=0.70,) #### something changed? anchor=(-1.0,0.5)) #ugly tweaking
        self.energy_selection= 'E=%.0f' %self.energy[ib] if ib is not None else 'E>100'
        sm = self.model_counts(self.source_name, ib)
        tot = self.model_counts('observed', ib)
        self.total_counts ='{:,d}'.format(int(np.sum(sm)))
        self.total_fraction = '%.1f' % (np.sum(sm)/np.sum(tot) * 100.)
        
        fig, ax = self.subplot_array(hsize, figsize=(12,6))
        def left(ax):
            if kwargs.pop('log', True):
                c,cbtext = np.log10(sm), 'log10(counts)'
            else: 
                c,cbtext = sm, 'counts'
            if ib is not None:
                return self.skyplot(c, ax=ax, cbtext=cbtext,  **Kwargs)
                 #   title='%s counts at %d MeV' % ( title, self.energy[ib]), **kwargs)
            else:
                return self.skyplot(c, ax=ax, cbtext=cbtext, **kwargs) #title=title,  **kwargs)
        def right(ax):
            if ib is not None:
                return self.skyplot(100*(sm/tot), ax=ax, cbtext='fraction (%)', **kwargs) #title='%s count fraction at %d MeV' % (title, self.energy[ib]),
            else:
                return self.skyplot(100*(sm/tot), ax=ax, cbtext='fraction (%)', **kwargs) #title=title,  cbtext='fraction (%)', **kwargs)
        for f, ax in zip((left,right), ax.flatten()): f(ax)
        return fig
            
#    def count_fraction(self,  title='', **kwargs):
#        """ Count Fraction for %(title)s
#        For each ROI, the fraction of %(title)s counts, 
#        for %(energy_selection)s MeV.
#        """
#        ib = kwargs.pop('ib', None)
#        self.energy_selection= 'E=%.0f' %self.energy[ib] if ib is not None else 'E>100'
#
#        sm = self.model_counts(self.source_name, ib)
#        tot = self.model_counts('observed', ib)
#        if ib is not None:
#            return self.skyplot(100*(sm/tot), title='%s count fraction at %d MeV' % (title, self.energy[ib]),
#                cbtext='fraction (%)', **kwargs)
#        else:
#            return self.skyplot(100*(sm/tot), title=title,  cbtext='fraction (%)', **kwargs)

    def skyplot_with_hist(self, values, xlabel, vmin, vmax, clip,  **kw):
    
        fig, ax = self.subplot_array( (1.0, 0.6, 1.5, 0.7), figsize=(10,5))
        stats = kw.pop('stats', True)
        def hist(ax):
            t = 'count %d\nmean  %.2f\nstd   %.2f'  %(len(values),values.mean(),values.std())
            ax.hist(values.clip(*clip), np.linspace(clip[0],clip[1], 51), label=t)
            ax.grid()
            ax.axvline(1.0, color='k')
            plt.setp(ax, xlabel=xlabel, xlim=clip)
            ax.legend( prop=dict(size=10))
                
        hist(ax[0,0])
        self.skyplot(values, ax=ax[0,1], vmin=vmin, vmax=vmax,  **kw)
        return fig

    
    def normalization(self, vmin=0.8, vmax=1.2, clip =(0.5,1.5), **kw):
        """ normalization factor for %(title)s
        The normalization should be nominally 1.0.
        Left: histogram: right: map
        """ 
        models = self.diffuse_models(self.source_name)
        norms = np.array([m.getp(0) if m is not None else np.nan for m in models])
        return self.skyplot_with_hist(norms, 'normalization', vmin, vmax, clip, **kw)
        
    def norm_unc(self, vmin=0, vmax=0.05, clip =(0, 0.05), **kw):
        """ normalization uncertainty for %(title)s
        The normalization should be nominally 1.0.
        Left: histogram: right: map
        """ 
        models = self.diffuse_models(self.source_name)
        unc = np.array([np.sqrt(m.get_cov_matrix()[0,0]) if m is not None else np.nan for m in models])
        return self.skyplot_with_hist(unc, 'normalization uncertainty', vmin, vmax, clip, **kw)

    def norm_vs_dec(self, vmin=0, vmax=90, size=15, ylim=(0.7,1.2), **kw):
        """ Normalization factor for %(title)s vs Dec
        The color represents the absolute Galactic latitude
        """
        models = self.diffuse_models(self.source_name)
        norms = [m.getp(0) if m is not None else np.nan for m in models]
        sindec = np.sin(np.radians(np.array(self.df.dec,float)))

        fig,ax = plt.subplots(figsize=(6,5))
        c=np.abs(self.df.glat.values)
        cut= c>vmin
        defaults =dict(edgecolors='none', s=size)
        scat=ax.scatter( sindec, norms, c=c, vmin=vmin, vmax=vmax, **defaults)
        plt.setp(ax, xlim=(-1,1), ylim=ylim, xlabel='sin(dec)', ylabel='normalization')
        ax.grid()
        cb =fig.colorbar(scat, ax=ax)
        cb.set_label('abs(b)')
        return fig
        
    def all_plots(self): #, other_html=None):
        self.runfigures(self.funcs, self.fnames, **self.plots_kw)
        
    def counts_dataframe(self, roi_index):
        """ return a dataframe with columns for the diffuse sources, free and fixed point sources,
            total, observed, and pull
        """
        c = self.df.ix[roi_index]['counts']
        cp = dict()
        scols = []
        for n,d in c['models']:
            scols.append(n)
            cp[n]=np.array(d).round(1)
        cols = 'total observed'.split()
        for n in cols:
            cp[n] = np.array(c[n]).round(1)
        df= pd.DataFrame(cp, index=np.array(c['energies'],int), columns=scols+cols)
        df['pull'] = ((df.observed-df.total)/np.sqrt(df.total)).round(1)
        return df