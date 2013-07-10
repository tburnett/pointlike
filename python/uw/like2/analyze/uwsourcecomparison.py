"""
Comparison with another UW model

$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/analyze/uwsourcecomparison.py,v 1.2 2013/07/09 23:30:17 burnett Exp $

"""

import os
import numpy as np
import pylab as plt
import pandas as pd

from . import sourceinfo
from . diagnostics import FloatFormat

class UWsourceComparison(sourceinfo.SourceInfo):
    """Comparision with another UW model: %(othermodel)s
    <br>Ratios are %(skymodel)s/%(othermodel)s.
    
    """
    def setup(self, othermodel='uw25'):
        super(UWsourceComparison,self).setup()
        self.plotfolder = 'comparison_%s' % othermodel

        otherfilename = '../%s/sources.pickle' %othermodel
        self.othermodel=othermodel
        assert os.path.exists(otherfilename), 'File %s not found' % otherfilename
        print 'loading %s' % otherfilename
        odf = pd.load(otherfilename)
        self.odf = odf[odf.ts>10]
        self.df = self.df[self.df.ts>10]
        self.df['pindex_old']=self.odf.pindex
        self.df['ts_old'] = self.odf.ts
        self.df['eflux_old']=self.odf.eflux
        self.df['a_old'] = self.odf.a
        self.df['skydir_old'] = self.odf.skydir


    def check_missing(self):
        """Missing sources, and possible replacements
        The following table has source names from %(othermodel)s that do not appear in %(skymodel)s,
        with closest %(skymodel)s source name and distance to it.
        %(missing_html)s
        """
        oldindex = set(self.odf.index)
        newindex = set(self.df.index)
        missed = oldindex.difference(newindex)
        if len(missed)==0:
            self.missing_html='<br>No missing sources!'
            print 'no missing sources'
            return
        print '%d missing sources' %len(missed)
        self.missing = self.odf.ix[missed]
        t = map(self.find_nearest_to, self.missing['skydir'].values)
        self.missing['nearest']=[x[0] for x in t]
        self.missing['distance']=[x[1] for x in t]
        self.missing['nearest_ts']=self.df.ix[self.missing.nearest.values]['ts'].values
        
        self.missing_html=self.missing[self.missing.ts>10]['ts ra dec nearest nearest_ts distance roiname'.split()]\
            .sort_index(by='distance').to_html(float_format=FloatFormat(2))
        
    def check_moved(self, tol=2):
        """Sources in old and new lists, which apparently moved
        Criterion: moved by more than %(move_tolerance).1f sigma
        %(moved_html)s
        """
        self.move_tolerance=tol
        skydir = self.df.skydir.values
        skydir_old = self.df.skydir_old.values
        self.df['moved']=[np.degrees(a.difference(b)) if b is not np.nan else np.nan for a,b in zip(skydir, skydir_old)]
        self.moved_html = self.df[self.df.moved>tol*self.df.a]['ts_old ts ra dec a moved roiname'.split()]\
            .sort_index(by='moved').to_html(float_format=FloatFormat(2))

    
    def compare(self, scat=True):
        """Ratios of values of various fit parameters
        """
        odf, df = self.odf, self.df
        plane = np.abs(df.glat)<5
        
        def plot_ratio(ax, y, cut, ylim, ylabel):
            ts = df.ts
            if scat:
                ax.semilogx(ts[cut], y.clip(*ylim)[cut], '.')
                ax.semilogx(ts[cut*plane], y.clip(*ylim)[cut*plane], '+r',label='|b|<5')
                plt.setp(ax, xlim=(10,1e4), ylim=ylim, ylabel=ylabel,)
            else:
                bins = np.logspace(1,4,13)
                x = np.sqrt(bins[:-1]*bins[1:])
                for c, fmt,label,offset in zip([cut*(~plane), cut*plane], ('ob', 'Dr'), ('|b|>5', '|b|<5'),(0.95,1.05)):
                    t = ts[c]
                    u = y.clip(*ylim)[c]
                    ybinned = np.array([ u[ (t >= bins[i])*(t < bins[i+1])] for i in range(len(bins)-1)])
                    ymean = [t.mean() for t in ybinned]
                    yerr = [t.std()/np.sqrt(len(t)) if len(t)>1 else 0 for t in ybinned] 
                    ax.errorbar(x*offset, ymean, yerr=yerr, fmt=fmt, label=label)
                sy = lambda y: 1+(y-1)/4.
                plt.setp(ax, xlim=(10,1e4), ylim=(sy(ylim[0]),sy(ylim[1])), ylabel=ylabel, xscale='log')
            ax.axhline(1, color='k')
            ax.legend(prop=dict(size=10))
            ax.grid()
            yhigh = y[cut*(df.ts>200)]
            #print '%-6s %3d %5.3f +/- %5.3f ' % (ylabel, len(yhigh), yhigh.mean(),  np.sqrt(yhigh.std()/len(yhigh)))

        def plot_pindex(ax,  ylim=(0.9,1.1)):
            cut=df.beta<0.01
            y = df.pindex/df.pindex_old
            plot_ratio(ax, y, cut, ylim, 'index')
            ax.set_xlabel('TS')
        def plot_ts(ax, rdts=(0.5,1.5)):
            y = df.ts/(df.ts_old +0.1)
            cut=df.ts>10
            plot_ratio(ax, y, cut, rdts,  'TS')
        def plot_flux(ax, ylim=(0.5, 1.5)):
            y = df.eflux/df.eflux_old
            cut = df.ts>10
            plot_ratio(ax, y, cut, ylim, 'Eflux')
            
        def plot_semimajor(ax, ylim=(0.5,1.5)):
            y = df.a/df.a_old
            cut = df.ts>10
            plot_ratio(ax, y, cut, ylim, 'r95')
            
        fig, ax = plt.subplots(4,1, figsize=(10,12), sharex=True)
        plt.subplots_adjust(hspace=0.05, left=0.1, bottom=0.1)
        for f, ax in zip((plot_ts, plot_flux, plot_pindex,plot_semimajor,), ax.flatten()): f(ax)
        fig.text(0.5, 0.05, 'TS', ha='center')
        return fig
    
    def compare_profile(self):
        """Ratios of values of various fit parameters: profile plots
        Same as the first plot, but showing means and errors for bins in TS.
        <br>Changes below TS=25 are due to a threshold selection effect, a bias towared higher TS for the Clean, as can be seen
        in the TS scatter plot above.
        """
        return self.compare(False)
    
    def band_compare(self):
        """Band flux ratios
        For each of the 12 energy bands from 100 MeV to 100 GeV, plot ratio of fits for each source in common
        """
        fnow=self.df.sedrec[0].flux
        fold=self.odf.sedrec[0].flux
        self.df['sedrec_old'] = self.odf.sedrec
        fnew = np.array([s.flux for s in self.df.sedrec])
        fold = np.array([s.flux if type(s)!=float else [np.nan]*14 for s in self.df.sedrec_old])
        energy = np.logspace(2.125, 5.125, 13) # 4/decade
        
        fig, axx = plt.subplots(3,4, figsize=(14,12), sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.1, bottom=0.1)
        def plotone(ib, ax):
            ok = fold[:,ib]>1
            r =fnew[:,ib][ok]/fold[:,ib][ok]
            ax.plot(fold[:,ib][ok].clip(1,1e3), r, '.');
            plt.setp(ax, xscale='log', ylim=(0.5,1.5))
            ax.axhline(1.0, color='k')
            ax.text( 50, 1.4 ,'%d MeV'% energy[ib], fontsize=12)

        for ib,ax in enumerate(axx.flatten()): plotone(ib, ax)
        axx[0,0].set_xlim(1,1e3)
        fig.text(0.5, 0.05, 'Energy flux (eV/cm**2/s)', ha='center')
        
        return fig
        
    def quality_comparison(self):
        """FIt quality comparison
        Compare the spectral fit quality of the reference model with this one. All sources in common with TS>50, are shown.
        """
        fig, ax = plt.subplots(figsize=(5,5))
        lim=(0,30)
        cut = self.df.ts>50
        self.df['fitqual_old'] = self.odf.fitqual
        new, old = self.df.fitqual.clip(*lim)[cut], self.df.fitqual_old.clip(*lim)[cut]
        ax.plot(old, new ,  '.')
        ax.plot(lim, lim, color='r')
        plt.setp(ax, xlim=lim, ylim=lim, xlabel='old fit quality', ylabel='new fit quality')
        return fig
        
    def all_plots(self):
        self.runfigures([self.check_missing, self.check_moved, self.compare, self.compare_profile, self.band_compare, self.quality_comparison ])