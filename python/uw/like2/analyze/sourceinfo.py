"""
Basic analyis of source spectra

$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/analyze/sourceinfo.py,v 1.33 2018/01/27 15:39:29 burnett Exp $

"""

import os
from collections import OrderedDict

import astropy.io.fits as pyfits
import cPickle as pickle
from collections import Counter
import numpy as np
import pylab as plt
import matplotlib.ticker as ticker
import pandas as pd

from uw.utilities import makepivot
from . import analysis_base, _html, fitquality
from .. import extended, configuration
from ..tools import create_jname, decorate_with 
from analysis_base import html_table, FloatFormat
from skymaps import SkyDir, Band

class SourceInfo(analysis_base.AnalysisBase): #diagnostics.Diagnostics):
    """Source spectral properties 
    <br>See <a href="../localization/index.html?skipDecoration"> localization </a> for localization plots.
    <br> Link to a csv file containing a subset of the info used here:
        <a href="../../%(csvfile)s?download=true">%(csvfile)s</a>
    
    """
    require='pickle.zip'

    def setup(self, **kwargs):
        version = os.path.split(os.getcwd())[-1]
        self.csvfile='sources_%s.csv' % version        
        self.plotfolder='sources' #needed by superclass
        filename = 'sources.pickle'
        self.quiet = kwargs.pop('quiet', True)
        cat=kwargs.get('cat', '3FGL')
        refresh = kwargs.pop('refresh', not os.path.exists(filename) or os.path.getmtime(filename)<os.path.getmtime('pickle.zip'))
        if refresh:
            files, pkls = self.load_pickles('pickle')
            assert len(files)==1728, 'Expected to find 1728 files'
            self.pkls = pkls # for debugging
            sdict= dict()
            # if cat=='3FGL':
            #     try:
            #         get_cat3fgl = Cat_3fgl()
            #         print 'loaded 3FGL'
            #     except Exception, msg:
            #         print 'Could not load 3FGL: %s' % msg
            #         get_cat3fgl=None
            # else: 
            #     print 'Not adding 3FGL equivalence'
            #     get_cat3fgl=None
                
            for pkl in pkls:
                roidir = pkl['skydir']
                for name, info in pkl['sources'].items():
                    model = info['model']
                    pars = np.empty(4); pars.fill(np.nan)
                    errs = np.empty(4); errs.fill(-2)
                    free = np.zeros(4, bool)
                    n = model.len()
                    pars[:n] = model.parameters
                    free[:len(model.free)] = model.free
                    try:
                        d = np.diag(model.get_cov_matrix()).copy()
                        d[d<0] =0
                        errs[:n] = np.sqrt(d)
                        errs[np.isnan(errs)]=-1
                        badfit = np.any(errs[model.free]<=0)
                        freebits= np.sum( int(b)*2**i for i,b in enumerate(model.free))
                        badbits = freebits &  np.sum( int(b)*2**i for i,b in enumerate(errs==0))
                        
                    except Exception, msg:
                        raise exception( 'fail errors for %s:%s' % (name, msg))
                        badfit = True
                        
                    try:
                        # should be fixed by setting emax for simulation?
                        dts = info['sedrec'].delta_ts
                        dts[np.isnan(dts)]=0
                        fitqual = round(sum(info['sedrec'].delta_ts),2)
                    except:
                        fitqual = np.nan
                    try:
                        fitndf  = sum(1-info['sedrec'].zero_fract)
                    except:
                        fitndf = 10
                    try:
                        npred = round(sum(info['sedrec'].npred),1)
                    except:
                        npred=np.nan
                    try:
                        ts_flat=info['sedrec'].ts_flat
                    except:
                        ts_flat= None
                    ellipse = info.get('ellipse', None)
                    moment  = info.get('moment', None)
                    has_moemnt = moment is not None
                    sdict[name] = info
                    pulsar = model.name.endswith('Cutoff')
                    betavalue = float(pars[2]) if not pulsar else np.nan
                    if pulsar: # factor to convert flux to prefactor
                        bvalue =1.0 if model.npar==3 else model['b']
                        prefactor = np.exp(-(model.e0/model['cutoff'])**bvalue)
                    else: prefactor = 1.0
                    sdict[name].update(
                        glat=info['skydir'].b(), glon=info['skydir'].l(),
                        npred=npred,

                        roiname=pkl['name'], 
                        pars= pars, errs=errs, free=free, badfit=badfit,
                        a = ellipse[2] if ellipse is not None else np.nan,
                        b = ellipse[3] if ellipse is not None else np.nan,
                        ang=ellipse[4] if ellipse is not None else np.nan,
                        locqual = round(ellipse[5],2) if ellipse is not None else np.nan,
                        delta_ts = ellipse[6] if ellipse is not None else np.nan,
                        ts_beta = info.get('ts_beta', None ),
                        moment=moment,
                        freebits= freebits,
                        badbits=badbits,
                        flux = prefactor * pars[0],
                        flux_unc = prefactor * errs[0],
                        pindex = pars[1],
                        pindex_unc = errs[1],
                        beta = betavalue,
                        beta_unc = errs[2] if not pulsar and pars[2]>0.002 else np.nan,
                        index2 = pars[3] if pulsar else pars[2],
                        index2_unc = errs[3] if pulsar and not np.isnan(pars[3]) else errs[2],
                        cutoff = pars[2] if pulsar else np.nan,
                        cutoff_unc = errs[2] if pulsar else np.nan,
                        e0 = model.e0,
                        modelname=model.name,
                        fitqual = fitqual,
                        fitndf  = fitndf,
                        eflux = prefactor*pars[0]*model.e0**2*1e6,
                        eflux_unc=prefactor*errs[0]*model.e0**2*1e6,
                        eflux100 = info.get('eflux', (np.nan,np.nan))[0],
                        eflux100_unc = info.get('eflux', (np.nan,np.nan))[1],
                        psr = name.startswith('PSR'),
                        profile=info.get('profile', None),
                        # cat3fgl = None if get_cat3fgl is None else get_cat3fgl(name),
                        transient= not info.get('fixed_spectrum', False) and not info['isextended'],
                        roi_dist= np.degrees(info['skydir'].difference(roidir)),
                        ts_flat = ts_flat,
                        )
            df = pd.DataFrame(sdict).transpose()
            assert len(df)>0, 'No sources found!'
            df.index.name='name'
            df['name'] = df.index
            ra = [x.ra() for x in df.skydir]
            dec = [x.dec() for x in df.skydir]
            df['ra'] = ra
            df['dec'] = dec
            df['jname']= map(create_jname,ra,dec)
            df.loc[df.isextended,'jname']=df.isextended[df.isextended].index

            # Set up some association summary: highest probability and its catalog name
            probfun = lambda x: x['prob'][0] if not pd.isnull(x) else 0
            df['aprob'] = np.array(map(probfun, df.associations))
            catfun =  lambda x: x['cat'][0] if not pd.isnull(x) else ''
            df['acat'] = np.array(map(catfun, df.associations))
            
            self.df = df.sort_values(by='ra')
            self.curvature(setup=True) # add curvature item
            self.df.to_pickle(filename)
            assert 'aprob' in df.columns

            if not self.quiet:
                print 'saved %s' % filename

        else:
            if not self.quiet: print 'loading %s' % filename
            self.df = pd.read_pickle(filename)
        #self.df['flux']    = [v[0] for v in self.df.pars.values]
        #self.df['flux_unc']= [v[0] for v in self.df.errs.values]
        localized = ~np.array(pd.isnull(self.df.delta_ts))
        extended = np.array(self.df.isextended, bool)
        self.df['unloc'] = ~(localized | extended)
        self.df['poorloc'] = (self.df.a>0.2) | (self.df.locqual>8) | (self.df.delta_ts>2)
        self.df['flags'] = 0  #used to set bits below
        pl = (self.df.poorloc | self.df.unloc) & (self.df.ts>10)
        flags = self.df.flags.values; flags[self.df.poorloc] +=8
        self.df.loc[:,'flags'] = flags ### bit 8 (avoid warning? MP)
        #print '%d sources flagged (8) as poorly or not localized' % sum(pl)


        sr = self.df.iloc[0]['sedrec'] 
        if sr is None: sr = self.df.iloc[1]['sedrec'] 
        if sr is None:
            self.energy = np.logspace(2.125, 5.875,16)
            print 'Warning. did not find a sedrec'
        else:
            self.energy = np.sqrt( sr.elow * sr.ehigh )
            
    def skyplot(self, values, proj=None, ax=None, ecliptic=False, df=None,
                labels=True, title='', colorbar=True, cbtext='', **scatter_kw):
        """ 
        Make a sky plot of some quantity for a selected set of sources
        Parameters:
        ----------
            values: a DataFrame column, posibly a subset: 
                expect to have source name index to get position
            proj: None, or a function to map values to colors
            s : float
                size of dot to plot
                
            df : None | DataFrame
                if None, use self.df to look up glat and glon
        """
        assert hasattr(values, 'index'), 'skyplot: values arg must have index attribute'
        
        if df is None:
            df = self.df
        assert len(set(values.index).intersection(df.index))==len(values), 'skyplot: index values unknown'
        # generate arrays of glon and singlat using index 
        sd = df.loc[values.index, ['glat', 'glon']] # see page 101
        glon = sd.glon
        glon[glon>180]-=360
        singlat = np.sin(np.radians(list(sd.glat)))

        c = values if proj is None else map(proj, values)

        if ax is None:
            fig, ax = plt.subplots(figsize = (6,5))
        else: fig = ax.figure
        
        scatter_kw_default=dict(s=20, vmin=None, vmax=None, edgecolor='none')
        scatter_kw_default.update(scatter_kw)
        
        scat = self.basic_skyplot(ax, glon, singlat, c=c,
                title=title, ecliptic=ecliptic, colorbar=colorbar,cbtext=cbtext, **scatter_kw_default)
        return fig
        
    def fluxinfo(self, ib=0, cut=None):
        """ extract flux info for energy bin ib, return as a DataFrame
        """
        if cut is None: cut=self.df.ts>25
        s = self.df[cut]
        energy = self.energy[ib]
        fdata = np.array([s.loc[i]['sedrec'].flux[0] for i in range(len(s))])
        udata = np.array([s.loc[i]['sedrec'].uflux[0] for i in range(len(s))])
        ldata = np.array([s.loc[i]['sedrec'].lflux[0] for i in range(len(s))])
        fmodel = np.array([s.loc[i]['model'](energy)*energy**2*1e6 for i in range(len(s))])
        return pd.DataFrame(dict(fdata=fdata, udata=udata, ldata=ldata, fmodel=fmodel, 
                glat=s.glat, glon=s.glon, roiname=s.roiname),
            index=s.index).sort_values(by='roiname')

    def cumulative_ts(self, latcut=10, loncut=90, dlim=900 ):
        """ Cumulative test statistic TS
        
        Plots of "logN-logS" for high and low latitudes, but using TS for source intensity. 
        Important thresholds at TS=10 and 25 are shown.
        The lower plots show the difference between the cumulative counts, and an expected distribution
        with a -3/2 slope.
        """
        df = self.df
        hilat = np.abs(df.glat)>latcut
        glon = df.glon.values
        glon[glon>180]-=360
        innergal = (np.abs(df.glat)<latcut) & (abs(glon)<loncut)

        fig, axx= plt.subplots(2,2, figsize=(12,6), sharex=False, gridspec_kw={'hspace':0.})
        fig.set_facecolor('white')
        axes=axx.flatten()
        axes[0].tick_params(labelbottom=False)
        left, bottom, width, height = (0.10, 0.10, 0.40, 0.85)
        fraction = 0.75
        axes[0].set_position([left, bottom+(1-fraction)*height, width, fraction*height])
        axes[1].set_position([left, bottom, width, (1-fraction)*height])
        axes[2].set_position([left+0.5, bottom+(1-fraction)*height, width, fraction*height])
        axes[3].set_position([left+0.5, bottom, width, (1-fraction)*height])

        self.cum_ts(ts=df.ts[~innergal], axes=axes[:2], title=r'Outer galaxy or $|b|>{}^\circ $'.format(latcut),dlim=dlim)
        self.cum_ts(ts=df.ts[innergal],  axes=axes[2:], title=r'Inner galaxy: $|b|<{}^\circ \ & \ |l|<{}^\circ $'.format(latcut, loncut),dlim=dlim)
        return fig

    def cum_ts(self, ts=None, axes=None, tscut=(10,25, 300),  legend=True, title='', dlim=520):
        """ Cumulative test statistic TS
        
        A logN-logS plot, but using TS. Important thresholds at TS=10 and 25 are shown.
        The lower plot shows the difference between the cumulative counts, and expected distribution
        with a -3/2 slope.
        """
        usets = self.df.ts if ts is None else ts
        df = self.df

        if axes is None:
            # set up axes, fig
            fig, axes= plt.subplots(2,1, figsize=(6,6), sharex=True, gridspec_kw={'hspace':0.})
            axes[0].tick_params(labelbottom=False)
            left, bottom, width, height = (0.15, 0.10, 0.75, 0.85)
            fraction = 0.75
            axes[0].set_position([left, bottom+(1-fraction)*height, width, fraction*height])
            axes[1].set_position([left, bottom, width, (1-fraction)*height])
        else:
            fig = axes[0].figure
        
        ax=axes[0]
        dom = np.logspace(np.log10(9),5,1601)
        ax.axvline(25, color='green', lw=1, ls='--',)#label='TS=25')
        hist_kw=dict(cumulative=-1, lw=2,  histtype='step')
        ht=ax.hist( np.array(usets,float) ,dom, color='k',   **hist_kw)
        # add logN-logS line with slope -2/3
        anchor=300 #100 
        y = sum(usets>anchor)
        ymin = sum(usets>1e3)
        b=-2/3.
        a = np.log(y) - b*np.log(anchor)
        popf = lambda x: np.exp(a + b*np.log(x))
        ax.plot(dom, popf(dom), '--r', label='-3/2 slope');
        
        ax.set( ylabel='# sources with greater TS', xlabel='TS',
            xscale='log', yscale='log', xlim=(9, 1e3), ylim=(ymin,None), title=title) #ylim=(200,20000))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda val,pos: { 10.0:'10', 100.:'100'}.get(val,'')))
            
        # label the plot with number at given TS
        for t in tscut:
            n = sum(usets>t) 
            f,ha = (2,'left') if t<300 else (0.5,'right')
            ax.plot([t,f*t], [n,n], '-k');
            ax.plot(t, n, 'og')
            ax.text(f*t, n, 'TS>%d: %d'%(t,n), fontsize=14, va='center', ha=ha)
                
        ax.grid(True, alpha=0.3)
        if  legend: 
            leg =ax.legend()
            for patch in leg.get_patches():
                pbox = patch; pbox._height=0; pbox._y=5

        # stack over difference from the logN-logS
        ax = axes[1]
        dom2 = dom[1:]
        ax.plot(dom2, ht[0] -popf(dom2) ,  lw=2,  )
        n25 = sum(usets>25)
        n = n25-popf(25)
        dn = int(100*n/float(n25))
        ax.plot([25,50], [n/2,-100], ':r')
        ax.plot([25,25], [n, 0], '-r', lw=4)
        txt = ' deficit: {} ({})%'.format(-int(n), dn) if n<0 else ' surplus: {} ({})%'.format(int(n),dn)
        ax.text(50, -dlim/4, txt, fontsize=14, va='center')

        ax.set( ylabel='difference', xlabel='TS',
            xscale='log',  xlim=(9, 1000),ylim=(-dlim,dlim) )
        ax.axvline(25, color='green', lw=1, ls='--')# ,label='TS=25')
        ax.axhline(0, color='gray')

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda val,pos: { 10.0:'10', 100.:'100'}.get(val,'')))

        ax.grid(True, alpha=0.3)
        fig.set_facecolor('white')
        return fig

    def cumulative_counts(self):
        #assume rois.pickle available
        # extract the model counts for each source from it, and add to the DF
        if not hasattr(self,'roi_df'):
            self.roi_df = pickle.load(open('rois.pickle'))
        def counts(src_df,roi_df, name):
            """return fit counts for source name"""
            roiname = src_df.loc[name]['roiname']
            roi=roi_df.loc[roiname]
            names = list(roi['counts']['names'])
            i = names.index(name)
            if i<0: print name, i
            try: 
                t =np.sum(roi['counts']['models'][i][1])
                return t
            except:
                print 'fail to get counts for source', name, roiname, i
                return 0
        self.df['counts'] = [counts(self.df, self.roi_df, name) for  name in self.df.index]
        
    def non_psr_spectral_plots(self, index_min=1.0,tsvals=(10,16,25,250), index_max=3.5,
            beta_min=-0.1, beta_max=1.0, tail_check=False, selection='modelname=="LogParabola"'):
        """ Plots showing spectral parameters for PowerLaw and LogParabola spectra
        From left to right:
        <br> energy flux in eV/cm**2/s. This is the differential flux at the pivot energy
        <br> spectral index.
        <br> curvature index 
        <br> Pivot energy
        %(tail_check)s
        %(beta_check)s
        """
        fig, axx = plt.subplots( 1,4, figsize=(16,4))
        plt.subplots_adjust(wspace=0.2, left=0.05,bottom=0.15)

        t = self.df.query(selection)['ts flux pindex beta beta_unc freebits e0 roiname'.split()]
        t['eflux'] = t.flux * t.e0**2 * 1e6
        ax = axx[0]
        hkw=dict(histtype='step', lw=2)
        for tscut in tsvals:
            #print 'tscut:', tscut
            ax.hist(np.array(t.eflux[t.ts>tscut],float).clip(4e-2,20), np.logspace(np.log10(4e-2),np.log10(20),26), 
                label='TS>%d' % tscut, **hkw) 
        plt.setp(ax, xscale='log', xlabel='energy flux', xlim=(4e-2,20)); ax.grid(alpha=0.5); 
        ax.legend(prop=dict(size=10))
        ax = axx[1]
        [ax.hist(np.array(t.pindex[t.ts>tscut],float).clip(index_min,index_max), np.linspace(index_min,index_max,26),
                 label='TS>%d' % tscut, **hkw) for tscut in tsvals ]
        plt.setp(ax, xlabel='spectral index'); ax.grid(alpha=0.5); ax.legend(prop=dict(size=10))
        ax = axx[2]
        [ax.hist(np.array(t.beta[t.ts>tscut],float).clip(beta_min,beta_max), np.linspace(beta_min,beta_max,26),
            label='TS>%d' % tscut, **hkw) for tscut in tsvals ]
        # sel=(t.ts>tscut)&(t.beta>0.01)
        # if sum(sel)>0:
        plt.setp(ax, xlabel='beta'); ax.grid(alpha=0.5); ax.legend(prop=dict(size=10))
 
        ax = axx[3]
        [ax.hist(np.array(t.e0[t.ts>tscut],float), np.logspace(2,5,31), label='TS>%d' % tscut, **hkw) for tscut in tsvals ]
        plt.setp(ax, xlabel='e0 [MeV]', xscale='log'); ax.grid(alpha=0.5);ax.legend(prop=dict(size=10))        
        # get tails
        tail_cut = (t.eflux<5e-2) | ((t.pindex<index_min) | (t.pindex>index_max))& (t.beta==0) | (t.beta>beta_max) | (t.beta<0)
        
        if sum(tail_cut)>0 and tail_check:
            tails=t[tail_cut]['ts eflux pindex beta freebits roiname'.split()].sort_values(by='roiname')
            filename = 'non_pulsar_tails.html'
            html_file = self.plotfolder+'/%s' % filename
            #html = tails.sort_values(by='roiname').to_html(float_format=FloatFormat(2))
            self.tail_check = html_table(tails, name=self.plotfolder+'/pulsar_tails', 
                heading='<h4>Table of %d sources on tails</h4>'%len(tails),
                float_format=FloatFormat(2))
            #open(html_file,'w').write('<head>\n'+ _html.style + '</head>\n<body>'+ html+'\n</body>')
            #self.tail_check = '<p><a href="%s?skipDecoration">Table of %d sources on tails</a>: '% (filename, len(tails))
            self.tail_check += 'Criteria: require index between 1 and 3.5 for powerlaw, beta<2.0 for log parabola'
            
        #     # flag sources
        #     tflags = self.df.flags[tail_cut]+1
        #     #tails = tails.index
        #     self.df.loc[tail_cut,'flags'] = t ### bit 1
        #     print '%d sources flagged (1) in tails of flux, index, or beta' % sum(tail_cut)
        else:
            self.tail_check ='<p>No sources on tails'

        # check errors, especially that beta is at least 2 sigma
        self.beta_check=''
        #beta_bad = (t.beta>0.001) & ((t.beta_unc==0) | (t.beta/t.beta_unc<2) | (t.freebits!=7))
        #if sum(beta_bad)>0:
        #    print '%d sources fail beta check' % sum(beta_bad)
        #    self.beta_check = html_table(t[beta_bad]['ts beta beta_unc freebits roiname'.split()], 
        #        name=self.plotfolder+'/beta_check',
        #        heading = '<h4>Table of %d sources failing beta 2-sigma check</h4>'%sum(beta_bad),
        #        float_format=FloatFormat(2))
            
        return fig
        
    def check_beta(self, xmax=25, lpthresh=0.001):
        """Check beta
        
        <p>Compare TS for power-law vs. log parabola
        
        Histograms of TS_beta, the TS difference between beta=0 and a best fit
        (beta_check_note)s
        """
        df=self.df
        check = np.array([np.isfinite(x) for x in self.df.ts_beta]);
        beta_check_note=''
        if sum(check)==0:
            print 'No ts_beta values set'
            self.beta_check_note='<p>No beta analysis was done'
            return
        
        logp = np.array([model.name=='LogParabola' for model in df.model],bool)
        PL = logp & (df.beta<lpthresh)
        LP = logp & (df.beta>lpthresh)
        fig,ax = plt.subplots(figsize=(6,6))
        dom=np.linspace(0,xmax,26)
        args= dict(log=True, histtype='step', lw=2)
        ax.hist(df.ts_beta[PL].clip(0,xmax), dom,  label='PowerLaw', **args)
        try:
            ax.hist(df.ts_beta[LP].clip(0,xmax),dom, color='orange', label='LogParabola', **args)
        except Exception,msg:
            print 'fail LP: {}'.format(msg)
        plt.setp(ax, xlabel='TS_beta', ylim=(1,None))
        ax.grid(); ax.legend()
        return fig
    
    def pulsar_spectra(self, index_min=0.0, index_max=2.5, cutoff_max=1e4, LATpsr=False, taillist=True):
        """ Distributions for  %(psr_selection)s

        
        For each plot, the subset with a poor fit is shown.
        %(pulsar_tail_check)s
        """
        fig, axx = plt.subplots( 1,4, figsize=(14,4))
        plt.subplots_adjust(wspace=0.3, left=0.05,bottom=0.15)
        psrmodel = (self.df.ts>10) & (self.df.modelname=='PLSuperExpCutoff')
        if LATpsr: 
            psrmodel= psrmodel & self.df.psr
            self.psr_selection='LAT pulsars'
        else:
            self.psr_selection='sources fit with PLSuperExpCutoff spectral model, including LAT pulsars'
        t = self.df.loc[psrmodel]\
            ['ts flux pindex cutoff e0 index2 index2_unc roiname freebits fitqual'.split()]
        t['eflux'] = t.flux * t.e0**2 * 1e6
        badfit = t.fitqual>30

        def plot1(ax, efmin=1e-1,efmax=1e3):
            bins = np.logspace(np.log10(efmin),np.log10(efmax),26)
            vals = np.array(t.eflux,float).clip(efmin,efmax)
            ax.hist(vals, bins )
            if sum(badfit)>0:
                ax.hist(vals[badfit], bins, color='red', label='poor fit')
            ax.set(xscale='log', xlabel='energy flux', xlim=(efmin,efmax)); ax.grid(alpha=0.5); 
            ax.legend(prop=dict(size=10))

        def plot2(ax):
            bins = np.linspace(index_min,index_max,26)
            vals = np.array(t.pindex,float).clip(index_min,index_max)
            ax.hist(vals, bins)
            if sum(badfit)>0:
                ax.hist(vals[badfit], bins, color='red', label='poor fit')
            ax.set( xlabel='spectral index'); ax.grid(alpha=0.5); 
            ax.legend(prop=dict(size=10))
            
        def plot3(ax):
            bins = np.logspace(2,4,26)
            vals = np.array(t.cutoff,float).clip(None,cutoff_max) 
            ax.hist(vals, bins)
            if sum(badfit)>0:
                ax.hist(vals[badfit], bins, color='red', label='poor fit')
            ax.set(xscale='log', xlabel='cutoff energy (GeV)'); ax.grid(alpha=0.5)
            ax.legend(prop=dict(size=10))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda val,pos: { 100:'0.1', 1000:'1', 10000:'10'}.get(val,'')))
            
        def plot4(ax):
            xvals = np.array(t.cutoff,float).clip(None, cutoff_max)
            yvals = np.array(t.pindex,float).clip(index_min,index_max)
            ax.plot(xvals, yvals, 'o')
            if sum(badfit)>0:
                ax.plot(xvals[badfit], yvals[badfit], 'or', label='poor fit')
            ax.set(xscale='log', xlabel='cutoff [GeV]', ylabel='spectral index',
                 ylim=(index_min-0.1, index_max+0.1),
                )
            ax.grid(alpha=0.5); 
            ax.legend(loc='lower right', prop=dict(size=10))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda val,pos: { 100:'0.1', 1000:'1', 10000:'10'}.get(val,'')))

        for f,ax in zip((plot1,plot2,plot3,plot4,), axx.flatten()): f(ax)
        flags = self.df.flags
        tail_cut = (t.pindex<=index_min) | (t.pindex>index_max) | (t.cutoff>cutoff_max)
        tails = t.loc[tail_cut].index

        print '%d pulsar sources found in tails of  index or cutoff' % sum(tail_cut)
        if taillist & (sum(tail_cut)>0) :
            tails=t[tail_cut]['ts eflux pindex cutoff freebits roiname'.split()]
            filename = 'pulsar_tails.html'
            html_file = self.plotfolder+'/%s' % filename
            #html = tails.sort_values(by='roiname').to_html(float_format=FloatFormat(2))
            html = html_table(tails.sort_values(by='roiname'), float_format=FloatFormat(2))
            open(html_file,'w').write('<head>\n'+ _html.style + '</head>\n<body>'+ html+'\n</body>')
            self.pulsar_tail_check = '<p><a href="%s?skipDecoration">Table of %d sources on tails</a>: '% (filename, len(tails))
            self.pulsar_tail_check += 'Criteria: require index between 0 and 2.5, cutoff < {:.1f} GeV'.format(cutoff_max*1e-3)
        else:
            self.pulsar_tail_check ='<p>No sources on tails'

        return fig
    
    def ecliptic_hist(self, ax=None, title=''):
        ea = map(self.ecliptic_angle, self.df.skydir)
        fig, ax = self.get_figure(ax)
        ax.hist( ea, np.linspace(-90, 90, 91))
        plt.setp(ax, xlim=(-90,90), xlabel='ecliptic latitude')
        ax.set_xticks([-90, -45, 0, 45, 90])
        if title: ax.set_title(title)
        ax.grid(True)
        return fig
        
    @decorate_with(fitquality.FitQualityPlots)
    def fit_quality(self, xlim=(0,30), ndf=(9,8,8), ):

        fq = fitquality.FitQualityPlots(self, xlim=xlim, ndf=ndf )
        fq.badfit()
        return plt.gcf()
      
    def poor_fit_positions(self):
        """ Positions of poorly-fit sources
        Selection: fitqual>30 or |pull|>3
        """
        s = self.df
        s['pull0'] = np.array([x.pull[0] if x is not None else 0 for x in s.sedrec])
        poor = ( (s.fitqual>30) | (np.abs(s.pull0)>3)) & (s.ts>10) 
        return self.skyplot(s.fitqual[poor], vmin=30, vmax=100, cbtext='TS')
        
    def pivot_vs_e0(self, xylim=(100, 1e5)):
        """ pivot vs e0
        The reference energy, e0, is fixed except by a special run that iterates until the measured pivot energy, 
        which is the energy at which the differential flux uncertainty is minimum is the same. This plot checks that by measuring the pivot energy, and comparing it with the current reference. Note that e0 is required to be in the range 200 MeV to 100 GeV.
        """
        fig, ax = plt.subplots(figsize=(5,5))
        s = self.df
        cut = s.ts>10
        df = self.df
        offset = df.pivot_energy/df.e0-1.0
        df['offset'] = offset
        not_converged = cut & ( (abs(offset)>0.3) &(df.e0<99999) & (df.e0>201) |
                                 (df.e0>99999) & (df.pivot_energy<df.e0*0.95) 
                                | (df.e0<201) & (df.pivot_energy>df.e0*1.05)
                               )
        print 'Pivot needs fixing: %d sources' % sum(not_converged)
        self.pivotfix = tofix=df[not_converged]['ts pivot_energy e0 offset roiname'.split()].sort_values(by='roiname')

        ax.plot(s.e0[cut].clip(*xylim), s.pivot_energy[cut].clip(*xylim), '.')
        plt.setp(ax, xscale='log',xlabel='e0', xlim=xylim, 
                    ylabel='pivot', yscale='log', ylim=xylim)
        ax.set_title('compare calculated pivot with e0', fontsize=10)
        ax.grid()
        return fig
        
    def fitquality(self, grid=True):
        """Fit Quality
        
        Left: fit quality histogram; right fit quality vs. TS'

        """
        fig, axs = plt.subplots(1,2, figsize=(7,3))
        plt.subplots_adjust(wspace=0.35)
        s = self.df
        fitqual = s.band_ts-s.ts
        from scipy import stats
        ndf=12
        chi2 = lambda x: stats.chi2.pdf(x,ndf)
        d = np.linspace(0,100,51); delta=d[1]-d[0]
        ax =axs[0]
        ax.hist(fitqual, d, log=False);
        ax.hist(fitqual[s.ts>500], d, label='TS>500');
        ax.plot(d, chi2(d)*len(fitqual)*delta/1.6, 'r', label=r'$\mathsf{\chi^2\ ndf=%d}$'%ndf)
        plt.setp(ax, xlabel='fit qual', ylim=(0,500))
        ax.grid(grid); ax.legend(prop=dict(size=10))
        ax = axs[1]
        ax.plot(s.ts, fitqual, '.'); 
        plt.setp(ax, xscale='log', xlabel='TS', xlim=(10,1e5),
             ylabel='fit qual',ylim=(1,1e3),yscale='log')
        ax.grid(grid)
        return fig

    def flux_uncertainty(self):
        """ flux uncertainty compared with TS
        
        """
        fig, axx = plt.subplots(1,2, figsize=(10,4))
        plots=[]
        relflux_unc= self.df.flux_unc/self.df.flux
        ts = np.asarray(self.df.ts, float)
        ru= np.array(relflux_unc*100.,float)

        def plot1(ax):   
            dom = np.logspace(0,2,26)
            ax.hist(ru[ts>9], dom, label='%d sources'% sum(ts>9))
            for tsmin in (25,100,1000):
                ax.hist(ru[ts>tsmin], dom, label='TS<%d' % tsmin )
            plt.setp(ax, xscale='log', xlabel='relative flux uncertainty (%)', xlim=(1,100))
            ax.set_xticklabels([1,10,100])
            ax.grid()
            ax.legend(loc='upper left', prop=dict(size=10))
        plots.append(plot1)
            
        def plot2(ax):
            ax.plot(ts, ru*np.sqrt(ts)/100, '.')
            plt.setp(ax, xlabel='TS', xlim=(10,10000), xscale='log',
                 yscale='log',ylabel='ru*sqrt(ts)', ylim=(0.8,4))
            ax.plot([0.1,100], [0.1,100],'-g')
            ax.grid()
        plots.append(plot2)
            
        for plotf, ax in zip( (plots), axx.flatten(),):
            plotf(ax)
        return fig
        
    def sed_info(self, iband=0):
        pass

    def spectral_fit_consistency_plots(self, energy=133., minflux=1.0, ldatacut=True,
            title = 'low energy fit consistency',
            n_plots=2, pub=False, query=None,
        ):
        """ Spectral fit consistency for the lowest energy bin
        
        These plots show the consistency of the lowest energy band with the spectrum
        defined by the full fit. Require TS>25, energy flux for the bin > 1 eV. Discard cases where 
        there is only an upper limit for the flux.<br>
        <b>Left</b>: distribution of the "pull" <br>
        <b>Right</b>: position in the sky of sources with |pull]>3 <br>
        "Low" means |b|<5 deg.
        """
        hassed = np.array([self.df.iloc[i]['sedrec'] is not None for i in range(len(self.df))]) 
        nosed = (self.df.ts>10) & ~ hassed
        if sum(nosed)>0:
            print '+++Warning: %d TS>10 sources without sed info' % sum(nosed)
            print self.df[~hassed]['ts roiname'.split()][:min(20, sum(nosed))]
        cut= (self.df.ts>10) & hassed
        if query is None:
            s = self.df[cut].copy()
        else:
            s=self.df.query(query)
            print 'Additional cut, {}, results in {} events'.format(query, len(s))

        print 'selected {} sources with TS>10 and with SED info'.format(len(s))
        
        sedrec= [s.iloc[i]['sedrec'] for i in range(len(s))]

        fdata = np.array([sr.flux[0] for sr in sedrec]);
        udata = np.array([sr.uflux[0] for sr in sedrec])
        ldata = np.array([sr.lflux[0] for sr in sedrec])
        pull = np.array([sr.pull[0] for sr in sedrec])
        fmodel = np.array([s.iloc[i]['model'](energy)*energy**2*1e6 for i in range(len(s))])
        glat = np.array([x.b() for x in s.skydir])
        fluxcut = (fmodel>minflux) 
        if ldatacut: fluxcut = fluxcut & (ldata>0) #  avoid lower limits
        print 'after cuts on low energy flux > {:.1f} and significance= {}: {} events'.format(
                 minflux, ldatacut,sum(fluxcut))

        if pub:
            latcut, cut_labels = abs(glat)> 10.0 ,('|b|>10', '|b|<10')
        else:
            latcut, cut_labels = abs(glat)> 5.0,  ('|b|>5', '|b|<5')

        hilat = fluxcut & (latcut)
        lolat = fluxcut & (~latcut)
        
        #lowebad = np.abs(pull)>3
        #s.flags[lowebad] += 4
        #print 'Tagged %d sources with lowebad bit (4)' % sum(lowebad)

        lowebad = np.asarray((np.abs(pull)>3) , bool)
        s['lowebad']=lowebad
        s['pull']=pull
        self.spec_con= s[fluxcut] # for interactive checks

        #self.df.loc[lowebad,'flags'] = np.array(self.df.flags[lowebad],int) | 4
        #print 'Tagged %d sources with lowebad, abs(pull0)>3, bit (4)' % sum(lowebad)

        y = fdata/fmodel
        ylower, yupper =[(fdata-ldata)/fmodel,(udata-fdata)/fmodel]
        xhi,yhi,yerrhi = fmodel[hilat], y[hilat], [ylower[hilat],yupper[hilat]]
        xlo,ylo,yerrlo = fmodel[lolat], y[lolat], [ylower[lolat],yupper[lolat]]
        
        def error_bar(ax):
            ax.errorbar(x=xhi, y=yhi, yerr=yerrhi, fmt='og', label='%d hilat sources'%sum(hilat))
            ax.errorbar(x=xlo, y=ylo, yerr=yerrlo, fmt='or', label='%d lowlat sources'%sum(lolat))
            plt.setp(ax, xlabel=r'$\mathsf{model\ flux\ (eV\ cm^{-2} s^{-1}})$', xscale='log', 
                ylabel='data/model', ylim=(0,2.5), xlim=(minflux, 100) )
            #ax.set_xticks([2,5,10,20,50,100])
            ax.set_title( title, fontsize='medium')
            ax.legend(loc='upper left', prop=dict(size=10))
            ax.grid()  

        def hist(ax):
            hist_kw=dict(bins=np.linspace(-3,3,25), lw=2, histtype='step')
            q=pull.clip(-3,3) 
            assert len(q)== len(hilat)#, len(lolat)
            ax.axvline(0, color='grey', ls='--')
            for name,color, cut in [(cut_labels[0],'g', hilat), (cut_labels[1],'r', lolat)]:
                vals = q[cut]
                ax.hist(vals, color=color,  label='{:5}{:5d}{:4.1f}{:4.1f}'.format(
                            name, len(vals), 
                            np.sqrt((vals[vals<0]**2).mean()), np.sqrt((vals[vals>0]**2).mean())), 
                        **hist_kw)
                print name, vals.std(), np.sqrt((vals[vals<0]**2).mean()), np.sqrt((vals[vals>0]**2).mean())
            ax.set_xlabel('normalized residual')

            ax.set_xlim((-3,3))
            
            leg=ax.legend(loc='upper left', prop=dict(size=10, family='monospace'),
                        #title='     type   #   RMS\n'
                        title='  selection #    RMS\n'
                              '                <0 >0',
                            )
            ltit = leg.get_title(); ltit.set_fontsize(10); ltit.set_family('monospace')
            if not pub: 
                ax.grid(alpha=0.5)
                ax.set_title( title, fontsize='medium')  
            # else:
            #     ax.set(ylim=(0,150))


        def skyplot(ax):

            pdf = s.query('lowebad==True')# to atatch indx
            self.skyplot(pdf.pull, ax=ax, vmin=-3, vmax=3,
                cmap=plt.get_cmap('coolwarm'), title=title, cbtext='pull')

        if n_plots==3:
            fig,ax = plt.subplots(1,3, figsize=(12,5))
            plt.subplots_adjust(wspace=0.3, left=0.05)
            for f, ax in zip( (hist, error_bar, skyplot), ax.flatten()):
                f(ax=ax)
        elif n_plots==2:
            fig,ax = plt.subplots(1,2, figsize=(12,5))
            plt.subplots_adjust(wspace=0.3, left=0.05)
            for f, ax in zip( (hist, skyplot), ax.flatten()):
                f(ax=ax)
        else:
            fig,ax = plt.subplots(1,1, figsize=(7,4))
            hist(ax)

        return fig
        
    def census(self, primary_prefix='P88Y', cols=[0,5,10,16,25]): #'P7R4'):
        """Census
        
        %(census_html)s
        <p>
        In this table of prefixes, the columns are the number of sources with TS greater than the header value. 
        The first set of columns, with an "H" in the column label, are for sources with |b|>5.

        The row labels are the first four characters of the source name, except 'ext' means extended.

        """
        df = self.df
        extended = np.asarray(df.isextended.values,bool)
        pointsource = ~extended

        def count(prefix, tsmin, cut=None):
            if tsmin==0: tsmin=-10 
            sel = df.ts>tsmin if cut is None else (df.ts>tsmin) & cut
            if prefix=='ext':
                return sum(extended & sel)
            elif prefix=='total':
                return sum(sel)
            names = df[pointsource & sel]['name'].values    
            return sum([n.startswith(prefix) for n in names])
        if count('ext',0)>0:
            prefixes = list(set( n[:3] for n in df[pointsource]['name'])) +['ext', 'total']
        else:
            prefixes = list(set( n[:3] for n in df[pointsource]['name'])) +['total']
        
        census = OrderedDict()
        prefixes = sorted(prefixes)

        highlat = np.abs(df.glat)>5
        for x in cols:
            census['{}H'.format(x)] = [count(prefix, x, highlat) for prefix in prefixes]
        for x in cols:
            census[x] = [count(prefix, x) for prefix in prefixes]

        self.census_data=pd.DataFrame(census, index=prefixes)
        self.census_html = '\n<h4>Prefixes</h4>\n'\
            +html_table(self.census_data, maxlines=20, href=False)
        
      
    def recent_origin(self, recent_adds='211 214 215'.split() ):
        """
        """
        df = self.df
        self.census()
        cs = self.census_data    
        t = cs.loc[recent_adds,25]
        t, np.cumsum(t.values)

        added = lambda nm: nm[:3] in recent_adds
        ts = df.ts[map(added, df.index)].values.astype(float); ts

        def flabel(n, x):
            return '{} ({})'.format(n, sum(x>25))
        tsall = df.ts.astype(float).clip(5,500)
        xlim = (5,500)
        ts = [ tsall[map(lambda  nm:  nm.startswith(key), df.index)] for key in recent_adds];
        fig, ax = plt.subplots(figsize=(8,5))
        fig.set_facecolor('white')
        hpkw = dict(bins=np.logspace(np.log10(5),np.log10(500),41), histtype='step', lw=2)
        ax.hist(tsall, label=flabel('all',tsall), **hpkw)
        hpkw.update(histtype='stepfilled')
        ax.hist(ts,  stacked=True,  label=[flabel('uw1'+n, ts[i]) for i,n in enumerate(recent_adds)], **hpkw);

        ax.set(xscale='log', ylim=(0.9,None), xlim=xlim, xlabel='TS', title='TS: total and by recent origin');
        xticks=[10,25,100]
        ax.set_xticks(xticks)
        ax.set_xticklabels( map(lambda x:'{:0.0f}'.format(x), xticks) )
        ax.axvline(25, color='grey', ls='--')
        ax.legend(title='set, n(TS>25)')
        return fig


    def flag_proc(self, make_pivot=False):
        """ Flagged source summary:
        %(flagged_link)s
        """
        # Generate summary table for flagged sources (except 
        t =self.df[(self.df.flags& 7 >0) & (self.df.ts>10)]['ra dec ts fitqual pull0 eflux pindex beta cutoff index2 flags roiname'.split()]
        t.to_csv('flagged_sources.csv')
        print 'wrote %d sources to flagged_sources.csv' % len(t)
        
        num=[sum(self.df.flags & 2**b > 0) for b in range(4)] 
        flagtable=pd.DataFrame(dict(number=num, description=('tails','poor fits','low energy bad', 'poor localization') ))
        flagtable.index.name='bit'
        self.flagged_link = """\
        <p>A number of these sources have been flagged to indicate potential issues with the spectral fit. 
        The flag bits and number flagged as such are:
        %s<br>  """ % html_table(flagtable, href=False)
        if not make_pivot: return None
        try:
            pc =makepivot.MakeCollection('flagged sources %s' % os.path.split(os.getcwd())[-1], 'sedfig', 'flagged_sources.csv')
            self.flagged_link += """\
            <p>These can be examined with a 
            <a href="http://deeptalk.phys.washington.edu/PivotWeb/SLViewer.html?cID=%d">Pivot browser</a>,
            which requires Silverlight."""  % pc.cId
        except Exception, msg: 
            print "**** Failed to make pivot table, perhaps need to run sedinfo first: %s" % msg
        return None

    def flux_table(self, source_name):
        """ Return a DataFrame describing the sed info for the given source
        Columns, with energy fluxes in eV units, are:
            flux, lflux, uflux : measured flux, lower and upper uncertainty, or 0,0, 95% limit
            mflux : predicted flux for this bin
            TS : Test Statistic for the signal
            pull : signed square root of the 
        """
        si = self.df.loc[source_name]['sedrec']
        pull = np.sign(si.flux-si.mflux) * np.sqrt(si.delta_ts.clip(0,100))
        return pd.DataFrame(dict(flux=si.flux.round(1), TS=si.ts.round(1), lflux=si.lflux.round(1), 
            uflux=si.uflux.round(1), model=si.mflux.round(1), pull=pull.round(1) ),
                index=np.array(np.sqrt(si.elow*si.ehigh),int), columns='flux lflux uflux model TS pull'.split())
    
    def find_nearest_to(self, *pars):
        """Given the ra, dec, or a skydir, find nearest source, return name, distance"""
        from skymaps import SkyDir

        sd = SkyDir(*pars) if len(pars)==2 else pars[0]
        dists = [x.difference(sd) for x in self.df.skydir]
        t= min(dists)
        i = dists.index(t)
        return self.df.index[i], np.degrees(t)

    def curvature(self, setup=False, cmax=1.0):
        """Curvature
        
        Distribution of the curvature per source, equivalent to the beta parameter for a LogParabola spectral model.
        """
        if setup:
            #expect to be called initially
            self.df['curvature']= np.array([model.curvature() for model in self.df.model])
            return
        assert 'curvature' in self.df, 'Curvature not calculated'
        df = self.df
        psr = np.asarray([n.startswith('PSR') for n in df.index], bool)
        pseed = np.array([ len(n)>3 and n[3]=='N' for n in df.index],bool);
        fig,ax = plt.subplots(figsize=(8,6))
        hkw = dict(bins=np.linspace(0,cmax,41), log=True, histtype='step', lw=2)
        ax.hist(df.curvature.clip(0,cmax), label='all sources', **hkw)
        ax.hist(df[pseed].curvature.clip(0,cmax), label='PSR seed', **hkw)
        ax.hist(df[psr].curvature.clip(0,cmax), label='PSR source', **hkw)
        plt.setp(ax, xlabel='Curvature', ylim=(0.5,None))
        ax.legend()
        ax.grid()
        return fig
    
    def roi_check(self):
        """Distance from ROI center
        Number of sources per ROI and distribution of source positions from the center of the ROI.
        %(roi_check_html)s
        """
        self.df['roinum'] = [int(s[-4:]) for s in self.df.roiname]
        pnroi = 1728
        pocc=np.histogram(self.df.roinum,np.linspace(0,pnroi,pnroi) )[0]
        pocc.min(), pocc.mean(), pocc.max()
        df = self.df
        df['actual_roi'] = map(Band(12).index, df.skydir)
        df['roi'] = map( lambda n:int(n[-4:]), df.roiname)
        df['rname'] = np.array(['HP12_{:04d}'.format(x) for x in df.roi])
        df['roi_dist'] = map(lambda s,r: np.degrees(s.difference(Band(12).dir(r))),df.skydir,df.roi)
        check = df.roi!=df.actual_roi; 
        print 'Found {} sources in the wrong ROI'.format(sum(check))
        fig,axx = plt.subplots(1,2,figsize=(10,4))
        ax=axx[0]
        ax.hist(pocc, np.linspace(1,80,80),log=True);
        ax.set_xlabel('Number of sources')
        hist_kw = dict(bins=np.linspace(0,8,33), histtype ='stepfilled', log=True)
        ax=axx[1]
        ax.hist(df.roi_dist.clip(0,8), **hist_kw)
        if sum(check)>0:
            ax.hist(df.roi_dist[check].clip(0.8), color='red', label='wrong ROI', **hist_kw)
        ax.grid(True, alpha=0.5);
        ax.legend()
        plt.setp(ax, xlabel='Distance (deg)', ylim=(0.8,None))
        to_move = df[check]['roi actual_roi ts'.split()].sort_values(by='roi')
        to_move['roi_dist'] = df.roi_dist
        self.roi_check_html = 'Check for sources outside ROI OK.'
        if sum(check)>0:
            self.roi_check_html = html_table(to_move, name=self.plotfolder+'/outside_roi', 
                heading='<h4>%d Sources that are outside the HEALPix ROI boundary</h4>' %len(to_move),
                float_format=FloatFormat(2))
        return fig

    def extended_table(self):
        """Extended table
        Table of information for fits, including links to SEDs for all extended sources.
        %(extended_table_html)s
        """
        
        df = self.df
        ext = df.isextended

        cols = 'ra dec ts fitqual pindex roiname'.split()
        extdf = pd.DataFrame(df[ext][cols]).sort_values(by='roiname')
        config = configuration.Configuration('.', quiet=True, postpone=True)
        ecat = extended.ExtendedCatalog(os.path.expandvars('$FERMI/catalog/')+config.extended)
        extdf['spatial_model'] = [ecat[name].dmodel.name for name in extdf.index]
        self.extended_table_html = html_table(extdf, name=self.plotfolder+'/extended_sources',                                        
                                        heading='<h4>Table of {} extended sources</h4>'.format(len(extdf)),
                                        float_format=FloatFormat(2))
        return None
    
    def get_source(self, name):
        """ return a sources.PointSource object"""
        from uw.like2 import sources
        try:
            s = self.df.loc[name]
        except:
            print 'Name "{}" not found'.format(name)
            #flag not found
            return sources.PointSource(name=name, model=None, skydir=None, sedrec=None, ts=-1)

        p=sources.PointSource(name=s.name, model=s.model, skydir=s.skydir)
        p.sedrec=s.sedrec
        p.ts = s.ts
        return p

    def plot_sed(self, name, ax=None, xlim=(1e2,3e4), ylim=(0.04, 20)):
        from uw.like2.plotting import sed
        p = self.get_source(name)
        if p is None: return
        sed.Plot(p)(axes=ax, galmap=p.skydir, axis=xlim+ylim)

    def plot_seds(self, namelist, row_size=5, **kwargs):
        from uw.like2.plotting import sed
        sed.plot_seds(self, namelist, row_size=row_size, **kwargs)

    def high_tail(self, tscut=50, high_tail_cut=12, curvature_cut=0.0):
        """High energy tail sources
        
        A test source with index 2.0 was inserted at the position of each source to detect, for each energy band,
        if there was a significant signal not accounted for by the source's spectral model. The histogram above 
        is of the total TS for this component above 100 GeV, for all non-LAT pulsar or extended sources above TS=50.
        The table has information for each source including TS, position and association. It has sources with a threshold 
        for the TS shown on the histogram.
        
        %(high_tail_table)s

        %(high_tail_psr_table)s
        
        """
        df = self.df
        psr_flag = [n.startswith('PSR') | n.startswith('Crab') for n in df.index]
        flat_ts = df.ts_flat.values;
        assert sum(np.array([x is not None  for x in flat_ts]))>0, 'No high-tail analysis was done'
        df['high_tail'] = np.array([sum(x[-2:]) for x in flat_ts],float)
        selection = (df.ts>tscut) & np.logical_not(psr_flag) & (df.curvature>curvature_cut) #& np.logical_not(df.isextended)
        dfhigh =df[selection & (df.high_tail>high_tail_cut)]['jname ts curvature high_tail acat glon glat'.split()].sort_values(by='glat')
        dfhigh_psr =df[(df.ts>tscut) & psr_flag & (df.high_tail>high_tail_cut)]['jname ts high_tail acat glon glat'.split()].sort_values(by='glat')

        fig, ax = plt.subplots(figsize=(7,5))
        ax.hist((df.high_tail[selection]).clip(0,25), np.linspace(0,25,26), log=True, histtype='step', lw=2);  
        ax.axvline(high_tail_cut, color='orange', label='threshold for table')
        ax.set(xlabel='TS for E>100 GeV')
        ax.legend()
        
        self.high_tail_table = html_table(dfhigh, name=self.plotfolder+'/high_tail_sources',                                        
                                            heading='<h4>Table of {} high-tail sources</h4>'.format(len(dfhigh)),
                                            float_format=FloatFormat(2))
        self.high_tail_psr_table = html_table(dfhigh_psr, name=self.plotfolder+'/high_tail_sources_psr',                                        
                                            heading='<h4>Table of {} PSR high-tail sources</h4>'.format(len(dfhigh_psr)),
                                            float_format=FloatFormat(2))
        return fig
    
    def all_plots(self):

        plt.close('all')
        probfun = lambda x: x['prob'][0] if not pd.isnull(x) else 0
        self.df['aprob'] = np.array([ probfun(assoc) for  assoc in self.df.associations])

        colstosave="""ra dec roiname ts aprob eflux100 modelname pars errs freebits fitqual e0 flux flux_unc pindex pindex_unc index2 index2_unc
                 cutoff cutoff_unc  eflux100_unc locqual delta_ts a b ang flags jname """.split()
        self.df.loc[(self.df.ts>10) | self.df.psr ][colstosave].to_csv(self.csvfile)
        print 'saved truncated csv version to "%s"' %self.csvfile
        
        self.runfigures([self.census, 
            self.cumulative_ts, 
            self.fit_quality,
            self.spectral_fit_consistency_plots, 
            #self.poor_fit_positions,
            self.non_psr_spectral_plots, 
            #self.beta_check, 
            self.pulsar_spectra, 
            self.curvature, 
            self.pivot_vs_e0, 
            self.high_tail,
            self.roi_check, 
            self.extended_table, 
            ]
        )
    
    def associate(self, df, angle_cut = 0.1 , tag_uw=False):
        """Make associations with a another list of source positions
        df : DataFrame with ra, dec or skydir members 
            Note: needs to be a copy if a subset of DataFrame
        angle_cut : float value use use to set flag
        tag_uw : bool
            if True, set values in the uw list
        """
        def differences(a,b):
            matrix = np.array([[x.difference(y) for y in b] for x in a], np.float32)
            return matrix
        def closest(t):
            n = t.argmin()
            return (n, (np.degrees(t[n])*3600).round()) #return rounded arcsec dist
        oth_skydir = map(SkyDir, np.array(df.ra,float),np.array(df.dec,float)) if 'skydir' not in df else df.skydir
        f95, quad = self.config['localization_systematics']
        print 'Applying factor of {:.2f} to localization errors, and adding {:.3g} arcmin in quadrature to r95'.format(f95, quad)
        dfuw = self.df
        diff_array =differences(oth_skydir, dfuw.skydir)
        cl_oth = np.array([closest(r) for r in diff_array[:,]], int)
        close_cut = 3600*angle_cut
        dfuw['r95'] = np.sqrt( np.array((2.45*f95)**2 *dfuw.a * dfuw.b + (quad/60)**2,float))
        df['dist'] = cl_oth[:,1]
        df['uw_name'] = [dfuw.index[i] for i in cl_oth[:,0]]
        df['uw_jname'] = [dfuw.jname[i] for i in cl_oth[:,0]]
        df['uw_ts'] = [dfuw.loc[i].ts for i in cl_oth[:,0]]
        df['uw_r95'] = [dfuw.loc[i].r95 for i in cl_oth[:,0]]
        df['uw_pindex'] = [dfuw.loc[i].pindex for i in cl_oth[:,0]]
        df['uw_roi'] = [int((dfuw.loc[i].roiname)[-4:]) for i in cl_oth[:,0]]
        df['uw_locqual'] = [dfuw.locqual[i] for i in cl_oth[:,0] ]
        df['uwok'] = (df.dist<close_cut)
        if not tag_uw: return
        cl_uw = np.array([closest(c) for c in diff_array[:,].T], int) 
        #todo when needed
         
    def __call__(self, newname):
        oldname = self.get_oldname(newname)
        if oldname not in self.index_3fgl: return None
        return self.cat.loc[oldname]

class ExtSourceInfo(SourceInfo):
    """ subclass invoked with a specific path
    """
    def __init__(self, model_path, quiet=True):
        curdir = os.getcwd()
        try:
            os.chdir(model_path)
            if not quiet: print os.getcwd()
            self.setup(refresh=False, quiet=quiet)
            self.plotfolder=model_path+'/plots/'+self.plotfolder
        finally:
            os.chdir(curdir)


