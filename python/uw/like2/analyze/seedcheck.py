"""Analyze seeds

$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/analyze/seedcheck.py,v 1.14 2018/01/27 15:39:29 burnett Exp $

"""
import os, glob
import numpy as np
import pylab as plt
import pandas as pd
import skymaps
from uw.like2.pub import healpix_map as hpm
import healpy
from astropy.io import fits
from .. import associate, maps
from . import sourceinfo, _html
from .analysis_base import FloatFormat, html_table

b12index = skymaps.Band(12).index

class SeedCheck(sourceinfo.SourceInfo):
    """Seed Analysis

    <p>This is a second stage after a run that produced a residual TS map. The first pipeline run, "sourcefinding", 
    generates nside=512 HEALPix tables of the residual plots for the following spectral templates:
    <ul type="circle">
        <li>flat</li>
        <li>hard</li>
        <li>soft</li>
        <li>peaked</li> 
        <li>psr</li>   
    </ul>.
    <p>
     The plot analysis "hptables" that follows it does a cluster analysis, generating a list of
    seed positions in the file "seeds.txt". A second pipeline run, "seedcheck", for each ROI, adds seeds falling in the central pixel to the model, 
    performing an inital fit to the flux only, then a localization, and finally a full fit. The fit and localization information are 
    written to a set of pickle files in the folder "seedcheck" or the zip file "seedcheck.zip".
    A selection is applied with these cuts: 
    %(cut_summary)s
    <br>and the remaining sources are analyzed here.
    """

    def setup(self, nside=512, **kw):
        # ??
        # import seaborn as sns
        self.nside=nside
        self.plotfolder='seedcheck' #needed by superclass
        super(SeedCheck, self).setup(**kw) 

        self.input_model=self.skymodel
        #???
        #self.load()

    def template_info(self):
        """ Template spectra

        """
        modelnames = [maps.table_info[key][1]['model'] for key in self.keys];
        models = [eval('maps.'+modelname) for modelname in modelnames]
        class Source(object):
            def __init__(self, name,  model):
                self.name=name;  self.spectral_model= model
        templates = map(Source,  self.keys, models) 
        ee = np.logspace(2,5)
        fig, ax = plt.subplots(figsize=(6,4))
        fig.set_facecolor('white')
        for tm in  templates:
            f = lambda e: tm.spectral_model(e)*e**2
            ax.loglog(ee, f(ee)/f(1e3), '-', label=tm.name, lw=2);
        ax.set(ylim=(2e-3,10),xlim=(ee[0],ee[-1]), ylabel='Energy flux', xlabel='Energy [MeV]', title='Spectral templates')
        ax.legend(loc='lower left');
        return fig

    # def seed_plots(self,  bcut=5, subset=None, title=None):
    #     """ Seed plots

    #     Results of cluster analysis of the residual TS distribution. Analysis of %(n_seeds)d seeds from file 
    #     <a href="../../%(seedfile)s">%(seedfile)s</a>. 
    #     <br>Left: size of cluster, in 0.15 degree pixels
    #     <br>Center: maximum TS in the cluster
    #     <br>Right: distribution in sin(|b|), showing cut if any.
    #     """
    #     z = self.seeddf #self.seeds if subset is None else self.seeds[subset]
    #     keys = list(set(z.key)); 
    #     fig,axx= plt.subplots(1,3, figsize=(12,4))
    #     plt.subplots_adjust(left=0.1)
    #     bc = np.abs(z.b)<bcut
    #     histkw=dict(histtype='step', lw=2)
    #     def all_plot(ax, q, dom, label, log=False):
    #         for key in keys:
    #             ax.hist(q[z.key==key].clip(dom[0],dom[-1]),dom, label=key, log=log, **histkw)
    #         ax.set( xlabel=label, xlim=(dom[0],dom[-1]))
    #         ax.grid(alpha=0.5)
    #         ax.legend(prop=dict(size=10, family='monospace'))
    #         if log: ax.set_ylim(ymin=0.9)
    #     all_plot(axx[0], z['size'], np.linspace(0.5,10.5,11), 'cluster size')
    #     all_plot(axx[1], z.ts.clip(15,50), np.linspace(15,50,16), 'TS', log=True)
    #     all_plot(axx[2], np.sin(np.radians(z.b)), np.linspace(-1,1,21), 'sin(b)')
    #     axx[2].axvline(0, color='k')
    #     fig.set(facecolor='white')
    #     return fig

    def tsmaps(self):
        """TS maps for the spectral templates

        Each has 12*nside**2 = 3.2M pixels. The following table has the number of pixels above the TS value labeling the column.
        %(tsmap_info)s
        """
        z = dict(); cuts = (10,16,25,100)
        keys=self.keys
        for key in keys:
            z[key]= [np.sum(self.tables[key]>x) for x in cuts]
        df=pd.DataFrame(z, index=cuts).T 
        self.tsmap_info= html_table(df, href=False, )

        fig, axx = plt.subplots(2,3, figsize=(18,15))
        plt.subplots_adjust(hspace=-0.4,wspace=0.05, left=0.05)
        for ax,key in zip(axx.flatten(), keys):
            plt.sca(ax)
            healpy.mollview(self.tables[key],hold=True, min=10, max=25, title=key, cbar=True, 
            cmap=plt.get_cmap('YlOrRd'))
            healpy.graticule(dmer=180, dpar=90, color='lightgrey')
        axx[1,2].set_visible(False)
        return fig

    def pixel_ts_distribution(self, tsmax=25):
        """The cumulative TS distribution
        For single pixels, one might expect the chi-squared distribution for the null hypothesis if 
        there were no resolvable sources, and the background model was correct.
        The shaded area is the difference.
        """
        def setup():
            hpname = 'hptables_*_{}.fits'.format(nside)
            gg = glob.glob(hpname)
            if len(gg)==0:
                print 'File {} not found'
                return False
            f = gg[0]; 
            keys = f.split('_')[1:-1];
            self.keys=keys #= ['ts', 'tsp', 'hard', 'soft'] #keys =stagedict.stagenames['sourcefinding']['pars']['table_keys']

            tt = fits.open('hptables_{}_{}.fits'.format('_'.join(keys), nside))
            self.tables = tt[1].data;

            self.seeds.index.name='name'
            print 'Loaded {} seeds from file "{}"'.format(len(self.seeds), seedfile)
            unique, counts = np.unique(self.seeds.key, return_counts=True)
            print 'seed keys: {}'.format(dict(zip(unique, counts)))
            self.cut_summary='[Not done yet]'
            return True

        def plot_one(seed_key, ax):
            bins = np.linspace(0,tsmax,501)
            tsvec=self.tables[seed_key]
            ax.hist(tsvec, bins, log=True, histtype='step', lw=2, cumulative=-1, label='data');
            # make array corresponding to the hist
            h = np.histogram(tsvec, bins, )[0]
            x = bins[:-1]
            yh = sum(h)-h.cumsum() 
            f = lambda x: np.exp(-x/2)
            ye=6e5*f(x)
            ax.plot(x, ye, '-g', lw=2, label='exp(-TS/2)')
            ax.fill_between(x,yh,ye,where=x>5, facecolor='red', alpha=0.6)
            plt.setp(ax, xscale='linear', xlabel='TS', ylim=(1,None), ylabel='# greater than TS')
            ax.legend()
            ax.set_title('seed type {}'.format(seed_key), fontsize=14)
            ax.grid(True, alpha=0.5)

        if not setup():
            return None

        fig, axx = plt.subplots(2,3, figsize=(15,12))
        for key,ax in zip(self.keys, axx.flatten()):
            plot_one(key, ax)
        axx[1,2].set_visible(False)
        fig.suptitle('Cumulative distribution of single-pixel TS values');        
        fig.set_facecolor('white')
        return fig


    def seed_plots(self, subset='all', bcut=5, title=None):
        """ Seed plots
        
        Results of cluster analysis of the residual TS distribution. Analysis of %(n_seeds)d seeds from file 
        <a href="../../%(seedfile)s">%(seedfile)s</a>. 
        <br>Left: size of cluster, in 0.15 degree pixels
        <br>Center: maximum TS in the cluster
        <br>Right: distribution in sin(|b|), showing cut if any.
        """
        # check for local, then in parent's folder
        seedfile = 'seeds/seeds_all.csv'
        if not os.path.isfile(seedfile):
            seedfile = os.path.join(self.config['input_model']['path'], seedfile)
    
        assert os.path.isfile(seedfile), 'seed file not found localally or in parent'
        self.seeds = pd.read_csv(seedfile, index_col=0)

        try:
            z = self.seeds if subset=='all' else self.seeds.query('key=="{}"'.format(subset))
        except Exception, msg:
            print 'Failed to find subset {}: {}'.format(subset, msg)
            raise
        bc = np.abs(z['b'])<bcut
        
        fig,axx= plt.subplots(1,3, figsize=(12,4))
        plt.subplots_adjust(left=0.1)
        histkw=dict(histtype='step', lw=2)
        def all_plot(ax, q, dom, label):
            ax.hist(q.clip(dom[0],dom[-1]),dom, label=None, **histkw)
            ax.hist(q[bc].values.clip(dom[0],dom[-1]),dom, color='orange', label='|b|<%d'%bcut, **histkw)
            plt.setp(ax, xlabel=label, xlim=(dom[0],dom[-1]))
            ax.grid()
            ax.legend(prop=dict(size=10))
        all_plot(axx[0], z['size'], np.linspace(0.5,10.5,11), 'cluster size')
        all_plot(axx[1], z.ts, np.linspace(10,50,21), 'TS')
        all_plot(axx[2], np.sin(np.radians(z.b)), np.linspace(-1,1,41), 'sin(b)')
        axx[2].axvline(0, color='k')
        fig.suptitle('{} {} seeds from model {}'.format( len(z), subset, self.input_model,)
             if title is None else title)
        fig.set_facecolor('white')
        return fig



    def load(self):
        files, sources = self.load_pickles(self.seedname)
        sdict={}
        assoc={}
        for source in sources:
            name = source.name
            model = source.model
            pars = np.empty(4); pars.fill(np.nan)
            errs = np.empty(4); errs.fill(-2)
            free = np.zeros(4, bool)
            n = model.len()
            pars[:n] = model.parameters
            free[:n] = model.free
            try:
                diag = np.diag(model.get_cov_matrix())
                errs[:n] = [np.sqrt(x) if x>0 else -1 for x in diag[:n]]
                badfit = np.any(errs[model.free]<=0)
            except Exception, msg:
                print 'fail errors for %s:%s' % (name, msg)
                badfit = True
            has_adict = hasattr(source,'adict') and source.adict is not None
            has_ellipse = hasattr(source, 'ellipse') and source.ellipse is not None
            sdict[name] = dict(
                ra =source.skydir.ra(), dec=source.skydir.dec(),
                ts=source.ts,
                delta_ts=source.ellipse[6] if has_ellipse else np.nan,
                r95 = 2.6*source.ellipse[2] if has_ellipse else np.nan,
                locqual=source.ellipse[5] if has_ellipse else np.nan,
                glat=source.skydir.b(), glon=source.skydir.l(),
                eflux=pars[0] * model.e0**2 *1e6,
                eflux_unc=errs[0] * model.e0**2 *1e6 if errs[0]>0 else np.nan,
                pindex = pars[1],
                pindex_unc = errs[1] if errs[1]>0 else np.nan,
                par2 = pars[2],
                par2_unc = errs[2] if errs[2]>0 else np.nan,
                e0 = model.e0,
                aprob = source.adict['prob'][0] if has_adict else 0,
                index = b12index(source.skydir),
                #gflux  = model.i_flux(), ## photon flux
                )
            assoc[name] = dict(
                acat = source.adict['cat'][0] if has_adict else None,
                aname= source.adict['name'][0] if has_adict else None,
                adelta_ts = source.adict['deltats'][0] if has_adict else None,
                aprob = source.adict['prob'][0] if has_adict else 0.,
                adict = source.adict if has_adict else None,
                )
        self.df = pd.DataFrame(pd.DataFrame(sdict).transpose(), 
            columns="""ra dec glat glon ts ellipse delta_ts eflux eflux_unc 
                    pindex pindex_unc par2 par2_unc e0 r95 locqual aprob index""".split() 
            )
        self.df.index.name='name'
        
        # analyze associations, make summary
        self.assoc = pd.DataFrame(assoc).transpose()
        self.assoc.index.name = 'name'
        if all(self.assoc.aprob==0):
            print "No associations found: running the standard logic"
            try:
                self.make_associations()
            except Exception, msg:
                print 'Association attempt failed: %s' % msg
                raise
        else:
            print "Using associations from uwpipeline run"
        
        # define good subset
        self.good = (self.df.ts>6)&(self.df.r95<0.6)&(self.df.locqual<8)
        self.df_good= self.df[self.good]
        
#        self.test = self.df_good.ix[self.df_good.aprob>0.8]
        
        acat=list(self.df_good.ix[self.df_good.aprob>0.8]['acat'].values)
        sa = list(set(acat))
        t = np.zeros(len(sa),int)
        for x in acat:
            t[sa.index(x)]+=1
        self.assoc_sum = zip(sa, t)
        self.cut_summary= """<p>Read in %d sources from file %s: <br>selection cut:
                    (self.df.ts>6)*(self.df.r95<0.6)*(self.df.locqual<8) : %d remain"""\
            % (len(sources), self.require, sum(self.good))
        print self.cut_summary
    
    def make_associations(self):
        """ run the standard association logic 
            Only apply to "good" sources
        """
        srcid = associate.SrcId()
        assoc={}
        print 'Note: making bzcat first if second'
        for name, s in self.df.iterrows():
            has_ellipse=  not np.isnan(s['r95'])
            if has_ellipse: 
                try:
                    adict = srcid(name, skymaps.SkyDir(s['ra'],s['dec']), s['r95']/2.6)
                except Exception, msg:
                    print 'Failed association for source %s: %s' % (name, msg)
                    adict=None
                has_ellipse = adict is not None
                if has_ellipse:
                    cats = adict['cat']; probs = adict['prob']
                    i=1 if len(cats)>1 and \
                        cats[1]=='bzcat' and probs[1]>0.8\
                        or cats[0]=='cgrabs' else 0
            assoc[name] = ad = dict(
                    acat =  adict['cat'][i] if has_ellipse else None,
                    aname=  adict['name'][i] if has_ellipse else None,
                    adelta_ts = adict['deltats'][i] if has_ellipse else None,
                    aprob = adict['prob'][i] if has_ellipse else 0.,
                    adict = adict if has_ellipse else None,
                    )
        
        self.assoc = pd.DataFrame(assoc).transpose()
        self.assoc.index.name = 'name'
        for col in 'acat aname adelta_ts aprob adict'.split():
            self.df[col] = self.assoc[col]
    
    def seed_cumulative_ts(self, cut=None, label=None):
        """ Cumulative TS distribution for seeds 
        """
        df = self.df_good
        prefixes = list(set([s[:4] for s in df.index]))
        n = len(prefixes)
        if n>2:
            print 'Warning: only set to plot two prefixes'
            n=2
        if n >1:
            print 'found prefixes %s' %prefixes
            x = [[s.startswith(prefixes[i]) for s in df.index] for i in range(n)]
            fig=self.cumulative_ts(df.ts[x[0]], label=prefixes[0],
                               other_ts=df.ts[x[1]], otherlabel=prefixes[1],
                              check_localized=False, tscut=[] )
        else:
            v = df.ts
            if cut is not None: v=v[cut]
            fig = self.cumulative_ts(v, label=label if label is not None else prefixes[0], 
                    check_localized=False,)
        ax = plt.gca()
        plt.setp(ax, ylim=(1,1000), xlim=(9,100))
        leg =ax.legend()
        for patch in leg.get_patches():
            pbox = patch; pbox._height=0; pbox._y=5
        return fig
        
    def unassoc_seed_cumulative_ts(self):
        """ Cumulative TS distribution for seed sources that are not associated
        """
        return self.seed_cumulative_ts(cut=self.assoc.aprob<0.8, label='unassociated')
        
    def histo(self, ax, v, bins):
        ax.hist(v, bins)
        ts = np.array(self.df_good.ts)
        ax.hist(v[ts>10], bins, label='TS>10')
        ax.hist(v[ts>25], bins, label='TS>25')
        ax.legend(prop=dict(size=10))
        ax.grid()
    
    def localization(self):
        """ Localization results
        <br>Left: r95; right; delta TS
        """
        fig, ax = plt.subplots(1,3, figsize=(12,5))
        plt.subplots_adjust(left=0.1)
        df = self.df_good
        def r95(ax):
            v = 60.*df.r95; bins = np.linspace(0, 25,26)
            self.histo(ax, v[~pd.isnull(v)], bins)
            plt.setp(ax, xlabel='r95 (arcmin)')
        def delta_ts(ax):
            v = np.sqrt(list(df.delta_ts)); bins = np.linspace(0,10,26)
            self.histo(ax, v, bins)
            plt.setp(ax, xlabel='sqrt(delta_ts)', xlim=(0,3))
        def locqual(ax):
            self.histo(ax, df.locqual.clip(0,8), np.linspace(0,8,26))
            plt.setp(ax, xlabel='localization quality', xlim=(0,8))
        for f, a in zip((r95, delta_ts,locqual), ax.flatten()):
            f(a)
        return fig

    def spectral_parameters(self, ax=None, index_lim=(1.5,3.0), flux_lim=(0.05,10)):
        """ Spectral fit parameters
        Flux vs. spectral index for %(spectral_type)s fit
        <br>histograms of sin(glat) and sqrt(delta_ts) for all, TS>10, and TS>25
        """
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        df = self.df_good
        good = df.ts>10
        super = df.ts>25
        def flux_index(ax, ylim=index_lim, xlim=flux_lim):
            for cut, c,label in zip((good, super), ('.b', 'or'), ('TS>10', 'TS>25')):
                ax.plot(df.eflux[cut].clip(*xlim), df.pindex[cut].clip(*ylim), c, label=label)
            ax.grid()
            ax.legend(loc='lower right', prop=dict(size=10))
            plt.setp(ax, ylim=ylim, xlim=xlim, xscale='log', ylabel='spectral index', 
                xlabel='Energy flux [eV]')
        def singlat(ax):
            v = np.sin(np.radians(list(df.glat))); bins=np.linspace(-1,1,26)
            self.histo(ax, v, bins)
            plt.setp(ax, xlabel='sin(glat)')
        def skyplot(ax):
            glon = df.glon
            glon[glon>180]-=360
            ax.plot(glon, np.sin(np.radians(list(df.glat))), 'o')
            plt.setp(ax, xlim=(180,-180), xlabel='glon', ylabel='sin(glat)')
        def index_vs_cutoff(ax):
            cutoff = df.par2
            for tsmin, marker in zip((10,25), ('.b', 'or')):
                cut = df.ts>tsmin
                ax.plot(cutoff[cut], df.pindex[cut],  marker, label='TS>%d'%tsmin)
            plt.setp(ax, ylabel='spectral index', xlabel='cutoff', ylim=(0.5,3.0), xlim=(0, 3000))
            ax.grid(); ax.legend( prop=dict(size=10))
        for f, a in zip((flux_index,  singlat,), ax.flatten()):
            f(a)
            
        return fig

    def locations(self, vmin=10, vmax=50):
        """ Positions
        Locations of the good candidates. Colors show TS value.
        """
        fig, ax = plt.subplots(figsize=(10,8))
        return self.skyplot(self.df_good.ts, ax=ax, vmin=vmin, vmax=vmax, cbtext='TS', s=50)
    
    def seed_list(self):
        """ Results of analysis of seeds
        %(info)s
        <p>A <a href="../../%(csv_file)s?download=true">csv file</a> is also available.
        """
        cols="""ra	dec	glon glat		ts		eflux	pindex r95	
            delta_ts locqual aprob acat aname index""".split()
        t = self.df_good[cols].sort_index(by='index')
        good_seeds = 'good_seeds.csv'
        self.csv_file = os.path.join(self.plotfolder,good_seeds)
        t.to_csv(self.csv_file)
        print 'wrote list that succeeded to %s' % self.csv_file
        filename = 'good_seeds.html'
        html_file = self.plotfolder+'/%s' % filename
        ## FIX LATER htmldoc = diagnostics.html_table(t, float_format=diagnostics.FloatFormat(2))
        open(html_file,'w').write('<head>\n'+ _html.style + '</head>\n<body>'+ htmldoc+'\n</body>')

        self.info = self.df_good['ts eflux pindex r95 locqual aprob'.split()].describe().to_html(float_format=FloatFormat(2)).replace('%', '%%')
        self.info += '<p><a href="%s?skipDecoration">Table of %d seeds</a> '% (filename, len(t))
        if len(self.assoc_sum)>0:
            self.info += '<p>Association summary for good seeds' 
            t = pd.DataFrame(self.assoc_sum, columns='catalog associations'.split()).sort_index(by='catalog')
            self.info += t.to_html(index=False)
            # Make a summary of the AGN types 
            ta=self.assoc[self.assoc.aprob>0.8]
            bznames = np.array(ta.aname[ta.acat=='bzcat'])
            if len(bznames!=0):
                self.info+='<p>BZCAT AGN type summary'
                bztypes = set([name[3] for name in bznames])
                td = dict()       
                for t in bztypes:
                    td[t]= 0
                for n in bznames:
                    t = n[3]
                    td[t] +=1
                self.info += pd.DataFrame(td.items(), columns='type count'.split()).to_html(index=False)
        else:
            self.info += '<p>No associations found'

    def all_plots(self):
        self.runfigures(
            [
                self.template_info,
                self.seed_plots,
                self.tsmaps,
                self.pixel_ts_distribution,
                #self.seed_list, self.seed_cumulative_ts, self.locations, 
                # self.spectral_parameters, 
                # self.localization,
            ])
