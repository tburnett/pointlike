"""
Comparison with a gtlike catalog output


"""

import os, glob
from collections import OrderedDict
import astropy.io.fits as pyfits
from astropy.table import Table
import numpy as np
import pylab as plt
import pandas as pd
from scipy import stats

from skymaps import SkyDir
from . import sourceinfo

class SourceComparison(sourceinfo.SourceInfo):
    """<p>Comparison with a gtlike analysis

    <p>Catalog name: %(catname)s
    """
    def setup(self, **kw):
        plt.style.use('seaborn-bright')
        catname = kw.pop('catname', 'psc8')#'3FGL')
        cat = kw.pop('cat','*pscP305*') #''3FGL-v13r3_v6r9p1_3lacv12p1_v7.fits', )
        self.catname=catname

        if cat[0]!='/':
            cat = os.path.expandvars('$FERMI/catalog/'+cat)
        if cat.find('*')!=-1:
            check = glob.glob(cat)
            if len(check)>0:
                cat=check[-1]
            else:
                raise Exception('Found no files using pattern {}'.format(cat))
        assert os.path.exists(cat), 'Did not find file %s' %cat
        ft = pyfits.open(cat)[1].data
        self.ft=ft # temp
        print 'loaded gtlike file %s with %d entries' % (cat, len(ft))
        
        super(SourceComparison, self).setup(**kw)
        self.plotfolder='comparison_%s' % catname
        if not os.path.exists('plots/'+self.plotfolder):
            os.mkdir('plots/'+self.plotfolder)
 
        self.gtlike_info=OrderedDict(
            [('filename',cat.split('/')[-1]), ('entries', len(ft)),
            ])
        self.setup_cat()

    def setup_cat(self):
        ft=self.ft
        fieldnames = ft.dtype.fields.keys()
        id_check = map(lambda n:n.startswith('ID_Prob'), fieldnames)
        if sum(id_check)==0:
            print 'warning: ID_Probability field not found' 
            id_prob = [np.nan]*len(ft)
        else:
            id_name = fieldnames[np.arange(len(fieldnames))[id_check][0]]
            print 'Using prob from field {}'.format(id_name)
            id_prob = ft[id_name]

        cat_skydirs = map (lambda x,y: SkyDir(float(x),float(y)), 
                           ft.RAJ2000, ft.DEJ2000)
        
        glat = [s.b() for s in cat_skydirs]
        glon = [s.l() for s in cat_skydirs]
        truncnames = [n.replace(' ','') for n in self.df.index]
        not_found=[]
        def nickfix(n):
            # put blanks into index
            if n in truncnames:
                return self.df.index[truncnames.index(n)]
            not_found.append(n)
            return n
        nicknames = ft.NickName_3FGL if 'NickName_3FGL' in ft.dtype.fields else ft.NickName
        sourcenames = ft.Source_Name if 'Source_Name' in ft.dtype.fields else nicknames
        index = map(nickfix, nicknames) #Source_Name 
        if len(not_found)>0:
            print '{} entries not found in {}, names starting with {}'.format(len(not_found), 
                self.skymodel, list(set(map(lambda n:n[:4], not_found))))
        self.not_found= not_found
        self.gtlike_info['missing'] = len(not_found)
        flags = np.asarray(ft.Flags_3FGL,int) if 'Flags_3FGL' in ft.dtype.fields else [0]*len(ft)
        self.cat = pd.DataFrame(dict(namec=sourcenames, #_3FGL_1, 
                #nickname=nicknames, #map(nickfix, nicknames), 
                ra=ft.RAJ2000,dec= ft.DEJ2000, 
                ts=ft.Test_Statistic,
                pindex=ft.PL_Index,
                unc_pindex =ft.Unc_PL_Index,
                beta=ft.LP_beta,
                pivot=ft.Pivot_Energy,
                flux = ft.Flux_Density,
                unc_flux = ft.Unc_Flux_Density, 
                #skydir=cat_skydirs,
                glat=glat, glon=glon, 
                #pivot=ft.Pivot_Energy, flux=ft.Flux_Density, 
                #modelname=ft.SpectrumType, 
                id_prob=id_prob,
                a95=ft.Conf_95_SemiMajor, 
                b95=ft.Conf_95_SemiMinor, 
                ang95=ft.Conf_95_PosAng,
		#ROI_num = ft.ROI_num,
		#ROI_dist = ft.ROI_dist,
                 ), 
             index=index, )
        self.cat.index.name='name'
        self.cat['eflux'] = self.cat.flux * self.cat['pivot']**2 * 1e6
        # set values for corresponding source model
        self.cat['pt_ts'] = self.df.ts
        self.cat['pt_ra'] = self.df.ra
        self.cat['pt_dec'] = self.df.dec
        self.cat['pt_beta']=self.df.beta
        self.cat['pt_index']=self.df.pindex
        self.cat['pt_index_unc']=self.df.pindex_unc
        self.cat['pt_eflux'] = self.df.eflux
        try:
            self.cat['pt_eflux_unc']=self.df.eflux_unc
        except:
            print 'No eflux_unc in input.'
            self.cat['pt_eflux_unc'] = np.nan
        self.cat['pt_pivot'] =self.df.pivot_energy
        self.cat['ispsr']= map(lambda name:name.startswith('PSR'), self.cat.index)
        self.cat['ismissing'] = [name in not_found for name in self.cat.index]

        extended_names = self.df.query('isextended==True').index
        self.cat['isextended'] = [name in extended_names for name in self.cat.index]
        self.gtlike_info['pulsars'] = sum(self.cat.ispsr)
        self.gtlike_info['extended']=sum(self.cat.isextended)

        print 'Found {} extended sources, {} pulsars'.format(sum(self.cat.isextended), sum(self.cat.ispsr))
         
        def find_close(A,B):
            """ helper function: make a DataFrame with A index containg
            columns of the
            name of the closest entry in B, and its distance
            A, B : DataFrame objects each with a skydir column
            """
            def mindist(a):
                d = map(a.difference, B.skydir.values)
                n = np.argmin(d)
                return (B.index[n], np.degrees(d[n]))
            return pd.DataFrame( map(mindist,  A.skydir.values),
                index=A.index, columns=('otherid','distance'))
                
        if self.catname=='2FGL' or self.catname=='3FGL':
            print 'generating closest distance to catalog "%s"' % self.catname
            closedf= find_close(self.df, self.cat)
            self.df['closest']= closedf['distance']
            self.df['close_name']=closedf.otherid
            closedf.to_csv(os.path.join('plots', self.plotfolder,
                                       'comparison_%s.csv'%self.catname))
            closest2 = np.degrees(np.array([min(map(sdir.difference, 
                     self.df.skydir.values)) for sdir in cat_skydirs]))
            self.cat['closest']= closest2
        elif self.catname=='psc8':
            self.cat['closest']=0

    def roi_info(self):
        """ROI information
            Check number of ROIs and sources per ROI
            %(gtlke_info_html)s

        """
        ft=self.ft
        nroi = max(ft.ROI_num); print 'found {} ROIs'.format(nroi)
        occ=np.histogram(ft.ROI_num,np.linspace(1,nroi+1,nroi+1) )[0]
        #occ.min(), occ.mean(), occ.max()
        fig, axx = plt.subplots(1,2, figsize=(10,4))
        ax=axx[0]
        ax.hist(occ, np.linspace(0.5,10.5,11), histtype='step', lw=2);
        ax.set_xlabel('Sources/RoI')
        #ax.hist(ft.ROI_num, np.linspace(1,nroi+1,nroi+1), histtype='step');
        ax=axx[1]
        ax.hist(ft.ROI_dist, np.linspace(0,5,26), histtype='step', lw=2);
        ax.set_xlabel('ROI distance')
        ax.set_xlim((0,5));
        
        s ='<p><dl>'
        self.gtlike_info['ROIs'] = nroi
        for k,v in self.gtlike_info.iteritems():
             s += '\n<dt>{}</dt> <dd>{}</dd>'.format(k,v)
        self.gtlke_info_html = s+ '</dl>'

        return fig
    
    def fit_comparison(self, select=[]): #select=['P88Y2704', 'P88Y0282', 'P88Y1919']):
        """Comparison of fit parameters

        """
        self.same=same = self.cat.query('ismissing==False and ispsr==False').copy()
        same['unc_eflux'] = same.unc_flux * same.eflux/same.flux
        self.same_count = len(same)
        print 'Comparing {} sources with LP fits'.format(self.same_count)
        fig, axx = plt.subplots(3,2, figsize=(15,10), sharex=True, sharey=True)
        plt.subplots_adjust(hspace=0.,wspace=0, top=0.96)
        good_sel = filter( lambda x: x in same.index, select)
        if len(good_sel)>0:
            print 'selecting sources {}'.format(good_sel)
        ratio_plots = OrderedDict([
            ('TS',        same.pt_ts/same.ts), 
            ('pivot',     same.pt_pivot/same['pivot']),
            ('eflux',     same.pt_eflux/same.eflux),
            ('eflux_unc', same.pt_eflux_unc/same.unc_eflux),
            ('index',     same.pt_index/same.pindex),
            ('index_unc', same.pt_index_unc/same.unc_pindex),
            ])
        for ax, title, ratio, in zip(axx.flatten(), ratio_plots.keys() ,ratio_plots.values()): 
            print title, ratio.mean()
            ax.semilogx(same.ts, ratio, '.')
            for i,sel in enumerate(good_sel):
                ax.semilogx(same.ts[sel], ratio[sel], color='red',marker='*v^<>1234sp'[i], 
                    markersize=15, label=sel)
            ax.set_ylim((0., 1.95));
            ax.axhline(1.0, ls= '--', color='orange');
            ax.text(8e4, 1.7, title, size=14, ha='right')
        ax.set_xlabel("TS");
        ax.set_xlim((20,1e5))
        fig.suptitle('Pointlike/gtlike ratios', size=16)
        axx[0,0].legend(loc='lower right', prop=dict(size=10,family='monospace'))
        return fig

    def missing_plots(self):
        """
        Gtlike sources not in current model

        Left: TS values
        Right: locations
        <p>The names are not in the model: newer seeds may be close, however.

        %(missing_table)s
        """
        missing = self.cat.query('ismissing==True')['ts ra dec glat glon pindex flux'.split()]

        fig,axx = plt.subplots(1,2, figsize=(12,5))
        ax = axx[0]
        ax.hist(missing.ts, np.logspace(1,3), histtype='step')
        ax.set_xscale('log');
        ax=axx[1]
        self.skyplot(missing.ts, ax=ax, df=missing, colorbar=False);
        return fig

    def history_check(self):
        """use the history to get the set of 3FGL names of sources
        that were not direct ancestors. Only applies to P7R4 sources
        """
        # the guys with 3FGL ancestors
        in3fgl = [x is not None for x in self.df.cat3fgl]
        t = self.df.cat3fgl[in3fgl].values
        inset = set([x.namec for x in t])

        #Make the set of all 3FGL names 
        fullset = set(self.cat.namec); 
        # lost: the difference
        lost = fullset.difference(inset);
        missing = [n in lost for n in self.cat.namec]
        missing_cat = self.cat[missing]
        really_missing=  [x.startswith('P7R4') for x in missing_cat.nickname]
        truly_missing = missing_cat[really_missing]
        return truly_missing
    
    def distance_to_cat(self, maxdist=0.5, tscuts=[10,50,500], nbins=26):
        """Associations of sources with 3FGL
        
        """
        fig,ax = plt.subplots( figsize=(5,5))
        hist_kw = dict(bins=np.linspace(0,maxdist,nbins), log=True,
                    histtype='step', lw=2)
        for tscut in tscuts:
            ax.hist(self.df.closest[self.df.ts>tscut].clip(0,maxdist),
                    label='TS>%d'%tscut, **hist_kw)
        ax.grid(True, alpha=0.5)
        ax.legend(prop=dict(size=10))
        plt.setp(ax, xlabel='closest distance to %s source'%self.catname,
                ylim=(0.8,None), )
        return fig
    
    def lost_plots(self, close_cut=0.25, minassocprob=0.8, maxts=250):
        """Catalog sources not present in new list
        Histogram of the catalog TS and Galactic latitude for those sources
        more than %(close_cut).2f deg from a skymodel source. 
        The subset of sources with associations (prob>%(minassocprob)s) is shown.
        <br>
        Left: Distribution vs. TS.<br>
        Right: Distribution vs sine of Galactic latitude.
        """ 
        self.minassocprob=minassocprob
        self.close_cut = close_cut
        fig,axx = plt.subplots(1,2, figsize=(8,4))
        self.lost = self.cat.closest>close_cut
        print '%d sources from %s further than %.2f deg: consider lost' % (sum(self.lost) , self.catname, close_cut )
        self.cat.ix[self.lost].to_csv(os.path.join(self.plotfolder,'3fgl_lost.csv'))
        print '\twrite to file "%s"' % os.path.join(self.plotfolder,'3fgl_lost.csv')
        lost_assoc = self.lost & (self.cat.id_prob>0.8)

        def left(ax):
            space = np.linspace(0,maxts,21)
            ax.hist(self.cat.ts[self.lost].clip(0,maxts), space, 
                    label='all (%d)'%sum(self.lost))
            ax.hist(self.cat.ts[lost_assoc].clip(0,maxts), space, 
                    color='orange', label='associated(%d)' %sum(lost_assoc) )
            ax.legend(prop=dict(size=10))
            ax.grid()
            plt.setp(ax, xlabel='TS of %s source' %self.catname)

        def right(ax):
            space = np.linspace(-1,1,51)
            singlat = np.sin(np.radians(self.cat.glat))
            ax.hist(singlat[self.lost], space, 
                    label='all (%d)'%sum(self.lost))
            #lost_assoc = self.lost & (self.cat.id_prob>0.8)
            ax.hist(singlat[lost_assoc], space, color='orange', 
                    label='associated(%d)' %sum(lost_assoc) )
            ax.legend(prop=dict(size=10))
            ax.grid()
            plt.setp(ax, xlabel='sin(glat) of %s source' 
                     %self.catname, xlim=(-1,1))
            return fig
        for f, ax in zip((left,right), axx.flatten()): 
            f(ax)
        return fig
        
    def poorly_localized(self):
        pass
    
    def properties_of_missing(self):
        """Properties of the missing 3FGL sources
        This defines missing from the history, looking at the %(number_lost)d P7R4 sources 
        used to start the 6-year list, which were used in 3FGL, but did not 
        survive to the end
        """
        lost = self.history_check()
        self.number_lost=len(lost)
        maxdist, nbins= 2.6, 53
        fig,axx = plt.subplots(2,1, figsize=(8,12))
        ax=axx[0]
        hist_kw = dict(bins=np.linspace(0,maxdist,nbins), log=True,
                    histtype='step', lw=2)
        ax.hist(lost.closest, **hist_kw)
        plt.setp(ax, xlabel='Closest distance to 6-year source', 
                 ylim=(0.8,None),title='3FGL sources missing from 6-year list')
        ax.grid(True, alpha=0.5)

        ax = axx[1]
        maxdist, nbins= 200, 53
        hist_kw = dict(bins=np.logspace(1,3,nbins), log=False,
                    histtype='step', lw=2)
        ax.hist(lost.ts.clip(0,1000), **hist_kw)
        plt.setp(ax, xlabel='TS', ylim=(0,None),xscale='log', xlim=(25,1000))
        ax.grid(True, alpha=0.5)
        return fig
        
    def all_plots(self):
        self.runfigures([ 
            self.roi_info,
            self.fit_comparison,
            #self.distance_to_cat, 
            #self.lost_plots,
            #self.properties_of_missing,
            ])
        
    def lookup_3fgl(self, namec):
        if namec[-1]!=' ' and namec[-1]!='c': namec=namec+' '
        fglnames = list(self.cat.namec)
        try:
            i = fglnames.index(namec)
            nick = self.cat.ix[i].name
            j=  list(self.df.close_name).index(nick)
            return self.df.ix[j]
        except Exception, msg:
            print 'Source %s not found (%s)' % (namec, msg)
            return None

class CompareSimulation(object):
    """Plots for comparison of uw8011t with FL8Y
    """
    def __init__(self, fl8y_file='gll_pscP305uw8011_v2.fit', uwmodel='uw8607', sim='uw8607s2'):
        os.chdir(os.path.expandvars('$FERMI/skymodels/P305_8years/{}'.format(sim)))
        self.uwmodel=uwmodel
        self.sim = sim
        dfm = pd.read_csv('sources_{}.csv'.format(sim), index_col=0); 
        print '{}:'.format(sim) ,len(dfm)
        dfa = pd.read_csv('../{0}/sources_{0}.csv'.format(uwmodel), index_col=0); 
        print '{}:'.format(uwmodel),len(dfa)
        self.dfm=dfm
        self.dfa=dfa
        def set_glat(df):
            from skymaps import SkyDir
            sds = map(SkyDir, df.ra, df.dec);
            glat = map(lambda x: x.b(), sds)
            df['glat'] = glat
            df['hilat'] = (df.glat<-5) | (df.glat>5)
        set_glat(dfa); 
        set_glat(dfm)

        #return
        # tag MC seed sources
        def seedcheck(name):
            if name.find( sim[-2:] )>0:
                return 'MC seed'
            else: return 'fit source'
        grouped =dfm.groupby([seedcheck])
        self.mc_seed= grouped.get_group('MC seed')

        # load gll version of FL8Y 
        file = os.path.expandvars('$FERMI/catalog/'+fl8y_file) 
        df_fl8y=Table.read(file, hdu=1).to_pandas()
        print 'FL8Y: {} w/ {} sources'.format(fl8y_file, len(df_fl8y))
        df_fl8y.index=df_fl8y.NickName
        del df_fl8y['NickName'];

        dfa['in_fl8y']=[name in df_fl8y.index for name in dfa.index]


    def scat_ts_pindex(self,):
        dfa, mc_seed = self.dfa, self.mc_seed
        def plotit(df, color, ax=None, title=None):
            if ax is None:
                fig, ax = plt.subplots(figsize=(8,8))
            else: fig=ax.figure
            ax.plot(df.ts, df.pindex.clip(0.5,3.5), '.', color=color);
            ax.axvline(25, ls=':', color='red')
            ax.set(xlim=(10,40), ylim=(0.5,3.5));
            ax.set(xlabel='TS', ylabel=r'$\Gamma$')
            ax.axhline(2.8, ls=':', color='red')
            ax.axhline(1.5, ls=':', color='red')
            if title is not None: ax.set_title(title)

        fig,axx = plt.subplots(2,2, figsize=(12,12), sharex=True, sharey=True)
        axf = axx.flatten()
        plotit(dfa, 'green', ax=axf[0], title=self.uwmodel)
        plotit(dfa.query('in_fl8y==False'), 'grey', ax=axf[2], title=self.uwmodel+' not in FL8Y')
        plotit(mc_seed, 'orange', ax=axf[3], title='MC seeds')
        plotit(dfa.query('in_fl8y==True'), 'blue', ax=axf[1], title='FL8Y')

        return fig

    def purity_plots(self, cut=None, tsbins=(16, 40, 13), ylim=(50,105)):
        mc_seed, dfa = self.mc_seed, self.dfa
    
        fig, axx =plt.subplots(3,2, figsize=(10,12), sharex=True,)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        hkw=dict(bins=np.linspace(*tsbins), histtype='step',log=True, lw=2)

        for i, cut, label in zip(range(3), [None,'pindex<1.6', '3.5>pindex>2.8'], ['All', 'Hard', 'Soft']):
            # get the MC group
            if cut is None:
                mc_ts=mc_seed.ts
                data_ts=dfa.ts
            else:
                mc_ts=mc_seed.query(cut).ts
                data_ts=dfa.query(cut).ts

            ax=axx[i,0]
            ax.text(0.1,0.9, label, fontsize=16, transform=ax.transAxes)
            tslim=tsbins[:2]
            ax.hist(data_ts.clip(*tslim), label=self.uwmodel,color='green', **hkw);
            ax.hist(mc_ts.clip(*tslim),label='MC seed', color='orange', **hkw);
            ax.legend();
            ax.grid(alpha=0.5)
            ax.set(xlim=tslim, ylim=(0.8,None), xlabel='TS');
            ax.axvline(25, ls=':', color='red')

            ax=axx[i,1]
            bins=hkw['bins']
            mc = np.histogram(mc_ts,bins=bins)[0]
            data = np.array(np.histogram(data_ts, bins=bins)[0], float)
            purity = 1-(mc/data).clip(0,1)
            delta=bins[1]-bins[0]
            t=bins[:-1]+0.5*delta
            yerr=100.*np.sqrt(mc*(mc+data)/data**3)
            ax.errorbar(x=t,y=purity*100, xerr=delta/2,yerr=yerr, fmt='o', marker='o')
            ax.set(xlabel='TS',  ylim=ylim)
            ax.set_ylabel(ylabel='purity [%]',fontsize=14)
 
            ax.axvline(25, ls=':', color='red')
            ax.grid(alpha=0.5);
            
        #fig.suptitle(suptitle, fontsize=14)
        return fig

    def latitude_dependence(self, cut='pindex<3.4 and locqual<10 and delta_ts<4'):
        mc_seed, dfa = self.mc_seed, self.dfa
        dfa_cut = dfa.query(cut)
        mc_cut = mc_seed.query(cut)
        titles = (self.uwmodel, self.uwmodel+ ' sources not in FL8Y', 'MC seed sources')

        fig,axx = plt.subplots(len(titles),1, figsize=(8,8), sharex=True, sharey=True)
        hkw = dict(bins=np.logspace(1,3,26), log=True, histtype='step', lw=2)
        for df,ax,title in zip((dfa_cut, dfa_cut[~dfa_cut.in_fl8y], mc_cut), axx.flatten(),titles):
            ax.hist(df[df.hilat].ts.clip(10,1e3), label='high lat', **hkw);
            ax.hist(df[~df.hilat].ts.clip(10,1e3), label='low lat', **hkw)
            ax.set(xlabel='TS', xscale='log', ylim=(0.8,None), title=title)
            ax.grid(alpha=0.5)
            ax.legend();  
        return fig

    def spectral_check(self, ):
        """Compare spectral parameters to check uncertainties
        """
        a, b = self.dfa, self.dfm.copy()
        b['ts_a']=a.ts
        b['flux_a'] = a.flux
        b['dflux'] = (b.flux-b.flux_a)/b.flux_unc
        b['eflux100_a'] = a.eflux100
        b['deflux'] = (b.eflux100-b.eflux100_a)/b.eflux100_unc
        b['pindex_a'] = a.pindex
        b['gdelta'] = (b.pindex-b.pindex_a)/b.pindex_unc

        fig,axx = plt.subplots(1,2, figsize=(12,6), sharey=True)
        hkw = dict(bins=np.linspace(-5,5,26), histtype='step', lw=2, density=True)

        cut =  (b.ts>50) & ~pd.isnull(b.deflux) & ~pd.isnull(b.gdelta) &\
                    (b.modelname=="LogParabola") & (b.pindex<3) & (b.e0>500) &(b.eflux100_unc>0)
        for ax, title, val in zip(axx.flatten(), ['Energy Flux', 'Spectral index'], [b.deflux, b.gdelta]):    

            df=val[cut]
            ax.hist(df.clip(-5,5), label='mean {:.2f} std {:.2f}'.format(df.mean(),df.std()), **hkw);
            ax.grid(alpha=0.5); 
            x=np.linspace(-4,4)
            ax.plot(x, stats.norm.pdf(x), '--g' );
            ax.set(xlabel='normalized fit deviation', title=title, );
        fig.suptitle('Normalized devations of fit from model', fontsize=16);

        return fig
