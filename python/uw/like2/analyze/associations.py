"""
Association analysis

"""
import os, glob, sys, warnings
import numpy as np
import pylab as plt
import pandas as pd
from astropy.io import fits as pyfits
from astropy.utils.exceptions import AstropyUserWarning

from skymaps import SkyDir, Band
from . import sourceinfo, fermi_catalog
from . analysis_base import FloatFormat, html_table


class Associations(sourceinfo.SourceInfo):
    """Analysis of associations
    <p>
    A feature of the UW pipeline is the application of the LAT association algorithm, with the same catalogs as
    were used for 2FGL, except:
    <ul> 
        <li>The latest LAT pulsar list</li>
        <li>All pulsars from the "bigfile" list maintained by D. Smith</li> 
        <li>The latest BZCAT list of blazars
    </ul>
    <p>Note that priors are not recalculated, we use the values determined for 2FGL. 
    The method is explained in the <a href="http://arxiv.org/pdf/1002.2280v1.pdf">1FGL paper</a>, see Appendix B.
    <p>
    <p>Also, the current pipeline has the Planck and WISE catalogs.
    """
    
    def setup(self, **kw):
        self.args = kw.pop('args', None)
        self.quiet = kw.get('quiet', False)
        super(Associations, self).setup(**kw)
        self.plotfolder='associations'
        df = kw.get('df', None)
        if df is not None:
            self.df = df
            print 'Associations: set subset, %d sources, to report on.' % len(df)
        self.load_assoc(self.args)
        # these suppresses warnings associated with headers
        warnings.filterwarnings('ignore', category=pyfits.verify.VerifyWarning, append=True)
        warnings.filterwarnings('ignore', category=AstropyUserWarning, append=True)
        plt.rc('font', size=14)
    
    def load_assoc(self, fromdf=None, minprob=0.8):
        if fromdf is not None:
            if not self.quiet:
                print 'loading associations from file %s' %fromdf
            df['altassoc']=pd.load(fromdf)
        else: 
            if not self.quiet:
                print 'using associations found in sourceinfo'
            df = self.df.copy()
        associations = df.associations if fromdf is None else self.df.altassoc
        probfun = lambda x: x['prob'][0] if not pd.isnull(x) else 0
        
        df['aprob'] = np.array([ probfun(assoc) for  assoc in associations])
        df['acat']  = np.array([ assoc['cat'][0] if not pd.isnull(assoc) else 'unid' for  assoc in associations])
        df['aname'] = np.array([ assoc['name'][0] if not pd.isnull(assoc) else 'unid' for  assoc in associations])
        df['aang']  = np.array([ assoc['ang'][0] if not pd.isnull(assoc) else np.nan for  assoc in associations])
        df['adeltats'] = np.array([assoc['deltats'][0] if not pd.isnull(assoc) else np.nan for assoc in associations])
        # make selection requiring association probabliliy
        total=len(df)
        self.indf=df
        self.df = df[(df.aprob>minprob) | (df.psr)]
        self.df10 = self.df.loc[self.df.ts>10]
        if not self.quiet:
            print 'associated: %d/%d' % (sum(self.df10.aprob>0.8), total)
        
    def association_vs_ts(self, aprob_min=0.5):
        """ Associations vs. TS
        
        <br>Left: histogram of TS, showing the fraction that have associations.
        <br>Right: The fractions themselves.
        """
        ts = np.array(self.indf.ts,float) #self.df10.ts
        lowlat = np.abs(self.indf.glat)<5
        assoc = self.indf.aprob>aprob_min
        def plota(ax, bins=np.logspace(1,5,41) ):
            histkw = dict(bins=bins, histtype='step', lw=2)
            ax.hist(ts, label='all sources', **histkw)
            ax.hist(ts[assoc], color='orange', label='associated', **histkw)
            plt.setp(ax, xscale='log', xlabel='TS', xlim=(10,1e4))
            ax.legend(prop=dict(size=10)); ax.grid()
        def plotb(ax, bins=np.logspace(1,4.0,12)):
            for tsvals, label,color in zip( (self.indf[~lowlat].ts, self.indf[lowlat].ts), 
                    ('|b|>5', '|b|<5'), ('blue','red')):
                all = np.array(np.histogram(tsvals, bins)[0],float)
                subset = np.histogram(tsvals[assoc], bins)[0]
                fraction = (subset/all)
                x = np.sqrt(bins[:-1]*bins[1:])
                yerr = np.sqrt(subset*(all-subset)/all )/all
                xerr = [x-bins[:-1], bins[1:]-x]
                ax.errorbar(x=x, y=fraction,xerr=xerr, yerr=yerr, fmt= 'o', color=color, label=label)
            plt.setp(ax, xscale='log', xlim=(bins[0],bins[-1]), ylim=(0,1), xlabel='TS', ylabel='associated fraction')
            ax.grid()
            ax.legend(loc='upper left', prop=dict(size=10))
            
        fig, axx = plt.subplots(1,2, figsize=(12,5))
        plt.subplots_adjust(left=0.1)
        for f,ax in zip((plota,plotb), axx.flatten()): f(ax) 
        return fig           
    
    def delta_ts_figure(self, df=None, title='', ax=None, ):
        """Delta ts for association
        
        """
        if df is None: df=self.df
        
        deltats = df.adeltats if 'adeltats' in df.columns else df.deltats
        
        if ax is None:
            fig,ax = plt.subplots(figsize=(5,5))
        else: fig = ax.figure
        hist_kw=dict(log=True, histtype='step',color='grey', lw=2)
        binsize=0.5; maxts=12
        bins = np.linspace(0,maxts,maxts/binsize+1)

        ax.hist( deltats.clip(0,12), bins,   label='all', **hist_kw)
        ax.hist( deltats[df.ts<25].clip(0,12), bins,  color='orange',
                label='TS<25',  **hist_kw)
        ax.plot(bins, len(deltats)*binsize*0.5*np.exp(-bins/2),ls='--', color='r', label='expected')

        plt.setp(ax, xlabel='delta TS', ylim=(1,1000), title=title)
        ax.grid(True, alpha=0.5); 
        leg=ax.legend()
        for box in leg.get_patches():
            box._height=0; box._y=0.5
        return fig

    def summary(self):
        """Summary
        %(summary_html)s
        """
        # load the catalogs used
        srcid_path=sys.path[0] = os.path.expandvars('$FERMI/catalog/srcid/')
        import classes
        cats = dict()
        for module_name in classes.__all__:
            cd = dict()
            # this is equivalent to "from classes import module_name"
            module = __import__('classes.'+module_name,  fromlist=['classes'])

            for var in 'catid catname prob_prior prob_thres figure_of_merit max_counterparts new_quantity selection'.split():
                cd[var] = module.__dict__[var] 
            cats[module_name]= cd
            if cd['catname'].find('*')>0:
                try:
                    t = glob.glob(srcid_path+'/cat/'+cd['catname'])[-1]
                    cd['catname']=os.path.split(t)[-1]
                except:
                    print 'File %s not found' %s
            
        self.catdf = pd.DataFrame(cats).T
        self.catdf['objects']=[len(pyfits.open(srcid_path+'cat/'+fname)[1].data) for fname in self.catdf.catname]
        # make dict of catnames with values the number of associations
        t = dict()
        for c in self.df10.acat:
            if c not in t: t[c]=1
            else: t[c]+=1
        
        self.catdf['associations'] = pd.DataFrame(t.items(), index=t.keys(), columns='name associations'.split())['associations']
        
        self.summary_html = html_table(
            self.catdf[~pd.isnull(self.catdf.associations)]['objects associations prob_prior prob_thres catname'.split()],
            dict(objects='Objects,number of entries in table',
                 prob_prior='Prior,prior probability', prob_thres='Threshold,threshold probability',
                 catname='Catalog,name of the FITS file in the folder %s'%(srcid_path+'/cat')),
            heading='<br>Table of catalogs with associations', name=self.plotfolder+'/associated_catalogs', 
            href=False, maxlines=100)
                 
    def bzcat_study(self, tsmax=10000, title='Integral TS distribution bzcat associations'):
        """BZCAT associations
        Study the AGN associations with a BZCAT entry.
        %(bzcat_html)s
        """
        df = self.df
        assoc = df.aprob>0.8; sum(assoc)
        dfa = df[assoc]; 
        agn = [ n in ('crates bzcat bllac agn cgrabs').split() for n in dfa.acat]; 
        print 'agns:', sum(agn)
        dfagn = dfa[agn]
        ###Look for cases where the second or third association is above 0.8, and is 'bzcat'
        test=[-1]* len(dfagn)
 
        # make a DataFrame of all BZCAT associations if the prob is >0.8
        bzdict = dict()
        for i,a in enumerate(dfagn.associations):
            prob = a['prob']
            for j,p in enumerate(prob):
                if p>0.8 and a['cat'][j]=='bzcat':
                    test[i]=j
                    bzdict[a['name'][j]] = \
                        dict(ts=dfagn.ts[i], 
                            sname=dfagn.index[i],
                            ra=a['ra'][j], dec=a['dec'][j],
                            deltats=a['deltats'][j],
                            locqual=dfagn.locqual[i],
                            ang=a['ang'][j],
                            pindex=dfagn.pindex[i],
                            # pivot=dfagn.pivot[i],
                            # curvature=dfagn.curvature[i],
                            )
        print 'bzdict length:', len(bzdict)
        self.bzdf = bzdf = pd.DataFrame(bzdict).T 
        bzdf_ts = np.array(bzdf.ts, float)       
        t=np.asarray(test)
        u =[sum(t==k) for k in range(-1,4)] ; print u
        self.bzcat_html= '<p>There is a BZCAT association in all but %d out of %d AGN associations' % (u[0], sum(u))
        bzdf['type'] = [n[3] for n in bzdf.index]
        type_names=dict(B='BL Lac', G='Radio Galaxy', U=None, Q='FSRQ', i=None)        
        
        # make a table of Blazar types, with and without TS cut
        types=filter(lambda x: x is not None, set(bzdf.type))
        tc_all = [sum(bzdf.type==type) for type in types]
        tc_25  = [sum((bzdf.type==type) & (bzdf_ts>25)) for type in types]
        rows=[tc_25, tc_all]
        dfT=pd.DataFrame(rows, index=['TS>25','all'], columns=[type_names[n] for n in types])
        dfT['total'] = [sum(x) for x in rows]

        self.bzcat_html += '<p>Frequencies by Blazar type: {}'.format(dfT.to_html())
        
        if len(bzdict)==0:
            print 'No BZCAT associations found'
            self.bzcat_html += '<p>No BZCAT associations: quitting'
            return fig

        fig, axx = plt.subplots(1,2, figsize=(16,5))
         
        hist_args=dict(bins=np.logspace(1,np.log10(tsmax),501),
            cumulative=-1, lw=2, histtype='step', log=True)
        #ax.hist(bzdf_ts,  label='all', **hist_args );

        # make an integral logTS plot 
        ax = axx[0]
        for type,label in zip('QBG', ['FSRQ', 'BL Lac', 'Galaxy']):
            sel = bzdf.type==type
            if sum(sel)>0:
                try:
                    ax.hist(bzdf_ts[sel],  label=label, **hist_args)
                except: pass
        plt.setp(ax, xscale='log', xlabel='TS', ylim=(1,None), xlim=(10, tsmax),
                title=title,)
        ax.grid(True, alpha=0.8);
        ax.axvline(25, color='k', ls='--', label='TS=25')
        leg =ax.legend()
        for patch in leg.get_patches():
            pbox = patch; pbox._height=0; pbox._y=5

        # check photon index distribution
        ax = axx[1]
        gt = self.bzdf.groupby('type')
        for name, group in gt:
            if type_names[name] is None: continue
            ax.hist(np.array(group.pindex,float), np.linspace(1.0, 3.5, 31), 
                histtype='step', lw=3, label=type_names[name])
        ax.legend();
        ax.set(title='Photon index by AGN type', xlabel='Photon spectral index');

        # save a summary file
        print 'Writing bzcat summary to %s ' %(self.plotfolder+'/bzcat_summary.csv')
        bzdf.to_csv(self.plotfolder+'/bzcat_summary.csv')
        return fig
        
    def no_pulsar_check(self):
        """Pulsar checks
        Pulsar analyses moved: see 
         <br>See <a href="../pulsars/index.html?skipDecoration"> pulsars</a> for analysis that was here.
        """
        return None


    def pulsar_check(self):
        """LAT pulsar check
        %(atable)s
        """
        # compare with LAT pulsar catalog     
        tt = set(self.df.name[self.df.psr])
        self.pulsar_lat_catname = sorted(glob.glob(os.path.expandvars('$FERMI/catalog/srcid/cat/obj-pulsar-lat_v1*')))[-1]
        print 'opening LAT catalog file %s' %self.pulsar_lat_catname
        psr_cat_data= pp = pyfits.open(self.pulsar_lat_catname)[1].data
        self.lat= lat = pd.DataFrame(pp, index=[n.strip() for n in pp.Source_Name])
        lat['ts'] = self.df[self.df.psr]['ts']
        lat['aprob'] = self.df[self.df.psr]['aprob']
        lat['ROI_index'] = [Band(12).index(SkyDir(float(ra),float(dec))) for ra,dec in zip(lat.RAJ2000,lat.DEJ2000)]

        lat['skydir'] = [SkyDir(float(ra),float(dec)) for ra,dec in zip(lat.RAJ2000, lat.DEJ2000)]
        #map(SkyDir, np.array(lat.RAJ2000,float), np.array(lat.DEJ2000,float))
        lat['sourcedir'] = self.df.skydir[self.df.psr]
        lat['delta'] = [np.degrees(s.difference(t)) if not type(t)==float else np.nan for s,t in zip(lat.skydir,lat.sourcedir)]

        far = lat.delta>0.25
        dc2names =set([name.strip() for name in psr_cat_data.Source_Name])
        # missing_from_catalog=np.asarray(list(tt.difference(dc2names)))
        # print 'sources with exp cutoff not in LAT catalog:', missing_from_catalog
        print 'Catalog entries not found:', list(dc2names.difference(tt))
        missing = np.array([ np.isnan(x) or x<10. for x in lat.ts])
        missing |= np.array((lat.aprob==0) & (lat.ts<1000) )
        
        missing_names = lat.index[missing]
        cols = 'RAJ2000 DEJ2000 ts aprob delta ROI_index'.split()
        self.latsel=latsel = pd.DataFrame( np.array([lat[id][missing] for id in cols]), index=cols, columns=missing_names).T

        #psrx = np.array([x in 'pulsar_fom pulsar_low msp pulsar_big'.split() for x in self.df.acat])
        #print '%d sources found in other pulsar catalogs' % sum(psrx)

        self.atable = '<h4>Compare with LAT pulsar catalog: %s</h4>' % os.path.split(self.pulsar_lat_catname)[-1]
        # self.atable += '<p>{} sources fit with exponential cutoff not in LAT pulsar catalog:\n{} '.format(
        #             len(missing_from_catalog), missing_from_catalog)
        self.atable += html_table(latsel.query('ts<10'),
                    dict(ts='TS,Test Statistic', aprob='aprob,Association probability', ROI_index='ROI Index,Index of the ROI, a HEALPix ring index'),
                    heading = '<p>LAT catalog entries with weak or no fit (TS<10)',
                    name=self.plotfolder+'/weak', maxlines=20,
                    float_format=(FloatFormat(2)))
        self.atable += html_table(latsel.query('ts>10'),
                    dict(ts='TS,Test Statistic', aprob='aprob,Association probability', ROI_index='ROI Index,Index of the ROI, a HEALPix ring index'),
                    heading = '<p>LAT catalog entries with nearby, but unassociated source ',
                    name=self.plotfolder+'/far', maxlines=20,
                    float_format=(FloatFormat(2)))
        # if sum(far)>0:
        #     far_names = lat.index[far]
        #     cols = 'delta ts RAJ2000 DEJ2000 ROI_index'.split()
        #     latfar = pd.DataFrame( np.array([lat[id][far] for id in cols]),index=cols, columns=far_names).T
        #     print 'LAT pulsar catalog entries found more than 0.25 deg from catalog:'
        #     print far_names
        #     self.atable += '<p>Pulsars located > 0.25 deg from nominal'\
        #             + latfar.to_html(float_format=FloatFormat(2))
     
    def pulsar_candidates(self, test=False):
        """Pulsar candidates
        Construct a list of all point sources associated with the BigFile pulsar list.
        If a name starts with "PSR", exclude it.
        %(pulsar_candidate_table)s
        """
        class BigFile(object):
            """"manage look up in the BigFile"""
            def __init__(self):
                ff = sorted(glob.glob(os.path.expandvars('$FERMI/catalog/srcid/cat/Pulsars_BigFile_*.fits')))
                t= pyfits.open(ff[-1])
                self.d = pd.DataFrame(t[1].data)
                self.names=[t.strip() for t in self.d.NAME.values]
                self.jnames=[t.strip() for t in self.d.PSRJ.values]

            def __call__(self, name):
                """Find the entry with given name"""

                if name in  self.names: i = self.names.index(name)
                elif name in self.jnames: i= self.jnames.index(name)
                else:
                    error = 'Data for source %s not found' %name
                    print error
                    raise ValueError(error)

                return self.d.iloc[i]
        
        not_psr = np.array([not n.startswith('PSR') for n in self.df.index],bool)
        psrx = np.array([x=='pulsar_big' for x in self.df.acat],bool) & not_psr

        print '%d sources found in BigFile pulsar catalog' % sum(psrx)
        pt = self.df[psrx]['aprob aname aang ts delta_ts locqual'.split()]
        
        # look it up in BigFile, add other stuff
        bf=BigFile()
        anames = self.df[psrx].aname

        # # Deal with sources with apparent BigFile association not in this BigFile version
        # aset=set(anames); bset=set(bf.names)
        # missing = list(aset.difference(bset))
        # if len(missing)>0:
        #     print 'Warning: associations not in this BigFile: {}'.format(missing)
        #     alist =list(anames)
        #     for x in missing:
        #         #print '\tremove {}'.format(x)
        #         alist.remove(x)
        #     anames = np.array(alist)
        
        pt['jname'] = [bf(n).PSRJ for n in anames]
        pt['history']= [bf(n).History[1:-1].replace("'","") for n in anames]
        pt['edot'] = ['%.2e'%bf(n).EDOT for n in anames]
        
        # make file table
        ptx = pt['jname edot history ts aprob aang delta_ts locqual'.split()]
        
        if len(ptx)>0:
            self.pulsar_candidate_table= \
                html_table(ptx,
                    dict(name='Source Name,click for link to SED',
                      jname='Pulsar name,J version',
                      history='History,history of the BigFile entry',
                      ts='TS,Test Statistic for the source', 
                      aprob='Probability,Association probability: not cut on',
                       aang='Angle,angular distance (deg)',
                      delta_ts='Delta TS,change in TS to the point source\n'
                                      'should be positive negative means peak of TS map was not at source',
                      locqual='Localization quality,measure of the goodness of the localization fit\n greater than 5 is questionable',
                      ),
                      float_format=FloatFormat(2), 
                      heading = """<p>%d sources with pulsar association not in LAT pulsar catalog. Note, no cut on 
                        association probability.""" % sum(psrx),
                      name=self.plotfolder+'/atable',
                      maxlines=80)        
        else:
            self.pulsar_candidates='No candidates found'
        return ptx if test else None    
    
    def localization_check(self, tsmin=100, dtsmax=9, qualmax=5, systematic=None, make_plots=True):
        r"""Localization resolution test
        
        The association procedure records the likelihood ratio for consistency of the associated location
        with the fit location, expressed as a TS, or the difference in the TS for source at the maximum,
        and at the associated
        source. The distribution in this quantity should be an exponential, $\exp{-\Delta TS/2/f^2}$,
        where $f$ is a scale factor to be measured from the distribution. 
        If the PSF is a faithful representation of the distribution of photons
        from a point source, $f=1$. For 1FGL and 2FGL we assumed 1.1. The plots show the results for AGN,
        LAT pulsars, and all other associations. They are cut off at 9, corresponding to 95 percent
        containment.
        <br><br>%(localization_html)s
        """
        self.localization_html = 'Cuts: TS>{}, Delta TS<{}, localization quality <{}'.format(tsmin, dtsmax, qualmax) 
        if systematic is None:
            systematic = self.config['localization_systematics']\
                if 'localization_systematics' in self.config.keys() else (1,0)
        if len(systematic)<2: systematic.append(1)

        txt = 'Applied systematic factor: {0:.2f} ({2:.2f} if |b|<5) and {1} arcmin '\
                'added in quadrature with r95.'.format(*systematic)
        print txt        
        self.localization_html+= '<br><br>'+txt

        t = self.df.acat
        df = self.df
        ts = np.array(df.ts,float)
        locqual = np.array(df.locqual,float)
        locqual[np.isnan(locqual)]=99
        r95 = 60* 2.45 * np.sqrt(np.array(df.a, float)*np.array(df.b,float))
        r95[np.isnan(r95)]=99
        
        # population selections from association
        agn = np.array([x in 'crates bzcat agn bllac'.split() for x in t])
        psr = np.array([x in 'pulsar_lat'.split() for x in t])
        unid= np.array([x in 'unid'.split() for x in t])
        otherid=~(agn | psr | unid)
        
        hilat = np.abs(df.glat.values)>5
        
        # adjust using the systematics
        sf = np.where(hilat, systematic[0], systematic[2])
        sa = systematic[1]
        deltats = df.adeltats / (sf**2 + (sa/r95)**2)

        def select(sel, rlim=(0,20) ,):
            cut = sel & (df.aprob>0.8) & (ts>tsmin) & (locqual<qualmax) & (r95>rlim[0]) & (r95<rlim[1])
            return deltats[cut]
        
        
        cases = [(agn, 'AGN strong', (0,1)), (agn,'AGN mod.', (1,2)), (agn,'AGN weak',(2,20)),
                 (psr&hilat, 'PSR high',(0,20)),(psr&~hilat, 'PSR low',(0,20)),  
                  (agn&hilat, 'AGN high',(0,20)),(agn&~hilat, 'AGN low',(0,20)),  
                 (otherid, 'other ids',(0,20))]
        zz=[]
        print '{:14} {:10} {:6} {:6}'.format(*'Selection range number factor'.split())
        for sel, name, rcut in cases:
            cut = select(sel, rcut)
            try:
                z = FitExponential(cut, name);
            except:
                z = None
            print '{:14} {:3}{:3}  {:6.0f} {:6.2f}'.format(name,
                     rcut[0],rcut[1], len(z.vcut), z.factor if z is not None else 99)
            zz.append(z)
        self.fit_summary=zz    
        if not make_plots: return
        fig, axx = plt.subplots(2,len(cases)//2, figsize=(12,8), sharex=True)
        plt.tight_layout()
        for ax, z in zip(axx.flatten(), zz):
            z.plot(ax=ax, xlabel=r'$\Delta TS$');
        #axx.flatten()[-1].set_visible(False)           
        return fig

    def unassociated_soft(self, query='isextended==False & pindex>2.2 & curvature<0.05 & psr==False',
            selection='aprob<0.5 & eflux100>3e-12 & a<0.1 & locqual<5', 
            flux_min=3e-12):
        """Unassociated soft sources

        This is an analysis of a set of unassociated soft (index>2.2) sources that have a concentration 
        in the inner galaxy region (|b|<10, |l|<50). We examine soft, near-power-law sources. 
        <p>The first row is for those that are associated, the second unassociated. 
        The left plot is energy flux vs. spectral index. It shows a line that is an energy flux
        minimum for the next two histograms.
        <p>A list selected with the queries "%(softie_query)s", and then "%(unid_query)s" which are
        not associated can be downloaded 
        <a href="../../%(softie_filename)s?download=true">here</a>, and viewed in the following table.
        <p>%(softie_table)s 

        """
        self.softie_query=query
        self.unid_query=selection
        q = self.indf.query(query)\
            ['name ra dec ts pindex glat glon eflux curvature aprob eflux100 a b ang locqual'.split()]
        q['singlat'] = np.sin(np.radians(np.array(q.glat,float)))

        fig,axx=plt.subplots(2,3, figsize=(12,8),
                            gridspec_kw={'wspace':0.2, 'hspace':0.40, 'left':0.01, 'top':0.98})

        def scatxy(ax, query='aprob<0.5',  xlim=(2.2,3.2), ylim=(1,40)):
            una = q.query(query)
            bcut= np.abs(una.singlat)<0.2
            x = una.pindex.clip(*xlim)
            y = (una.eflux100*1e12).clip(*ylim)
            ax.plot(x[~bcut], y[~bcut], '.r', label='high lat');
            ax.plot(x[bcut],  y[bcut],  '+g', label='low lat' );
            ax.set(xlim=xlim, xlabel='Spectral index',
                ylim=ylim, ylabel=r'$Energy\ FLux \times\ 10^{12}$', yscale='log')
            ax.legend();
            ax.axhline(flux_min*1e12, color='gray')

        scatxy(axx[0,0], 'aprob>0.5')
        scatxy(axx[1,0])

        idquery = 'aprob>0.5 & eflux100>{}'.format(flux_min)
        def hist1(ax, query=idquery):
            df = q.query(query)
            ax.hist(df.singlat, bins=np.linspace(-1,1,21), histtype='step', lw=2)
            ax.set(xlabel='sin(glat)')
        def hist2(ax, query=idquery):
            df = q.query(query)
            ax.hist(np.array(df.pindex,float), bins=np.linspace(2.2,3.2,21), histtype='step', lw=2)
            ax.set(xlabel='Spectral index')
        def hist3(ax, query=idquery):
            df = q.query(query)
            ax.hist(np.array(df.eflux100,float)*1e12, bins=np.logspace(0,2, 21), histtype='step', lw=2)
            ax.set(xlabel=r'$Energy\ FLux \times\ 10^{12}$', xscale='log')

        hist1(axx[0,1])
        hist1(axx[1,1],selection)
        hist2(axx[0,2])
        hist2(axx[1,2],selection)
        # hist3(axx[0,3])
        # hist3(axx[1,3],selection)
         
        fig.text(-0.05, 0.99, 'associated', fontsize=16)
        fig.text(-0.05, 0.49, 'unassociated',fontsize=16);
        
        self.softie_filename = self.plotfolder+'/softies.csv'
        dfu=q.query(self.unid_query)['ra dec ts pindex curvature glat glon eflux100 a b ang locqual'.split()]
        dfu.eflux100 *= 1e12 # for readability
        #scale the ellipse 1-sigma radii with multiplicative systematic factor
        factor = self.config.get('localization_systematics', [1.,0])[0]
        dfu.a *= factor
        dfu.b *= factor
        # add a column with circular radius in arc min
        dfu['r95'] = np.sqrt(np.array(dfu.a*dfu.b,float))*2.45*60

        # add 4FGL info: 
        fermi_catalog.check4FGL(dfu, pattern=self.config['gllcat'])

        dfu.to_csv(self.softie_filename)
        self.softies=dfu
        print 'Wrote list of sources to {}'.format(self.softie_filename)

        self.softie_table= \
            html_table(dfu,                
                    float_format=FloatFormat(2), 
                    heading = """<p>Table of %d unassociated soft sources """ % len(dfu),
                    name=self.plotfolder+'/softie_table',
                    maxlines=10)        

        return fig

    def softie_geometry(self, vmax=8, vmin=0, bmax=10, vcut=None, loncut=90):
        """Softie geometry

        For a point source population, the square root of the flux should be inversely proportional 
        to distance. These plots show it vs. |b|, and in a scatter plot vs. (l,b),

        """
        df=self.softies
        l,b,f = df.glon.copy(), df.glat, np.array(df.eflux100,float)
        sf = np.sqrt(f).clip(0,vmax)
        l[l>180]-=360

        fig, axx=plt.subplots(1,2, figsize=(15,5))

        def fig1(ax):
            ax.plot(np.abs(b), sf,  'o', label=r'$|l|>{}$'.format(loncut));
            lonsel = np.abs(l)<loncut
            ax.plot(np.abs(b)[lonsel], sf[lonsel],  'o', label=r'$|l|<{}$'.format(loncut));
            if vcut is not None: ax.axhline(vcut, color='grey', ls='--')
            ax.legend(loc='upper right')
            ax.set(xlim=(0,bmax), ylim=(0,vmax), xlabel='$|b|$', ylabel='sqrt(Energy Flux)');

        def fig2(ax):
            ax = plt.gca()
            scat=ax.scatter(l,b, c=np.sqrt(f),cmap=plt.get_cmap('YlOrRd'), vmax=vmax, vmin=vmin) ;
            ax.set(xlim=(90,-90), ylim=(-bmax,bmax), xlabel='longitude', ylabel='latitude');
            cb = plt.colorbar(scat)
            cb.set_label('sqrt(Enegy Flux)')
            ax.axvline(0, color='lightgrey'); ax.axhline(0, color='lightgrey')

        for ff, ax in zip([fig1,fig2], axx.flatten()): ff(ax)
        return fig

    def all_plots(self):    
        self.runfigures([
            self.summary, 
            self.no_pulsar_check,
            #self.pulsar_check, self.pulsar_candidates, 
            self.association_vs_ts, self.localization_check, 
            self.bzcat_study, 
            self.unassociated_soft, self.softie_geometry,
        ])


class FitExponential(object):
    """manage the fit to an exponential distribution, with no background
    """
    
    def __init__(self, v, label, vmax=9,  binsize=0.25):
        from scipy import optimize

        self.vmax, self.binsize, self.label = vmax, binsize, label
        #self.vcut=vcut = v[(v<vmax) & (v>=0)].clip(0,vmax)
        self.vcut=vcut = v[(v<vmax) ].clip(0,vmax)
        self.vmean = vmean = vcut.mean() 
        # find factor that has same average over the interval
        self.factor = optimize.brentq( lambda x : self.cfactors(x)[3]-self.vmean, 0.9,2.0 )
        beta, c0, c1, r = self.cfactors(self.factor)
        self.alpha = len(vcut) / c0 * binsize
        self.beta=beta
        
    def cfactors(self, f):
        """for factor f, return beta, c0, c1, and c1/c0
        where c0 and c1 are integrals(o,vmax) of exp(-x/beta) and x*exp(-x/beta)
        Thus c1/c0 is <x> over the interval
        """
        beta = 2 * f**2
        u = self.vmax/beta
        c0 = beta * (1 - np.exp(-u))
        c1 = beta**2 * (1 - np.exp(-u) * (1+u)) 
        return beta, c0, c1, c1/c0
    
    def plot(self, ax=None, xlabel=''):
        if ax is None:
            fig,ax = plt.subplots(figsize=(5,5))
        else:fig = ax.figure
        x = np.linspace(0, self.vmax, int(self.vmax/self.binsize)+1) 
        ax.plot(x, self(x), '-r', lw=2,  label='factor=%.2f'% self.factor)
        ax.hist( self.vcut, x, label='%d %s'%(len(self.vcut),self.label), 
                edgecolor='black', color='lightgrey',
                histtype='stepfilled')
        ax.set_ylim(ymin=0)

        ax.grid(); ax.legend()
        plt.setp(ax, ylim=(0, 1.1*self.alpha), xlabel=xlabel)

    def __call__(self, x):
        return self.alpha * np.exp(-x/self.beta)

class ExtAssociations(Associations):
    """ subclass invoked with a specific path
    """
    def __init__(self, model_path, **kw):
        curdir = os.getcwd()
        try:
            os.chdir(model_path)
            self.setup(refresh=False, **kw)
            self.plotfolder=model_path+'/plots/'+self.plotfolder
        finally:
            os.chdir(curdir)

