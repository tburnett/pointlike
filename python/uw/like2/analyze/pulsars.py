"""
Pulsar search analysis

"""

import os, glob
import numpy as np
import pylab as plt
import matplotlib.ticker as ticker
import pandas as pd
from astropy.io import fits 
from skymaps import SkyDir, Band
from . import (sourceinfo, associations, _html, fermi_catalog)
from .. import tools
from analysis_base import html_table, FloatFormat

from astropy.table import Table

def bigfile( path='$FERMI/catalog/srcid/cat/Pulsars_BigFile_*.fits'):
    """"Return BigFile as a DataFrame.
        Index is the PSRJ name
    """

    ff = sorted(glob.glob(os.path.expandvars(path)))
    filename = ff[-1]
    version = filename.split('_')[-1][:-5]
    t= fits.open(filename)
    df = pd.DataFrame(t[1].data)
    names=[t.strip() for t in df.NAME.values]
    jnames=[t.strip() for t in df.PSRJ.values]
    psrnames = map(lambda s:'PSR '+s, jnames)
    df.index = psrnames
    return df

def lat_psr(path='$FERMI/catalog/srcid/cat/obj-pulsar-lat_v1*'):
    """get the LAT pulsar table as a DataFrame, with index as the name
    """
    from astropy.table import Table
    lcat = glob.glob(os.path.expandvars(path))[-1]
    filename= os.path.split(lcat)[-1]
    version = filename.split('.')[0][-4:]
    print 'Loading LAT pulsar catalog {}'.format(filename)
    lcatdf = df =Table.read(lcat, hdu=1).to_pandas()
    lcatdf['msec']=msec = np.array([code.find('m')>-1 for code in df.CHAR_Code ], bool) # was PSR_Code
    print 'Found {} entries, {} millisecond pulsars'.format(len(df), sum(msec))
    df.index = map(lambda name: name.strip(), df.Source_Name.values)
    del df['Source_Name']
    return df

class Pulsars(sourceinfo.SourceInfo):
    """Pulsar plots and analysis
    """

    def setup(self, **kw):
        super(Pulsars, self).setup(**kw)
        self.plotfolder='pulsars'
        self.psr = np.asarray([s.startswith('PSR') for s in self.df.index],bool)
        plt.rc('font', size=14)

        # get the LAT pulsar table as a DataFrame, with index as the name
        self.lcat=lcat = glob.glob(os.path.expandvars('$FERMI/catalog/srcid/cat/obj-pulsar-lat_v1*'))[-1]
        filename= os.path.split(lcat)[-1]
        self.version = filename.split('.')[0][-4:]
        print 'Loading LAT pulsar catalog {}'.format(filename)
        self.lcatdf = df =Table.read(lcat, hdu=1).to_pandas()
        self.lcatdf['msec']=msec = np.array([code.find('m')>-1 for code in df.CHAR_Code ], bool) # was PSR_Code
        print 'Found {} entries, {} millisecond pulsars'.format(len(df), sum(msec))
        self.latpsr_info = 'From file {}: {} entries, {} millisecond pulsars'.format(filename,len(df), sum(msec))
        df.index= map(lambda name: name.strip(), df.Source_Name.values)

        # msec designation to corresponding entries in full source list 
        self.df['msec'] = self.lcatdf.msec

        def load_assoc(df):
            # add association info to the data frame
            associations = df.associations 
            probfun = lambda x: x['prob'][0] if not pd.isnull(x) else 0            
            df['aprob'] = np.array([ probfun(assoc) for  assoc in associations])
            df['acat']  = np.array([ assoc['cat'][0] if not pd.isnull(assoc) else 'unid' for  assoc in associations])
            df['aname'] = np.array([ assoc['name'][0] if not pd.isnull(assoc) else 'unid' for  assoc in associations])
            df['aang']  = np.array([ assoc['ang'][0] if not pd.isnull(assoc) else np.nan for  assoc in associations])
            df['adeltats'] = np.array([assoc['deltats'][0] if not pd.isnull(assoc) else np.nan for assoc in associations])

        load_assoc(self.df)
        keep = self.df[self.psr]
        keep.to_csv('pulsars.csv')

    def check4FGL(self, pattern=None):

            if hasattr(self, 'gdf'): return
        
            # add 4FGL info to dataframe of pointlike sources
            df=self.df
            cindex = [n.replace(' ','') for n in self.df.index]
            systematic = self.config['localization_systematics']
            f95, quad = 2.45*systematic[0], systematic[1]/60. 
            self.df['r95'] = (f95**2*(self.df.a * self.df.b) + quad**2)** 0.5
            
            # get the catalog "gll" entries as a DataFrame and set corresponding values
            if pattern is None:
                pattern=self.config['gllcat']
            if not pattern.startswith('/'):
                pattern = '$FERMI/catalog/'+pattern
            self.fgl_name = pattern.split('/')[-2]
            filename = sorted(glob.glob(os.path.expandvars(pattern)))[-1]
            try:
                fcat = fermi_catalog.GLL_PSC2(filename)
            except Exception(msg):
                print 'Failed to load {}: {}'.format(filename, msg)
                raise
            self.fhl_file = fcat.filename.split('/')[-1]
            self.gdf = gdf=  fcat.df
            gdf['uw_ts']    = self.df.ts
            gdf['uw_r95']   = self.df.r95
            gdf['uw_pindex']= self.df.pindex
            gdf['uw_eflux100']=self.df.eflux100

            # add boolean for in FL8Y 
            self.df['fl8y'] = np.isin(cindex, gdf.index )
            print '{} of {} have nicknames in pointlike list'.format(sum(df.fl8y), len(gdf))

            # for sources not already tagged via the pointlike name being the same as the gtlike nickname
            # look for nearest 4FGL source: add name, its distance to DataFrame
            ok  = df.fl8y==True
            added = np.logical_not(ok)

            df.loc[df.index[ok],'otherid']= df[ok].name
            df.loc[df.index[ok], 'distance']=0

            # look for nearest 4FGL source in rejected list: add name, distance to DataFrame
            print 'Searching {} for nearest source to the {} not found in it...'.format(self.fgl_name,sum(added)),
            close = tools.find_close(df[added], self.gdf)

            df.loc[df.index[~ok],'otherid'] = close.otherid
            df.loc[df.index[~ok], 'distance'] = close.distance
            df['b4fgl'] = df.distance<0.015

            df['otherts'] = [self.gdf.loc[s.otherid.replace(' ','')].ts for name,s in df.iterrows() ]
            df['other_extended'] = [self.gdf.loc[s.otherid.replace(' ','')].extended for name,s in df.iterrows() ]

            print 'done.'
    
    def LATpulsars(self):
        """ LAT pulsar information
        
        %(latpsr_info)s
        """

        df = self.lcatdf
        msec = np.array(df.msec, bool)


        def fig1(ax):
            ax.loglog(-df.F1[msec],df.F0[msec],  'o', label='Millisecond');
            ax.loglog(-df.F1[~msec],df.F0[~msec],  'o', label='Young');

            ax.set(xlabel='Frequency derivative [Hz/s]', ylabel='Frequency [Hz]')
            ax.grid(alpha=0.5)
            ax.legend()
 
        def fig2(ax):
            sd = map(SkyDir, df.RAJ2000, df.DEJ2000)
            sinb = np.sin(np.radians(map(lambda s:s.b(), sd)))
            hkw= dict(bins=np.linspace(-1,1,21), histtype='step', lw=2 )
            ax.hist(sinb[msec], label='msec', **hkw)
            ax.hist(sinb[~msec], label='young', **hkw)
            ax.set(xlabel='sin(b)')
            ax.legend(); ax.grid(alpha=0.5)
        
        fig, axx = plt.subplots(1,2, figsize=(12,6))
        map( lambda f,ax: f(ax), [fig1,fig2], axx.flatten() )
        fig.suptitle('LAT pulsars v{}'.format(self.version))
        return fig

  
    def spectra(self, index_min=0.0, index_max=2.5, cutoff_max=1e4, taillist=True):
        """ Spectral distributions
        
        Spectral parameters for %(spectral_fits)d pulsars with significant fits (TS>16)

        %(pulsar_tail_check)s
        """

        psrmodel = (self.df.ts>16) & (self.df.modelname=='PLSuperExpCutoff') & self.df.psr
        self.spectral_fits = sum(psrmodel)
        t = self.df.loc[psrmodel]\
            ['ts flux pindex cutoff e0 index2 index2_unc pivot_energy roiname freebits fitqual msec'.split()]
        t['eflux'] = t.flux * t.e0**2 * 1e6
        msec = np.array(t.msec.values,bool)  


        def histit(ax, bins, vals):
            hkw = dict(histtype='stepfilled', alpha=0.5, lw=2)
            ax.hist(vals[msec], bins, label='msec', color='lightblue',edgecolor='blue',  **hkw )
            ax.hist(vals[~msec], bins, label='young',color='pink', edgecolor='red', **hkw)
            
        def plot1(ax, efmin=1e-2,efmax=1e3):
            bins = np.logspace(np.log10(efmin),np.log10(efmax),26)
            vals = np.array(t.eflux,float).clip(efmin,efmax)
            histit(ax, bins, vals)

            ax.set(xscale='log', xlabel='energy flux', xlim=(efmin,efmax)); ax.grid(alpha=0.5); 
            ax.legend(prop=dict(size=10))

        def plot3(ax):
            bins = np.linspace(index_min,index_max,16)
            vals = np.array(t.pindex,float).clip(index_min,index_max)
            histit(ax, bins, vals)

            ax.set( xlabel='spectral index'); ax.grid(alpha=0.5); 
            ax.legend(prop=dict(size=10))
            
        def plot2(ax):
            bins = np.logspace(2,4,26)
            vals = np.array(t.cutoff,float).clip(None,cutoff_max) 
            histit(ax,bins, vals)

            ax.set(xscale='log', xlabel='cutoff energy (GeV)'); ax.grid(alpha=0.5)
            ax.legend(prop=dict(size=10))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda val,pos: { 100:'0.1', 1000:'1', 10000:'10'}.get(val,'')))
            
        def plot4(ax):
            xvals = np.array(t.cutoff,float).clip(None, cutoff_max)
            yvals = np.array(t.pindex,float).clip(index_min,index_max)
            ax.plot(xvals[msec], yvals[msec], 'o', color='blue', label='msec')
            ax.plot(xvals[~msec], yvals[~msec], 'D', color='orange', label='young')
            ax.set(xscale='log', xlabel='cutoff [GeV]', ylabel='spectral index',
                 ylim=(index_min-0.1, index_max+0.1),
                )
            ax.grid(alpha=0.5); 
            ax.legend(loc='lower right', prop=dict(size=10))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda val,pos: { 100:'0.1', 1000:'1', 10000:'10'}.get(val,'')))
        def plot5(ax):
            bins = np.logspace(2,4,41)
            histit(ax, bins, vals=t.pivot_energy.astype(float ))
            ax.set(xscale='log', xlabel='pivot energy')
            ax.grid(alpha=0.5)
            

        fig, axx = plt.subplots( 2,3, figsize=(12,12))
        plt.subplots_adjust(wspace=0.3, left=0.05,bottom=0.15)
        map(lambda f,ax:f(ax),(plot1,plot2,plot3,plot4,plot5), axx.flatten()[:5])

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
 
    def pulsar_check(self):
            """LAT pulsar check

            %(atable)s
            %(missing_table)s
           """
             
            
            # compare with LAT pulsar catalog     
            lat=self.lcatdf
            psrdf = self.df[self.df.psr].copy()
            diff = list(set(lat.index).difference(set(psrdf.index)))
            if len(diff)>0:
                print 'Missing from model: {}'.format(diff)
            lat['ts'] = self.df[self.df.psr]['ts']
            lat['aprob'] = self.df[self.df.psr]['aprob']
            lat['ROI_index'] = [Band(12).index(SkyDir(float(ra),float(dec))
                                              ) for ra,dec in zip(lat.RAJ2000,lat.DEJ2000)]

            lat['skydir'] = [SkyDir(float(ra),float(dec))
                             for ra,dec in zip(lat.RAJ2000, lat.DEJ2000)]

            lat['sourcedir'] = self.df.skydir[self.df.psr]
            lat['delta'] = [np.degrees(s.difference(t)) if not type(t)==float else np.nan
                            for s,t in zip(lat.skydir,lat.sourcedir)]
            psrdf.loc[:,'r95']= 2.5*np.sqrt(np.array(psrdf.a.values**2+psrdf.b.values**2,float))
            lat.loc[:,'r95'] = psrdf.r95

            far = lat.delta>0.25
            dc2names =set(self.lcatdf.index)
            tt = set(self.df.name[self.df.psr])
            print 'Catalog entries not found:', list(dc2names.difference(tt))
            missing = np.array([ np.isnan(x) or x<10. for x in lat.ts])
            missing |= np.array((lat.aprob==0) & (lat.ts<1000) )
            
            missing_names = lat.index[missing]
            cols = 'RAJ2000 DEJ2000 ts delta r95 ROI_index'.split()
            self.latsel=latsel = pd.DataFrame( np.array([lat[id][missing]
                                    for id in cols]), index=cols, columns=missing_names).T

            self.atable = '<h4>Compare with LAT pulsar catalog id: {}</h4>'.format( self.version)

            label_info= dict(ts='TS,Test Statistic from full fit', 
                             RAJ2000='RAJ2000, position of pulsar',
                             delta='delta,distance to source position (deg)',
                             r95='R95,95 radius (deg), NaN if not calculated',
                                ROI_index='ROI Index,Index of the ROI, a HEALPix ring index')
            # Table of weak or not fit
            weak_df = self.weak_df = latsel.query('ts<10')
            self.atable += html_table(weak_df, label_info,
                        heading = '<p>{} LAT catalog entries with weak or no fit (TS<10); positioned at Pulsar'.format(len(weak_df)),
                        name=self.plotfolder+'/weak', maxlines=20,
                        float_format=(FloatFormat(2)))
            self.unassoc_df = latsel.query('ts>10')
            self.atable += html_table(self.unassoc_df, label_info,
                        heading = '<p>LAT catalog entries with nearby, but currently unassociated source',
                        name=self.plotfolder+'/far', maxlines=20,
                        float_format=(FloatFormat(2)))
            self.missing_table='All LAT pulsasrs are in the current model!'
            self.dfm = dfm = self.latsel[pd.isna(self.latsel.ts)].copy()
            if len(dfm>0):
                dfm.pop('ts'); dfm.pop('delta')
                dfm.rename(columns=dict(RAJ2000='ra', DEJ2000='dec'), inplace=True)
                close =tools.find_close(dfm, self.df)
                dfm.loc[:,'distance'] = close.distance
                dfm.loc[:,'pointlike_name'] = close.otherid
                self.missing_df = dfm
                
                self.missing_table = html_table(dfm, label_info, 
                        heading='<p>LAT catalog entries not in this model',
                        name=self.plotfolder+'/missing',
                        float_format=(FloatFormat(2)))
            

    def bigfile_associations(self, test=False):
        """BigFile Associations
        Construct a list of non LAT pulsar point sources associated with the BigFile pulsar list 
        (Version %(bigfile_version)s).
        <br>Exclude sources with poor localization (quality>5) and BigFile pulsars in clusters. 
        <ul>
        <li>%(bigfile_hi_table)s </li>
 
        <li>%(bigfile_lo_table)s </li>
        </ul>
        """
        class BigFile(object):
            """"manage look up in the BigFile"""
            def __init__(self):
                ff = sorted(glob.glob(
                    os.path.expandvars('$FERMI/catalog/srcid/cat/Pulsars_BigFile_*.fits')))
                t= fits.open(ff[-1])
                print 'Read file {}'.format(ff[-1])
                self.version = ff[-1].split('_')[-1].split('.')[0]
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
        self.psrx = psrx = np.array(
            [x=='pulsar_big' for x in self.df.acat], bool) & not_psr & (self.df.locqual<5)

        print '%d sources associated with BigFile pulsar list' % sum(psrx)
        pt = self.df[psrx]['aprob aname aang ts glat glon pivot_energy curvature locqual'.split()]
        
        # look it up in BigFile, add other stuff
        self.bf = bf=BigFile()
        self.bigfile_version = bf.version
        
        # check to see if current BigFile may be different from the one used to make association
        lost = np.isin(pt.aname, bf.names, invert=True)
        if sum(lost)>0:
            print 'Current BigFile missing entries : {}'.format(list(pt.aname[lost].index))
            pt = pt.iloc[~lost]
        
        anames = pt.aname
        
        pt['jname'] = jname = [bf(n).PSRJ for n in anames]
        # test for the jname not ending in numeric character
        not_incluster = [n[-1] in '0123456789' for n in pt.jname]
        print '  Selected {} out of {} associations not in clusters'.format(sum(not_incluster), len(pt))
        
        def hmax(n):
            t = bf(n)
            return max(t.Hall32, t.Hall36, t.Hall40, t.Hval32, t.Hval36, t.Hval40)
        pt['Hmax']  = [hmax(n) for n in anames]
        pt['history']= [bf(n).History[1:-1].replace("'","") for n in anames]
        pt['edot'] = ['%.2e'%bf(n).EDOT for n in anames]
        pt['P0'] =   ['{:.3f}'.format(bf(n).P0) for n in anames]
        
        # make file table
        ptx = pt[not_incluster][
            'jname glat glon edot P0 history Hmax ts aprob aang curvature pivot_energy locqual'.split()].sort_values(by='name')
        hilat = abs(ptx.glat)>5
        if len(ptx)>0:
            colinfo=dict(name='Source Name,click for link to SED',
                    jname='Pulsar name,J version',
                    edot='Edot, rate of energy change',
                    Hmax='Hmax, max(Hall32, Hall36, Hall40, Hval32, Hval36, Hval40)',
                    pivot_energy='Pivot Energy,Energy of zero correlation between spectral index and normalization ',
                    history='History,BigFile history entry',
                    ts='TS,Test Statistic for the source', 
                    aprob='Probability,Bayesian association probability',
                    aang='Angle,angular distance (deg)',
                    #curvature='curvature,?',
                    locqual='Localization quality,measure of the goodness of the localization fit\n greater than 5 is questionable',
                    )
            self.bigfile_hi_table= \
                html_table(ptx[hilat], colinfo, float_format=FloatFormat(2), 
                      heading = """<b>Table of %d high-latitude (|b|>5) associations.</b>""" % sum(hilat),
                      name=self.plotfolder+'/hilat_table',
                      maxlines=10)   
            self.bigfile_lo_table= \
                html_table(ptx[~hilat], colinfo,   float_format=FloatFormat(2), 
                      heading = """<b>Table of %d low-latitude (|b|<5) associations.</b> """ % sum(~hilat),
                      name=self.plotfolder+'/lolat_table',
                      maxlines=10)          
        else:
            self.bigfile_hi_table= self.bigfile_lo_table=''
        return ptx if test else None    

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
        fig,ax = plt.subplots(figsize=(8,6))
        hkw = dict(bins=np.linspace(0,cmax,41), log=True, histtype='step', lw=2)
        ax.hist(df.curvature.clip(0,cmax), label='all sources', **hkw)
        ax.hist(df[df.psr].curvature.clip(0,cmax), label='EC model', **hkw)
        ax.hist(df[psr].curvature.clip(0,cmax), label='PSR souce', **hkw)
        plt.setp(ax, xlabel='Curvature', ylim=(0.5,None))
        ax.legend()
        ax.grid()
        return fig

    def candidates_in_4fgl(self):
        """Potential pulsar candidates in 4FGL
        Make a list of sources with the selections
        <ul>
            <li in 4FGL
            <li>not associated or a LAT Pulsar
         </ul>
        The plots are for sources from this list, showing the effects of subsequent cuts:
        <ol>
            <li>0.15 < curvature < 0.75
            <li>pivot energy < 3 GeV
            <li>R95 < 15 arcmin
        </ol>
          
        <h4>%(candidate_table_in_4fgl)s</h4>
        <br>A csv file of the above, including both cuts is <a href="../../%(pulsar_candidate_filename_in_4fgl)s?download=true">here</a>
        """

        return self.new_candidates(in_4fgl=True)


    def new_candidates(self, in_4fgl=False):
        """Potential pulsar candidates
        Make a list of sources with the selections
        <ul>
            <li>not associated
            <li>not in 4FGL or withinn 0.5 deg of one 
            <li>nearest 4FGL source is extended or has TS<1000
         </ul>
        The plots are for sources from this list, showing the effects of subsequent cuts:
        <ol>
            <li>0.15 < curvature < 0.75
            <li>pivot energy < 3 GeV
            <li>R95 < 15 arcmin
        </ol>
          
        <h4>%(candidate_table)s</h4>
        <br>A csv file of the above, including both cuts is <a href="../../%(pulsar_candidate_filename)s?download=true">here</a>
        """
        # add info about 4FGL
        self.check4FGL(pattern=None)

        df=self.df

        # select subset not in 4FGL and not associated and not close to a 4FGL source and that the closest is very strong
        if in_4fgl: 
            # alternative: only in 4FGL
            dfx = df.query('fl8y==True & aprob<0.8 & locqual<8 & psr==False')
            dfx['sname']= self.gdf.loc[dfx.index, 'sname']
        else:
            dfx = df.query('fl8y==False & aprob<0.8 & locqual<8 & distance>0.5 & other_extended==False & otherts<1000')
        # values to display
        ts = dfx.ts.astype(float).clip(0,250)
        singlat = np.sin(np.radians(dfx.glat.astype(float)))
        curvature= dfx.curvature.astype(float).clip(0,1)
        r95_arcmin = dfx.r95.astype(float)*60
        pivot = dfx.pivot_energy.astype(float)
        eflux =dfx.eflux.astype(float)
        #curvature selection
        curvature_cut = np.logical_and(curvature<0.75, curvature>0.15)
        r95_cut = r95_arcmin<15 #7.9
        pivot_cut = pivot<3e3
        cuts=[]; 
        cut_labels=['curvature', 'pivot', 'R95']
        cuts.append( curvature_cut)
        cuts.append(cuts[-1] & pivot_cut)
        cuts.append(cuts[-1] & r95_cut)

        label_info = dict(name='Source Name,click for link to SED',
                          pindex='Spectral index, at pivot energy',
                          pivot_energy='pivot energy,',
                          curvature='curvature,at the pivot energy',
                          r95='localization,in degrees, 95%',
                          distance='distance,in degrees to nearest 4FGL source',
                          otherid='4FGL nick name,',
                          otherts='4FGL TS,',
                          locqual='locqual,localization quality',
                         )                    
        dfcut = dfx[cuts[-1]][
            '''ra dec ts glat pindex pivot_energy curvature locqual r95 pars errs
                distance otherid otherts'''.split()].sort_values(by='ts', ascending=False)

        if in_4fgl:
            dfcut['sname'] =  self.gdf.loc[dfcut.index, 'sname']
            self.df_pulsar_candidates_in_4fgl = dfcut 
            self.candidate_table_in_4fgl = html_table(dfcut, label_info,
                heading = '''<b>Table of {} pointlike sources not in {}, 
                    not assocated and with curvature selection</b>'''.format(self.fgl_name,len(dfcut)),
                name=self.plotfolder+'/candidates_in_4fgl', maxlines=20,
                float_format=(FloatFormat(2)))
            self.pulsar_candidate_filename_in_4fgl=self.plotfolder+'/pulsar_candidates_in_4fgl.csv'
            dfcut.to_csv(self.pulsar_candidate_filename_in_4fgl)
        else:
            self.df_pulsar_candidates = dfcut 
            self.candidate_table = html_table(dfcut, label_info,
                heading = '''<b>Table of {} pointlike sources not in {}, 
                    not assocated and with curvature selection</b>'''.format(self.fgl_name,len(dfcut)),
                name=self.plotfolder+'/candidates', maxlines=20,
                float_format=(FloatFormat(2)))
            self.pulsar_candidate_filename = self.plotfolder+ '/pulsar_candidates.csv'
            dfcut.to_csv(self.pulsar_candidate_filename)

        fig, axx = plt.subplots(2,3, figsize=(12,10))
        ax1,ax2,ax3,ax4,ax5,ax6 = axx.flatten()
        hkw = dict(histtype='step', lw=2)
        def doit(ax, x, bins, xlabel, xlog=False):
            ax.hist(x, bins, label='', **hkw)  
            for i,cut in enumerate(cuts):
                ax.hist(x[cut], bins, label=cut_labels[i], **hkw)
            ax.set(xlabel=xlabel, xscale='log' if xlog else 'linear')
            ax.legend()

        doit(ax1, curvature, np.linspace(0,1,21), 'curvature')
        doit(ax2, pivot, np.logspace(np.log10(200),np.log10(2e4),21), 'pivot energy', xlog=True)
        doit(ax3, r95_arcmin, np.linspace(0,25,26),'R95 (arcmin)')
        doit(ax4, ts, np.logspace(1,np.log10(250),25), 'TS', xlog=True)
        doit(ax5, singlat, np.linspace(-1,1,21), 'sin(b)')
        doit(ax6, eflux, np.logspace(np.log10(3e-2),np.log10(3), 21), 'energy flux', xlog=True) 

        fig.tight_layout()
        fig.set_facecolor('white')
        return fig


    def all_plots(self):
        self.runfigures([
            self.LATpulsars,
            self.spectra,
            self.pulsar_check,
            self.bigfile_associations,
            self.new_candidates,
            self.candidates_in_4fgl,
        ])

def simple_plot(name, rec):
    from uw.like2 import plotting
    
    class XSource(object):
        def __init__(self, name,rec):
            self.name= name
            self.spectral_model = rec['model']
            self.sedrec = rec['sedrec']
    x = XSource(name, rec)
    ps = plotting.sed.Plot(x)
    fig, ax = plt.subplots(figsize=(2.5,2.5))
    ps(axes = ax, axis=(100,3.2e4, 0.05, 4))
    ax.set(xlabel='', ylabel='', title='', xticklabels=[], yticklabels=[]);
    ax.text(0.3, 0.85, name,transform=ax.transAxes, fontsize=16 );
    return fig

def save_pulsar_candidate_stuff(refresh=True, ):
    import pickle  

    def sed_info(name):
        roi= int(self.df.loc[name]['roiname'][-4:]); roi
        pkl = self.pkls[roi]
        rec = pkl['sources'][name]; 
        fig = simple_plot(name, rec)

        plt.savefig('plots/pulsars/candidates/{}.jpg'.format(name))
        plt.close(fig)
        return dict(model = rec['model'], sedrec= rec['sedrec'])
    
    # Grab spectral model object, sed info from ROI pickles 
    if not os.path.isdir('plots/pulsars/candidates'):
        os.makedirs('plots/pulsars/candidates')
    self=Pulsars(refresh=refresh)
    self.new_candidates()
    self.candidates_in_4fgl()

    all_sed_info = {}
    for pdf in [self.df_pulsar_candidates, self.df_pulsar_candidates_in_4fgl]:
        for name in pdf.index:
            all_sed_info[name] = sed_info(name)

    with open('pulsar_candidate_sed_info.pkl', 'wr') as out:
        pickle.dump(all_sed_info, out)


#=================================================================================================
#  Old stuff, may be useful
    def efratio(self, e1=2000, e2=20000):
        """Examine energy flux ratio
        
        Ratio of the energy flux at 20 GeV to that at 2 GeV.
        The identified pulsar subset is shown.
        """
        df=self.df
        efr = df['eflux_ratio']=np.asarray([model(e2)/model(e1)*(e2/e1)**2 for model in self.df.model])
        fig,ax = plt.subplots(figsize=(5,5))
        xlim = (1e-2,10)
        dom = np.logspace( np.log10(xlim[0]),np.log10(xlim[1]) ,31)  
        ax.hist(efr.clip(*xlim), dom ,log=True);
        ax.hist(efr[self.psr].clip(*xlim), dom, log=True, color='orange', label='PSR');
        plt.setp(ax, xscale='log', xlabel='eflux(20 GeV)/eflux(2 GeV)')
        ax.legend()
        ax.grid();
        return fig
        
    def selection(self, curvature_cut=0.1, ts_cut=10):
        """Select candidates.
        
        %(selection_info)s
        """
        self.curvature_cut=curvature_cut
        self.ts_cut=ts_cut
        df=self.df
        probfun = lambda x: x['prob'][0] if not pd.isnull(x) else 0
        aprob = np.array([ probfun(assoc) for  assoc in self.df.associations])
        no3fgl = np.asarray([s is None for s in self.df.cat3fgl]);
        self.keep= keep = no3fgl &(~self.psr) \
            & (self.df.curvature>curvature_cut) & (self.df.ts>ts_cut) & (self.df.locqual<8) &(aprob<0.1)
        
        self.total=sum(keep)
        self.cvsname='pulsar_candidates.csv'
        t = self.df[keep]['ra dec glat ts pivot_energy pindex eflux_ratio curvature roiname'.split()]
        t.to_csv(self.cvsname)
        print 'wrote %d sources to %s' % (len(t), self.cvsname)
        self.selection_info="""\
        Cuts: non-3FGL, non-LAT PSR, association probability < 0.1, curvature>%(curvature_cut)s, TS>%(ts_cut)s<br>, 
        <br>Total:%(total)s
        <br>%(html_list)s
        <br>
         Link to csv format table:
         <a href="../../%(cvsname)s?download=true">%(cvsname)s</a></li>
        """
        self.html_list = html_table(t, name=self.plotfolder+'/candidates', 
                heading='<h4>%d Candidate pulsar sources</h4>' % len(t), 
                    float_format=FloatFormat(2))
        

    def no_curvature(self, prefix='S966', ts_high_cut=2):
        """Weak new sources with PW fits
        
        %(weak_list)s
        """
        df = self.df
        pcut = np.array([n.startswith(prefix) for n in df.index],bool); 
        cut = (df.ts>10) &  (df.locqual<8) & (df.curvature<0.01) & pcut & (df.ts_high<ts_high_cut) & (df.ts_low<5)
        t = self.df[cut]['ra dec glat ts pivot_energy pindex  fitqual locqual ts_low ts_med ts_high roiname'.split()]
        self.noc_df=t.sort_index(by='roiname')
        print 'selected %d %s sources' % (len(t), prefix)
        self.weak_list = html_table(t, name=self.plotfolder+'/weak_pl_sources', 
                heading='<h4>%d weak new power-law sources</h4>' % len(t), 
                    float_format=FloatFormat(2))

    
    def load_list(self, filename):
        df = self.df
        print 'Reading the LOFAR list "{}"'.format(filename)
        self.lofar=lofar= pd.read_csv(filename, index_col=0)
        lofarset=set(lofar.index)
        print '{}/{} found in uw7000'.format(len(lofarset.intersection(set(df.index))), len(lofarset))
        dfb = pd.read_pickle('beta_fts.pkl')
        for col in 'TS_beta beta beta_fit roiname'.split():
            lofar[col]=dfb[col]
        print 'reading 6-year variability table...',
        s6y=pd.read_pickle('../../P301_monthly/six_yr_source_variability.pkl')
        s6yset= set(s6y.index)
        print 'Found variability info for {}/{} sources'.format(len(s6yset.intersection(lofarset)),len(lofarset))
        for col in 'mean ngood nmonths pull_rms tsmax tssum'.split():
            lofar[col]=s6y[col]
            lofar['ts6y']=s6y['ts']
        return lofar

    def check_a_list(self, filename='../uw7002/7yr_LOFAR_uw7000.txt'):
        """Make a table with curvature and variability
        %(check_list)s
        """
        lofar = self.load_list(filename)
        self.check_list = html_table(lofar, name=self.plotfolder+'/checked_sources',
            heading='<h4>%d checked sources</h4>'%len(lofarset),
            float_format=FloatFormat(2))

            