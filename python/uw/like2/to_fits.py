"""
Code to generate a standard Fermi-LAT catalog FITS file
also, see to_xml, to generate XML for the sources
$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/to_fits.py,v 1.16 2017/02/09 19:02:30 burnett Exp $
"""
import os, argparse, glob
from astropy.io import fits as pyfits
from skymaps import SkyDir
import numpy as np
import pandas as pd
from .analyze import fermi_catalog, sourceinfo

#catdir  = config.catalog_path
#catalog = config.default_catalog
#assoc   = 'gll_psc18month_uw8_assoc.fits' # file from Jean with associations
##newcat  = '24M_%s.fits' %outdir

#def get_rec(outdir,name='sources'):    
#    return pickle.load(open('%s_%s.rec'%(name, outdir))) 
#def full_path(fn):
#    return os.path.join(catdir, fn)
#
#
#def to_xml(outdir):
#    recfile ='sources_%s.rec'%outdir 
#    assert os.path.exists(recfile), 'pickled rec file %s not found' %recfile
#    cat = CatalogManager(recfile)
#    ps = map(cat.point_source, cat.dirs, cat.names, cat.models)
#    stack=xml_parsers.unparse_point_sources(ps)
#    xmlfile = '24M_%s.xml'%outdir
#    xml_parsers.writeXML(stack, xmlfile, '24M_%s source library'%outdir)
#    print 'wrote out %d source entries XML file %s ' % (len(stack), xmlfile)
#
#class Assoc(object):
#    def __init__(self,cat, curcat, cat_ref=assoc):
#        self.hdu = pyfits.open(full_path(curcat))
#        self.data = self.hdu[1].data
#        self.cat_ref = pyfits.open(full_path(cat_ref))[2]
#    def __call__(self, name):
#        return self.data[self.data.field('NickName')==name]
#
# default column definitions 
coldata ="""\
    SourceName             20A None
    NickName               20A None
    RAJ2000                  E deg
    DEJ2000                  E deg
    GLON                     E deg
    GLAT                     E deg
    Conf_95_SemiMajor        E deg
    Conf_95_SemiMinor        E deg
    Conf_95_PosAng           E deg
    Test_Statistic           E None
    Pivot_Energy             E MeV
    Cutoff_Energy            E MeV
    Flux_Density             E photon/cm**2/MeV/s
    Unc_Flux_Density         E photon/cm**2/MeV/s
    Spectral_Index           E None
    Unc_Spectral_Index       E None
    Flux1000                 E photon/cm**2/s
    Unc_Flux1000             E photon/cm**2/s
    Energy_Flux100           E erg/cm**2/s
    Unc_Energy_Flux100       E erg/cm**2/s
    SpectralFitQuality       E None
    ID_Number                I None
    ID_Name               520A None
    ID_Probability         26E None
    ID_RA                  26E deg
    ID_DEC                 26E deg
    ID_Angsep              26E deg
    ID_Catalog             26I None
    SpectrumType           18A None
    Extended                 L None
    beta                     E None
    Unc_beta                 E None
    Spectral_Fit_Quality     E None
    Flags                    I None""".split('\n')
coldict = {}
for name,format,unit in [c.split() for c in coldata]:
    coldict[name]= dict(format=format, unit = unit)

def makecol(name, array):
    if name in coldict:
        format = coldict['name']['format']
        unit = coldict['name']['unit']
        if unit=='None': unit=''
    elif name[:4] == 'Flux' or name[:8]=='Unc_Flux':
        format, unit = 'E', 'photon/cm**2/s'

    else:
        format, unit = 'E', ''
    return pyfits.Column(name=name, format=format, unit=unit, array=array)
        
def band_cols():
    elist = (100,300,1000,3000,10000,100000)
    for i in range(len(elist)-1):
        e1,e2 = elist[i:i+2]
        print '%d_%d' % (e1,e2)
  
def get_data():
    return pipeline.load_rec_from_pickles(outdir) 
   
def source_class(z):
    source = [{'1F':'1FGL', 'PG':'PGW', 'MR':'MRF', 'UW':'UW', 'MS':'MST', 'SE':'SEED', '18':'18M',
        'Cy':'bin', 'LS':'bin','PS':'PSR',}[n[:2]] for n in z.name]
    return source
    
def move(z):
    source = [{'1F':'1FGL', 'PG':'PGW', 'MR':'MRF', 'UW':'UW', 'MS':'MST', 'SE':'SEED', '18':'18M',
        'Cy':'bin', 'LS':'bin','PS':'PSR',}[n[:2]] for n in z.name]
    fgl = np.array(source)=='1FGL'
    em  = np.array(source)=='18M'
    canmove = em+fgl
    print 'moving %d sources' % sum(canmove)
    z.ra[canmove]=z.fit_ra[canmove]
    z.dec[canmove]=z.fit_dec[canmove]


def model_info(model):
    """ helper function to interpret the model object
    Returns tuple of: 
            pnorm pindex cutoff 
            pnorm_unc pindex_unc cutoff_unc
            e0 pivot_energy 
            flux flux_unc
            eflux eflux_unc
            beta beta_unc
            index2 index2_unc
            modelname 
    """
    data = []
    eflux = list(np.array(model.i_flux(e_weight=1, error=True, emax=1e5, quiet=True))*1e6)
    if np.isnan(eflux[0]):
        import pdb; pdb.set_trace()
    p,p_relunc = model.statistical()
    p_unc = p*p_relunc
    psr_fit =  model.name.endswith('Cutoff')
    data += [p[0],     p[1],     p[2] if psr_fit else np.nan, ]
    data += [p_unc[0], p_unc[1] ,p_unc[2] if psr_fit else np.nan,]
    pivot_energy=model.e0
    e0 = model.e0 if model.name!='LogParabola' else p[3]
    flux = model(e0)
    flux_unc = flux*p_relunc[0]
    data += [e0, pivot_energy]
    data += [flux, flux_unc]
    
    # energy flux from model e < 1e5, 1e-6 MeV units
    data += eflux
    if model.name=='ExpCutoff':
        data += [np.nan,np.nan, 1.0, np.nan, 'ExpCutoff']
    elif model.name=='PLSuperExpCutoff':
        data += [np.nan,np.nan, p[3], p_unc[3], model.name]
    elif p[2]<0.01:
        data += [0.0, np.nan,    np.nan, np.nan, 'PowerLaw'] 
    else:
        data += [p[2], p_unc[2], np.nan, np.nan, 'LogParabola']
    return data                

def test(s):
    """  return a DataFrame with model info
        first step takes a long time """
    t =  map( makecat.model_info, s.models)
    df = pd.DataFrame(t, index=s.index ,columns="""pnorm pindex cutoff 
            pnorm_unc pindex_unc cutoff_unc
            e0 pivot_energy 
            flux flux_unc
            eflux eflux_unc
            beta beta_unc
            index2 index2_unc
            modelname""".split() 
        )
    return df
    
class MakeCat(object):
    
    def __init__(self, z,   add_assoc=False):
        self.z = z  
        self.add_assoc = add_assoc
        si = sourceinfo.SourceInfo()
        self.fcat = fermi_catalog.CreateFermiFITS(si.df,names=list(z.index))
        
    def add(self, name, array, fill=0):
        #print ' %s ' % name ,
        if name in coldict:
            format = coldict[name]['format']
            unit = coldict[name]['unit']
        else:
            format, unit = 'E', ''
        t = array
        if self.check:
            t = array.copy()
            t[self.bad] = fill
        self.cols.append(pyfits.Column(name=name, format=format, unit=unit, array=t))
        
        
    def __call__(self, outfile, localization_systematic=(1.1, 5e-3) ):
        self.cols = []
        z = self.z
        z.sort_values(by='ra')
        self.check=False
        self.bad = z.ts<9
        self.add('SourceName', z.jname)
        self.add('NickName', z.index)
        self.add('RAJ2000', z.ra)
        self.add('DEJ2000', z.dec)
        sdir = map(SkyDir, z.ra, z.dec)
        self.add('GLON', [s.l() for s in sdir])
        self.add('GLAT', [s.b() for s in sdir])
        
        # localization 
        # f95, quad = 2.45*localization_systematic[0], localization_systematic[1] # 
        # print 'Applying factor of %.2f to localization errors, and adding %.3g deg in quadrature' % localization_systematic
        self.add('LocalizationQuality', z.locqual)
        major, minor, posangle = z.a, z.b, z.ang 
        flags = z.flags
        if 'ax' in z.columns:
            refit = ~pd.isnull(z.ax)
            print 'applying alternate ellipses to %d sources, setting 16 flag bit' % sum(refit) 
            major[refit] = z[refit].ax
            minor[refit] = z[refit].bx
            posangle[refit] =  z[refit].angx
            flags[refit] += 16
        # self.add('Conf_95_SemiMajor', np.sqrt((f95*major)**2+quad**2) )
        # self.add('Conf_95_SemiMinor', np.sqrt((f95*minor)**2+quad**2) )
        self.add('Conf_95_SemiMajor', 2.45 * major * z.systematic)
        self.add('Conf_95_SemiMinor', 2.45 * minor * z.systematic)
        self.add('Conf_95_PosAng',    posangle)
        
        # determine which modelname to use, how to interpret index2
        notpsr = z.modelname=='LogParabola'
        psr = z.modelname=='PLSuperExpCutoff'
        logpar = notpsr & (z.index2>0)
        powerlaw = notpsr & (z.index2==0)
        has_exp_index = psr & (z.index2<1)
        exp_cutoff = psr & (z.index2==1)
        extended = pd.isnull(z.locqual)
        print 'found %d logparabola, %d exp cutoff, %d super cutoff, %d extended'\
               % (sum(logpar), sum(exp_cutoff), sum(has_exp_index), sum(extended))
        self.add('Test_Statistic',    z.ts)
        
        # Spectral details
        self.add('SpectrumType',      np.where(powerlaw, 'PowerLaw', np.where(exp_cutoff, 'PLExpCutoff', z.modelname)))
        self.add('Pivot_Energy',      z.e0)  # note that pivot_energy is the measured value
        self.add('Flux_Density',      z.flux)
        self.add('Unc_Flux_Density',  z.flux_unc)
        self.add('Spectral_Index',    z.pindex)
        self.add('Unc_Spectral_Index',z.pindex_unc)
        self.add('Exp_Index',         np.where(psr, z.index2, np.nan))
        self.add('Unc_Exp_Index',     np.where(has_exp_index, z.index2_unc, np.nan))
        self.add('Cutoff_Energy',     z.cutoff) 
        self.add('Unc_Cutoff_Energy', z.cutoff_unc) 
        self.add('beta',              np.where(logpar, z.index2, np.nan))
        self.add('Unc_beta',          np.where(logpar, z.index2_unc, np.nan))
        self.add('Energy_Flux100',    z.eflux100 )
        self.add('Unc_Energy_Flux100',z.eflux100_unc)
        self.add('SpectralFitQuality',z.fitqual) 
        self.add('Extended',          extended)
        self.add('Flags',             flags)
        
        # make the FITS stuff, adding the *_Band columns and the EnergyBounds table
        table = pyfits.BinTableHDU.from_columns(self.cols + self.fcat.make_bands(z.index))
        table.header['ERPOSFAC']= localization_systematic[0], 'Systematic factor applied to conf95'
        table.header['ERPOSABS']= localization_systematic[1], 'systematic value (deg) added in quadrature conf95'
        table.name = 'LAT_Point_Source_Catalog' 
        if os.path.exists(outfile):
            os.remove(outfile)
        self.hdus =  [pyfits.PrimaryHDU(header=None),  #primary
                 table,      # this table
                 pyfits.BinTableHDU.from_columns(self.fcat.make_ebounds(),name='EnergyBounds'),
                ]
        if self.add_assoc:
            self.hdus += [assoc.cat_ref,]    # the catalog reference (copied)
            
        self.finish(outfile)
        
    def finish(self, outfile):
        pyfits.HDUList(self.hdus).writeto(outfile)
        print '\nwrote FITS file to %s, with %d columns and %d entries' % (outfile, len(self.cols), len(self.z))
        
def to_reg(fitsfile, filename=None, color='green'):
    """ generate a 'reg' file from a FITS file, write to filename
    """
    if filename is None: filename = fitsfile.replace('.fits', '.reg')
    s = pyfits.open(fitsfile)[1].data
    out = open(filename, 'w')
    print >> out, "# Region file format: DS9 version 4.0 global color=%s" % color
    for t in zip(s.RA,s.DEC,
                          s.Conf_95_SemiMinor,s.Conf_95_SemiMajor,s.Conf_95_PosAng,
                          s.NickName):
        print >>out, "fk5; ellipse(%.4f, %.4f, %.4f, %.4f, %.4f) #text={%s}" % t
                        
    out.close()
    print 'wrote reg file to %s' % filename


def main(outfile, infile='sources_*.csv', cuts='(sources.ts>10)', localization_systematic=(1,0)):
    infiles = glob.glob(infile)
    assert len(infiles)>0, 'Input file "%s" not found' % infile
    infile = infiles[-1]
    sources = pd.read_csv(infile, index_col=0)
    print 'Loaded DataTable from file %s' % infile
    selected = sources[eval(cuts)]
    print 'applied cuts %s: %d -> %d sources' % (cuts, len(sources), len(selected))
    t = MakeCat(selected)
    if outfile is None:
        outfile = '_'.join(os.path.abspath('.').split('/')[-2:])+'.fits'
        # for example, 'P202_uw10.fits'
    t(outfile, localization_systematic=localization_systematic)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='create a FITS file')
    parser.add_argument('filename', nargs='*', help='output FITS file' )
    parser.add_argument('--cuts', 
        default='(sources.ts>10)', 
        help='selection cuts')
    parser.add_argument('--infile', default='sources*.csv')
    args = parser.parse_args()
    filename = args.filename[0] if len(args.filename)>0 else None
    main(filename, infile=args.infile, cuts=args.cuts)
