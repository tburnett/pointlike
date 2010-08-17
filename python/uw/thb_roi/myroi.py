"""
User interface to SpectralAnalysis
----------------------------------
$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/thb_roi/myroi.py,v 1.17 2010/08/06 18:15:11 burnett Exp $

"""

import numpy as np
import numpy as N  # for Kerr compatibility
from scipy import optimize
import pylab as plt
import os, pickle, math, types 

from uw.like import roi_analysis, roi_localize, roi_plotting,  Models, sed_plotter
from uw.utilities import makerec, fermitime, image
from skymaps import SkyDir,  PySkyFunction


def spectralString(band,which=None):
    """Return a string suitable for printSpectrum.

       which -- if not None, an array of indices for point sources to fit
    """
    self=band
    which = which or [0]
    r     = []
    for w in which:
       try:
          self.bandFit(which=w)
       except np.linalg.LinAlgError:
          return 'singular matrix'
       self.m.p[0] = N.log10(self.uflux)
       ul = sum( (b.expected(self.m) for b in self.bands) )
       if self.flux is None:
          r += [0,ul,0]
       else:
          n = ul*self.flux/self.uflux
          r += [n,ul - n, n - ul*self.lflux/self.uflux]
       r += [self.ts]

    rois = self.__rois__()
    ph   = int(sum( (b.photons for b in self.bands) ))
    gal  = sum( (b.bg_counts[0] for b in self.bands) )
    iso  = sum( (b.bg_counts[1] for b in self.bands) )
    values = tuple([int(round(self.emin)),rois[0],rois[1],ph,gal,iso, ph-gal-iso] + r)
    format = '  '.join(['%6i','%6.1f','%6.1f','%7i','%8.1f','%9.1f','%6.1f']+['%7.1f +%5.1f -%5.1f %6.1f' ]*len(which))
    return format%values


class MyROI(roi_analysis.ROIAnalysis):
    """ create an ROIAnalysis subclass 

        Parameters

        ps_manager :  PointSourceManager
        bg_manager :  BackgroundManager
        roifactory :  ROIfactory

    """

    def __init__(self,roi_dir, ps_manager,bg_manager,roifactory,**kwargs):
        """
        Parameters

        ps_manager :  
        bg_manager : 
        roifactory : 
       
        change default free_radius to turn this off

        optional:
            bgfree [True, False, True]
            prune_radius [0.1]
            free_radius  [0]
        """
        bgfree = kwargs.pop('bgfree', [True,False,True])
        if 'fit_bg_first' in kwargs:
            self.fit_bg_first = kwargs['fit_bg_first']
        super(MyROI, self).__init__(roi_dir, ps_manager, bg_manager, roifactory, **kwargs)
        self.bgm.models[0].free = np.array(bgfree[:2])
        if len(self.bgm.models)>1:
            self.bgm.models[1].free = np.array([bgfree[2]])
        self.name = self.psm.point_sources[0].name # default name
        self.center= roi_dir

    def fit(self, **kwargs):
        """ invoke base class fitter, but insert defaults first 
        """
        fit_bg_first = self.fit_bg_first
        if 'fit_bg_first' in kwargs: 
            fit_bg_first = kwargs.pop('fit_bg_first')
        if 'use_gradient' not in kwargs: kwargs['use_gradient']=self.use_gradient
        pivot = kwargs.pop('pivot', False)
        ts = 0
        ignore_exception = kwargs.pop('ignore_exception', True)
        try:
            super(MyROI, self).fit(fit_bg_first=fit_bg_first, **kwargs)
            ts = self.TS()
            if not self.quiet: print self
            if pivot:
                ## refit to determine parameters at the pivot energy
                m = self.psm.models[0]
                #reassign, but keep in range
                e0= np.min(np.max(m.pivot_energy(), self.fit_emin[0]), self.fit_emax[0])
                assert e0>100 and e0< 1e5, "check that it worked"
                m.set_e0(e0)
                
                if not self.quiet:
                    print 'Refit only first source with e0=%.0f' % m.e0
                fr = self.get_free()
                fr[:]=False
                fr[:2]=True
                
                super(MyROI, self).fit(**kwargs)
                if not self.quiet: 
                    print m
                    print '---- predicted pivot_energy: %.0f'  % self.psm.models[0].pivot_energy()
        except Exception, msg:
            if not self.quiet: print 'Fit failed: %s' % msg
            if not ignore_exception: raise
        return ts

    def get_free(self):
        """ return array of the free variables, point sources followed by background"""
        t =np.hstack((self.psm.models, self.bgm.models))
        return np.hstack([m.free for m in t])
 
    def set_free(self, free):
        """ save the array retrieved by get_free
        """
        i = 0
        t =np.hstack((self.psm.models, self.bgm.models))
        for m in t:
            j = i+len(m.free)
            m.free = free[i:j]
            i = j
     
    def band_ts(self, which=0):
        """ return the sum of the individual band ts values
        """
        self.setup_energy_bands()
        ts = 0
        for eb in self.energy_bands:
            eb.bandFit(which)
            ts += eb.ts
        return ts

    def localize(self,which=0, tolerance=1e-3,update=False, verbose=False, bandfits=True, seedpos=None):
        """Localize a source using an elliptic approximation to the likelihood surface.

          which     -- index of point source; default to central 
                      **if localizing non-central, ensure ROI is large enough!**
          tolerance -- maximum difference in degrees between two successive best fit positions
          update    -- if True, update localization internally, i.e., recalculate point source contribution
          bandfits  -- if True, use a band-by-band (model independent) spectral fit; otherwise, use broabband fit
          seedpos   -- use for a modified position (pass to superclass)

         return fit position, change in TS
        """
        try:
            quiet, self.quiet = self.quiet, not verbose # turn off details of fitting
            loc, i, delta, deltaTS= super(MyROI,self).localize(which=which,bandfits=bandfits,
                            tolerance=tolerance,update=update,verbose=verbose, seedpos=seedpos)
            self.quiet = quiet
            if not self.quiet: 
                name = self.psm.point_sources[which].name if type(which)==types.IntType else which
                print 'Localization of %s: %d iterations, moved %.3f deg, deltaTS: %.1f' % \
                    (name,i, delta, deltaTS)
                self.print_ellipse()
        except Exception, e:
            print 'Localization failed! %s' % e
            self.qform=None
            loc, deltaTS = None, 99 
        #self.find_tsmax()
        return loc, deltaTS

    def print_ellipse(self, label=True, line=True):
        if not self.qform: return
        labels = 'ra dec a b phi qual'.split()
        if label: print (len(labels)*'%10s') % tuple(labels)
        if not line: return
        p = self.qform.par[0:2]+self.qform.par[3:]
        print len(p)*'%10.4f' % tuple(p)
        
    def print_associations(self, srcid, thresh=0.5):
        ra,dec= self.qform.par[0:2] 
        error =self.qform.par[3:6]
        t = srcid( SkyDir(ra,dec), error)
        if len(t)==0:
            print 'No associations'
        else:
            print 'Associations:   prob  source name      catalog '
            for i,a in enumerate(t):
                if a[2]<thresh:
                    print '\t\t... (skipping %d with prob<%.2f)' % (len(t)-i, thresh)
                    break #assume sorted on prob.
                print '\t%12.2f  %-15s  %s' % (a[2],a[0],a[1])
    

    def dump(self, sdir=None, galactic=False, maxdist=5, title=''):
        """ formatted table point sources positions and parameter in the ROI
        Parameters
        ----------
            sdir : SkyDir, optional, default None for center
                for center: default will be the first source
            galactic : bool, optional, default False
               set true for l,b
            maxdist : float, optional, default 5
               radius in degrees

        """
        self.print_summary(sdir, galactic, maxdist, title)

    

    def tsmap(self, which=0, bandfits=True):
        """ return function of likelihood in neighborhood of given source
            tsm = roi.tsmap(which)
            size=0.25
            tsp = image.TSplot(tsm, center, size, pixelsize =size/20, axes=plt.gca())
            tsp.plot(center, label=name)
            tsp.show()

        """
        self.localizer = roi_localize.ROILocalizer(self, which, bandfits=bandfits)
        return PySkyFunction(self.localizer)


    def plot_spectra(self, which=0, axes=None,axis=None, outfile=None, fignum=1, **kwargs):
        """generate a spectral plot
        sets up a figure, if fignum is specified.
        """

        if axes is None:
            fig=plt.figure(fignum);
            plt.clf()
            axes = plt.gca()
        return roi_plotting.band_fluxes(self, which,axes, axis,outfile, **kwargs)

    def pickle(self, name, outdir, **kwargs):
        """ Write a dictionary with parameters
           name: name for source, used as filename (unless fname in kwargs)
            outdir: ouput directory
            **kwargs: anything else to add to the dictionanry
        """
        name = name.strip()
        fname = kwargs.pop('fname', name)
        output = dict()
        output['name'] = name
        output['ra']   = self.center.ra()
        output['dec']  = self.center.dec()
        
        # get source fit parameters, relative uncertainty
        p,p_unc = self.psm.models[0].statistical()
        output['src_par'] = p 
        output['src_par_unc'] = p*p_unc 
        output['pivot_energy'] = self.psm.models[0].e0 #this is 1000 by default, but catalog will be different
        output['src_model'] = self.psm.models[0] # simpler to save it all
        output['bgm_par'] = np.hstack([m.statistical()[0] for m in self.bgm.models] )
        output['bgm_par_unc'] = np.hstack([m.statistical()[1] for m in self.bgm.models] )
        try:
            output['qform_par'] = self.qform.par if 'qform' in self.__dict__ else None
        except AttributeError:
            output['qform_par'] = None
        output['tsmax'] = None if 'tsmax' not in self.__dict__ else [self.tsmax.ra(),self.tsmax.dec()]
        output.update(kwargs) # add additional entries from kwargs

        if not os.path.exists(outdir):
            os.mkdir(outdir)
        f = file(os.path.join(outdir,fname+'.pickle'),'wb')
        pickle.dump(output,f)
        f.close()

    def printSpectrum(self,sources=None):
        """Print total counts and estimated signal in each band for a list of sources.

        Sources can be specified as PointSource objects, source names, or integers
        to be interpreted as indices for the list of point sources in the roi. If
        only one source is desired, it needn't be specified as a list. If no sources
        are specified, all sources with free fit parameters will be used."""
        if sources is None:
         sources = [s for s in self.psm.point_sources if N.any(s.model.free)]
        elif type(sources) != type([]): 
         sources = [sources]
        bad_sources = []
        for i,s in enumerate(sources):
         if type(s) == roi_analysis.PointSource:
            if not s in self.psm.point_sources:
               print 'Source not found in source list:\n%s\n'%s
               bad_sources += [s]
         elif type(s) == int:
            try:
               sources[i] = self.psm.point_sources[s]
            except IndexError:
               print 'No source #%i. Only %i source(s) specified.'\
                     %(s,len(self.psm.point_sources))
               bad_sources += [s]
         elif type(s) == type(''):
            names = [ps.name for ps in self.psm.point_sources]
            try:
               sources[i] = self.psm.point_sources[names.index(s)]
            except ValueError:
               print 'No source named %s'%s
               bad_sources += [s]            
         else:
            print 'Unrecognized source specification:', s
            bad_sources += [s]
        sources = set([s for s in sources if not s in bad_sources])
        indices = [list(self.psm.point_sources).index(s) for s in sources]
        self.setup_energy_bands()

        fields = ['  Emin',' f_ROI',' b_ROI' ,' Events','Galactic','Isotropic','Excess']\
                +[' '*15+'Signal']*len(sources)
        outstring = 'Spectra of sources in ROI about %s at ra = %.2f, dec = %.2f\n'\
                    %(self.psm.point_sources[0].name, self.center.ra(), self.center.dec())
        outstring += ' '*54+'  '.join(['%21s'%s.name for s in sources])+'\n'
        outstring += '  '.join(fields)+'\n'
        print outstring
        for eb in self.energy_bands:
        #         print eb.spectralString(which=indices)
         # local
            print spectralString(eb, which=indices)

    def get_spectrum(self):
        """
        return a recarry with spectral details
        """
        
        fields = 'emin  f_ROI b_ROI events galactic isotropic excess'.split();
        rec = makerec.RecArray(fields)
        def get_band_info(band):
            """ copied from spectralString """
            self = band
            rois = self.__rois__()
            ph   = int(sum( (b.photons for b in self.bands) ))
            gal  = sum( (b.bg_counts[0] for b in self.bands) )
            iso  = sum( (b.bg_counts[1] for b in self.bands) )
            values = [int(round(self.emin)),rois[0],rois[1],ph,gal,iso, ph-gal-iso]
            return values
        self.setup_energy_bands()
        for eb in self.energy_bands:
            rec.append( *get_band_info(eb) )
        return rec()

    def printMySpectrum(self,sources=None, out=None):
        """Print total counts and estimated signal in each band for a list of sources.

        Sources can be specified as PointSource objects, source names, or integers
        to be interpreted as indices for the list of point sources in the roi. If
        only one source is desired, it needn't be specified as a list. If no sources
        are specified, all sources with free fit parameters will be used.
        
        """
        #super(MyROI,self).printSpectrum(sources)
        if sources is None:
         sources = [s for s in self.psm.point_sources if N.any(s.model.free)]
        elif type(sources) != type([]): 
         sources = [sources]
        bad_sources = []
        for i,s in enumerate(sources):
            if type(s) == roi_analysis.PointSource:
               if not s in self.psm.point_sources:
                  print 'Source not found in source list:\n%s\n'%s
                  bad_sources += [s]
            elif type(s) == int:
               try:
                  sources[i] = self.psm.point_sources[s]
               except IndexError:
                  print 'No source #%i. Only %i source(s) specified.'\
                        %(s,len(self.psm.point_sources))
                  bad_sources += [s]
            elif type(s) == type(''):
               names = [ps.name for ps in self.psm.point_sources]
               try:
                  sources[i] = self.psm.point_sources[names.index(s)]
               except ValueError:
                  print 'No source named %s'%s
                  bad_sources += [s]            
            else:
               print 'Unrecognized source specification:', s
               bad_sources += [s]
        sources = set([s for s in sources if not s in bad_sources])
        spectra = dict.fromkeys(sources)
        for s in sources:
         spectra[s] = roi_plotting.band_spectra(self,source=self.psm.point_sources.index(s))
        iso,gal,src,obs = roi_plotting.counts(self)[1:5]
        fields = ['  Emin',' f_ROI',' b_ROI' ,' Events','Galactic','Isotropic','   excess']\
                +[' '*10+'Signal']*len(sources)
        outstring = 'Spectra of sources in ROI about %s at ra = %.3f, dec = %.3f\n'\
                    %(self.psm.point_sources[0].name, self.center.ra(), self.center.dec())\

        outstring += ' '*68+'  '.join(['%-18s'%s.name for s in sources])+'\n'
        outstring += '  '.join(fields)+'\n'
        for i,band in enumerate(zip(self.bands[::2],self.bands[1::2])):
            values = (band[0].emin, band[0].radius_in_rad*180/N.pi,
                      band[1].radius_in_rad*180/N.pi,obs[i],gal[i],iso[i], obs[i]-gal[i]-iso[i])
            for s in sources:
                values+=(spectra[s][1][i],.5*(spectra[s][3][i]-spectra[s][2][i]))          
            string = '  '.join(['%6i','%6.2f','%6.2f','%7i','%8.1f','%9.1f','%9.1f']+
                               ['%8.1f +/-%6.1f']*len(sources))%values
            outstring += string+'\n'
        print >>out, outstring

    def get_photons(self, emin=1000): 
        """ access binned photon data

        """
        roi = self
        ra,dec = roi.center.ra(), roi.center.dec() 
        energy=[]; etype=[]; dra=[]; ddec=[]

        cosdec=math.cos(math.radians(dec))
        for band in roi.bands:
            e = band.e
            if e<emin: continue
            t = band.b.event_class() & 3 # note: mask off high bits!
            for wsd in band.wsdl:
                for i in range(wsd.weight()):
                    energy.append(e)
                    etype.append(t)
                    x =wsd.ra()-ra
                    if x<-180: x+=360
                    dra.append(x*cosdec)
                    ddec.append(wsd.dec()-dec)
                    #print '%6.0f %8.3f %8.3f' % (e, (wsd.ra()-ra)*cosdec,wsd.dec()-dec)

        return np.rec.fromarrays([energy,etype,dra,ddec],
                           names='energy etype dra ddec'.split())         

    def plot_tsmap(self, name=None, center=None, size=0.5, pixelsize=None, outdir=None, 
            which=0, catsig=99, axes=None, fignum=99, 
            bandfits=True,
            galmap=True, galactic=False,
            assoc = None,
            notitle = False,
            nolegend = False,
            markercolor='blue', markersize=12,
            primary_markercolor='green', primary_markersize=14,
             **kwargs):
        """ create a TS map for the source

        Optional keyword arguments:

  =========   =======================================================
  Keyword     Description
  =========   =======================================================
  name        [None]  -- provide name for title, and to save figure if outdir set
  center      [None] -- center, default the roi center
  outdir       [None] if set, save sed into <outdir>/<source_name>_tsmap.png if outdir is a directory, save into filename=<outdir> if not.  
  catsig      [99]  -- if set and less than 1.0, draw cross with this size (degrees)
  size        [0.5]  -- width=height (deg)
  pixelsize   [None] -- if not set, will be 20 x20 pixels
  galmap      [True] -- if set, draw a galactic coordinate image with the source position shown
  galactic    [False] -- plot using galactic coordinates
  which       [0]    -- chose a different source in the ROI to plot
  assoc       [None] -- if set, a list of tuple of associated sources 
  notitle     [False] -- set to turn off (allows setting the current Axes object title)
  nolegend    [False]
  markersize  [12]   -- set 0 to not plot nearby sources in the model
  markercolor [blue]
  =========   =======================================================

        returns the image.TSplot object for plotting positions, for example
        """
        roi = self
        kwargs={} #fix later
        name = self.name if name is None else name
        tsm = roi.tsmap(which=which, bandfits=bandfits)
        sdir = center if center is not None else self.center
        if axes is None: 
            plt.figure(fignum,figsize=(5,5)); plt.clf()
        
        tsp = image.TSplot(tsm, sdir, size, pixelsize =pixelsize if pixelsize is not None else size/20. , 
                    axes=axes, galactic=galactic, galmap=galmap, **kwargs)
        if 'qform' in roi.__dict__ and roi.qform is not None:
            sigma = math.sqrt(roi.qform.par[3]*roi.qform.par[4]) # why do I need this?
            qual = roi.qform.par[6]
            if sigma<1 and qual <50:
                tsp.overplot(roi.qform, sigma)
            else:
                print 'bad fit sigma %g, >1 or qual %.1f >50' % (sigma, qual)
        tsp.show(colorbar=False)
        if catsig<1:
            tsp.cross(sdir, catsig, lw=2, color='grey')
            
        # plot the primary source, any nearby from the fit
        x,y = tsp.zea.pixel(sdir)
        tsp.zea.axes.plot([x],[y], '*', color=primary_markercolor, label=name, markersize=primary_markersize)
        marker = 'ov^<>1234sphH'; i=k=0
        if markersize!=0: 
            for ps in self.psm.point_sources: # skip 
                x,y = tsp.zea.pixel(ps.skydir)
                if ps.name==name or x<0 or x>tsp.zea.nx or y<0 or y>tsp.zea.ny: continue
                tsp.zea.axes.plot([x],[y], marker[k%12], color=markercolor, label=ps.name, markersize=markersize)
                k+=1
        
        tsp.plot(tsp.tsmaxpos, symbol='+', color='k') # at the maximum
        if not notitle: plt.title( name, fontsize=24)

        if assoc is not None:
            # eventually move this to image.TSplot
            last_loc,i=SkyDir(0,90),0
            for aname, loc, prob, catid in zip(assoc['name'],assoc['dir'],assoc['prob'],assoc['cat']):
                #print 'associate with %s, prob=%.2f' % (aname.strip(),prob)
                if catid in ('ibis',): 
                    print '---skip gamma cat %s' % catid
                    continue
                if i>8:
                    print '---skip because too many for display'
                    continue
                x,y = tsp.zea.pixel(loc)
                diff = np.degrees(loc.difference(last_loc)); last_loc=loc
                if diff>1e-3: k+=1 # new marker only if changed place
                tsp.zea.axes.plot([x], [y], marker=marker[k%12], color='green', linestyle='None',
                    label='%s[%s] %.2f'%(aname.strip(), catid, prob ), markersize=markersize)
                i+=1
        
        fs = plt.rcParams['font.size']
        plt.rcParams.update({'legend.fontsize':7, 'font.size':7})
        # put legend on left.
        if not nolegend: tsp.zea.axes.legend(loc=2, numpoints=1, bbox_to_anchor=(-0.15,1.0))
        plt.rcParams['font.size'] = fs

        if outdir is not None: 
          if os.path.isdir(outdir):
            plt.savefig(os.path.join(outdir,'%s_tsmap.png'%name.strip()))
          else :
            plt.savefig(outdir)
        return tsp

    def plot_FITS(self, fitsfile,  size=0.5, bandfits=True,):
        """ kluge to make a fits file
        """
        z = image.ZEA(self.center, size=size, pixelsize=size/20, fitsfile=fitsfile)
        z.fill(self.tsmap(bandfits=bandfits))
        del(z) 

    def rescan(self, tsm, threshold=1):
        """ scan the tsmap for a secondary peak
        tsm is a map generated by tsm = roi.plot_tsmap...
        return position of peak position, if delta TS >threshold
        """
        M = tsm.zea.image
        self.tsmap_max = M.max()
        maxpixel = None
        if self.tsmap_max >threshold:
            # there is a secondary peak
            t = np.where(M==M.max())
            x,y=t[1][0],t[0][0]
            maxpixel = tsm.zea.skydir(x,y)
            if not self.quiet: print 'found maximum %.1f at (%.3f,%.3f)'\
                % (self.tsmap_max, maxpixel.ra(),maxpixel.dec())
        return maxpixel
        

    def band_info(self):
        """ return dictionary of the band ts values and photons, diffuse  
        """
        self.setup_energy_bands()
        bands = self.energy_bands
        return BandDict(
            ts =      [band.bandFit()                             for band in bands],
            photons = [int(sum( (b.photons for b in band.bands))) for band in bands],
            galactic= [sum((b.bg_counts[0] for b in band.bands))  for band in bands],
            isotropic=[sum((b.bg_counts[1] for b in band.bands))  for band in bands],
            flux   =  [band.flux for band in bands],
            lflux  =  [band.lflux for band in bands],
            uflux =   [band.uflux for band in bands],
            signal =  [sum( (b.expected(band.m) for b in band.bands) )for band in bands],
            )
            
    def plot_sed(self, which=0, fignum=5, axes=None,
            axis=None, #(1e2,1e6,1e-8,1e-2),
            data_kwargs=dict(linewidth=2, color='k',),
            fit_kwargs =dict(lw=2,        color='r',),
            butterfly = True,
            use_ergs = True,
            outdir = None,
            galmap = True,
            ):
        """Plot a SED
        ========     ===================================================
        keyword      description
        ========     ===================================================
        which        [0] index of source to plot
        fignum       [5] if set, use (and clear) this figure. If None, use current Axes object
        axes         [None] If set use this Axes object
        axis         None, (1e2, 1e5, 1e-8, 1e-2) depending on use_ergs
        data_kwargs  a dict to pass to the data part of the display
        fit_kwargs   a dict to pass to the fit part of the display
        butterfly    [True] plot model with a butterfly outline
        use_ergs     [True] convert to ergs in the flux units and use GeV on the x-axis
        outdir       [None] if set, save sed into <outdir>/<source_name>_sed.png if outdir is a directory, 
                            save into filename=<outdir> if not.
        galmap       [True] plot position on galactic map if set
        ========     ===================================================
        
        """
        return sed_plotter.plot_sed(self,which,fignum, axes, axis, data_kwargs, 
            fit_kwargs, butterfly, use_ergs, outdir, galmap)


class BandDict(dict):
    """ a dictionary of Band stuff with access functions, like __str__
    """
    def __str__(self):
        n = len(self['ts'])
        #          0    1371   231.0   528.6  2242.0
        return 'band photons     gal     iso      TS\n'\
            +'\n'.join(['%4i%8i%8.1f%8.1f%8.1f' \
                    %(i, self['photons'][i],self['galactic'][i],self['isotropic'][i] ,self['ts'][i])for i in range(n)]) \
            +'\n Sum%33.1f' % sum(self['ts'])
        
if __name__=='__main__':
    pass
