"""
Code to plot TS maps

$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/plotting/tsmap.py,v 1.16 2015/12/03 17:33:45 burnett Exp $

"""
import math, os, sys
import numpy as np
from uw.utilities import image
import pylab as plt
from skymaps import SkyDir

def plot(localizer, name=None, center=None, size=0.5, pixelsize=None, outdir=None, 
        which=0, catsig=99, axes=None, fignum=99, 
        bandfits=True,
        galmap=True, galpos=(0.8,0.85),
        galactic=False,
        assoc = None,
        notitle = False,
        nolegend = False,
        nooverplot = False,
        markercolor='blue', markersize=8,
        primary_markercolor='green', primary_markersize=12,
         **kwargs):
    """ create a TS map for the source. These are localization style
        TS maps (where the source is not in the background model) and
        are useful for source localization.

    localizer: a Localization object

        Optional keyword arguments:

      =========   =======================================================
      Keyword     Description
      =========   =======================================================
      name        [None]  -- provide name for title, and to save figure if outdir set
      center      [None] -- center, default the location
      outdir       [None] if set, save sed into <outdir>/<source_name>_tsmap.png if outdir is a directory, save into filename=<outdir> if not.  
      catsig      [99]  -- if set and less than 1.0, draw cross with this size (degrees)
      size        [0.5]  -- width=height (deg)
      pixelsize   [None] -- if not set, will be 20 x20 pixels
      galmap      [True] -- if set, draw a galactic coordinate inset image with the source position shown
      galactic    [False] -- plot using galactic coordinates
      assoc       [None] -- if set, a list of tuple of associated sources 
      notitle     [False] -- set to turn off (allows setting the current Axes object title)
      nolegend    [False] -- set to turn off legend
      nooverplot  [False] -- set to turn off overplot of fit, source positions
      markersize  [10]   -- set 0 to not plot nearby sources in the model
      markercolor [blue]
      tsfits      [False] -- set True to also create a FITS-format file with the image
      ax or axes  [None]
      =========   =======================================================

    returns the image.TSplot object for plotting positions, for example
    """
    source = localizer.tsm.source
    if name is None: name=source.name
    sdir = center if center is not None else source.skydir
    axes =  kwargs.pop('ax', axes)
    if axes is None: 
        fig = plt.figure(fignum,figsize=(4,4)); 
        fig.clf()
        fig.set_facecolor('white')
        axes = fig.gca()
    maxsize = kwargs.pop('maxsize', 2.0)
    if size >maxsize:
        print 'setting size from %.2f to %.1f' % (size,maxsize)
        size = maxsize # prevent too big for reasonable ?
        pixelsize= size/15.
    tsfits = kwargs.pop('tsfits', False)
    tsp = image.TSplot(localizer.TSmap, sdir, size, 
                pixelsize=pixelsize if pixelsize is not None else size/20. , 
                axes=axes, galactic=galactic, galmap=galmap, galpos=galpos, **kwargs)
    if hasattr(source, 'ellipse') and source.ellipse is not None and not nooverplot: 
        loc = source.ellipse
        sigma = np.sqrt(loc[2]*loc[3]) #loc['a']*loc['b']) #?? scale factor needed?
        qual = loc[5] #'qual']
        if sigma<1 and qual <50:
            tsp.overplot(loc)
        else:
            print 'bad fit sigma %g, >1 or qual %.1f >50' % (sigma, qual)
    tsp.show(colorbar=False)
    if catsig<1:
        tsp.cross(sdir, catsig, lw=2, color='grey')
        
    # plot the primary source, any nearby from the fit
    i=k=0
    marker = 'ov^<>1234sphH'
    if not nooverplot:
        x,y = tsp.zea.pixel(sdir)
        tsp.zea.axes.plot([x],[y], '*', color=primary_markercolor, label=name, markersize=primary_markersize)
        if markersize!=0:
            # plot nearby sources in the ROI 
            for ps in localizer.tsm.blike.sources: 

                if ps.skydir is None: continue
                x,y = tsp.zea.pixel(ps.skydir)
                if ps.name==name or x<0 or x>tsp.zea.nx or y<0 or y>tsp.zea.ny: continue
                tsp.zea.axes.plot([x],[y], marker[k%12], color=markercolor, label=ps.name, markersize=markersize)
                if hasattr(ps, 'ellipse') and ps.ellipse is not None:
                    # this draws a line, perhaps shaded
                    tsp.zea.ellipse(SkyDir(*ps.ellipse[:2]), ps.ellipse[2:5])
                    #tsp.zea.ellipse(ps.skydir, ps.ellipse[2:5])
                k+=1
    
    tsp.plot(tsp.tsmaxpos, symbol='+', color='k') # at the maximum

    if assoc is not None:
        # eventually move this to image.TSplot
        last_loc,i=SkyDir(0,90),0
        for aname, loc, prob, catid in zip(assoc['name'],assoc['dir'],assoc['prob'],assoc['cat']):
            if prob<0.10: continue
            #print 'associate with %s, prob=%.2f' % (aname.strip(),prob)
            if catid in ('ibis',): 
                print '---skip gamma cat %s' % catid
                continue
            if i>4:
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
    if not nolegend: 
        leg=tsp.zea.axes.legend(loc=2, numpoints=1, bbox_to_anchor=(-0.15,1.0))
        leg.get_frame().set_alpha(1.0)
    plt.rcParams['font.size'] = fs

    if not notitle:
        tsp.zea.axes.set_title('%s'% name, fontsize=16)  # big title
    if outdir is not None:
        filename = name.replace(' ','_').replace('+','p')
        fout = os.path.join(outdir, ('%s_tsmap.jpg'%filename) )
        print 'saving tsplot to %s...' % fout , ; sys.stdout.flush()
        plt.savefig(fout, bbox_inches='tight', padinches=0.2) #cuts off outherwise
        print
        if tsfits: 
            fitsname = os.path.join(outdir, '%s_tsmap.fits' % filename)
            tsp.zea.skyimage.reimage(tsp.zea.center,fitsname , pixelsize, size)
            print 'saved fits format to %s' % fitsname
    return tsp
