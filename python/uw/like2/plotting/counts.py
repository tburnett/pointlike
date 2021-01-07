"""
Code to generate an ROI counts plot 
$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/plotting/counts.py,v 1.16 2016/03/21 18:56:11 burnett Exp $

Authors M. Kerr, T. Burnett

"""
import os, sys
import numpy as np
import pylab as plt
import matplotlib.ticker as ticker
import pandas as pd
from ..import diffuse

from uw.utilities import image

def get_counts(roi, event_type=None, tsmin=10, emax=None):
    """
    return a dictionary with counts information for plotting
    
    roi : an ROIstat object
    event_type : None or integer
        if integer, 0/1 for front/back
    tsmin : float
        used to select sources if bandts info in sedrec exists
        the bandts values are in the dict, None if no info
    """
    bands = roi.selected
    if event_type is not None:
        bands = [b for b in bands if b.band.event_type==event_type]
        num_types = 1
    else:
        num_types = len(roi.config.event_type_names)
    assert len(bands)>0, 'get_counts: no bands found'
    all_sources = np.array(roi.sources)
    global_mask = np.array([s.skydir is None for s in roi.sources])
    free_mask = np.array([s.skydir is not None and np.any(s.model.free) for s in roi.sources])
    global_sources = all_sources[global_mask]
    free_sources = all_sources[free_mask] 
    fixed_source_mask = ~free_mask & ~global_mask
    strong_filter= lambda s: (s.sedrec.ts.sum()>tsmin if hasattr(s,'sedrec') and s.sedrec is not None else True) and s in free_sources
    strong_mask = np.array([strong_filter(s) for s in all_sources])
    weak_mask = free_mask * (~strong_mask)
    
    names = np.array([s.name for s in np.hstack([global_sources,free_sources])])
    bandts = np.array([s.sedrec.ts.sum() if hasattr(s,'sedrec') and s.sedrec is not None else None for s in free_sources])
    energies = roi.energies #sorted(list(set([b.energy for b in bands])))
    if emax is not None: energies = filter(lambda e:e<emax, energies)
    nume = len(energies)
    numsrc = sum(global_mask)# | strong_mask)
    observed = np.zeros(nume)
    fixed = np.zeros(nume)
    fixed_source = np.zeros(nume)
    free_source = np.zeros(nume)
    weak = np.zeros(nume)
    model_counts = np.zeros((nume, numsrc))
    total = np.zeros(nume)
    utotal = np.zeros(nume)
    uobserved = np.zeros(nume)
    inside = np.zeros((nume,5)) # for data, gal, iso, sun, sources, 

    def pixel_counts(s, imask): # helper needed to avoid invoking pix_counts
        if s.counts==0: return 0
        return s.pix_counts[imask].sum()
    
    for i, energy in enumerate(energies):
        for b in bands:
            if b.band.energy!=energy: continue
            counts = np.array([s.counts for s in b]) # array of all model counts
            weak[i]  += sum(counts[weak_mask]) 
            fixed_source[i] += sum(counts[fixed_source_mask])
            free_source[i] += sum(counts[free_mask])
            observed[i] += sum(b.data)
            fixed[i] += b.fixed_counts            
            model_counts[i,:]  += counts[global_mask]# | strong_mask]
            total[i] += b.counts
            # model, photons with unweight
            utotal[i] += b.unweight * b.counts
            uobserved[i] += b.unweight * sum(b.data)


            # # total counts for active part of ROI
            # imask = b.inside
            # npix = len(imask); datapix=sum(imask)
            # inside[i,-1] =  datapix
            # if len(imask)==0: #no data pixels inside.
            #     continue
            # data = b.data[imask].sum()
            # diffuse = np.array([ x.pix_counts[imask].sum() for x in b[:3]], np.float32)

            # sources = np.array([ pixel_counts(x,imask) for x in b[3:]], np.float32).sum()
            # inside[i, :5] += np.hstack([data, diffuse, sources])

    
    models = [(names[j] , model_counts[:,j]) for j in range(numsrc)]
    #if sum(weak)>0: models.append( ('TS<%.0f'%tsmin, weak))
    #models.append( ('(fixed)', fixed))
    models.append( ('free_sources', free_source))
    models.append( ('fixed_sources', fixed_source)) 
    chisq = ((observed-total)**2/total).sum()
    uchisq= ((uobserved-utotal)**2/utotal).sum()
    return dict(energies=energies, observed=observed, models=models, 
        names=names, total=total, bandts=bandts, chisq=chisq, 
        uchisq=uchisq, utotal=utotal, uobserved=uobserved,
        # inside=inside,
        )
    
def get_npred(roi, source_name, event_type=None):
    """
    Return array of npred for a given source
    """
    bands = roi.selected
    if event_type is not None:
        bands = [b for b in bands if b.band.event_type==event_type]
        num_types = 1
    else:
        num_types = len(roi.config.event_type_names)
    assert len(bands)>0, 'get_counts: no bands found'
    all_sources = np.array(roi.sources)
    energies = roi.energies #sorted(list(set([b.energy for b in bands])))
    nume = len(energies)
    npred= np.zeros(nume)
    sindex = [s.name for s in all_sources].index(source_name)
    for i, energy in enumerate(energies):
        for b in bands:
            if b.band.energy==energy:
                npred[i] += b[sindex].counts
    return npred

def counts_dataframe(roi, event_type=None,):
    c = get_counts(roi, event_type)
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
    
def plot_counts(roi,fignum=1, event_type=None, outfile=None,
        max_label=10, #merge_non_free=True,
        #merge_all=False,
        axes = None,
        **kwargs
        ):
    """Make counts and residual plots for the ROI roi
    keyword parameters
        fignum    : integer 
        outfile   : filename, or None,
        max_label : integer
        merge_non_free : bool
            if True (default) combine all sources that are not free
        merge_all: bool
            if True (False default) combine *all* sources
        axes : array of one or two Axes instances, or None 
            If None (default), create standard horizontal layout with figure number fignum
            First is filled with the log plot, second, if there, with the residual 
    """

    def plot_counts_and_models(ax, count_data,
                model_kw = dict(linestyle='-', marker='', lw=2),
                total_kw = dict(linestyle='steps-mid', color='black', linewidth=2),
                obs_kw= dict(linestyle=' ', marker='o', color='black',),
                ):
        ax.set_xscale('log')
        ax.set_yscale('log')
        en,obs,tot = count_data['energies'],count_data['observed'],count_data['total']
        
        rel=1.0
        def get_data(name):
            for n, data in count_data['models']:
                if n==name:
                    return data.copy()
            raise Exception('Counts from model "{}" not found for relative plots'.format(name))
       
        if relto is not None:
            if relto=='isotropic':
                # special case to scale the isotrop model with the ratio of this roi's factors
                rel = get_data('isotrop')

                if 'factors' in roi.config['diffuse']['isotrop']:
                    global_iso =roi.config['diffuse']['isotrop']['factors']['front']
                    local_iso=diffuse.normalization['iso']['front']
                    ratio=local_iso/global_iso
                    rel[:8] /= ratio
                    print 'using special isotropic normalizatio...\n {}'.format(ratio.round(3))
                else: ratio=1.0

            else:  
                rel = get_data(relto)  
 
            ax.axhline(1.0, color='grey', ls='--') 

        for name, data in count_data['models']:
            if name==relto: continue
            if np.any(data<=0): continue # ignore models with no predicted counts
            assert len(en)==len(data), 'energy, data mismatch'
            if len(name)>20: name=name[:17]+'...'
            tmodel_kw = model_kw.copy()
            if name.startswith('iso'): tmodel_kw.update(linestyle=':', lw=2)
            elif name.startswith('ring'):tmodel_kw.update(linestyle='--', lw=2)
            elif name=='SunMoon': tmodel_kw.update(color='grey')
            elif name.startswith('fixed'): tmodel_kw.update(linestyle='-.', color='g', lw=3)
            elif name.startswith('free'): tmodel_kw.update(linestyle='-.', color='r', lw=3)
            ax.loglog(en, data/rel, label=name, **tmodel_kw)
        
        ax.loglog( en, tot/rel, label='Total Model', **total_kw)
        err = obs**0.5/rel
        low_err = np.where(obs-err <= 0, 0.99*obs, err)/rel
        ax.errorbar(en,obs/rel,yerr=[low_err,err], label='Counts', **obs_kw )
        
        if relto is not None:
            # format relative count 
            ax.set_ylabel('Counts relative to {}'.format(relto), fontsize=12)
            ax.set( ylim=(1e-1,200.),  ) #default, can be overridden
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda val,pos: { 1.0:'1', 10.0:'10', 100.:'100'}.get(val,''))) 
        else:
            ax.set_ylabel('Counts per Bin', fontsize=12)
            ax.set( ylim=(100.,None), )

        """ make it look nicer """
        def gev_tf(val, pos=0):
            lookup ={100.:'0.1', 1e3:'1', 1e4:'10', 1e5:'100'}
            return lookup.get(val, '') 
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(gev_tf))  
        ax.set_xlabel(r'$\mathsf{Energy\ (GeV)}$', fontsize=12)

        ax.legend(loc=0,prop=dict(size=10))
        ax.grid(b=True, alpha=0.5)

        
    def plot_residuals(ax, count_data, 
                plot_kw=dict( linestyle=' ', marker='o', color='black',),
                show_chisq=True, plot_pulls=True,
                ):
        energy, obs,tot = np.array(count_data['energies']),count_data['observed'],count_data['total']
        ax.set_xscale('log')
        if not plot_pulls:
            fdev = 100*(obs-tot)/(tot)
            ax.errorbar(energy, fdev, yerr=100*(tot**-0.5), **plot_kw)
            ybound=4.0 
            ylabel='fract. dev'
            ax.set(xscale='log', ylabel='fract. dev (%)', ylim=(-ybound*1.1,ybound*1.1))
            nhigh = sum(fdev>ybound)
            if nhigh>0:  ax.plot(energy[fdev>ybound], [ybound]*nhigh, '^r', markersize=10) 
            nlow = sum(fdev<-ybound)
            if nlow >0:  ax.plot(energy[fdev<-ybound], [-ybound]*nlow, 'vr', markersize=10)
            ax.set_yticks([-2,0, 2])
        else:
            ylabel = 'pull'
            ybound = 3.5
            pull = (obs-tot)/np.sqrt(tot)
            axes[1].plot(energy, pull, 'ko')

            nhigh = sum(pull>3)
            if nhigh>0:  ax.plot(energy[pull>3], [3]*nhigh, '^r', markersize=10) 
            nlow = sum(pull<-3)
            if nlow >0:  ax.plot(energy[pull<-3], [-3]*nlow, 'vr', markersize=10) 
            ax.set( xscale='log', ylabel='pull', ylim=(-3.5,3.5) )
            ax.set_yticks([-2,0,2])


        ax.axhline(0, color = 'grey', ls='--')

        def gevticklabel(x):
            if x<100 or x>1e5: return ''
            elif x==100: return '0.1'
            return '%d'% (x/1e3)
        """ make it look nicer """
        #ax.set_xticklabels(map(gevticklabel, ax.get_xticks()))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda val,pos: { 100.:'0.1', 1e3:'1', 1e4:'10', 1e5:'100'}.get(val,'')))
        ax.set_xlabel(r'$\mathsf{Energy\ (GeV)}$')
        ax.set(xlim=(95.,None)) # to get first tick
        ax.grid(b=True,alpha=0.3)
        if show_chisq :
            ax.text(0.75, 0.8,'chisq={:.0f}'.format(count_data['chisq']), 
                transform = ax.transAxes, fontsize=10)


    if axes is None:
        plt.close(fignum) # close it if exists
        fig, axes = plt.subplots(1,2, sharex=True, num=fignum, figsize=(12,6))

    relto = kwargs.pop('relto', None)
    tsmin = kwargs.pop('tsmin', 10)
    emax = kwargs.pop('emax', None)
    count_data = get_counts(roi, event_type, tsmin=tsmin, emax=emax) #, integral=integral, merge_non_free=merge_non_free, merge_all=merge_all)

    plot_counts_and_models(axes[0], count_data)
    if len(axes>1): plot_residuals(axes[1], count_data, 
            show_chisq=kwargs.pop('show_chisq', True), 
            plot_pulls=kwargs.pop('plot_pulls', True))
    if outfile is not None: 
       print 'saving counts plot to %s...' %outfile ; sys.stdout.flush()
       plt.savefig(outfile)


def stacked_plots(roi, counts_dir=None, fignum=6, title=None, **kwargs):
    """ 
    Make stacked plots
        
        roi : A ROIstat object
            Uses the name as a title unless title specified
        counts_dir : None or the name of a folder
            In the folder case, makes a file name from the ROI name
            
        Creates the two Axes objects, and returns them
    """
    plt.close(fignum)
    oldlw = plt.rcParams['axes.linewidth']
    plt.rcParams['axes.linewidth'] = 2
    xlim, ylim = [kwargs.pop(x, None) for x in ('xlim','ylim')]
    figsize = kwargs.pop('size', (4,6))

    fig, axes = plt.subplots(2,1, sharex=True, num=fignum)

    fig.subplots_adjust(hspace=0)
    axes[0].tick_params(labelbottom=False)
    left, bottom, width, height = (0.15, 0.10, 0.75, 0.85)
    fraction = 0.8

    axes[0].set_position([left, bottom+(1-fraction)*height, width, fraction*height])
    axes[1].set_position([left, bottom, width, (0.999-fraction)*height])
    plot_counts(roi, axes=axes, outfile=None, **kwargs)
    
    plt.rcParams['axes.linewidth'] = oldlw

    axes[0].set_xlabel('') 

    if title is None:
        l,b = roi.roi_dir.l(), roi.roi_dir.b()
        if l>180: l-=360
        if hasattr(roi,'name'): fig.suptitle('{} ({:.0f},{:.0f})'.format(roi.name, l,b))
    else: fig.suptitle(title)
    if counts_dir is not None:
        if os.path.isdir(counts_dir) and hasattr(roi,'name'):
            fout = os.path.join(counts_dir, ('%s_counts.jpg'%roi.name) )
        else:
            fout = counts_dir
        print 'saving counts plot to %s ...' % fout, ; sys.stdout.flush()
        fig.savefig(fout)
        print 
    fig.set_facecolor('white')
    if figsize is not None:
            fig.set(figwidth=figsize[0], figheight=figsize[1])        
    if xlim is not None: fig.axes[0].set(xlim=xlim)
    if ylim is not None: feg.axes[0].set(ylim=ylim)
    return fig


def ROI_residuals(roi, ):
    """ Multiple plots of the residuals for the pixels in bands of an ROI
    """

    def plot_residual(roi, index=0, ax=None, **kwargs):

        b=roi[index] # the BandLike object
        band_label = '{:.0f} Mev {}'.format(b.band.energy, ['Front','Back'][b.band.event_type])
        
        # get pixel info: counts, model, positions 
        data = b.data
        factor = sum(data)/sum(b.model_pixels) #scale factor
        model = b.model_pixels*factor #rescaled model count distribution
        pixel_dirs = b.pixel_dirs
        
        # the chisquared: useful if counts high enough
        chi2 = sum((data-model)**2/model)

        # create a ZEA image object to get the transformation to image coordinates,
        #  and set up axes for display
        z = image.ZEA(roi.roi_dir, size=13, galactic=True, axes=ax )
        pix = np.array([z.pixel(sdir) for sdir in pixel_dirs])
        ax=z.axes
         
        scat=z.axes.scatter(pix[:,0],pix[:,1], c=data/model-1, marker='D', edgecolor='none', 
                        s=20000/len(pix), **kwargs);

        ax.text(0.05,0.93,band_label, transform=ax.transAxes)
        ax.text(0.05,0.05,'factor={:.3f}, chi2/ndf = {:.0f}/{}'.format(factor, chi2, len(b.data)-3),transform = ax.transAxes)
        
        return scat

    fig, axx = plt.subplots(2,4, figsize=(18,8), sharex=True, sharey=True)
    plt.subplots_adjust(right=0.9, hspace=0.15, wspace=0.1)

    for i,ax in enumerate(axx.flatten()):
        j = [0,2,4,6,1,3,5,7][i]
        scat=plot_residual(roi, j ,ax=ax, vmin=-0.1, vmax=0.1)
    cbax = fig.add_axes((0.92, 0.15, 0.02, 0.7) )
    cb=plt.colorbar(scat, cbax, orientation='vertical')
    cb.set_label('fractional deviation', fontsize=12)

    fig.suptitle('Residuals for {}'.format(roi.name), fontsize=14)
    fig.set_facecolor('white')
    
def ROI_pixel_counts(roi, ):
    """
    Multiple plots of the counts per pixel for first 4 bands, front and back
    """

    def plot(roi, index=0, ax=None, **kwargs):

        b=roi[index] # the BandLike object
        band_label = '{:.0f} Mev {}'.format(b.band.energy, ['Front','Back'][b.band.event_type])
        
        # get pixel info: counts, model, positions 
        data = b.data
        pixel_dirs = b.pixel_dirs
        

        # create a ZEA image object to get the transformation to image coordinates,
        #  and set up axes for display
        z = image.ZEA(roi.roi_dir, size=13, galactic=True, axes=ax )
        pix = np.array([z.pixel(sdir) for sdir in pixel_dirs])
        ax=z.axes
         
        scat=z.axes.scatter(pix[:,0],pix[:,1], c=data, marker='D', edgecolor='none', 
                        s=25000/len(pix), **kwargs);

        ax.text(0.05,0.93,band_label, transform=ax.transAxes)
        return scat

    fig, axx = plt.subplots(2,4, figsize=(18,8), sharex=True, sharey=True)
    plt.subplots_adjust(right=0.9, hspace=0.15, wspace=0.1)

    for i,ax in enumerate(axx.flatten()):
        j = [0,2,4,6,1,3,5,7][i]
        scat=plot(roi, j ,ax=ax,)
    cbax = fig.add_axes((0.92, 0.15, 0.02, 0.7) )
    cb=plt.colorbar(scat, cbax, orientation='vertical')
    cb.set_label('counts', fontsize=12)

    fig.suptitle('Counts for {}'.format(roi.name), fontsize=14)
    fig.set_facecolor('white')
