"""
Output the ROI info to as a pickle file.
$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/to_healpix.py,v 1.14 2017/08/02 23:06:41 burnett Exp $
"""
import os, pickle, time
import numpy as np
from skymaps import Band

def pickle_dump(roi,  pickle_dir, dampen, ts_min=0, **kwargs):
    """ dump the source information from an ROI constructed from the sources here
    ts_min : float
        threshold for saving. But if name starts with 'PSR' or extended save anyway
    """
    assert os.path.exists(pickle_dir), 'output folder not found: %s' %pickle_dir
    name = roi.name.strip()
    fname = kwargs.pop('fname', name)
    filename=os.path.join(pickle_dir,fname+'.pickle')
    
    # start output record, a dict
    output = dict()
    output['name'] = name
    output['skydir']  = roi.roi_dir
    output['parameters'] = roi.sources.parameters[:] 
    output['gradient'] = np.asarray(roi.gradient(), np.float32)
    output['correlations'] = getattr(roi,'correlations', None)

    # add a record to the history read in originally
    history = roi.sources.history if hasattr(roi.sources, 'history') else []
    history.append( 
        dict(stream=kwargs.get('stream', '-1'),
            time=time.asctime(),
            logl=roi.log_like(),
            dampen= dampen,
            )
        )
    output['history'] = history
    
    # add diffuse and extended info
    diffuse_sources =  [s for s in roi.sources if s.isglobal or s.isextended]
    
    output['diffuse'] = [s.spectral_model for s in diffuse_sources] #roi.dsm.models
    output['diffuse_names'] = [s.name for s in diffuse_sources]  #roi.dsm.names
    try:
        output['diffuse_normalization'] = roi.sources.diffuse_normalization
    except:
        print 'No diffuse normalization DataFrame to save'
        
    # a dict for all variable sources
    sources=dict()
    output['sources']= sources
    
    inside = lambda sdir: Band(12).index(sdir)==Band(12).index(roi.roi_dir)
    for s in filter(lambda s: np.any(s.model.free) or s.isextended and inside(s.skydir),
         roi.sources[:]):

        try:
            pivot_energy = s.spectral_model.pivot_energy()
        except: # if not fit
            pivot_energy = None 
        if pivot_energy is None: pivot_energy=s.spectral_model.e0 # could be a pulsar?
        
        if 'sedrec' not in s.__dict__ or s.sedrec is None:
            print 'warning: no sedrec in %s' %s.name
            s.sedrec = None
        sedrec = s.sedrec
        loc = s.__dict__.get('loc', None)
        ts = s.ts if hasattr(s, 'ts') else roi.TS(s.name)
        if ts < ts_min and (s.name[:3]!='PSR' and not s.isextended): 
            print 'Not saving source %s: ts=%.1f < %.1f' % (s.name, ts, ts_min)
            continue
        sources[s.name]=dict(
            skydir=s.skydir, 
            model=s.spectral_model,
            isextended=s.isextended,
            ts = ts,
            sedrec = sedrec,
            profile=s.__dict__.get('profile',None),
            band_ts=0 if sedrec is None else sedrec.ts.sum(),
            ts_beta= s.__dict__.get('ts_beta', np.nan),
            pivot_energy = pivot_energy,
            # if ellipse or adict not done, but already in pickle, keep them
            ellipse= s.__dict__.get('ellipse', None), 
            moment= s.__dict__.get('ellipsex', None), #results, if any, of moment localization analysis
            associations = s.__dict__.get('associations',None),
            fixed_spectrum=s.__dict__.get('fixed_spectrum', False), #flag that spectrum should not be refit
            eflux = s.spectral_model.i_flux(e_weight=1, cgs=True, error=True),
            )
    output.update(kwargs) # add additional entries from kwargs
    with open(filename,'wb') as f:  #perhaps overwrite
        pickle.dump(output,f)
    print 'saved pickle file to %s' % filename
        
