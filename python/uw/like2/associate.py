"""
 Manage associations
 $Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/associate.py,v 1.5 2015/12/03 17:08:06 burnett Exp $
 author: T. Burnett <tburnett@uw.edu>
"""
import os, glob
import numpy as np
from skymaps import SkyDir
from uw.like import srcid  # gtsrcid work-alike from Eric

#### late in the day, wire in corrections for uw1409
class ApplySystematic(object):
    def __init__(self, systematic):
        self.systematic = systematic
    def __call__(self, a,b, glat):
        sf =  self.systematic[0] if np.abs(glat)>5 else self.systematic[2]
        sa = self.systematic[1]
        r95 = 60* 2.45 * np.sqrt(a*b)
        return np.sqrt(sf**2 + (sa/r95)**2)
sfunc = ApplySystematic([1.075, 0.475, 1.48])

class SrcId(srcid.SourceAssociation):
    """
    adapter to Eric Wallace's source association code
    """
    def __init__(self, catalog_path='$FERMI/catalog/', srcid='srcid', classes='all_but_gammas', 
        apply_systematic=False,
        quiet=True):
        """ 
        catalog_path : string
            path to the catalogs, expect to find srcid/classes under it
        clases: ['all' | 'all_but_gammas' | list ]
            list of classes to apply,
        
        """
        self.apply_systematic = apply_systematic
        self.classes = classes
        catalog_path = os.path.expandvars(catalog_path)
        d = os.path.join(catalog_path, srcid, 'classes')
        q = glob.glob(os.path.join(d, '*.py'))
        assert len(q)>0, 'no association classes found in folder %s' % d
        allclasses =[os.path.split(u)[-1].split('.')[0] for u in q if '__init__' not in u]
        if classes=='all': 
            # special tag to really get everything
            q = glob.glob(os.path.join(d, '*.py'))
            self.classes = allclasses
        elif self.classes=='all_but_gammas' and srcid=='srcid':
            self.classes = ['agn', 'bllac', 'bzcat', 'cgrabs', 'crates', 'crates_fom', 'dwarfs', 
            'galaxies', 'globular', 'hmxb', 'ibis', 'lbv', 'lmxb',  'ocl', 'ostar', 
             #'pulsar_fom',
            'pulsar_lat', 'pulsar_big', #'msp', 'pulsar_high',  'pulsar_low', 'pulsar_nonATNF', 
            'pwn', 'qso', 'seyfert', 'seyfert_rl', 'snr', 'snr_ext', 'starbursts', 'tev']
        elif srcid=='lott_srcid':
            self.classes=['agn', 'arxas', 'at20g', 'bat', 'bllac', 'bzcat', 'cgrabs',
               'crates', 'ecc', 'ercsc030', 'ercsc044', 'ercsc070', 'ercsc100',
               'ercsc143', 'ercsc217', 'ercsc353', 'ercsc545', 'ercsc857', 'esz',
               'fgl2us', 'gal', 'gc', 'hmxb', 'ibis', 'iras', 'lbv',
               'lmxb', 'msp', 'ocl', 'ostar', 'pulhigh', 'pulother', 'pulsar_lat',
               'pwn', 'qso', 'sey', 'seyrl', 'snr', 'vcs', 'wise', 'wisestrip',
               'wr']
        else:
            self.classes=classes
        for c in self.classes:
            if c not in allclasses:
                txt = 'class %s not in set classes: %s' % (c, allclasses)
                raise Exception(txt)
        super(SrcId, self).__init__(os.path.join(catalog_path, srcid),quiet=quiet)
        self.class_list = self.classes # will be used by the id method
     
    def __str__(self):
        return 'SrcId(%s)' %self.classes
    
    def __repr__(self):
        return '%s.%s: %s' % (self.__module__, self.__class__.__name__, self.classes)
        
    def __call__(self, name, pos, error):
        """ name: source name, ignored, for reference
            pos: a SkyDir object |; (ra,dec) cuple
            error: [float | tuple]
                a or tuple: (a,b,ang) or, (ra,dec,a,b,ang,)...
                a,b: 1-sigma elipse in deg; ang orientation degrees
            returns None, or a dictionary consistent with Association above. (elements sorted by prob.)
        """
        if not hasattr(error, '__iter__'):
            error = (error,error,0)
        else:
            if len(error)==7: 
                error = error[3:6] 
            assert len(error)==3, 'wrong length for error ellipse specification'
        if not isinstance(pos, SkyDir):
            pos = SkyDir(*pos)

        # late in the day: apply systematic correction to a,b here
        if self.apply_systematic:
            factor = sfunc(error[0],error[1], pos.b())
            error[0] *= factor
            error[1] *= factor

        source_ass = self.id(pos,error)
        # select first association per catalog, rearrange to sort on over-all prob.
        items =  source_ass.items()
        candidates = [(v[0][1], v[0][0], v[0][2], key, v[0][3]) for key,v in items if v!={}]
        if len(candidates)==0: return None
        candidates.sort(reverse=True)
        # format as a dict to conform to Association above (TODO: a recarray would be cleaner)
        d = {
            'name':  [a[1] for a in candidates],
            'cat':   [a[3] for a in candidates],
            'dir':   [a[2] for a in candidates],
            'prob':  [a[0] for a in candidates],
            'ra':    [a[2].ra() for a in candidates],
            'dec':   [a[2].dec() for a in candidates],
            'ang':   [np.degrees(a[2].difference(pos)) for a in candidates],
            'deltats':[a[4] for a in candidates],
            }
        # stuff depending on the catalog 
        cats = [self.catalogs[a[3]] for a in candidates]
        d['prior'] =   [cat.prior for cat in cats]
        d['density'] = [cat.local_density(pos) for cat in cats]
        #d['fom'] =     [cat.fom for cat in cats]
        return d
        
    def positional_likelihood(self, name, pos, error):
        a,b,ang = error

class BzcatId(SrcId):

    def __init__(self):
        super(BzcatId, self).__init__(classes=['bzcat'])
        #self.catalogs['bzcat'].prob_threshold=0.1
        
        
def make_association(source, tsf, associate, quiet=False):
    if not quiet: print ' %s association(s) ' % source.name,
    try:    ell = source.ellipse
    except: ell = None
    if ell is None:
        if not quiet: print '...no localization'
        source.associations = None
        return
    assert len(ell)>6, 'invalid ellipse for source %s' % source.name
    try:
        adict = associate(source.name, SkyDir(ell[0],ell[1]), ell[2:5]) 
    except srcid.SrcidError, msg:
        if not quiet: print 'Association error for %s: %s' % (source.name, msg)
        adict=None
    except Exception, msg:
        if not quiet: print 'Exception associating %s: %s' %( source.name, msg)
        adict=None
        raise
    source.associations = adict 
    if adict is not None:
    
        ts_local_max=tsf( SkyDir(ell[0],ell[1]) )
        adict['deltats'] = [ts_local_max-tsf(d) for d in adict['dir']]
        if not quiet: print '\n   cat         name                  ra        dec         ang     prob    Delta TS'
        #       15 Mrk 501               253.4897   39.7527    0.0013      0.41
        fmt = '   %-11s %-20s%10.4f%10.4f%10.4f%8.2f%8.1f' 
        for i,id_name in enumerate(adict['name']):
            tup = (adict['cat'][i], id_name, adict['ra'][i], adict['dec'][i], adict['ang'][i], 
                    adict['prob'][i],adict['deltats'][i])
            if not quiet: print fmt % tup
    else:
        if not quiet: print '...None  found'
