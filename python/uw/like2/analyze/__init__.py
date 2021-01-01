"""
Package containing source analysis code
$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/analyze/__init__.py,v 1.21 2017/11/18 22:26:37 burnett Exp $

"""
# this is a list of modules in this package that have a class implementing an 'all_plots()' method.
# see app.py, which has an "import *" to include all of them
# (comment out those not relevant)
__all__=[
 'associations',
# 'components',

 'counts',

 #'collection',
 'config',
 #'cputime',
 #'data',
 'diffuse_fits',
 'diffuse_correction',
 'environment',
 'export',
 #'find_peak',
 #'fluxcorr',
 # 'fluxcorriso',
 #'frontback',
 #'galactic',
 #'galacticspectra',
 'gtlikecomparison',
 'hptables',
 #'isotropic',
 #'isotropicspectra',
 #'limb',
 #'limbrefit',
 'localization',
 'isotropic',
 #'localization1k',
 # 'pgwseedcheck',
 # 'maps',
 # 'pgwave',
 #'ptstable',
 #'pulsarseedcheck',
'pulsars', 
 #'roi_info',
 'residuals',
 'seedcheck',
 'sourcecomparison',
 'sourceinfo',
 #'sourcetotal',
 #'sunmoon',
 #'sunmoonrefit',
 #'uwsourcecomparison',
 'transientinfo',
 'transient_catalog',
 'residual_maps',
 'gc_comparison',
 ]

