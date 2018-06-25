import pynbody
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pynbody.analysis.profile as profile
import glob
import math
import pynbody.filt as filt
import pynbody.analysis.halo as halo
import pickle

galaxies = ['g8.26e11']

for gal in galaxies:
    galaxy = '/scratch/nhs284/galaxies/' + gal + '/'
    datafiles = sorted(glob.glob(galaxy + '/g?.??e??' +  '.0????'))

    mbh = []
    mstar = []
    time = []
    loaded = []
    first = True
    for i in range(0, len(datafiles)):
        sim = pynbody.load(datafiles[i])
        loaded.append(sim)
        if first:
            try:
                h = sim.halos()
                h1 = h[1]
                false = False
            except:
                print('Error with Halos')
                        continue
        else:
            b = pynbody.bridge.OrderBridge(loaded[i-1], loaded[i])
            try:
                h = sim.halos()
                h1 = b(h[1])
            except:
                print('Error with Halos')
                continue

        h1.physical_units()
        print('Halo ID: ' + str(h1.properties['halo_id']))
        try:
            halo.center(h1,mode='pot')
        except:
            print('Not enough particles to center')

        z =sim.properties['z']
        r = h1.properties['Rvir']
        print(z)
        time.append(z)
        mbh_f = filt.LowPass('tform','0 yr')
        star_f = filt.HighPass('tform', '0 yr')
        r_filt = filt.Sphere( str(r) + ' kpc' )
        stellar_f = star_f and r_filt
        #print('r: ' + str(r))

        try:
            new = np.amax(h1.star[mbh_f]['mass'].in_units('Msol'))
            mbh.append(new)
        except:
            #   print("no black holes")
            mbh.append(0)
        try:
            new = np.sum(h1.star[stellar_f]['mass'].in_units('Msol'))
            mstar.append(new)
            #  print(new)
        except:
            #print('no stars')
            mstar.append(0)

    to_pickle = {'z': time,'mbh':mbh, 'mstar' : mstar}

    with open('mbh-mstar-' + gal + '-redshift.pkl', 'wb') as fp:
        pickle.dump(to_pickle, fp)


