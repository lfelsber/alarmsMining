# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
# from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
# from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small

import matplotlib.pyplot as plt
import numpy as np
import os
try:
    import PIL.Image as im
except ModuleNotFoundError :
    import Pillow.Image as im
import shutil
from datetime import datetime
import pandas as pd

###############################################################################
# Download utilities
###############################################################################


def download(url, filename):
    if not os.path.exists(filename):
        print("Download: %s ---> %s" % (url, filename))
        response = six.moves.urllib.request.urlopen(url)
        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)


###############################################################################
# Plot utility
###############################################################################


def load_image(path, size):
    ret = im.open(path)
    ret = ret.resize((size, size))
    ret = np.asarray(ret, dtype=np.uint8).astype(np.float32)
    if ret.ndim == 2:
        ret.resize((size, size, 1))
        ret = np.repeat(ret, 3, axis=-1)
    return ret


def get_imagenet_data(size=224):
    base_dir = os.path.dirname(__file__)

    # ImageNet 2012 validation set images?
    with open(os.path.join(base_dir, "images", "ground_truth_val2012")) as f:
        ground_truth_val2012 = {x.split()[0]: int(x.split()[1])
                                for x in f.readlines() if len(x.strip()) > 0}
    with open(os.path.join(base_dir, "images", "synset_id_to_class")) as f:
        synset_to_class = {x.split()[1]: int(x.split()[0])
                           for x in f.readlines() if len(x.strip()) > 0}
    with open(os.path.join(base_dir, "images", "imagenet_label_mapping")) as f:
        image_label_mapping = {int(x.split(":")[0]): x.split(":")[1].strip()
                               for x in f.readlines() if len(x.strip()) > 0}

    def get_class(f):
        # File from ImageNet 2012 validation set
        ret = ground_truth_val2012.get(f, None)
        if ret is None:
            # File from ImageNet training sets
            ret = synset_to_class.get(f.split("_")[0], None)
        if ret is None:
            # Random JPEG file
            ret = "--"
        return ret

    images = [(load_image(os.path.join(base_dir, "images", f), size),
               get_class(f))
              for f in os.listdir(os.path.join(base_dir, "images"))
              if f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")]
    return images, image_label_mapping


def plot_image_grid(grid,
                    row_labels_left,
                    row_labels_right,
                    col_labels,
                    file_name=None,
                    figsize=None,
                    dpi=224):
    n_rows = len(grid)
    n_cols = len(grid[0])
    
    col_labels[0]+='; x:alarm type; y:time (upwards)'
    
    if figsize is None:
        imshape=grid[0][0].shape
        figsize = (n_cols*3, n_rows*1.1*max(1,imshape[0]/imshape[1]))

    #plt.clf()
    plt.rc("font", family="sans-serif")

    plt.figure(figsize=figsize)
    
    for r in range(n_rows):
        for c in range(n_cols):
            ax = plt.subplot2grid(shape=[n_rows+1, n_cols], loc=[r+1, c])
            # TODO controlled color mapping wrt all grid entries,
            # or individually. make input param
            if grid[r][c] is not None:
                ax.imshow(grid[r][c], cmap='Blues', interpolation='none')
#                plt.colorbar(ac,ax=ax)
            else:
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            # column labels
            if not r:
                if col_labels != []:
                    ax.set_title(col_labels[c],
                                 rotation=22.5,
                                 horizontalalignment='left',
                                 verticalalignment='bottom')

            # row labels
            if not c:
                if row_labels_left != []:
                    txt_left = [l+'\n' for l in row_labels_left[r]]
                    ax.set_ylabel(
                        ''.join(txt_left),
                        rotation=0,
                        verticalalignment='center',
                        horizontalalignment='right',
                    )

            if c == n_cols-1:
                if row_labels_right != []:
                    txt_right = [l+'\n' for l in row_labels_right[r]]
#                    ax2 = ax.twinx()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_ylabel(
                        ''.join(txt_right),
                        rotation=0,
                        verticalalignment='center'
#                        horizontalalignment='right'
                    )
                    ax.yaxis.set_label_coords(2,.5)
                    
                   
#    plt.tight_layout()
#    ax.set_title('x-alarm type; y-time (upwards)')
#    plt.title('x-alarm type; y-time (upwards)')

    if file_name is None:
        plt.show()
    else:
#        plt.show()
        print('Saving figure to {}'.format(file_name))
        plt.savefig(file_name, orientation='landscape', dpi=dpi)
#        plt.savefig(file_name)
        
def plot_image_grid_w_legend(grid,
                    row_labels_left,
                    row_labels_right,
                    col_labels,
                    file_name=None,
                    signals_list=None,
                    figsize=(26,24),
                    scaler=None,
                    plot_factor=1,
                    dpi=324,
                    zoom_relevant=False,
                    times_idx=None):
    n_rows = len(grid)
    n_cols = len(grid[0])
    
    col_labels[0]+='; x:alarm type; y:time (upwards)'
    
    if figsize is None:
        imshape=grid[0][0].shape
        # figsize = (n_cols*4, n_rows*1.1*max(1,imshape[0]/imshape[1])+3)
        figsize = (plot_factor*n_cols*1*imshape[1]/imshape[0]+12, plot_factor*n_rows*2*imshape[0]/imshape[1]+10)

    #plt.clf()
    # plt.rc("font", family="sans-serif")

    plt.figure(figsize=figsize)
    plt.rcParams['font.family']='serif'
    
    for r in range(n_rows):
        for c in range(n_cols):
            ax = plt.subplot2grid(shape=[n_rows+1, n_cols], loc=[r+1, c])
            # TODO controlled color mapping wrt all grid entries,
            # or individually. make input param
            if grid[r][c] is not None:
                
                if c ==0 or zoom_relevant==False:
                    input_relevance=grid[r][c]
                    X,Y=np.meshgrid(np.arange(input_relevance.shape[1]+1), np.arange(input_relevance.shape[0]+1))
                    plt.pcolormesh(X, np.flip(Y,axis=0), input_relevance, cmap='Blues')
                    # ax.imshow(grid[r][c], cmap='Blues', interpolation='none')
                else:
                    ac_im=grid[r][c]
                    indmax=np.argpartition(np.max(ac_im,axis=0),-15)[-15:]
                    # ax.imshow(grid[r][c][:,indmax], cmap='Blues', interpolation='none')
                    input_relevance=grid[r][c][:,indmax]
                    X,Y=np.meshgrid(np.arange(input_relevance.shape[1]+1), np.arange(input_relevance.shape[0]+1))
                    plt.pcolormesh(X, np.flip(Y,axis=0), input_relevance, cmap='Blues')
                ax.set_yticks(np.arange(len(grid[r][c][:,0])))
                try:
                    ax.set_yticks(np.linspace(0,input_relevance.shape[0],10))
                    start_time=signals_list[2][times_idx[r]][0]
                    end_time=signals_list[2][times_idx[r]][1]
                    ax.set_yticklabels(pd.date_range(start_time,end_time,periods=10+1), fontsize=3)
                    # ax.set_yticklabels([datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in list(scaler.inverse_transform(grid[r][0])[:,[i for i, s in enumerate(signals_list[0]) if 'UT' in s]])], fontsize=3)
                except:
                    ('xlabeling does not work yet with importance zoom')

#                plt.colorbar(ac,ax=ax)
            else:
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
            ax.set_xticks([])
            # ax.set_yticks([])
            
            if zoom_relevant==True:
                ax.set_xticks(np.arange(len(indmax))+.5)
                ax.set_xticklabels([signals_list[0][i] for i in indmax], rotation=90, fontsize=5)

            # column labels
            if not r:
                if col_labels != []:
                    ax.set_title(col_labels[c],
                                 rotation=22.5,
                                 horizontalalignment='left',
                                 verticalalignment='bottom')

            # row labels
            if not c:
                if row_labels_left != []:
                    txt_left = [l+'\n' for l in row_labels_left[r]]
                    ax.set_ylabel(
                        ''.join(txt_left),
                        rotation=0,
                        verticalalignment='center',
                        horizontalalignment='right',
                    )

            if c == n_cols-1:
                if row_labels_right != []:
                    txt_right = [l+'\n' for l in row_labels_right[r]]
#                    ax2 = ax.twinx()
                    ax.set_xticks([])
                    # ax.set_yticks([])
                    ax.set_ylabel(
                        ''.join(txt_right),
                        rotation=0,
                        verticalalignment='center'
#                        horizontalalignment='right'
                    )
                    ax.yaxis.set_label_coords(2,.5)
                    
            if zoom_relevant ==  False:
                ax.set_xticks(np.arange(len(signals_list[0]))+.5)
                ax.set_xticklabels(signals_list[0], rotation=90, fontsize=5)
                
                
                    
                   
#    ax.set_title('x-alarm type; y-time (upwards)')
    plt.suptitle('Predicting '+str(signals_list[1]))
    plt.tight_layout()


    if file_name is None:
        plt.show()
    else:
#        plt.show()
        print('Saving figure to {}'.format(file_name))
        plt.savefig(file_name, orientation='landscape', dpi=dpi)
#        plt.savefig(file_name)
    plt.close() #saves memory