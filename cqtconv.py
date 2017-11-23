#!/usr/bin/env python 
import numpy as np
# https://github.com/librosa/librosa
import librosa
import librosa.display
import argparse
import os
from PIL import Image
from PIL import PngImagePlugin
import json
import scipy


FLAGS = None
# ------------------------------------------------------
# get any args provided on the command line
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--conversion', type=str, choices=['spec2cqt','cqt2spec'], help='direction of conversion', default='spec2cqt')
parser.add_argument('--filename', type=str, help='For single mode, enter filename', default=None) 
parser.add_argument('--rootdir', type=str, help='Roots folder where class folders containing audio files are kept', default='./') 
parser.add_argument('--outdir', type=str, help='Output directory', default='./output')
#parser.add_argument('--sr', type=int, help='Samplerate', default=None) 
parser.add_argument('--fftsize', type=int, help='Size of fft window in samples', default=1024)
#parser.add_argument('--hopsize', type=int, help='Size of frame hop through sample file', default=256)

FLAGS, unparsed = parser.parse_known_args()
print('\n FLAGS parsed :  {0}'.format(FLAGS))

#filetypes = ['.wav','.mp3'] not yet implimented. manually change fileExtList in listDirectory

def get_subdirs(a_dir):
    """ Returns a list of sub directory names in a_dir """ 
    return [name for name in os.listdir(a_dir)
            if (os.path.isdir(os.path.join(a_dir, name)) and not (name.startswith('.')))]


def listDirectory(directory, fileExtList):                                        
    """Returns a list of file info objects in directory that extension in the list fileExtList - include the . in your extension string"""
    fnameList = [os.path.normcase(f)
                for f in os.listdir(directory)
                    if (not(f.startswith('.')))]            
    fileList = [os.path.join(directory, f) 
               for f in fnameList
                if os.path.splitext(f)[1] in fileExtList]  
    return fileList , fnameList


def logSpect2PNG(outimg, fname, lwinfo=None) :
    
    info = PngImagePlugin.PngInfo()
    lwinfo = lwinfo or {}
    lwinfo['fileMin'] = str(np.amin(outimg))
    lwinfo['fileMax'] = str(np.amax(outimg))
    info.add_text('meta',json.dumps(lwinfo)) #info required to reverse scaling
    
    shift = int(lwinfo['scaleMax']) - int(lwinfo['scaleMin'])
    SC2 = 255*(outimg-int(lwinfo['scaleMin']))/shift
    savimg = Image.fromarray(np.flipud(SC2))

    pngimg = savimg.convert('L')  
    pngimg.save(fname,pnginfo=info)
    

def PNG2LogSpect(fname,scalemin,scalemax):

    """
    Read png spectrograms, expand to original scale and return numpy array.
    If not stored in one of png metadata, the values needed to undo previous scaling are required to be specified.
    """
    img = Image.open(fname)
    #info = PngImagePlugin.PngInfo()
    
    try:       
        img.text = img.text
        lwinfo = json.loads(img.text['meta'])
    except:
        print('PNG2LogSpect: no img.text, using user specified values!')
        lwinfo = {}
        lwinfo['scaleMin'] = scalemin #require to pass in
        lwinfo['scaleMax'] = scalemax
        #info.add_text('meta',json.dumps(lwinfo))
   
    minx, maxx = float(lwinfo['scaleMin']), float(lwinfo['scaleMax'])
    #minx, maxx = float(lwinfo['oldmin']), float(lwinfo['oldmax'])
    
    img = img.convert('L')
    outimg = np.asarray(img, dtype=np.float32)
    outimg = (outimg - 0)/(255-0)*(maxx-minx) + minx

    return np.flipud(outimg), lwinfo


def logfmap(I, L, H) :
    """
    % [M,N] = logfmap(I,L,H)
    I - number of rows in the original spectrogram
    L - low bin to preserve
    H - high bin to preserve
    
    %     Return a maxtrix for premultiplying spectrograms to map
    %     the rows into a log frequency space.
    %     Output map covers bins L to H of input
    %     L must be larger than 1, since the lowest bin of the FFT
    %     (corresponding to 0 Hz) cannot be represented on a 
    %     log frequency axis.  Including bins close to 1 makes 
    %     the number of output rows exponentially larger.
    %     N returns the recovery matrix such that N*M is approximately I
    %     (for dimensions L to H).
    %     
    % Ported from MATLAB code written by Dan Ellis:
    % 2004-05-21 dpwe@ee.columbia.edu
    """
    ratio = (H-1)/H;
    opr = np.int(np.round(np.log(L/H)/np.log(ratio))) #number of frequency bins in log rep + 1
    print('opr is ' + str(opr))
    ibin = L*np.exp(list(range(0,opr)*-np.log(ratio)))  #fractional bin numbers (len(ibin) = opr-1)
    
    M = np.zeros((opr,I))
    eps=np.finfo(float).eps
    
    for i in range(0, opr) :
        # Where do we sample this output bin?
        # Idea is to make them 1:1 at top, and progressively denser below
        # i.e. i = max -> bin = topbin, i = max-1 -> bin = topbin-1, 
        # but general form is bin = A exp (i/B)

        tt = np.multiply(np.pi, (list(range(0,I))-ibin[i]))
        M[i,:] = np.divide((np.sin(tt)+eps) , (tt+eps));

    # Normalize rows, but only if they are boosted by the operation
    G = np.ones((I));
    print ('H is ' + str(H))
    G[0:H] = np.divide(list(range(0,H)), H)
    
    N = np.transpose(np.multiply(M,np.matlib.repmat(G,opr,1)))
                   
    return M,N

def spect2CQT(topdir, outdir, fftSize, lowRow=1):
    """ 
        Creates psuedo constant-Q spectrograms from linear frequency spectrograms. 
        Creates class folders in outdir with the same structure found in topdir.
        
        Parameters
            topdir - the dir containing class folders containing png (log magnitude) spectrogram files. 
            outdir - the top level directory to write psuedo constantQ files to (written in to class subfolders)
            lowRow is the lowest row in the FFT that you want to include in the psuedo constant Q spectrogram
    """ 
    
    # First lets get the logf map we want
    LIN_FREQ_BINS = int(fftSize/2+1) #number of bins in original linear frequency mag spectrogram
    LOW_ROW = lowRow
    LOG_FREQ_BINS = int(fftSize/2+1) #resample the lgfmapped psuedo consantQ matrix to have this many frequency bins
    M,N = logfmap(LIN_FREQ_BINS,LOW_ROW,LOG_FREQ_BINS)
    
    
    subdirs = get_subdirs(topdir)
    count = 0
    for subdir in subdirs:

        fullpaths, _ = listDirectory(topdir + '/' + subdir, '.png')
            
        for idx in range(len(fullpaths)) : 
            fname = os.path.basename(fullpaths[idx])
            D, pnginfo = PNG2LogSpect(fullpaths[idx],None,None)  
            
            # Here's the beef
            MD = np.dot(M,D)
            MD = scipy.signal.resample(MD, LIN_FREQ_BINS) #downsample to something reasonable 

            #save
            #info={}
            pnginfo["linFreqBins"] =  LIN_FREQ_BINS
            pnginfo["lowRow"] = LOW_ROW
            pnginfo["logFreqBins"] = LOG_FREQ_BINS
            
            try:
                os.stat(outdir + '/' + subdir) # test for existence
            except:
                os.makedirs(outdir + '/' + subdir) # create if necessary
            
            print(str(count) + ': ' + subdir + '/' + os.path.splitext(fname)[0])
            logSpect2PNG(MD, outdir+'/'+subdir+'/'+os.path.splitext(fname)[0]+'.png',lwinfo=pnginfo)               
            
            count +=1
    print("COMPLETE")
    
def CQT2spec(topdir,outdir):
    
    subdirs = get_subdirs(topdir)
    count = 0
    for subdir in subdirs:

        fullpaths, _ = listDirectory(topdir + '/' + subdir, '.png')
            
        for idx in range(len(fullpaths)) : 
            fname = os.path.basename(fullpaths[idx])
            D, pnginfo = PNG2LogSpect(fullpaths[idx],None,None) 
            M,N = logfmap(pnginfo["linFreqBins"],pnginfo["lowRow"],pnginfo["logFreqBins"])
            resampledD = scipy.signal.resample(D, M.shape[0]) #upsample
            
            # Here's the beef
            ND = np.dot(N,resampledD)
            
            try:
                os.stat(outdir + '/' + subdir) # test for existence
            except:
                os.makedirs(outdir + '/' + subdir) # create if necessary
            
            print(str(count) + ': ' + subdir + '/' + os.path.splitext(fname)[0])
            logSpect2PNG(ND, outdir+'/'+subdir+'/'+os.path.splitext(fname)[0]+'.png',lwinfo=pnginfo)               
            
            count +=1
    print("COMPLETE")
    
if FLAGS.conversion == 'spec2cqt':
    spect2CQT(FLAGS.rootdir,FLAGS.outdir,FLAGS.fftsize,lowRow=10)
elif FLAGS.conversion == 'cqt2spec':
    CQT2spec(FLAGS.rootdir,FLAGS.outdir)