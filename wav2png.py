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
import math


FLAGS = None
# ------------------------------------------------------
# get any args provided on the command line
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('mode', type=str, help='Choices: folder | single. folder mode for wav files in directory structure, single mode for single files', default='folder')
parser.add_argument('--filename', type=str, help='For single mode, enter filename', default=None) 
parser.add_argument('--rootdir', type=str, help='Roots folder where class folders containing audio files are kept', default='./') 
parser.add_argument('--outdir', type=str, help='Output directory', default='./output')
parser.add_argument('--sr', type=int, help='Samplerate', default=None) 
parser.add_argument('--fftsize', type=int, help='Size of fft window in samples', default=1024)
parser.add_argument('--hopsize', type=int, help='Size of frame hop through sample file', default=512)
parser.add_argument('--dur', type=int, help='Make files this duration in sec. If unspecified keep original duration', default=None)

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


def wav2stft(fname, srate, fftSize, fftHop, dur) :
    try:
        audiodata, samplerate = librosa.load(fname, sr=srate, mono=True, duration=dur)
        #print(np.amax(np.abs(audiodata)))
        #print(np.amin(np.abs(audiodata)))
        #print(audiodata[50:70])
    except:
        print('can not read ' + fname)
        return
    
    #if srate == None:
    #    print('Using native samplerate of ' + str(samplerate))
    S = np.abs(librosa.stft(audiodata, n_fft=fftSize, hop_length=fftHop, win_length=fftSize,  center=False))                
    return S


def log_scale(x):
    output = np.log1p(x)
    return output


def findMinMax(img) :
    return np.amin(img),np.amax(img)


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
 
 
def checkScaling(topdir, outdir, srate, fftSize, fftHop, dur) :
    """ 
        Returns the max and min values of the log magnitude after STFT in the whole dataset.
        This is to provide a known and standardized mapping from [min,max] -> [0,255] when saving as a png image.
        
        Parameters
            topdir - the dir containing class folders containing wav files. 
            outdir - the top level directory to write wave files to (written in to class subfolders)
            dur - (in seconds) all files will be truncated or zeropadded to have this duration given the srate
            srate - input files will be resampled to srate as they are read in before being saved as wav files            
    """ 
    print("Now determining maximum log magnitude value in dataset for scaling to png...") 
    subdirs = get_subdirs(topdir)
    max_mag = 0
    min_mag = 0
    for subdir in subdirs:
        
        fullpaths, _ = listDirectory(topdir + '/' + subdir, '.wav') 
        
        for idx in range(len(fullpaths)) : 
            fname = os.path.basename(fullpaths[idx])
            
            D = wav2stft(fullpaths[idx], srate, fftSize, fftHop, dur)
            D = log_scale(D)
            minM,maxM = findMinMax(D)
            if minM > min_mag:
                min_mag = minM
            if maxM > max_mag:
                max_mag = maxM
    
    print("Dataset: min magnitude=",min_mag,"max magnitude=",max_mag)
    minScale = int(math.floor(min_mag))
    maxScale = int(math.ceil(max_mag)) 
    #print("New scale: min magnitude=",minScale,"max magnitude=",maxScale)
    print("Using [{0},{1}] -> [0,255] for png conversion".format(minScale,maxScale))
    pnginfo = {}
    pnginfo['datasetMin'] = str(min_mag)
    pnginfo['datasetMax'] = str(max_mag)
    pnginfo['scaleMin'] = str(minScale)
    pnginfo['scaleMax'] = str(maxScale)
    return pnginfo

    
def wav2Spect(topdir, outdir, srate, fftSize, fftHop, dur, pnginfo) :
    """ 
        Creates spectrograms for subfolder-labeled wavfiles. 
        Creates class folders for the spectrogram files in outdir with the same structure found in topdir.
        
        Parameters
            topdir - the dir containing class folders containing wav files. 
            outdir - the top level directory to write wave files to (written in to class subfolders)
            dur - (in seconds) all files will be truncated or zeropadded to have this duration given the srate
            srate - input files will be resampled to srate as they are read in before being saved as wav files            
    """ 
    
    subdirs = get_subdirs(topdir)
    count = 0
    for subdir in subdirs:
        
        fullpaths, _ = listDirectory(topdir + '/' + subdir, '.wav') 
        
        for idx in range(len(fullpaths)) : 
            fname = os.path.basename(fullpaths[idx])
            
            D = wav2stft(fullpaths[idx], srate, fftSize, fftHop, dur)
            D = log_scale(D)
            
            try:
                os.stat(outdir + '/' + subdir) # test for existence
            except:
                os.makedirs(outdir + '/' + subdir) # create if necessary
            
            print(str(count) + ': ' + subdir + '/' + os.path.splitext(fname)[0])

            logSpect2PNG(D, outdir+'/'+subdir+'/'+os.path.splitext(fname)[0]+'.png', lwinfo=pnginfo)
            
            count +=1
    print("COMPLETE") 
    
    
def wav2Spect_single(filename, srate, fftSize, fftHop, dur) :
    """ 
        Creates spectrograms from single wavfiles. 
    """    
    D = wav2stft(filename, srate, fftSize, fftHop, dur)
    D = log_scale(D)
    minM,maxM = findMinMax(D)
    
    print("Dataset: min magnitude=",minM,"max magnitude=",maxM)
    minScale = int(math.floor(minM))
    maxScale = int(math.ceil(maxM)) 
    print("Using [{0},{1}] -> [0,255] for png conversion".format(minScale,maxScale))
    pnginfo = {}
    pnginfo['datasetMin'] = str(minM)
    pnginfo['datasetMax'] = str(maxM)
    pnginfo['scaleMin'] = str(minScale)
    pnginfo['scaleMax'] = str(maxScale)
        
    print(str(0) + ': ' + os.path.splitext(filename)[0])
    logSpect2PNG(D, os.path.splitext(filename)[0] +'.png',pnginfo)

    print("COMPLETE") 


if FLAGS.mode == 'folder':
    pnginfo = checkScaling(FLAGS.rootdir,FLAGS.outdir,FLAGS.sr,FLAGS.fftsize,FLAGS.hopsize,FLAGS.dur)
    wav2Spect(FLAGS.rootdir,FLAGS.outdir,FLAGS.sr,FLAGS.fftsize,FLAGS.hopsize,FLAGS.dur,pnginfo)
elif FLAGS.mode == 'single':
    wav2Spect_single(FLAGS.filename,FLAGS.sr,FLAGS.fftsize,FLAGS.hopsize,FLAGS.dur)
else:
    raise ValueError("Not an acceptable mode! Choices: folder | single. folder mode for wav files in directory structure, single mode for single files")

