# Read DICOM IMAGES
# Read CSV Masks
from pydicom import dcmread
import pandas as pd
import cv2
from skimage.draw import polygon, polygon_perimeter
from skimage.transform import resize
import numpy as np
import glob
import os
from PIL import Image
from wand.image import Image as wi

inputPath = 'Patients'
outputPath = 'dataset'
imagesPath = outputPath + "/image"
labelsPath = outputPath + "/label"
singleMasksPath = outputPath + "/SingleMasks"
masksOrder = ['CZ', 'PZ', 'TZ', 'TUM']
path_TIFF = outputPath + "/TIFF"
img_TIFF = path_TIFF + "/image"
label_TIFF = path_TIFF + "/label"
path_PNG = outputPath + "/PNG"
img_PNG = path_PNG + "/images"
label_PNG = path_PNG + "/labels"
path_JPG = outputPath + "/JPG"
img_JPG = path_JPG + "/images"
label_JPG = path_JPG + "/labels"

if not os.path.exists(outputPath):
    os.makedirs(outputPath)
if not os.path.exists(imagesPath):
    os.makedirs(imagesPath)
if not os.path.exists(labelsPath):
    os.makedirs(labelsPath)
if not os.path.exists(singleMasksPath):
    os.makedirs(singleMasksPath)

if not os.path.exists(path_TIFF):
    os.makedirs(path_TIFF)
if not os.path.exists(img_TIFF):
    os.makedirs(img_TIFF)
if not os.path.exists(label_TIFF):
    os.makedirs(label_TIFF)

if not os.path.exists(path_PNG):
    os.makedirs(path_PNG)
if not os.path.exists(img_PNG):
    os.makedirs(img_PNG)
if not os.path.exists(label_PNG):
    os.makedirs(label_PNG)

if not os.path.exists(path_JPG):
    os.makedirs(path_JPG)
if not os.path.exists(img_JPG):
    os.makedirs(img_JPG)
if not os.path.exists(label_JPG):
    os.makedirs(label_JPG)

def dcmtojpg(ds, outputPath, section, paciente,contador):
    dcmimage = ds.pixel_array
    dcmimage = resize(dcmimage, (256, 256), anti_aliasing=True)
    scaled_image = (np.maximum(dcmimage, 0) / dcmimage.max()) * 255.0
    scaled_image = np.float32(scaled_image)
    scaled_image /= 255.0
    np.save(imagesPath + '/IMG_' + contador + '.npy', scaled_image)


def mergeMasks(imagesPath, outputPath, masksOrder):
    basemask = np.zeros((256, 256))
    for finalImage in os.listdir(imagesPath):
        #print(finalImage.rsplit(".", 1)[0])
        finalMergedMask = basemask.copy()
        for idx, mask in enumerate(masksOrder):
            #print(idx, mask)
            # maskShade = (idx * 40 + 50)
            maskShade = (idx + 1)                               # Fondo=0, CZ=1, PZ=2, TZ=3, TUM=4
            importedMask = cv2.imread(singleMasksPath + '/' + finalImage.rsplit(".", 1)[0] + '_' + mask + '.png', cv2.IMREAD_GRAYSCALE)
            finalMergedMask[importedMask >= 128] = maskShade
            # finalMergedMask = (finalMergedMask + importedMask)
            # finalMergedMask[finalMergedMask > maskShade] = maskShade
        finalMergedMask = resize(finalMergedMask, (256, 256), anti_aliasing=True)
        # finalMergedMask = np.uint8(finalMergedMask)
        # finalMergedMask -= 1
        # finalMergedMask[finalMergedMask > 250] = 0
        #finalImage_x = Image.fromarray(finalMergedMask)
        #finalImage_x = finalImage_x.convert('L')
        #finalImage_x.save(finalMaskPath + '/IMG_' + contador + '_' + masksOrder[idx] + '.png')
        np.save(labelsPath + '/' + finalImage.rsplit(".", 1)[0] + '.npy', finalMergedMask)


def generateMasks(ds, groundtruths, outputPath, masksOrder, section, paciente, contador):
    dcmimage = ds.pixel_array.astype(float)
    basemask = np.zeros((dcmimage.shape[0], dcmimage.shape[1]))
    for idx, gttype in enumerate(groundtruths):
        if not gttype.empty:
            currentmask = basemask.copy()
            X = list(map(int, gttype[0].tolist()))
            Y = list(map(int, gttype[1].tolist()))
            points_r, points_c = Y, X
            interior_r, interior_c = polygon(points_r, points_c)
            perimeter_r, perimeter_c = polygon_perimeter(points_r, points_c)
            currentmask[perimeter_r, perimeter_c] = 255
            currentmask[points_r, points_c] = 255
            currentmask[interior_r, interior_c] = 255
        else:
            currentmask = basemask.copy()
        currentmask = resize(currentmask, (256, 256), anti_aliasing=True)
        currentImage = Image.fromarray(currentmask)
        currentImage = currentImage.convert('L')
        currentImage.save(singleMasksPath + '/IMG_' + contador + '_' + masksOrder[idx] + '.png')



contador = "0".zfill(4)
for paciente in os.listdir(inputPath):
    if os.path.isdir(inputPath + '/' + paciente):
        #print(paciente)
        for section in os.listdir(inputPath + '/' + paciente):
            if os.path.isdir(inputPath + '/' + paciente + '/' + section):
                #print(paciente, section)
                fpath = glob.glob(inputPath + '/' + paciente + '/' + section + '/input/*.dcm')[0]
                ds = dcmread(fpath)
                mpath = inputPath + '/' + paciente + '/' + section + '/csv'
                contador = str(int(contador) + 1).zfill(4)

                dcmtojpg(ds, outputPath, section, paciente, contador)
                czpath = mpath + '/CZ.csv'
                pzpath = mpath + '/PZ.csv'
                tzpath = mpath + '/TZ.csv'
                tumpath = mpath + '/TUM.csv'
                emptyDF = pd.DataFrame()
                czcoord, pzcoord, tzcoord, tumcoord = emptyDF, emptyDF, emptyDF, emptyDF
                if os.path.exists(czpath):
                    czcoord = pd.read_csv(czpath, header=None)
                if os.path.exists(pzpath):
                    pzcoord = pd.read_csv(pzpath, header=None)
                if os.path.exists(tzpath):
                    tzcoord = pd.read_csv(tzpath, header=None)
                if os.path.exists(tumpath):
                    tumcoord = pd.read_csv(tumpath, header=None)

                groundtruths = [emptyDF, emptyDF, emptyDF, emptyDF]
                dcmimage = ds.pixel_array.astype(float)
                try:
                    groundtruths[0] = czcoord
                except:
                    pass

                try:
                    groundtruths[1] = pzcoord
                except:
                    pass

                try:
                    groundtruths[2] = tzcoord
                except:
                    pass

                try:
                    groundtruths[3] = tumcoord
                except:
                    pass

                generateMasks(ds, groundtruths, outputPath, masksOrder, section, paciente, contador)


mergeMasks(imagesPath, outputPath, masksOrder)


image_files = os.listdir(imagesPath)
label_files = os.listdir(labelsPath)

def get_tiff(imagesPath, labelsPath, img_TIFF, label_TIFF, image_files, label_files):
    for i in range(0, len(image_files)):
        original = np.load(imagesPath + '/' + str(image_files[i]))
        label = np.load(labelsPath + '/' + str(label_files[i]))
        img_orig = Image.fromarray(np.float32(original))
        img_label = Image.fromarray(np.uint8(label))
        image = img_orig.save(f"{img_TIFF}/{i}.tiff")
        label = img_label.save(f"{label_TIFF}/{i}.tiff")

def get_PNG(image_files, label_files, img_PNG, label_PNG):
    for i in range(0, len(image_files)):
        with wi(filename = f'{img_TIFF}/{i}.tiff') as Sampleimg:  
            Sampleimg.format = 'png' 
            Sampleimg.save(filename = os.path.join(img_PNG,f"{i}.png"))
        with wi(filename = f'{label_TIFF}/{i}.tiff') as Sampleimg:  
            Sampleimg.format = 'png' 
            Sampleimg.save(filename = os.path.join(label_PNG,f"{i}.png"))    

def get_JPG(image_files, label_files, img_JPG, label_JPG):
    for i in range(0, len(image_files)):
        with wi(filename = f'{img_TIFF}/{i}.tiff') as Sampleimg:  
            Sampleimg.format = 'jpg' 
            Sampleimg.save(filename = os.path.join(img_JPG,f"{i}.jpg"))
        with wi(filename = f'{label_TIFF}/{i}.tiff') as Sampleimg:  
            Sampleimg.format = 'jpg' 
            Sampleimg.save(filename = os.path.join(label_JPG,f"{i}.jpg"))

get_tiff(imagesPath, labelsPath, img_TIFF, label_TIFF, image_files, label_files)
get_PNG(image_files, label_files, img_PNG, label_PNG)
get_JPG(image_files, label_files, img_JPG, label_JPG)