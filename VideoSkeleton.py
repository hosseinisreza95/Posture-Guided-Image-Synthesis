import numpy as np
import cv2
import os
import pickle
import sys
import math
import gc

from VideoReader import VideoReader
from Skeleton import Skeleton


def filename_change_ext(filename_full, nouvelle_extension):
    # Divise le nom de fichier en base et extension
    base = os.path.basename(filename_full)
    base = os.path.splitext(base)
    base = base[0]
    _, extension_actuelle = os.path.splitext(filename_full)
    path = os.path.dirname(filename_full)
    nouveau_nom_fichier = path + "/" + base + nouvelle_extension
    return nouveau_nom_fichier, path, base



def combineTwoImages(image1, image2):
    height = max(image1.shape[0], image2.shape[0])
    # Calculate the combined width for the two images
    combined_width = image1.shape[1] + image2.shape[1]
    # Create an empty canvas with the combined width and height
    combined_image = np.zeros((height, combined_width, 3), dtype=np.uint8)
    # Copy the first image to the left side of the canvas
    combined_image[:image1.shape[0], :image1.shape[1]] = image1
    # Copy the second image to the right side of the canvas
    combined_image[:image2.shape[0], image1.shape[1]:] = image2
    return combined_image




class VideoSkeleton:
    """ 
    Class that associate a skeleton to each frame of a video
       self.im : nparray<str> => im[i] filename of the image
       self.ske : nparray<Skeleton> => ske[i] the skeleton
       Proc draw() : display all the frame image+skeleton
    """
    def __init__(self, filename, forceCompute=False, modFrame=10, newVideoWidth=800, cropRatio=1.0, isCrop=True):
        """
        filename : str => video filename
        forceCompute : boolean => recompute the skeleton if True
        modFrame : int => only one frame every modFrame is processed
        new_video_width : int => video is resized to this width (the height is computed with the ratio), -1 to keep the original size
        isCrop : boolean => crop the image and skeleton to the skeleton bounding box
        """
        self.mod_frame = modFrame                 # only one frame every mod_frame is processed (taichi1.mp4 has 14000 frames with 20=>700 frames))

        filename_pkl, filename_dir, filename_base = filename_change_ext(filename, ".pkl")
        print("directory: "+filename_dir+"/"+filename_base)
        if not os.path.exists(filename_dir+"/"+filename_base):
            print("create directory: "+filename_dir+"/"+filename_base)
            os.makedirs(filename_dir+"/"+filename_base)
        
        self.path = os.path.dirname(filename)
        if os.path.exists(filename_pkl) and os.path.exists(filename_dir) and not forceCompute:
            print("===== read precompute: "+filename)
            vs = VideoSkeleton.load(filename_pkl)
            self.ske = vs.ske
            self.im = vs.im
            image = self.readImage(0)
            self.setWidthHeigh(image, newVideoWidth, cropRatio)
            return
    
        print("===== compute: "+filename)
        video = VideoReader(filename)
        print("read: "+filename+ " #frame="+str(video.getTotalFrames()))
        self.ske = [] #np.empty( 0, dtype=Skeleton)
        self.im = []
        widthIsSet = False
        for i in range(video.getTotalFrames()):
            image = video.readFrame()

            if image is None:
                continue

            if widthIsSet == False:
                self.setWidthHeigh(image, newVideoWidth, cropRatio)
                widthIsSet = True
                print("video shape="+str(image.shape) )
                print("new_video_width="+str(self.new_video_width) + " new_video_height="+str(self.new_video_height))
                print(" ske_width_crop="+str(self.ske_width_crop) + " ske_height_crop="+str(self.ske_height_crop))

            if (i%self.mod_frame == 0):
                ske = Skeleton()
                isSke, image, ske = self.cropAndSke(image, ske, isCrop)
                if isSke:
                    filename_im = filename_base + "/image" + str(i) + ".jpg"
                    filename_imsave = filename_dir + "/" + filename_im
                    cv2.imwrite(filename_imsave, image)
                    #self.ske = np.append(self.ske, ske)
                    self.ske.append( ske )
                    self.im.append(filename_im )
                    print("frame "+str(i)+"/"+str(video.getTotalFrames()) + "   filename="+filename_im 
                                + "  save="+filename_imsave + " sizeof="+str(sys.getsizeof(self.ske)) )
                                #+ "  ske="+str(ske.boundingBox()) )
            #         del filename_im
            #     del ske
            # del image
            # gc.collect()
            # cv2.destroyAllWindows()
        video.release()
        #self.im = self.im.reshape((self.ske.shape[0],) + image_shape)
        #self.ske = np.array(self.ske, dtype=Skeleton)
        skenp = np.empty( len(self.ske), dtype=Skeleton)
        for i in range(len(self.ske)):
            skenp[i] = self.ske[i]
        self.ske = skenp
        self.im = np.array(self.im)
        print("#skeleton="+str(self.ske.shape) + " #image="+str(self.im.shape))
        self.save( filename_pkl )


    def setWidthHeigh(self, image, newVideoWidth, cropRatio):
        if newVideoWidth == -1:
            self.new_video_width = image.shape[1]               # video is not resized
            self.new_video_height = image.shape[0]
        else:
            self.new_video_width = newVideoWidth               # video is resized to this width before skeleton extraction
            self.new_video_height = int(image.shape[0] * self.new_video_width / image.shape[1])
        self.ske_width_crop = int(cropRatio*self.new_video_width)       # skeleton image is resized to this width
        self.ske_height_crop = int(cropRatio*self.new_video_height)                # skeleton image is resized to this height    


    def cropAndSke(self, image, ske, isCrop=True):
        """ crop image and skeleton """
        #new_video_height = int(image.shape[0] * self.new_video_width / image.shape[1])
        image = cv2.resize(image, (self.new_video_width, self.new_video_height))
        if ske.fromImage( image ):      # a skeleton is found
            # crop
            if isCrop:
                xm, ym, xM, yM = ske.boundingBox()
                center_x = self.new_video_width * (xm + xM) / 2
                center_y = self.new_video_height * (ym + yM) / 2
                xm = int(center_x-self.ske_width_crop/2)
                xM = int(center_x+self.ske_width_crop/2)
                ym = int(center_y-self.ske_height_crop/2)
                yM = int(center_y+self.ske_height_crop/2)
                image = image[ ym:yM, xm:xM ]           # image crop arround skeleton
                #print("crop: xm="+str(xm) + " xM="+str(xM) + " ym="+str(ym) + " yM="+str(yM))
                ske.crop(xm/self.new_video_width, ym/self.new_video_height, self.ske_width_crop/self.new_video_width, self.ske_height_crop/self.new_video_height)   # skeleton crop
            return True, image, ske
        else:
            return False, image, ske        


    def save(self,filename):
        with open(filename, "wb") as fichier:
            pickle.dump(self, fichier)
        print("save: "+filename)


    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as fichier:
            objet_charge = pickle.load(fichier)
        print("VideoSkeleton::load: "+filename + " #skeleton="+str(objet_charge.ske.shape) + " #image="+str(objet_charge.im.shape))
        return objet_charge


    def __str__(self):
        return str("VideoSkeleton: nbframe="+str(self.ske.shape))


    def imagePath(self, idx):
        return self.path + "/" + self.im[idx]


    def readImage(self, idx):
        return cv2.imread( self.imagePath(idx) )
    

    def skeCount(self):
        return self.ske.shape[0]


    def draw(self):
        """ draw skeleton on image """
        print(os.getcwd())
        for i in range(self.skeCount()):
            empty = np.zeros((self.ske_height_crop, self.ske_width_crop, 3), dtype=np.uint8)
            im = self.readImage(i)
            self.ske[i].draw(empty)
            resim = combineTwoImages(im, empty)
            cv2.imshow('Image', resim)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()



        


if __name__ == '__main__':
    force = True
    # force = False
    modFRame = 25           # 10=>1440 images, 25=>560 images, 100=>140 images, 500=>280 images
    # modFRame = 100

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
            if len(sys.argv) > 3:
                modFRame = int(sys.argv[3])
    else:
        filename = "data/taichi1.mp4"
    print("Current Working Directory: ", os.getcwd())
    print("Filename=", filename)

    s = VideoSkeleton(filename, force, modFRame, 400, 1.0, False)  # 0.75)
    print(s)
    s.draw()

    # s.draw(image)
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
