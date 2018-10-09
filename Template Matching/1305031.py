
import cv2
import math
import time
import  numpy as np
class Exhaustive_Search:
    def __init__(self,ref_image,test_image):
        self.ref_img = ref_image
        self.tst_img = test_image
        self.row_ref = self.ref_img.shape[0]
        self.col_ref = self.ref_img.shape[1]
        self.mismatch_row = self.tst_img.shape[0]-self.row_ref+1
        self.mismatch_col = self.tst_img.shape[1]-self.col_ref+1

    def minimization(self, D_mis):
        index = []
        index.append(-1)
        index.append(-1)
        minimum = 1.7976931348623157e+308

        i = 0
        while i<self.mismatch_row:
            j= 0
            while j<self.mismatch_col:
                minimum = min(D_mis[i][j],minimum)
                if minimum == D_mis[i][j]:
                    index[0]=i
                    index[1]=j
                j+=1
            i+=1

        return  index

    def det_cell_D(self,m,n):
        cell = 0
        row_lim = m+self.row_ref
        col_lim = n+self.col_ref

        i = m
        while i<row_lim:
            j= n
            while j<col_lim:
                t = float(self.tst_img[i][j])
                r = float(self.ref_img[i-m][j-n])
                cell += (t-r)*(t-r)
                j+=1
            i+=1

        return  cell
    def exhaustive(self):
        print("in class exhaustive")
        D_mismatch = [[0 for j in range(self.mismatch_col)] for i in range(self.mismatch_row)]

        i = 0
        while i<self.mismatch_row:
            j= 0
            while j<self.mismatch_col:
                D_mismatch[i][j]=self.det_cell_D(i,j)
                j+=1
            i+=1


        return self.minimization(D_mismatch)


class LogarithmicSearch:
    def __init__(self,ref_image,test_image,centerX,centerY,box,which):
        self.ref_img = ref_image
        self.tst_img = test_image
        self.row_ref = self.ref_img.shape[0]
        self.col_ref = self.ref_img.shape[1]
        self.mismatch_row = self.tst_img.shape[0]-self.row_ref+1
        self.mismatch_col = self.tst_img.shape[1]-self.col_ref+1
        self.centerX = centerX
        self.centerY = centerY
        self.box_sz = box
        self.algo = which

    def determine_k(self,rectangle):
        k = math.log2(rectangle)
        k = round(k)
        return  k

    def populate_neighbor(self,centr, distance):
        neighbors = []
        neighbors.append([centr[0]+distance,centr[1]-distance])
        neighbors.append([centr[0],centr[1]-distance])
        neighbors.append([centr[0]-distance,centr[1]-distance])
        neighbors.append([centr[0],centr[1]+distance])
        neighbors.append([centr[0]+distance,centr[1]+distance])
        neighbors.append([centr[0]+distance,centr[1]])
        neighbors.append([centr[0],centr[1]])
        neighbors.append([centr[0]-distance,centr[1]])
        neighbors.append([centr[0]-distance,centr[1]+distance])
        return neighbors

    def populate_neighbor_rectangleWindow(self, centr, distancex, distancey):
        neighbors = []
        neighbors.append([centr[0]+distancex,centr[1]-distancey])
        neighbors.append([centr[0],centr[1]-distancey])
        neighbors.append([centr[0]-distancex,centr[1]-distancey])
        neighbors.append([centr[0],centr[1]+distancey])
        neighbors.append([centr[0]+distancex,centr[1]+distancey])
        neighbors.append([centr[0]+distancex,centr[1]])
        neighbors.append([centr[0],centr[1]]) #
        neighbors.append([centr[0]-distancex,centr[1]])#
        neighbors.append([centr[0]-distancex,centr[1]+distancey])
        return neighbors


    def padding(self,img,padding_X,padding_Y):
        x_dim = img.shape[0]+padding_X
        y_dim = img.shape[1]+padding_Y
        padded_image = np.zeros(shape=(x_dim,y_dim))
        i = 0
        while i< x_dim:
            if i < img.shape[0]:
                padded_image[i] = np.append(img[i],np.zeros(padding_Y))
            else:
                padded_image[i]=np.zeros(y_dim)
            i += 1
        return padded_image

    def det_cell_D(self,m,n):
        cell = 0
        row_lim = m+self.row_ref
        col_lim = n+self.col_ref
        if self.algo == "log":
            self.tst_img =self.padding(self.tst_img,self.row_ref,self.col_ref)
        i = m
        while i<row_lim:
            j= n
            while j<col_lim:
                t = float(self.tst_img[i][j])
                r = float(self.ref_img[i-m][j-n])
                cell += (t-r)*(t-r)
                j+=1
            i+=1

        return  cell

    def init_rectangle_dimension(self):
        temp =  ( self.tst_img.shape[0] - self.centerX)
        rectangle_x =temp*.5
        temp =  ( self.tst_img.shape[1] - self.centerY)
        rectangle_y =temp*.5
        return  rectangle_x,rectangle_y

    def logarithm(self):

        centr = []
        centr.append(int(self.centerX))
        centr.append(int(self.centerY))

        if self.algo == "d1":
            distance = 1
            neighbors = self.populate_neighbor(centr,distance)
            minimum =  1.7976931348623157e+308

            limit = len(neighbors)

            i = 0
            result = centr
            while i<limit:

                if(int(neighbors[i][0])<self.mismatch_row and  int(neighbors[i][1])<self.mismatch_col):

                    cell_D = self.det_cell_D(int(neighbors[i][0]), int(neighbors[i][1]))

                    minimum = min(cell_D,minimum)
                    if(cell_D == minimum):
                        result = neighbors[i]

                i +=1
            centr = result
        else:

            rectangle_x , rectangle_y = self.init_rectangle_dimension()

            distancex = 1.7976931348623157e+308
            distancey = 1.7976931348623157e+308

            while(True):
                #print("in while")
                if distancex<=1 or distancey<=1 :
                    break

                distancex = math.pow(2,self.determine_k(rectangle_x)-1)
                distancey = math.pow(2,self.determine_k(rectangle_y)-1)

                neighbors = self.populate_neighbor_rectangleWindow(centr,distancex,distancey)


                minimum =  1.7976931348623157e+308
                limit = len(neighbors)

                i = 0
                result = centr
                while i<limit:
                    if(int(neighbors[i][0])<self.mismatch_row and  int(neighbors[i][1])<self.mismatch_col):
                        cell_D = self.det_cell_D(int(neighbors[i][0]), int(neighbors[i][1]))

                        minimum = min(cell_D,minimum)
                        if(cell_D == minimum):
                            result = neighbors[i]

                    i +=1


                centr = result
                rectangle_x /=  2
                rectangle_y /= 2


        return centr


class HierarchicalSearch:
    def __init__(self,ref_image,test_image,centerX,centerY,box):
        self.ref_img = ref_image
        self.tst_img = test_image
        self.row_ref = self.ref_img.shape[0]
        self.col_ref = self.ref_img.shape[1]
        self.mismatch_row = self.tst_img.shape[0]-self.row_ref+1
        self.mismatch_col = self.tst_img.shape[1]-self.col_ref+1
        self.centerX = centerX
        self.centerY = centerY
        self.box_sz = box

    def low_pass_filter(self,image):
        kernel = (5,5)
        img = cv2.GaussianBlur(image, kernel, 0)
        img =  cv2.resize(img, None, fx=1/2, fy=1/2)
        return img
    def image_filters(self):
        lvl1_ref = self.low_pass_filter(self.ref_img)
        lvl1_tst = self.low_pass_filter(self.tst_img)

        lvl2_ref = self.low_pass_filter(lvl1_ref)
        lvl2_tst = self.low_pass_filter(lvl1_tst)
        return  lvl1_ref,lvl1_tst,lvl2_ref,lvl2_tst

    def hierarchy(self):

        lvl1_ref,lvl1_tst,lvl2_ref,lvl2_tst = self.image_filters()

        ##L2
        rectangleSz = self.box_sz*.25
        centerX = self.centerX*.25
        centerY = self.centerY*.25
        #lgS = LogarithmicSearch(lvl2_ref,lvl2_tst,centerX,centerY,rectangleSz,"log")
        lgs = Exhaustive_Search(lvl2_ref,lvl2_tst)
        lvl1 = lgs.exhaustive()

        ##L1
        lgS = LogarithmicSearch(lvl1_ref,lvl1_tst,lvl1[0]*2,lvl1[1]*2,1,"d1")
        lvl2 = lgS.logarithm()

        ##L0
        lgS = LogarithmicSearch(self.ref_img,self.tst_img,lvl2[0]*2,lvl2[1]*2,1,"d1")
        lvl1 = lgS.logarithm()


        return lvl1




class FileReading:
    def __init__(self,filename):
        self.x = 0
        self.filename = filename

    def imageReading(self):
        img =  cv2.imread(self.filename)
        color_img = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img,color_img, img.shape

## Read the images

f = FileReading("ref2.jpg")
reference,color_img,reference_sz = f.imageReading()


f = FileReading("test2.jpg")
main_image,color_img,main_sz = f.imageReading()



def drawFinalImg(optimum_pt):
    print("drawing out :")
    r = reference_sz[0]
    c = reference_sz[1]
    pt1 =  (int(optimum_pt[1]), int(optimum_pt[0]))
    pt2 =  (int(optimum_pt[1]) + c, int(optimum_pt[0]) + r)
    color = 10
    thickness = 2
    cv2.rectangle(color_img,pt1,pt2,color, thickness)
    cv2.imshow("Result", color_img)
    cv2.waitKey(0)
def chooseOption():
    print("3 algo: ")
    print("1. Exhaustive Search")
    print("2. Logarithmic Search")
    print("3. Hierarchical Search")
    option = int(input("Choose One: "))
    if option == 1:
        s = "Exhaustive"
    elif option == 2:
        s  ="Logarithmic"
    elif option == 3:
        s = "Hierarchical"
    print("you've chosen ",s," search")
    return  option,s

opt,st = chooseOption()

def algorithm(option):
    cent_x = main_sz[0]/2
    cent_y = main_sz[0]/2
    if option == 1:
        ex = Exhaustive_Search(reference,main_image)
        s =ex.exhaustive()
    elif option == 2:

        lg = LogarithmicSearch(reference,main_image,cent_x,cent_y,32,"log")
        s  =lg.logarithm()
    elif option == 3:
        hier = HierarchicalSearch(reference,main_image,cent_x,cent_y,32)
        s = hier.hierarchy()
    return  s

def implement(option,option_name):
    start = time.time()
    results = algorithm(option)
    end = time.time()
    print("Execution Time: ",end-start)
    print(results)

    string = option_name +" search: "+str(results[0])+" "+str(results[1])+" Time: "+str(end-start)+" seconds\n"
    f=open("result.txt", "a+")
    f.write(string)
    f.close()

    drawFinalImg(results)
implement(opt,st)

