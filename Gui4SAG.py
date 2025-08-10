# -*- coding: utf-8 -*-
import sys
import time
import cv2

from PyQt5 import QtCore
from PyQt5.QtCore import (
    QPointF,
    Qt
)
from PyQt5.QtGui import (
    QBrush,
    QPainter,
    QPen,
    QPixmap,
    QKeySequence,
    QPen,
    QBrush,
    QColor,
    QImage,
    QFont,
)
from PyQt5.QtWidgets import (
    QFileDialog,
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPixmapItem,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QShortcut,
    QLineEdit,
    QGraphicsTextItem,
    QButtonGroup,
    QRadioButton,
)

import numpy as np
from skimage import transform, io
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image

from SAMraw import sam_model_registry
import sag_cfg as sacfg

from sam_lora import LoRA_Sam


# freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)



Seisam_CKPT_PATH = "Your_Logs/loraSAM.ckpt"


SAM_MODEL_FILE_MODE = Seisam_CKPT_PATH.split("_")[-2]
SAM_MODEL_TYPE = "vit_%s"%SAM_MODEL_FILE_MODE

Seisam_IMG_INPUT_SIZE = 1024
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


@torch.no_grad()
def Seisam_inference(Seisam_model, img_embed, box_1024, height, width):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = Seisam_model.sam.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = Seisam_model.sam.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=Seisam_model.sam.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )


    ## the confidence of mask
    confidence_mask = 0.7

    low_res_pred = Seisam_model.postprocess_masks(low_res_logits, (height,width), low_res_logits.shape[2:])
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    Seisam_seg = (low_res_pred > confidence_mask).astype(np.uint8)

    return Seisam_seg

@torch.no_grad()
def Seisam_inference_point(Seisam_model, img_embed, prompt_1024,bbox_1024, height, width):
    point_1024,label_1024 = prompt_1024
    point_torch = torch.as_tensor(point_1024, dtype=torch.float, device=img_embed.device)
    label_torch = torch.as_tensor(label_1024, dtype=torch.float, device=img_embed.device)
    if bbox_1024 != None:
        bbox_1024 = np.array(bbox_1024).flatten().reshape((1,-1))
        bbox_torch = torch.as_tensor(bbox_1024, dtype=torch.float, device=img_embed.device)

    if len(point_torch.shape) == 2:
        point_torch = point_torch[None,:,:]
        # point_torch = point_torch[:, None, :]  # (B, 1, 4)

    label_torch = label_torch.transpose(0,1)
    # print(point_torch.shape)
    # print(label_torch.shape)


    sparse_embeddings, dense_embeddings = Seisam_model.sam.prompt_encoder(
        points=(point_torch,label_torch),
        boxes=bbox_1024,
        masks=None,
    )
    low_res_masks, _ = Seisam_model.sam.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=Seisam_model.sam.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_masks)  # (1, 1, 256, 256)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    return low_res_pred

@torch.no_grad()
def Seisam_inference_bbox(Seisam_model, img_embed, bbox_1024, height, width):

    bbox_1024 = np.array(bbox_1024).flatten().reshape((1,-1))
    bbox_torch = torch.as_tensor(bbox_1024, dtype=torch.float, device=img_embed.device)

    sparse_embeddings, dense_embeddings = Seisam_model.sam.prompt_encoder(
        points=None,
        boxes=bbox_torch,
        masks=None,
    )
    low_res_masks, _ = Seisam_model.sam.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=Seisam_model.sam.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_masks)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    return low_res_pred

print("Loading SAM model, a sec.")
tic = time.perf_counter()

# set up model
args = sacfg.parse_args()
args.sam_checkpoint = None
Seisam_model = sam_model_registry['vit_b']().to(device)
ckptName = Seisam_CKPT_PATH
ckptDict = dict(
    checkpoint_path=ckptName,
    sam_model= Seisam_model, 
    r=32,
    args=args,
    q=True,
    k=True,
    v=True,
    out=True,
    lora_layer=None,
)
Seisam_model = LoRA_Sam.load_from_checkpoint(**ckptDict,map_location='cpu')
Seisam_model = Seisam_model.eval()



print(f"Done, took {time.perf_counter() - tic}")


def np2pixmap(np_img):
    height, width, channel = np_img.shape
    bytesPerLine = 3 * width
    qImg = QImage(np_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)

## the color of mask
colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 255, 255),
    (192, 192, 192),
    (64, 64, 64),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (0, 0, 127),
    (192, 0, 192),
]


class Window(QWidget):
    def __init__(self):
        super().__init__()

        # configs
        self.half_point_size = 5  # radius of bbox starting and ending points

        # app stats
        self.image_path = None
        self.color_idx = 0
        self.bg_img = None
        self.is_mouse_down = False
        
        self.use_mode = "Point"
        self.bbox_arr = []
        self.raw_bbox_arr = []

        self.rect = None
        self.point_size = self.half_point_size * 2
        self.start_point = None
        self.end_point = None
        self.start_pos = (None, None)
        self.embedding = None
        self.prev_mask = None

        self.threshold_mask = 0.7

        self.view = QGraphicsView()
        self.view.setRenderHint(QPainter.Antialiasing)

        pixmap = self.load_image()

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.view)

        load_button = QPushButton("Load Image")
        save_button = QPushButton("Save Mask")
        well_button = QPushButton("Load Well")
        reset_button = QPushButton("Reset")

        hbox = QHBoxLayout(self)
        hbox.addWidget(load_button)
        hbox.addWidget(save_button)
        hbox.addWidget(well_button)
        hbox.addWidget(reset_button)


        self.threshold_edit = QLineEdit()
        self.threshold_edit.setText("%f"%self.threshold_mask)
        set_ts_button = QPushButton("Set the threshold")

        hboxL = QHBoxLayout(self)
        
        hboxL.addWidget(self.threshold_edit)
        hboxL.addWidget(set_ts_button)



        hboxR = QHBoxLayout(self)
        groupBox = QButtonGroup(self)
        self.checkBoxPoint = QRadioButton("&Point")
        self.checkBoxPoint.setChecked(True)
        self.checkBoxPoint.toggled.connect(lambda: self.btnstate(self.checkBoxPoint))

        self.checkBoxBbox = QRadioButton("Bbox")
        self.checkBoxBbox.toggled.connect(lambda: self.btnstate(self.checkBoxBbox))

        self.newCategoryButton = QPushButton("New")
        self.newCategoryButton.clicked.connect(self.newCategory)



        groupBox.setExclusive(True)
        hboxR.addWidget(self.checkBoxPoint)
        hboxR.addWidget(self.checkBoxBbox)
        hboxR.addWidget(self.newCategoryButton)

        vbox.addLayout(hbox)
        vbox.addLayout(hboxL)
        vbox.addLayout(hboxR)

        self.setLayout(vbox)

        # keyboard shortcuts
        self.quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.quit_shortcut.activated.connect(lambda: quit())

        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)

        self.new_shortcut = QShortcut(QKeySequence("Ctrl+A"), self)
        self.new_shortcut.activated.connect(self.newCategory)

        load_button.clicked.connect(self.load_image)
        save_button.clicked.connect(self.save_mask)
        well_button.clicked.connect(self.load_well)
        reset_button.clicked.connect(self.reset_image)
        set_ts_button.clicked.connect(self.set_ts)




        self.pointArr = []
        self.labelArr = []

    def undo(self):
        if self.prev_mask is None:
            print("No previous mask record")
            return

        # self.color_idx -= 1

        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask = Image.fromarray(self.prev_mask.astype("uint8"), "RGB")
        img = Image.blend(bg, mask, 0.2)

        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img)))

        self.mask_c = self.prev_mask
        self.prev_mask = None
        
        

    def loadNpy(self,filename):
        image = np.load(filename).astype('uint8')
        eimg = np.zeros(image.shape)
        for ind in range(3):
            eimg[:,:,ind] = cv2.equalizeHist(image[:,:,ind])
        return eimg.astype('uint8')

    def load_image(self):
        # file_path, file_type = QFileDialog.getOpenFileName(
        #     self, "Choose Image to Segment", ".", "Image Files (*.png *.jpg *.bmp *.npy)"
        # )

        # file_path, file_type = QFileDialog.getOpenFileName(
        #     self, "Choose Image to Segment", "./testPNG/predPNG/", "Image Files (*.png *.jpg *.bmp *.npy)"
        # )

        # file_path, file_type = QFileDialog.getOpenFileName(
        #     self, "Choose Image to Segment", "./testPNG/Examples/", "Image Files (*.png *.jpg *.bmp *.npy)"
        # )

        # file_path, file_type = QFileDialog.getOpenFileName(
        #     self, "Choose Image to Segment", "./testPNG/fieldExamples/", "Image Files (*.png *.jpg *.bmp *.npy)"
        # )

        file_path, file_type = QFileDialog.getOpenFileName(
            self, "Choose Image to Segment", "/home/gaohang/Pictures/fieldExamples/", "Image Files (*.png *.jpg *.bmp *.npy)"
        )

        if file_path is None or len(file_path) == 0:
            print("No image path specified, plz select an image")
            exit()

        if file_path.endswith(".npy"):
            img_np = self.loadNpy(file_path)
        else:
            img_np = io.imread(file_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np

        if img_3c.shape[0]<512:
            H,W,_ = img_3c.shape
            scalingRatio = 512/H+0.3
            nH,nW = int(scalingRatio*H),int(scalingRatio*W)
            img_3c = cv2.resize(img_3c,(nW,nH))

        self.img_3c = img_3c

        self.image_path = file_path
        self.get_embeddings()
        pixmap = np2pixmap(self.img_3c)

        H, W, _ = self.img_3c.shape

        self.scene = QGraphicsScene(0, 0, W, H)
        self.end_point = None
        self.rect = None
        self.bg_img = self.scene.addPixmap(pixmap)
        
        self.bg_img.setPos(0, 0)
        self.mask_c = np.zeros((*self.img_3c.shape[:2], 3), dtype="uint8")
        self.view.setScene(self.scene)

        # events
        self.scene.mousePressEvent = self.mouse_press
        self.scene.mouseReleaseEvent = self.mouse_release



        # clear the Point & Label Array
        self.pointArr = []
        self.labelArr = []

    def reset_image(self):
        pixmap = np2pixmap(self.img_3c)
        H, W, _ = self.img_3c.shape
        self.scene = QGraphicsScene(0, 0, W, H)
        self.end_point = None
        self.rect = None
        self.bg_img = self.scene.addPixmap(pixmap)
        
        self.bg_img.setPos(0, 0)
        self.mask_c = np.zeros((*self.img_3c.shape[:2], 3), dtype="uint8")
        self.view.setScene(self.scene)

        # events
        self.scene.mousePressEvent = self.mouse_press
        self.scene.mouseReleaseEvent = self.mouse_release

        # clear the Point & Label Array
        self.pointArr = []
        self.labelArr = []
        self.bbox_arr = []

        self.color_idx = 0

    def btnstate(self,btn):
        if btn.text()=="&Point":
            self.use_mode = "Point"
        elif btn.text()=="Bbox":
            self.use_mode = "Bbox"
        else:
            pass


    def mouse_press(self, ev):
        x, y = ev.scenePos().x(), ev.scenePos().y()
        self.is_mouse_down = True
        self.start_pos = ev.scenePos().x(), ev.scenePos().y()
        self.start_point = self.scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("red")),
            brush=QBrush(QColor("red")),
        )

    def mouse_click_bbox(self,ev):
        x, y = ev.scenePos().x(), ev.scenePos().y()
        H, W, _ = self.img_3c.shape        
        pointArr = [x,y]
        point_np = np.array(pointArr)
        
        point_1024 = point_np / np.array([W, H]) * 1024

        if len(self.bbox_arr) == 0:
            self.bbox_arr.append(point_1024)
            self.raw_bbox_arr.append(pointArr)
        else:
            self.bbox_arr.append(point_1024)
            self.raw_bbox_arr.append(pointArr)

            sam_mask = Seisam_inference_bbox(Seisam_model, self.embedding, self.bbox_arr, H, W)
            # print(sam_mask.max(),sam_mask.min())
            sam_mask = (sam_mask > self.threshold_mask).astype(np.uint8)

            if len(sam_mask.shape)==3:
                sam_mask = sam_mask[-1,...]


            self.prev_mask = self.mask_c.copy()
            self.mask_c[sam_mask != 0] = colors[self.color_idx % len(colors)]

            bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
            mask = Image.fromarray(self.mask_c.astype("uint8"), "RGB")

            img = Image.blend(bg, mask, 0.6)

            self.scene.removeItem(self.bg_img)
            self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img)))
            x1,y1 = self.raw_bbox_arr[0]
            x2,y2 = self.raw_bbox_arr[1]
            self.scene.addRect(x1,y1,
                               abs(x2-x1),abs(y2-y1),
                               pen=QPen(QColor("yellow")),
                               brush=QBrush(QColor(0,0,0,50))
                               )
            
            # for x,y in self.bbox_arr:
            #     x,y = float(x),float(y)
            #     self.scene.addEllipse(
            #         x - self.half_point_size,
            #         y - self.half_point_size,
            #         self.point_size,
            #         self.point_size,
            #         pen=QPen(QColor("red")),
            #         brush=QBrush(QColor("red")),
            #     )

            self.bbox_arr = []
            self.raw_bbox_arr = []

    def mouse_release(self, ev):
        if self.use_mode=="Bbox":
            self.mouse_click_bbox(ev)
        elif self.use_mode=="Point":
            self.mouse_release_single(ev)
        else:
            pass
        
    def mouse_release_single(self, ev):
        x, y = ev.scenePos().x(), ev.scenePos().y()
        self.is_mouse_down = False
        H, W, _ = self.img_3c.shape
        self.pointArr.append([x,y])
        point_np = np.array(self.pointArr)
        point_1024 = point_np / np.array([W, H]) * 1024

        if ev.button() == QtCore.Qt.RightButton:
            self.labelArr.append([-1])
        else:
            self.labelArr.append([1])
        label_1024 = self.labelArr

        prompt_1024 = (point_1024,np.abs(label_1024))
        if len(self.bbox_arr)==0:
            bbox_1024 = None
        else:
            bbox_1024 = self.bbox_arr

        print("pt size: ",point_1024.shape)

        sam_mask = Seisam_inference_point(Seisam_model, self.embedding, prompt_1024,bbox_1024, H, W)
        sam_mask = (sam_mask > self.threshold_mask).astype(np.uint8)

        if len(sam_mask.shape)==3:
            sam_mask = sam_mask[-1,...]
        

        self.prev_mask = self.mask_c.copy()
        self.mask_c[sam_mask != 0] = colors[self.color_idx % len(colors)]
        # self.color_idx += 1

        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask = Image.fromarray(self.mask_c.astype("uint8"), "RGB")
        
        ## the alpha of image after adding points
        # img = Image.blend(bg, mask, 0.2)
        img = Image.blend(bg, mask, 0.6)

        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img)))

        for x,y in self.pointArr:
            x,y = float(x),float(y)
            self.scene.addEllipse(
                x - self.half_point_size,
                y - self.half_point_size,
                self.point_size,
                self.point_size,
                pen=QPen(QColor("red")),
                brush=QBrush(QColor("red")),
            )

    def load_well(self):
        wellName = self.image_path.replace(".npy",".txt")
        xArr,yArr,labArr =  self.readWellFile(wellName)
        self.runModelWithPos(xArr,yArr,labArr)

    def runModelWithPos(self,locX,locY,labWell):
        H, W, _ = self.img_3c.shape

        for x,y in zip(locX,locY):
            x,y = float(x),float(y)
            self.pointArr.append([x,y])
            self.labelArr.append([1])

        point_np = np.array(self.pointArr)

        point_1024 = point_np / np.array([W, H]) * 1024

        label_1024 = self.labelArr
        
        # print(label_1024)

        prompt_1024 = (point_1024,np.abs(label_1024))

        if len(self.bbox_arr)==0:
            bbox_1024 = None
        else:
            bbox_1024 = self.bbox_arr
        
        sam_mask = Seisam_inference_point(Seisam_model, self.embedding, prompt_1024,bbox_1024, H, W)
        
        sam_mask = (sam_mask > self.threshold_mask).astype(np.uint8)

        # print("sam_mask")
        # print(sam_mask.shape)
        if len(sam_mask.shape)==3:
            # sam_mask = sam_mask[-1,...]
            # sam_mask = sam_mask[0,...]
            sam_mask = self.mergeMat(sam_mask,np.array(label_1024))

        self.prev_mask = self.mask_c.copy()
        self.mask_c[sam_mask != 0] = colors[self.color_idx % len(colors)]
        # self.color_idx += 1

        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask = Image.fromarray(self.mask_c.astype("uint8"), "RGB")
        
        ## the alpha of image after adding points
        # img = Image.blend(bg, mask, 0.2)
        img = Image.blend(bg, mask, 0.6)

        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img)))

        for x,y,lb in zip(locX,locY,labWell):
            x,y = float(x),float(y)
            self.scene.addEllipse(
                x - self.half_point_size,
                y - self.half_point_size,
                self.point_size,
                self.point_size,
                pen=QPen(QColor("red")),
                brush=QBrush(QColor("red")),
            )
            tmpLb = QGraphicsTextItem(lb)
            tmpLb.setHtml('<div style="background:#ffffff;">%s</p>'%lb)
            tmpLb.setFont(QFont("Arial", 15, QFont.Bold))
            tmpLb.setPos(QPointF(x + 3*self.half_point_size,y - 0*self.half_point_size))
            self.scene.addItem(tmpLb)
            

    def mergeMatBak(self,maskArr,labArr):
        rawShape = maskArr.shape
        maskArr = maskArr.reshape((rawShape[0],-1))
        sumArr = maskArr*labArr
        sumArr = sumArr.reshape(rawShape)
        sumArr = np.sum(sumArr,axis=0)
        sumArr = np.clip(sumArr,0,1)
        sumArr = np.squeeze(sumArr)
        return sumArr
    
    def mergeMat(self,maskArr,labArr):
        rawShape = maskArr.shape
        maskArr = maskArr.reshape((rawShape[0],-1))
        sumArr = maskArr*labArr
        sumArr = sumArr.reshape(rawShape)
        sumArr = np.sum(sumArr,axis=0)
        sumArr = np.clip(sumArr,0,1)
        sumArr = np.squeeze(sumArr)
        return sumArr


    def readWellFile(self,fname):
        xArr,yArr,labArr = [],[],[]
        with open(fname,'r') as f:
            taf = f.readlines()
            for tt in taf:
                tmpx,tmpy,tmpl = tt.strip().split()
                xArr.append(tmpx)
                yArr.append(tmpy)
                labArr.append(tmpl)
        return xArr,yArr,labArr
    

    def set_ts(self):
        self.threshold_mask = float(self.threshold_edit.text())

    def save_mask(self):
        out_path = f"{self.image_path.split('.')[0]}_mask.png"
        io.imsave(out_path, self.mask_c)

    def newCategory(self):
        self.pointArr = []
        self.labelArr = []
        self.bbox_arr = []
        self.color_idx += 1

    @torch.no_grad()
    def get_embeddings(self):
        print("Calculating embedding, gui may be unresponsive.")
    
        img_1024 = transform.resize(
            self.img_3c[...,:3], (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)

        
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        

        # pixel_mean=[123.675, 116.28, 103.53]
        # pixel_std=[58.395, 57.12, 57.375]
        # img_1024 = (img_1024-pixel_mean)/pixel_std

        # import matplotlib.pyplot as plt
        # plt.imshow(img_1024)
        # plt.savefig('./trans.png')

        # convert the shape to (3, H, W)
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )

        # if self.embedding is None:
        with torch.no_grad():
            self.embedding = Seisam_model.sam.image_encoder(
                img_1024_tensor
            )  # (1, 256, 64, 64)
        print("Done.")


app = QApplication(sys.argv)

w = Window()
w.show()

app.exec()
