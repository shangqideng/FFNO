import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os

if __name__ == '__main__':
    ######################################### V12 #########################################
    # model = YOLO('ultralytics/cfg/models/12-RGBT/yolo12-RGBT-earlyfusion.yaml')
    # model = YOLO('ultralytics/cfg/models/12-RGBT/yolo12-RGBT-earlyfusion-ffno.yaml')  

    # model = YOLO('ultralytics/cfg/models/12-RGBT/yolo12-RGBT-midfusion.yaml') 
    # model = YOLO('ultralytics/cfg/models/12-RGBT/yolo12-RGBT-midfusion-ffno.yaml')  

    # model = YOLO('ultralytics/cfg/models/12-RGBT/yolo12-RGBT-latefusion.yaml') 
    # model = YOLO('ultralytics/cfg/models/12-RGBT/yolo12-RGBT-latefusion-ffno.yaml') 
    ######################################### V12 #########################################

    ######################################### V11 #########################################
    # model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11-RGBT-earlyfusion-RGBRGB6C.yaml')
    # model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11-RGBT-earlyfusion-RGBRGB6C-ffno.yaml')  

    # model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-RGBRGB6C.yaml') 
    # model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-RGBRGB6C-ffno.yaml')  

    # model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11-RGBT-latefusion-RGBRGB6C.yaml') 
    # model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11-RGBT-latefusion-RGBRGB6C-ffno.yaml') 
    ######################################### V11 #########################################

    ######################################### V10 #########################################
    # model = YOLO('ultralytics/cfg/models/v10-RGBT/yolov10n-RGBT-early-fusion.yaml')
    # model = YOLO('ultralytics/cfg/models/v10-RGBT/yolov10n-RGBT-early-fusion-ffno.yaml')  

    # model = YOLO('ultralytics/cfg/models/v10-RGBT/yolov10n-RGBT-midfusion.yaml') 
    # model = YOLO('ultralytics/cfg/models/v10-RGBT/yolov10n-RGBT-midfusion-ffno.yaml')  

    # model = YOLO('ultralytics/cfg/models/v10-RGBT/yolov10n-RGBT-latefusion.yaml') 
    # model = YOLO('ultralytics/cfg/models/v10-RGBT/yolov10n-RGBT-latefusion-ffno.yaml') 
    ######################################### V10 #########################################

    ######################################### V9 #########################################
    # model = YOLO('ultralytics/cfg/models/v9-RGBT/yolov9t-RGBT-early-fusion.yaml')
    # model = YOLO('ultralytics/cfg/models/v9-RGBT/yolov9t-RGBT-early-fusion-ffno.yaml')  

    # model = YOLO('ultralytics/cfg/models/v9-RGBT/yolov9t-RGBT-midfusion.yaml') 
    # model = YOLO('ultralytics/cfg/models/v9-RGBT/yolov9t-RGBT-midfusion-ffno.yaml')  

    # model = YOLO('ultralytics/cfg/models/v9-RGBT/yolov9t-RGBT-latefusion.yaml') 
    # model = YOLO('ultralytics/cfg/models/v9-RGBT/yolov9t-RGBT-latefusion-ffno.yaml') 
    ######################################### V9 #########################################

    ######################################### V8 #########################################
    # model = YOLO('ultralytics/cfg/models/v8-RGBT/yolov8-RGBT-earlyfusion.yaml')
    # model = YOLO('ultralytics/cfg/models/v8-RGBT/yolov8-RGBT-earlyfusion-ffno.yaml')  

    # model = YOLO('ultralytics/cfg/models/v8-RGBT/yolov8-RGBT-midfusion.yaml') 
    # model = YOLO('ultralytics/cfg/models/v8-RGBT/yolov8-RGBT-midfusion-ffno.yaml')  

    # model = YOLO('ultralytics/cfg/models/v8-RGBT/yolov8-RGBT-latefusion.yaml') 
    # model = YOLO('ultralytics/cfg/models/v8-RGBT/yolov8-RGBT-latefusion-ffno.yaml') 
    ######################################### v8 #########################################

    ######################################### V7 #########################################
    # model = YOLO('ultralytics/cfg/models/v7-RGBT/yolov7-tiny-RGBT-earlyfusion.yaml')
    # model = YOLO('ultralytics/cfg/models/v7-RGBT/yolov7-tiny-RGBT-earlyfusion-ffno.yaml')  

    # model = YOLO('ultralytics/cfg/models/v7-RGBT/yolov7-tiny-RGBT-midfusion.yaml') 
    # model = YOLO('ultralytics/cfg/models/v7-RGBT/yolov7-tiny-RGBT-midfusion-ffno.yaml')  

    # model = YOLO('ultralytics/cfg/models/v7-RGBT/yolov7-tiny-RGBT-latefusion.yaml') 
    model = YOLO('ultralytics/cfg/models/v7-RGBT/yolov7-tiny-RGBT-latefusion-ffno.yaml') 
    ######################################### v7 #########################################

    ######################################### world #########################################
    # model = YOLO('ultralytics/cfg/models/v7-RGBT/yolov7-tiny-RGBT-earlyfusion.yaml')
    ######################################### world #########################################

    # model.load('yolov7.pt') # loading pretrain weights



    model.train(data=R'ultralytics/cfg/datasets/FLIR_aligned.yaml',
                cache=False,
                imgsz=640,
                epochs=30,
                batch=6,
                close_mosaic=0,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBRGB6C",
                channels=6,  #
                project='runs/FLIR_yolov',
                name='FLIR-yolov7-RGBT-latefusion-ffno',
                )
    
# CUDA_VISIBLE_DEVICES=3 python train_RGBRGB.py
    