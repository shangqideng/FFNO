import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr-RGBT/rtdetr-resnet50-RGBT-midfusion.yaml')
    model.load('rtdetr-l.pt') # loading pretrain weights
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
                channels=6,
                project='runs/FLIR_rtdetr',
                name='FLIR-rtdetr-RGBT-midfusion',
                )
    
# CUDA_VISIBLE_DEVICES=0 python train-rtdetr.py