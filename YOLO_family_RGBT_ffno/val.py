import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'/home/dsq/MFOD/Methods/YOLOv11-RGBT-master/runs/FLIR_yolov11/FLIR-yolov11-RGBT-midfusion/weights/best.pt')
    model.val(data=r'ultralytics/cfg/datasets/FLIR_aligned.yaml',
              split='val',
              imgsz=640,
              batch=16,
              use_simotm="RGBT",
              channels=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val/FLIR_yolov11',
              name='FLIR-yolov11-RGBT-midfusion',
              )