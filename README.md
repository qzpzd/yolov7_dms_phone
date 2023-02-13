# yolov7_dms_phone
yolov7关于打电话检测应用
这里主要是修改了了data文件下的coco数据集路径：替换为打电话数据集

## train
sh train.sh
选择yolo_tiny轻量级模型框架进行训练得到 [**yolov7_phone.pt**](https://github.com/qzpzd/yolov7_dms_phone/tree/yolov7-change/best.pt)
## detect
替换对应训练好的模型，添加对应的图片路径即可
python detect_change.py --source /yolov7/inference/images/call
