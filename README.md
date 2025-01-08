### Create datasets
1. Split into train and validation
```python
project.utils.split_ds("dlp-object-detection")
```
2. Resize images
```python
project.utils.copy_and_resize_ds("dlp-object-detection", 224) # for fasterrcnn
project.utils.copy_and_resize_ds("dlp-object-detection", 640) # for yolo
```

### Train models
```bash
python3 -m project.run --mode train --model fasterrcnn  # for fasterrcnn
```
```bash
python3 -m project.run --mode train --model fasterrcnn  # for yolo
```


### Test models and save results
```bash
python3 -m project.run --mode test --model fasterrcnn --checkp-path /path/to/checkpoint  # for fasterrcnn
```
```bash
python3 -m project.run --mode train --model fasterrcnn --checkp-path /path/to/checkpoint # for yolo
```