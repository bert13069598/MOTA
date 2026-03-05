

```bash
mkdir -p 0720
ffmpeg -i DJI_0720.MP4 0720/%06d.jpg
```


```bash
uv run main.py -m yolov8s -o -p vsai --show \
--dirs /home/deepet/PycharmProjects/MOTA/0720
```
