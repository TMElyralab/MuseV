## 单机多进程并行
```bash
python ./scripts/extract_feature/extract_video_emb_with_multi_process.py  -task_path ./datasets/webvid/csvs/train_webvid_10M_train_portrait_and_action_w=512_h=320_10.csv   -h5py_dir ./datasets/webvid/emb/train/ -video_dir ./datasets/webvid/video/train -target_width 512 -target_height 320 -source webvid --sep , --n_process 1
```

