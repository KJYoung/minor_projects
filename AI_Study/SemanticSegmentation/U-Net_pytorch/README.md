### From Scratch
[Source](https://youtu.be/IHq1t7NxS8k?si=T7iWtx50SZgPT6Ew)
[Dataset](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data)
    - After download, first 50 images from train dataset is selected for val dataset 

### Trained 5epochs. 3epochs first, And loaded 3 epochs model to train last 2 epochs
```shell
=> Loading Checkpoint
    /Applications/anaconda3/lib/python3.9/site-packages/torch/cuda/amp/grad_scaler.py:120: UserWarning: torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.
    warnings.warn("torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.")
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 158/158 [03:23<00:00,  1.29s/it, loss=0.0412]
    => Saving Checkpoint
    Got 1911339/1920000 with acc 99.55%
    Dice Score: 0.9877797365188599
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 158/158 [03:36<00:00,  1.37s/it, loss=0.0349]
    => Saving Checkpoint
    Got 1912054/1920000 with acc 99.59%
```

### Albumentation: Data Augmentation Library
### Carvana Dataset(kaggle competition dataset)

## Memos
[nn.ReLU inplace=True?](https://keepgoingrunner.tistory.com/79)
[pin_memory](https://mopipe.tistory.com/191)
[fp16 Training(CUDA)](https://youtu.be/ks3oZ7Va8HU?si=Jf2mWdJhJE1DECUZ)