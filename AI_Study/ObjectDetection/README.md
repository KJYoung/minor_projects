# YoloV1 From Scratch
[Source](https://youtu.be/n9_XyCGr-MI?si=cBLS4TzexliYWuni)

# YoloV3 From Scratch

## Memos
### Runtime Error
```shell
    RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [MPSFloatType [8, 7, 7, 2]], which is output 0 of AsStridedBackward0, is at version 1; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
```

```python
    box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
        torch.abs(box_predictions[..., 2:4]) + 1e-6
    )
```
위 코드를 아래와 같이 바꾸니 해결되었음.
```python
    box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
        torch.abs(box_predictions[..., 2:4] + 1e-6)
    )
```