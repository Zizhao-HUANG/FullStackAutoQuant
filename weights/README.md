# Model Weights

Pre-trained model weights are not included in the repository due to file size.

## Download

Download the pre-trained weights from [GitHub Releases](https://github.com/Zizhao-HUANG/FullStackAutoQuant/releases):

```bash
# Download params.pkl and state_dict_cpu.pt
# Place them in this directory (weights/)
```

## Files

| File | Size | Description |
|------|------|-------------|
| `params.pkl` | ~3.1 MB | Full Qlib GeneralPTNN serialized model (includes optimizer state) |
| `state_dict_cpu.pt` | ~3.1 MB | PyTorch state dictionary only (lighter, for inference) |

## Training Your Own

To train a new model, refer to [Microsoft RD-Agent](https://github.com/microsoft/RD-Agent) for the automated model training pipeline.
