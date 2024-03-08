
# Few-shot RAW Image Denoising @ MIPI-challenge Solution

This README provides a brief overview of our solution for the Few-shot RAW Image Denoising challenge at the MIPI-challenge. Our method includes pre-trained model parameters, training and fine-tuning inference stage log files, training parameters, and the final test results submitted to the server, all of which have been uploaded to a Google Drive link provided below.

## Google Drive Link
- Pre-trained model parameters, training logs, fine-tuning inference logs, training parameters, and final test results: [Google Drive](https://drive.google.com/drive/folders/1N4Ql_g2NJ1-6Hph2M8YmDpgCYIFWqRJe?usp=drive_link)

## Environment Setup
For setting up the environment, please refer to the instructions in the official LED repository:
- [LED Official Repository](https://github.com/Srameo/LED)

## Training and Evaluation

### Pre-training Phase
To pre-train the model, execute the following command:
```bash
python led/nafnettrain.py -opt led/LED+NAFNet_Pretrain.yaml
```

### Fine-tuning and Inference Phase
For fine-tuning and inference, you can run:
```bash
python led/trainc1.py -opt led/branchnaf1.yaml
```
And:
```bash
python led/trainc2.py -opt led/branchnaf2.yaml
```


