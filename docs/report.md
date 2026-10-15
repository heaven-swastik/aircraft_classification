# Aircraft manufacturer classification: project report

Swastik, B.Tech CSE (Cyber Security), UEM Kolkata

## 1. Goal

The task is to predict which of 30 manufacturers built the aircraft in a photograph. I picked it because it is a fine-grained problem with a small, uneven dataset, which makes it a good place to learn transfer learning and to see where accuracy alone is misleading. I used PyTorch and Google Colab (free T4 GPU) throughout.

## 2. Data

I used FGVC-Aircraft 2013b from the Visual Geometry Group at Oxford, manufacturer level, with the official split: 3,334 training, 3,333 validation and 3,333 test images, 30 classes. I kept the official split rather than re-splitting. Checks on the split files found no missing images, malformed lines or duplicate ids, and no image id appears in two splits.

The class distribution is very uneven. Boeing has 733 training images (22.0% of the set) and the smallest class, Cirrus Aircraft, has 33, a ratio of 22.2. A model that always answers "Boeing" is therefore right about 22% of the time, which is why I also track macro-averaged metrics.

Each image has a 20 px copyright banner at the bottom. I crop it for every split so the network cannot use the text as a shortcut. I did not measure how much this matters.

## 3. Method

**Preprocessing.** Images are converted to RGB, the banner is cropped and the image is shrunk once so the short side is 256 px (cached on disk to speed up training). When loaded, images are resized to 224x224 and normalised with the ImageNet mean and standard deviation, which is what the pretrained weights expect. Training uses a random resized crop (scale 0.7 to 1.0) and a horizontal flip. Validation and test use no randomness.

**Data loading.** A custom `Dataset` checks columns, file paths and label ranges before training, and raises an error naming the split, index and image id if an image cannot be read. All splits share one class-to-index mapping taken from `manufacturers.txt`. I confirmed that labels coming out of the DataLoader match the DataFrame order.

**Model.** ResNet-18 with ImageNet weights, final layer replaced by a 512 to 30 linear layer (about 11.19 million parameters in total). With only 3,334 training images, training from scratch would overfit, so I reuse the pretrained features.

**Training.** Cross-entropy loss, AdamW with weight decay 0.01, cosine learning-rate decay, batch size 64. A checkpoint is saved only when validation macro-F1 improves, and training stops early if it does not improve for 5 epochs. Frozen layers are kept in eval mode so their BatchNorm statistics stay fixed. Seeds were fixed for Python, NumPy and PyTorch.

**Baselines.** Always predicting the most common training class, and logistic regression on 32x32 RGB pixels (about a minute to fit).

## 4. Experiments

All three use seed 42 and one run each.

| experiment | what is trained | learning rates | class weights |
|---|---|---|---|
| exp1 | new output layer only | 1e-3 | no |
| exp2 | `layer4` and new output layer | 1e-3 (new layer), 1e-4 (`layer4`) | no |
| exp3 | same as exp2 | same as exp2 | yes, N / (30 x class count) |

Validation results:

| experiment | best epoch | val macro-F1 | val accuracy | val loss | train time |
|---|---|---|---|---|---|
| exp1 | 12 | 0.4431 | 0.4650 | 1.728 | 5.4 min |
| exp2 | 12 | 0.7857 | 0.7579 | 0.781 | 7.7 min |
| exp3 | 14 | 0.7801 | 0.7240 | 0.827 | 8.0 min |

Exp2 had the highest validation macro-F1 and was selected. The frozen run (exp1) was still improving at its last epoch, so part of its gap to the fine-tuned models is probably due to a short training budget; I did not run a longer frozen version. Class weights (exp3) lowered validation accuracy and did not raise macro-F1.

## 5. Test results

Evaluated once, after selection:

| model | accuracy | macro precision | macro recall | macro F1 | weighted F1 |
|---|---|---|---|---|---|
| majority class (Boeing) | 0.2202 | 0.0073 | 0.0333 | 0.0120 | 0.0795 |
| logistic regression, 32x32 pixels | 0.2637 | 0.2935 | 0.2175 | 0.2418 | 0.2621 |
| ResNet-18, `layer4` fine-tuned | 0.7636 | 0.8240 | 0.7704 | 0.7935 | 0.7637 |

For the selected model, weighted precision is 0.7684 and weighted recall 0.7636. 788 of 3,333 images (23.6%) are misclassified. Predicting the full test set took 10.75 s (3.2 ms per image including image loading).

Weighted F1 is almost the same as accuracy because both are dominated by the large classes. Macro-F1 (0.794) is higher than accuracy because several small classes do well. Macro precision (0.824) is higher than macro recall (0.770), so on average the model is more reliable when it names a class than at finding all images of it.

Per class (test F1): weakest are McDonnell Douglas 0.621 (233 images), Lockheed Corporation 0.651 (66) and Dornier 0.712 (33); strongest are Robin 0.943, Ilyushin 0.939 and Cirrus Aircraft 0.912 (about 34 images each).

## 6. Error analysis

- **Boeing and Airbus absorb many errors.** 358 of the 788 errors (45%) are predictions of Boeing or Airbus, and 247 (31%) are predictions of Boeing. Boeing's precision is 0.708 but its recall is 0.815, so it is over-predicted. The most frequent confusions (true -> predicted) are Airbus -> Boeing 92 (21% of Airbus test images), Boeing -> Airbus 60, McDonnell Douglas -> Boeing 51 (22% of that class), Boeing -> McDonnell Douglas 41, Douglas Aircraft Company -> Boeing 27, Embraer -> Boeing 19, Embraer -> Airbus 15, Lockheed Corporation -> Boeing 13. Confusions between the two Douglas classes are rare (2 and 3 images).
- **Class size.** The Spearman correlation between training-set size and per-class F1 is -0.47 (p = 0.009): larger classes scored lower. I had expected rare classes to suffer. Possible reasons, which I did not test: large manufacturers cover very varied aircraft, and some small classes (Robin, Cirrus) look nothing like airliners. With about 33 test images per small class, one image shifts recall by roughly 3 points, so single-class scores are noisy.
- **Confidence.** Mean confidence is 0.851 on correct and 0.620 on wrong predictions, so confidence carries some signal, but the five most confident mistakes are between 0.994 and 0.999. In the gallery of high-confidence errors there are what look like 747s and a 777 predicted as Airbus, and rear-engine jets mixed up between McDonnell Douglas and Boeing.
- **Overfitting.** Exp2's training accuracy reaches about 99% by epoch 11 while validation accuracy stays around 75-76% from epoch 7 on. The model memorises the small training set; I did not try stronger augmentation or unfreezing fewer layers.

## 7. Limitations

- One seed and one run per setup, no confidence intervals; small differences may be noise.
- The validation set was used for checkpoint choice and for choosing among three setups, so validation scores are slightly optimistic. The test set was used once.
- Per-class test counts are small (about 33 for the smaller classes).
- No photos from outside the dataset were tested. The model may use airline liveries or backgrounds rather than aircraft shape; I did not check.
- Predicted probabilities are not calibrated.

## 8. Next steps

Repeat the best setup with several seeds; give the frozen backbone a longer or higher learning-rate run for a fairer comparison; try unfreezing `layer3`; try stronger augmentation to reduce overfitting; try a larger backbone; and try the family (70 classes) and variant (100 classes) tasks as separate models.

## References

1. S. Maji, E. Rahtu, J. Kannala, M. Blaschko, A. Vedaldi. Fine-Grained Visual Classification of Aircraft. arXiv:1306.5151, 2013.
2. K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. CVPR, 2016.
3. I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization. ICLR, 2019.
