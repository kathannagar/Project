# Project
### Introduction

Face verification is an important identity authentication technology used in more and more mobile and embedded applications such as device unlock, application login, mobile payment and so on. Some mobile applications equipped with face verification technology, for example, smartphone unlock, need to run offline. To achieve user friendliness with limited computation resources, the face verification models deployed locally on mobile devices are expected to be not only accurate but also small and fast. However, modern high-accuracy face verification models are built upon deep and big convolutional neural networks (CNNs) which are supervised by novel loss functions during training stage. The big CNN models requiring high computational resources are not suitable for many mobile and embedded applications. Several highly efficient neural network architectures, for example, MobileNetV1 ShuffleNet, and MobileNetV2, have been proposed for common visual recognition tasks rather than face verification in recent years. It is a straight-forward way to use these common CNNs unchanged for face verification, which only achieves very inferior accuracy compared with state-of-the-art results according to our experiments

### Face Detection
Face detection can be regarded as a specific case of object-class detection. In object-class detection, the task is to find the locations and sizes of all objects in an image that belong to a given class. Examples include upper torsos, pedestrians, and cars. Face-detection algorithms focus on the detection of frontal human faces. It is analogous to image detection in which the image of a person is matched bit by bit. Image matches with the image stores in the database. Any facial feature changes in the database will invalidate the matching process.
In order to recognize a face, we would first need to detect a face from an image. There are many ways to do so. We have explored multiple face detectors. These include Face-recognition package (containing Histogram of Oriented Gradients (HOG) and Convolutional Neural Network (CNN) detectors), MTCNN, Yoloface, Faced, and a ultra light face detector released recently. We found that while Yoloface has the highest accuracy and most consistent execution time, the Ultra-light face detector was unrivalled in terms of speed and produces a relatively good accuracy.

### Facial Recognition

A facial recognition system is a technology capable of identifying or verifying a person from a digital image or a video frame from a video source. There are multiple methods in which facial recognition systems work, but in general, they work by comparing selected facial features from given image with faces within a database. It is also described as a Biometric Artificial Intelligence based application that can uniquely identify a person by analyzing patterns based on the person's facial textures and shape.
While initially a form of computer application, it has seen wider uses in recent times on mobile platforms and in other forms of technology, such as robotics. It is typically used as access control in security systems and can be compared to other biometrics such as fingerprint or eye iris recognition systems. Although the accuracy of facial recognition system as a biometric technology is lower than iris recognition and fingerprint recognition, it is widely adopted due to its contactless and non-invasive process. Recently, it has also become popular as a commercial identification and marketing tool. Other applications include advanced human-computer interaction, video surveillance, automatic indexing of images, and video database, among others.


## Approach


### Related Work

Tuning deep neural architectures to strike an optimal balance between accuracy and performance has been an area of active research for the last several years. For common visual recognition tasks, many efficient architectures have been proposed recently. Some efficient architectures can be trained from scratch. For example, SqueezeNet uses a bottleneck approach to design a very small network and achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters (i.e., 1.25 million). MobileNetV1 uses depth wise separable convolutions to build lightweight deep neural networks, one of which, i.e., MobileNet-160 (0.5x), achieves 4% better accuracy on ImageNet than SqueezeNet at about the same size. ShuffleNet utilizes pointwise group convolution and channel shuffle operation to reduce computation cost and achieve higher efficiency than MobileNetV1. MobileNetV2 architecture is based on an inverted residual structure with linear bottleneck and improves the state-of-the-art performance of mobile models on multiple tasks and benchmarks. The mobile NASNet model, which is an architectural search result with reinforcement learning, has much more complex structure and much more actual inference time on mobile devices than MobileNetV1, ShuffleNet, and MobileNetV2. However, these lightweight basic architectures are not so accurate for face verification when trained from scratch.

Accurate lightweight architectures specifically designed for face verification have been rarely researched. presents a light CNN framework to learn a compact embedding on the large-scale face data, in which the Light CNN-29 model achieves 99.33% face verification accuracy on LFW with 12.6 million parameters. Compared with MobileNetV1, Light CNN-29 is not lightweight for mobile and embedded platforms. Light CNN-4 and Light CNN-9 are much less accurate than Light CNN-29. proposes ShiftFaceNet based on ShiftNet-C model with 0.78 million parameters, which only achieves 96.0% face verification accuracy on LFW. In, an improved version of MobileNetV1, namely LMobileNetE, achieves comparable face verification accuracy to state-of-the-art big models. But LMobileNetE is actually a big model of 112MB model size, rather than a lightweight model. All above models are trained from scratch

Another approach for obtaining lightweight face verification models is compressing pretrained networks by knowledge distillation. In , a compact student network (denoted as MobileID) trained by distilling knowledge from the teacher network DeepID2+  achieves 97.32% accuracy on LFW with 4.0MB model size. In, several small MobileNetV1 models for face verification are trained by distilling knowledge from the pretrained FaceNet  model and only face verification accuracy on the authors’ private test dataset are reported. Regardless of the small student models’ accuracy on public test datasets, our MobileFaceNets achieve comparable accuracy to the strong teacher model FaceNet on LFW and MegaFace.



### The Weakness of Common Mobile Networks for Face Verification

There is a global average pooling layer in most recent state-of-the-art mobile networks proposed for common visual recognition tasks, for example, MobileNetV1, ShuffleNet, and MobileNetV2. For face verification and recognition, some researchers have observed that CNNs with global average pooling layers are less accurate than those without global average pooling. However, no theoretical analysis for this phenomenon has been given. Here we make a simple analysis on this phenomenon in the theory of receptive field


### MobileFaceNet Architectures 

Now we describe our MobileFaceNet architectures in detail. The residual bottlenecks proposed in MobileNetV2 are used as our main building blocks. For convenience, we use the same conceptions as those in. The detailed structure of our primary MobileFaceNet architecture is shown in Table 1. Particularly, expansion factors for bottlenecks in our architecture are much smaller than those in MobileNetV2. We use PReLU as the non-linearity, which is slightly better for face verification than using ReLU. In addition, we use a fast downsampling strategy at the beginning of our network, an early dimension-reduction strategy at the last several convolutional layers, and a linear 1 × 1 convolution layer following a linear global depthwise convolution layer as the feature output layer. Batch normalization is utilized during training and batch normalization folding is applied before deploying. Our primary MobileFaceNet network has a computational cost of 221 million MAdds and uses 0.99 million parameters. We further tailor our primary architecture as follows. To reduce computational cost, we change input resolution from 112 × 112 to 112 × 96 or 96 × 96. To reduce the number of parameters, we remove the linear 1 × 1 convolution layer after the linear GDConv layer from MobileFaceNet, the resulting network of which is called MobileFaceNet-M. From MobileFaceNet-M, removing the 1 × 1 convolution layer before the linear GDConv layer produces the smallest network called MobileFaceNet-S. These MobileFaceNet networks’ effectiveness is demonstrated by the experiments in the next section










### Training settings and accuracy comparison on LFW and ClassDataset

We have used two datasets LFW and our own dataset collected from students of our class 



As shown in Table 2, compared with the baseline models of common mobile networks, our MobileFaceNets achieve significantly better accuracy with faster inference speed. Our primary MobileFaceNet achieves the best accuracy and MobileFaceNet with a lower input resolution of 96 × 96 has the fastest inference speed. Note that our MobileFaceNets are more efficient than those with larger expansion factors such as MobileFaceNet (expansion factor ×2) and MobileNetV2- GDConv.

To pursue ultimate performance, MobileFaceNet, MobileFaceNet (112 ×96), and MobileFaceNet (96 ×96) are also trained by ArcFace loss on the cleaned training set of MS-Celeb-1M database with 3.8M images from 85K subjects. The accuracy of our primary MobileFaceNet is boosted to 99.55% and 96.07% on LFW and AgeDB30, respectively. The three trained models’ accuracy on LFW is compared with previous published face verification models in Table 3.


### References

1. Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., et al.: Mobilenets: Efficient convolutional neural networks for mobile vision applications. CoRR, abs/1704.04861 (2017) 

2. Zhang, X., Zhou, X., Lin, M., Sun, J.: Shufflenet: An extremely efficient convolutional neural network for mobile devices. CoRR, abs/1707.01083 (2017) 

3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.C.: MobileNetV2: Inverted Residuals and Linear Bottlenecks. CoRR, abs/1801.04381 (2018)

4. MobileFaceNets: Efficient CNNs for Accurate RealTime Face Verification on Mobile Devices. Sheng Chen1,2, Yang Liu2 , Xiang Gao2 , and Zhen Han1, abs/1804.07573

5. Real time face recognition with CPU: towards data science blog by Yirui Feng


