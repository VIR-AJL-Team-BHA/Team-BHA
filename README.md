# ✨ Equitable AI for Dermatology

## 📚 Table of Contents
* [Team Members](#-team-members)
* [Project Highlights](#-project-highlights)
* [Setup & Execution](#%EF%B8%8F-setup--execution)
* [Project Overview](#-project-overview)
* [Data Exploration](#-data-exploration)
* [Model Development](#%EF%B8%8F-model-development)
* [Results & Key Findings](#-results--key-findings)
* [Impact Narrative](#-impact-narrative)
* [Next Steps](#-next-steps)

## 🤝 Team Members
* [Shelby Servis](https://github.com/shelbyydiane): data preprocessing
* [Nandini Shah](https://github.com/nshah47): exploratory data analysis
* [Rosemarie Nasta](https://github.com/rosemarie-17): modeling
* [Aayat Alsweiti](https://github.com/Aayat-yuh): modeling
* [Pranavi Reddi](https://github.com/pranavireddi): data preprocessing
* [Carly Kiang](https://github.com/carlykiang): documentation

## 🌟 Project Highlights

This project was developed as part of a Kaggle competition that encourages students to explore how different machine learning models can classify skin conditions across diverse skin tones, ensuring that underrepresented groups are fairly represented. 

For this project, we implemented an image-classification model using ResNet-18 to categorize skin conditions. A pre-trained ResNet-18 model is fine-tuned by replacing its final layer to match the number of classes. The model is trained using cross-entropy loss and Adam optimization for 10 epochs.

One major finding includes the effectiveness of transfer learning in improving model accuracy. We also discovered the importance of diverse and inclusive datasets, as they ensure fair and equitable AI models for medical applications.
## ⚙️ Setup & Execution
1. **Clone Repository**
   ```
   git clone https://github.com/VIR-AJL-Team-BHA/Team-BHA
   cd team-bha
   ```
2. **Install Dependencies**
   ```
   pip install torch torchvision pandas numpy matplotlib
   ```
3. **Download & Prepare Dataset**

   1. Download the dataset:
   ```
   kaggle competitions download -c bttai-ajl-2025
   ```
   2. Ensure the data is structured as:
    ```
    /root folder/
    │── train/                  # Training images
    │   ├── class_1/
    │   ├── class_2/
    │   ├── ...
    │── test/                   # Test images
    │── train.csv               
    │── test.csv                
    ```
## 📋 Project Overview
This Kaggle competition encourages students to explore how different machine learning models can be used to classify skin conditions across diverse skin tones to ensure underrepresented and marginalized groups are equitably represented in the machine learning space. This Kaggle competition connects to the Break Through AI Program since both their missions are centered on increasing inclusivity and fairness in the fields of AI/ML. 

## 🔍 Data Exploration
The dataset is a subset of the FitzPatrick17k dataset, which is a collection of around 4,500 labeled images of 21 dermatological conditions out of the 100+ in the full FitzPatrick set. The images are scored on the FitzPatrick skin tone scale and they were sourced from reputable dermatology websites. 

In order to explore the data we were given, our group first examined the different kinds of data we were given and the balance within the overall dataset. We also focused on examining relationships between the different variables and data we were given to see how we could leverage those relationships to best identify statistically significant results with our final model.

![image 1](https://github.com/VIR-AJL-Team-BHA/Team-BHA/blob/main/visualizations/image1.png) 

Image of the visualization above aims to understand data composition and balance.

![image 4](https://github.com/VIR-AJL-Team-BHA/Team-BHA/blob/main/visualizations/image4.png) 

Image of the visualization above aims to explore relationships between variables given in the data.

![image 3](https://github.com/VIR-AJL-Team-BHA/Team-BHA/blob/main/visualizations/image3.png) 

Image of the visualization above aims to explore the distribution of skin types documented in the data.

![image 2](https://github.com/VIR-AJL-Team-BHA/Team-BHA/blob/main/visualizations/image2.png) 

Image of the visualization above aims to explore the distribution of skin types per diagnosis.

## 🛠️ Model Development

1. **Model Selection: ResNet-18**

    Initially, we experimented with a custom Convolutional Neural Network (CNN) designed from scratch. However, the model struggled to achieve high accuracy due to:
   
      ❌ Insufficient feature extraction
   
      ❌ Limited dataset size
   
      ❌ Overfitting on training data
        
    Despite adjusting hyperparameters and adding more layers, the CNN failed to generalize well, leading to overfitting and poor performance on unseen data. This prompted us to shift toward transfer learning with ResNet-18, which significantly improved accuracy by leveraging pre-trained feature representations. We chose ResNet-18 for the following reasons:

      ✔️ Strong performance on image classification tasks

      ✔️ Pre-trained on ImageNet for robust feature extraction

      ✔️ Ability to learn complex features efficiently

   To adapt ResNet-18 to our dataset, we replaced the fully connected (FC) layer to match the number of skin condition classes.
3. **Hyperparameter Tuning**

   We experimented with different settings and found the following values to be the most effective:
   |⚙️ **Hyperparameter** |🔢 **Value**|
   |:---:|:---:|
   |Learning Rate|0.001|
   |Optimizer|Adam|
   |Loss Function|CrossEntropyLoss|
   |Batch Size|32|
   |Number of Epochs|10|

   These hyperparameters were chosen after evaluating validation loss trends and accuracy improvements across different settings.
4. **Training Approach**

    Since ResNet-18 has been pre-trained on ImageNet, we leverage transfer learning:
   
     **Step 1.** Freezing the early layers to retain general feature extraction capabilities.
   
     **Step 2.** Fine-tuning the FC layers to specialize in skin condition classification.

    The training process involves:
     - <ins>Foward pass:</ins> compute model predictions on batches of images.
     - <ins>Loss computation:</ins> measure how far predictions are from actual labels.
     - <ins>Backpropagation:</ins> update weights using Adam optimizer.
     - <ins>Epoch evaluation:</ins> track loss over epochs to monitor model convergence.
6. **Reason for Approach**

   *Why Transfer Learning?*
    * Faster convergence and better generalization than training from scratch.
    * Prevents overfitting.
      
   *Why Adam Optimizer?*
    * Adaptive learning rate adjusts to different layers, speeding up convergence.
    * It handles sparse gradients well.
      
   *Why Batch Size = 32?*
    * Provides a good balance between stability and training efficiency
    * Larger batches require more memory.
## 📊 Results & Key Findings

To evaluate the model, we used accuracy as the primary metric and supplemented it with precision, recall, and F1-score to assess performance.
|📊 **Metric**|🔢 **Value**|
|:---:|:---:|
|✅ Training Accuracy|91%|
|🎯 Precision (Avg.)|91%|
|🔄 Recall (Avg.)|91%|
|📈 F1-score (Avg.)|91%|

The final model achieved 91% accuracy, demonstrating strong generalization across skin condition classes.

**Confusion Matrix:**
![confusion_matrix](https://github.com/VIR-AJL-Team-BHA/Team-BHA/blob/main/visualizations/confusion-matrix.png)

Here are some key findings of the confusion matrix above:

1. 👍 Strong Performance for Some Classes
   - The model performs well in classifying basal-cell carcinoma (319 correct), squamos-cell carcinoma (403 correct), and melanoma (176 correct), which most predictions along the diagonal.
   - Other classes like folliculitis (232 correct) and eczema (136 correct) also show strong classification performance.
2. 👎 Misclassification Trends
   - Acne is frequently confused with acne vulgaris, with 11 misclassified cases.
   - Acne vulgaris is also confused for basal cell carcinoma (15 misclassifications), dermatomyositis (14 misclassifications), and folliculitis (34 misclassifications, the most of any class).

The model performs well in distinguishing common skin conditions like folliculitis, actinic keratosis, and squamous-cell carcinoma. However, misclassifications occur in closely related conditions, particularly in malignant melanoma vs. melanoma and acne vs. acne vulgaris.

## 🌍 Impact Narrative

Imagine a dermatologist in training, excited to use AI-powered tools to diagnose skin conditions. They pull up a model trained on a vast dataset but soon realize it struggles to identify conditions on darker skin tones, a common flaw in medical AI.

⚠️This isn’t just a technical issue; it’s a matter of health equity, trust, and real-world impact.

**Bridging the Gap**

This project, built for a Kaggle competition focused on skin condition classification across diverse skin tones, aimed to bridge that gap by ensuring underrepresented and marginalized groups are equitably represented in AI models.

**Addressing Model Fairness**

We recognized early on that bias in training data could lead to unfair model predictions, disproportionately affecting darker skin tones. To mitigate this, we took these steps:

✅ <ins>Diverse Dataset Representation:</ins> The dataset included a broad range of skin tones, allowing the model to learn features across different demographics.

✅ <ins>Balanced Data Processing:</ins> We checked for class imbalances and considered techniques such as oversampling & weighted loss functions to prevent bias toward majority classes.

By aligning with Break Through Tech AI's mission to promote fairness in AI/ML, we reinforced the need for inclusive and ethical AI development.

💡 *AI is only as good as the data and ethics behind it.*
## 🚀 Next Steps

While our ResNet-18 model performed well, we recognize several limitations and opportunities for improvement to make it more fair and generalizable.

1. **🗂️ Expanding the Dataset**

   Our dataset was relatively small, which limited the model's ability to learn rare skin conditions and generalize to real-world scenarios.

   <ins>Next Step:</ins> We aim to expand the dataset by exploring additional open-source medical image datasets to improve coverage of underrepresented conditions.
2. **🎯 Improving Model Generalization**

   The model occasionally misclassified visually similar conditions, meaning it struggles with some feature distinctions.

   <ins>Next Step:</ins> We plan to fine-tune deeper architectures, such as ResNet-50, for better feature extraction.
3. **⏲️ Reducing Training Time**

   Currently, the model takes at least 30 minutes per training cycle, which could be optimized to make the model more practical for retraining.

   <ins>Next Step:</ins> Use mixed-precision training to improve training speed without compromising accuracy
