# Movement Detection Using Radar Data
This project, developed at IIT Kharagpur under the guidance of Prof. Koustav Rudra, focuses on building an end-to-end deep learning model for human movement detection using Doppler radar data. The goal is to classify movement into three distinct states: 'no person', 'approaching', and 'moving away', enabling robust, real-time monitoring.
Objective
The primary objective is to create a reliable system for classifying human movement by analyzing radar signals. This involved collecting raw radar data, pre-processing it to create a clean and uniform dataset, and then developing and training various deep learning models to achieve high classification accuracy. The GRU-based model was ultimately selected for its superior performance.
Project Workflow
## 1. Data Collection & Labeling
Raw Doppler radar data was collected in a lab environment for the three movement scenarios.
The collected data was manually labeled to create a supervised learning dataset.
## 2. Signal Pre-processing
Visualization: The raw signals were visualized using Matplotlib to understand their characteristics.
Noise Removal: Filtering techniques were applied to remove noise from the radar signals.
Normalization: The data was normalized to ensure all signal sequences had a similar scale, which is crucial for stable model training.
Truncation & Padding: Sequences were truncated or padded to a uniform length to create a consistent input size for the deep learning models.
## 3. Model Development & Training
Architectures Explored: Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU) models were developed and evaluated.
Fine-Tuning: The models were fine-tuned by adjusting hyperparameters like the number of layers, hidden units, dropout rate, and learning rate.
Training: The models were trained on the processed dataset to learn the patterns corresponding to each movement class.
## 4. Evaluation & Results
The models were evaluated based on their accuracy on the training and validation datasets. The final optimized GRU model achieved:
98% Training Accuracy
95% Validation Accuracy
This indicates a highly effective model with good generalization capabilities for unseen data.

## How to Use This Code
Clone the repository:
git clone [https://github.com/your-username/radar-movement-detection.git](https://github.com/your-username/radar-movement-detection.git)
cd radar-movement-detection


Install dependencies:
Make sure you have Python and pip installed. Then, install the required libraries using the requirements.txt file.
pip install -r requirements.txt


Prepare the Data:
Place your processed and separated radar data files (.zip) in the appropriate directory as referenced in the code.
Update the file paths in the notebook/script to point to your dataset location.
Run the Notebook:
Open the Jupyter Notebook RADAR_PROJECT_CODE_RNN_LSTM_GRU.ipynb.
Execute the cells sequentially to pre-process the data, define the models, train them, and evaluate their performance.
Model Architecture
The final model is a GRU-based classifier built with PyTorch. Key features of the architecture include:
Bidirectional GRU Layers: To capture temporal information from both forward and backward directions in the sequence.
Dropout: To prevent overfitting.
Batch Normalization: To stabilize and speed up the training process.
Fully Connected Layers: To map the GRU output to the final class probabilities.
This architecture proved to be the most effective in learning the complex patterns within the radar signal data for this classification task.



