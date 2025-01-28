# Thyroid tumor classification



## Introduction

This project is about classifying thyroid tumors based on ultrasound photos. It uses not only CNN with custom architecture, but also some pre-trained models such as VGG16 or MobileNetV2. Project contains models evaluations on test and validation datasets using confusion matrix and epoch plots for CNN.

## Instalation
1. Clone the Repository
   - Clone the project repository from GitHub:
     
     ```shell
     git clone https://git.comtegra.pl/michalstrus/thyroid_tumor_classification.git
     ```
2. Create Virtual Environment
   - Go to thyroid_tumor_classification and create virtual environment named 'venv' there
     
     ```shell
     cd thyroid_tumor_classification
     python3 -m venv venv
     ```
3. Activate Virtual Environment
    - Activation on Windows
    ```cmd
     venv\Scripts\activate
     ```
    - On macOS/Linux
    ```cmd
     source venv/bin/activate
     ```
     You should now see that your virtual environment is active
4. Install required packages
    ```shell
     pip install -r requirements.txt
     ```

## Downloading the dataset
1. Zipped dataset is already inside the project and will be unzipped in the notebook code

## Running the notebook
1. Run the jupyter notebook
     ```shell
     jupyter notebook
     ```
2. Acces the file
    - Open 'actual_Thyroid.ipynb'
3. Execute the cells in notebook and uncomment unzipping part to load the dataset

## Additional notes
Ensure you have followed the steps in the given order to avoid any issues.




