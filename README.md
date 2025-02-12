# **ENVI Implementation**

## **Overview**  
This repository contains the implementation of **ENVI** data. 

## **Dataset**  
The dataset used in this project can be downloaded using the following commands:  
```bash
!wget https://dp-lab-data-public.s3.amazonaws.com/ENVI/sc_data.h5ad
!wget https://dp-lab-data-public.s3.amazonaws.com/ENVI/st_data.h5ad
```
- **Original Dataset:** 8.5GB, 71K cells (Updated from previous 1.5GB, 17K cells dataset).
- 
## **Repository Structure**  

- `UMAP/` - Contains UMAP results and related files.  
- `Colab.ipynb` - Merged notebook for running the implementation on Google Colab.  
- `ENVI.ipynb` - Main notebook for ENVI implementation and UMAP generation.  
- `file_reduction.ipynb` - Script for reducing dataset file size.  
- `pre_processing.ipynb` - Preprocessing code to make the dataset manageable.  
- `requirements.txt` - List of dependencies required to run the project.  

## **How to Use**  

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/Undergraduate-Project.git
   cd envi-umap
   ```  
2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```  
3. **Run preprocessing script** (`pre_processing.ipynb`) to reduce dataset size.  
4. **Train the model** using `ENVI.ipynb` to generate UMAP embeddings.  
5. **Run on Colab** using `Colab.ipynb` (Note: Large datasets may cause session restarts).  

## **Future Improvements**  
- Train the model for **16,000 iterations** using a GPU server to improve results.  
- Optimize dataset handling for large-scale training.  
- Improve model performance and fine-tune hyperparameters.  

## **Contributors**  
- **Anushka Singh**  
- **Shubh**  

