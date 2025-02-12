# **ENVI Implementation for UMAP**

## **Overview**  
This repository currently contains the implementation of ENVI. 
## **Repository Structure**  

- `UMAP/` - Contains UMAP results and related files.  
- `Colab.ipynb` - Merged notebook for running the implementation on Google Colab.  
- `ENVI.ipynb` - Main notebook for ENVI implementation and UMAP generation.  
- `file_reduction.ipynb` - Script for reducing dataset file size.  
- `pre_processing.ipynb` - Preprocessing code to make the dataset manageable.  
- `requirements.txt` - List of dependencies required to run the project.  


## **Dataset**  
The dataset used in this project can be downloaded using the following commands:  
```bash
!wget https://dp-lab-data-public.s3.amazonaws.com/ENVI/sc_data.h5ad
!wget https://dp-lab-data-public.s3.amazonaws.com/ENVI/st_data.h5ad

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


## **Contributors**  
- **Anushka Singh**  
- **Shubh**  

