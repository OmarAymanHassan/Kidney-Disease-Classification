# Kidney-Disease-Classification
A deep learning-based project to classify kidney images as normal or tumorous. It includes a complete MLOps pipeline using MLflow, DVC, Docker, and CI/CD tools for reproducibility, tracking, and deployment in real-world healthcare scenarios.



# 🧠 Kidney Tumor Classification with Deep Learning & MLOps

## 📝 Description

This project presents a full MLOps pipeline for classifying **kidney tumors** from structured or image-based data using a **Deep Learning model**. The classification task involves determining whether a patient's kidney is **normal** or shows signs of a **tumor**. The primary goal is to deliver a reliable, reproducible, and deployable AI solution that could assist in clinical decision-making.

The project goes beyond model training by integrating modern **MLOps tools** to support data versioning, experiment tracking, pipeline automation, and deployment. Tools like **MLflow**, **DVC**, **Git**, and **Docker** are used to ensure the project is production-ready and maintainable.

The pipeline includes preprocessing, training, evaluation, and model tracking as modular components versioned with **DVC**. **MLflow** logs model metrics, parameters, and artifacts. The model is containerized using **Docker**, enabling easy deployment via **FastAPI** or **Streamlit**. CI/CD integration (e.g., GitHub Actions) ensures continuous testing and delivery.

This project is ideal for showcasing end-to-end MLOps practices in a healthcare setting where reproducibility, traceability, and explainability are key.

---

## 📌 Project Objectives

- Classify kidneys as **normal** or **tumorous** using Deep Learning.
- Version data and models using DVC.
- Track experiments with MLflow.
- Build an automated pipeline using DVC stages.
- Package and deploy the model with Docker and FastAPI/Streamlit.
- Integrate CI/CD for automation and reproducibility.

---




## Workflows

To maintain and update the pipeline properly, follow these steps in order:

1. **Update `config.yaml`**  
   Define paths, artifact directories, and pipeline-related configurations.

2. **Update `secrets.yaml` [Optional]**  
   Store and manage sensitive credentials (e.g., API keys, database URIs).

3. **Update `params.yaml`**  
   Modify training-related parameters (e.g., epochs, learning rate, batch size).

4. **Update the entity**  
   Adjust the data classes inside `entity/config_entity.py` to match the new structure.

5. **Update the Configuration Manager**  
   Modify `src/config/configuration.py` to load and manage updated configs/entities.

6. **Update the components**  
   Refactor or add logic inside `src/components/` as per new requirements.

7. **Update the pipeline**  
   Adjust pipeline scripts inside `src/pipeline/` to incorporate component updates.

8. **Update `main.py`**  
   Link and trigger the entire pipeline logic through the main driver script.

9. **Update `dvc.yaml`**  
   Define or modify the DVC pipeline stages to reflect the new process flow.

10. **Update `app.py`**  
   Build or revise the FastAPI/Flask web app used for model inference or testing.

