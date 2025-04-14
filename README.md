# SIADM - Système Intelligent d'Analyse de Données Multimodales

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7.0-orange)
![Transformers](https://img.shields.io/badge/Transformers-4.30.0-yellow)
![MLflow](https://img.shields.io/badge/MLflow-2.3.0-blueviolet)
![Docker](https://img.shields.io/badge/Docker-24.0-lightblue)

Un projet **end-to-end** combinant **Computer Vision**, **NLP** et **MLOps** pour la surveillance et l'analyse en temps réel de flux vidéo et textuels.

## 📌 Fonctionnalités

- **🎥 Computer Vision**  
  - Détection d'objets/personnes avec YOLOv8.  
  - Reconnaissance d'actions (3D CNN).  
- **📝 NLP**  
  - Analyse de sentiment (BERT/Transformers).  
  - Scraping automatique (articles, tweets).  
- **⚙️ MLOps**  
  - Tracking des modèles avec MLflow.  
  - Déploiement conteneurisé (Docker + FastAPI).  
- **📊 Dashboard**  
  - Visualisation des données avec Streamlit/Grafana.  


## 🚀 Utilisation

- **Lancement de l'application**  
   ```bash
   docker-compose up --build    

- **Module Computer Vision**
```bash
python src/cv/realtime_detection.py --source 0  # Webcam

- **Module NLP**
```bash
python src/nlp/sentiment_analysis.py --text "Entrez votre texte ici !! "

- **Accès au dashboard**
```bash
streamlit run src/dashboard/app.py


   
