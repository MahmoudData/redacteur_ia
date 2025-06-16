# Documentation Technique du Projet : Rédacteur IA

## 1. Introduction

Ce projet permet de générer automatiquement un document Word structuré à partir :

* d'un titre + finalité du document
* d'un sommaire fourni par l'utilisateur
* de documents sources PDF

Il utilise des techniques d'IA avancées (LLMs, embeddings, vector search) pour produire un contenu pertinent, basé exclusivement sur les données fournies.

---

## 2. Architecture du projet

```
Streamlit UI
   |
   v
Formulaire utilisateur
   |
   v
Pipeline (pipeline.py)
   |
   v
- Parsing du sommaire
- Extraction texte via LLM Sherpa
- Embedding via Scaleway LLM
- Stockage vectoriel (Qdrant)
- Génération de contenu par section
- Compilation DOCX
```

---

## 3. Détails de la pipeline

### a. Parsing du sommaire

Utilisation de regex pour structurer le sommaire en sections, sous-sections, etc.

### b. Extraction de texte

Via `LLMSherpaFileLoader`, les documents PDF sont découpés par section.

### c. Vectorisation

Les chunks de texte sont encodés avec `Scaleway OpenAI-compatible API` (modèle : `bge-multilingual-gemma2`).

### d. Stockage vectoriel

Les embeddings sont insérés dans `Qdrant`, une base de données vectorielle.

### e. Recherche contextuelle

Pour chaque section à rédiger, on recherche les meilleurs chunks associés dans Qdrant.

### f. Rédaction avec LLM

Le modèle `llama-3.3-70b-instruct` génère le contenu des sections sur la base du contexte extrait.

### g. Génération Word

Tous les textes rédigés sont injectés dans un `Document Word (.docx)`.

---

## 4. Interface utilisateur (Streamlit)

* Formulaire avec titre, finalité, sommaire, fichiers PDF.
* Affichage de l'état d'avancement via `st.status()`.
* Téléchargement final du document via `st.download_button()`.
* Authentification vérifiée avec `st.session_state["authentication_status"]`

---

## 5. Configuration et Secrets

* `api_key`: clé Scaleway pour LLM et embeddings
* `llmsherpa_api_url`: URL de l'API LLM Sherpa
* `qdrant_url`: URL du serveur Qdrant

Ces données sont stockées dans `st.secrets`.

---

## 6. Modèles et APIs utilisés

* **LLM de génération**: LLaMA 3.3 - 70B (Scaleway)
* **Embedding**: `bge-multilingual-gemma2` (Scaleway)
* **Vector store**: `Qdrant`
* **Parsing PDF**: `LLMSherpa`

### Hébergement des services API

* **Qdrant** : hébergé sur un **VPS (Hostinger)**, pour le stockage et la recherche d'embedding.
* **LLM Sherpa API** : également déployée sur **VPS (Hostinger)**, permet l'extraction de texte structuré à partir de PDF.

---

## 7. Environnement & Dépendances

### Principaux packages Python :

* `streamlit`
* `openai`
* `langchain`
* `qdrant-client`
* `docx`
* `uuid`, `datetime`, `time`, `json`

