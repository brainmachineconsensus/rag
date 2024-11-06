FROM python:3.10-slim

# Définition du répertoire de travail
WORKDIR /app

# Copie du fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie de tout le code de l'application dans le conteneur
COPY . .

# Exposition du port 8080
EXPOSE 8080

# Configuration d'une variable d'environnement pour le port attendu par Cloud Run
ENV PORT=8080

# Démarrer l'application avec Uvicorn
CMD ["uvicorn", "rag.asgi:application", "--host", "0.0.0.0", "--port", "8080"]