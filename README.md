# RagBot: Docker Rebuild & Azure Deployment Guide

## 1. Prerequisites

- Docker and Docker Compose installed
- Azure account (for deployment)
- (Optional) Azure CLI installed (`az`)

## 2. Project Structure

```
ragbot-backend/      # Flask backend (Python)
ragbot-frontend/     # React frontend (Node/React)
docker-compose.yml   # Multi-service orchestration
```

## 3. Local Rebuild (Full Stack)

### a. Rebuild and Start Everything

```bash
docker-compose down
docker-compose up --build -d
```
- This will rebuild both backend and frontend images and start them in detached mode.
- Backend: http://localhost:50505
- Frontend: http://localhost

### b. Rebuild Only the Frontend

```bash
cd ragbot-frontend
docker build -t ragbot-frontend .
```

### c. Rebuild Only the Backend

```bash
cd ragbot-backend
docker build -t ragbot-backend .
```

## 4. Building the React Frontend for Production

If you want to serve the React build from the backend (for single-container deployment):

```bash
cd ragbot-frontend
npm install
npm run build
# Copy the build to the backend's static folder:
cp -r build/* ../ragbot-backend/frontend/
```
- The backend Flask app is configured to serve static files from `ragbot-backend/frontend/`.

## 5. Deploying to Azure

### a. Using Azure Web App for Containers

1. **Push your images to a registry** (Azure Container Registry or Docker Hub):

   ```bash
   # Tag and push backend
docker tag ragbot-backend <your-registry>/ragbot-backend:latest
docker push <your-registry>/ragbot-backend:latest

   # Tag and push frontend
docker tag ragbot-frontend <your-registry>/ragbot-frontend:latest
docker push <your-registry>/ragbot-frontend:latest
   ```

2. **Create Azure Web App for Containers** (one for backend, one for frontend):

   ```bash
   # Example using Azure CLI
   az webapp create --resource-group <group> --plan <plan> --name <backend-app-name> --deployment-container-image-name <your-registry>/ragbot-backend:latest
   az webapp create --resource-group <group> --plan <plan> --name <frontend-app-name> --deployment-container-image-name <your-registry>/ragbot-frontend:latest
   ```

3. **Configure environment variables** in Azure Portal or using `az webapp config appsettings set`.

4. **Set up networking** so the frontend can reach the backend (use environment variables or Nginx config to point `/api` to the backend).

### b. Using Azure Container Instances (for quick testing)

```bash
az container create --resource-group <group> --name ragbot-backend --image <your-registry>/ragbot-backend:latest --ports 50505
az container create --resource-group <group> --name ragbot-frontend --image <your-registry>/ragbot-frontend:latest --ports 80
```

## 6. Notes on React/Backend Integration

- **Development:** React runs on its own (port 3000), backend on 50505.
- **Production:** React build is served by Nginx (in the frontend container) or can be copied to the backend's static folder for a single-container deployment.
- **API Proxy:** The frontend is configured to use `/api` as the API root. In production, ensure Nginx or Azure routes `/api` to the backend.

## 7. Troubleshooting

- If you see "no image" or missing files, make sure you've rebuilt both containers and that volumes are correctly mapped.
- For persistent data, Docker volumes are mapped in `docker-compose.yml`.

---

**Summary:**  
- Use `docker-compose up --build -d` to rebuild and run locally.
- For Azure, push images to a registry and deploy using Azure Web App for Containers or Container Instances.
- To serve React from Flask, build React and copy the build to the backend's static folder. 