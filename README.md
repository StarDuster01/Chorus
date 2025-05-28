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


## 4. Building the React Frontend for Production

If you want to serve the React build from the backend (for single-container deployment):

```bash
cd ragbot-frontend
npm install
npm run build
# Copy the build to the backend's static folder:
cp -r build/* ../ragbot-backend/frontend/
```
on windows 

xcopy /E /I /Y ragbot-frontend\build ragbot-backend\frontend

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

### Common Issues

**"Container name already in use" error:**
- Follow the container management steps in section 5 above.

**Frontend crashes when asking for images ("Error processing your request"):**
- This was caused by a regex error in the backend. The fix is included in the latest code.
- If you're still experiencing this issue, rebuild and restart your container:
  ```bash
  docker stop ragbot-local-instance
  docker rm ragbot-local-instance
  cd ragbot-backend
  docker build -t ragbot-local-unified .
  cd ..
  docker run -d -p 50505:50505 --name ragbot-local-instance ragbot-local-unified
  ```

**Images not displaying properly:**
- Make sure your datasets contain images and they were uploaded successfully.
- Check that the image files exist in the `ragbot-backend/uploads/images` directory.
- Verify the dataset type is set to "image" or "mixed" if you want to upload images.

**General troubleshooting:**
- If you see "no image" or missing files, make sure you've rebuilt both containers and that volumes are correctly mapped.
- For persistent data, Docker volumes are mapped in `docker-compose.yml`.
- Check container logs: `docker logs ragbot-local-instance`

## 8. Local Single-Container Testing (Frontend served by Backend)

This method allows you to test the application locally in a way that mirrors a single-container deployment, where the Flask backend serves the React frontend. This does not interfere with Azure deployment steps.

**Complete Rebuild Process (Exact Order of Operations):**

1.  **Stop and Remove Existing Docker Container:**
    ```bash
    # Stop the existing container
    docker stop ragbot-local-instance
    
    # Remove the stopped container
    docker rm ragbot-local-instance
    ```

2.  **Rebuild the React Frontend:**
    ```bash
    cd ragbot-frontend
    npm install  # If you haven't already or dependencies changed
    npm run build
    cd ..
    ```

3.  **Delete Existing Frontend Files in Backend:**
    ```bash
    # For Windows:
    rmdir /S /Q ragbot-backend\frontend
    # For Linux/macOS:
    # rm -rf ragbot-backend/frontend
    ```

4.  **Copy New Frontend Build to Backend:**
    ```bash
    # Create the target directory
    mkdir ragbot-backend\frontend
    
    # Copy the build files
    # For Windows:
    xcopy ragbot-frontend\build ragbot-backend\frontend\ /E /I /Y
    # For Linux/macOS:
    # mkdir -p ragbot-backend/frontend/
    # cp -r ragbot-frontend/build/* ragbot-backend/frontend/
    ```

5.  **Build the New Backend Docker Image:**
    ```bash
    cd ragbot-backend
    docker build -t ragbot-local-unified .
    cd ..
    ```

6.  **Run the New Unified Docker Container:**
    ```bash
    docker run -d -p 50505:50505 --name ragbot-local-instance ragbot-local-unified
    ```
    - Access the application at `http://localhost:50505`.

**Quick Commands for Different Scenarios:**

**If you get a "container name already in use" error (container management only):**
```bash
# Stop the existing container
docker stop ragbot-local-instance

# Remove the stopped container
docker rm ragbot-local-instance

# Now run the new container
docker run -d -p 50505:50505 --name ragbot-local-instance ragbot-local-unified
```

**To restart with a fresh container (removes all data, no rebuild):**
```bash
docker stop ragbot-local-instance
docker rm ragbot-local-instance
docker run -d -p 50505:50505 --name ragbot-local-instance ragbot-local-unified
```

---

**Summary:**  
- Use `docker-compose up --build -d` to rebuild and run locally (multi-container).
- For local single-container testing (Flask serving React): build React, copy to backend, then build and run backend Docker image.
- For Azure, push images to a registry and deploy using Azure Web App for Containers or Container Instances.
- To serve React from Flask, build React and copy the build to the backend's static folder. 