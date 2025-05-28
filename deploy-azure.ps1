az login
az account set --subscription "Azure subscription 1"
az account show
cd ragbot-frontend
npm install
npm run build
cd ..
if (Test-Path "ragbot-backend\frontend") { Remove-Item -Recurse -Force "ragbot-backend\frontend" }
New-Item -ItemType Directory -Path "ragbot-backend\frontend" -Force
Copy-Item -Path "ragbot-frontend\build\*" -Destination "ragbot-backend\frontend\" -Recurse -Force
az acr update --name viriditytech --resource-group viridity_tech --admin-enabled true
az acr credential show --name viriditytech --resource-group viridity_tech
az acr login --name viriditytech
cd ragbot-backend
docker build -t viriditytech.azurecr.io/ragbot-unified:latest .
docker push viriditytech.azurecr.io/ragbot-unified:latest
cd ..
az containerapp registry set --name chorusbotfull --resource-group viridity_tech --server viriditytech.azurecr.io --identity system
az containerapp update --name chorusbotfull --resource-group viridity_tech --image viriditytech.azurecr.io/ragbot-unified:latest 