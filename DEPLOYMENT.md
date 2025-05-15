# RagBot Deployment Guide

This guide explains how to deploy RagBot to Azure Container Apps.

## Prerequisites

- Azure account with an active subscription
- Azure CLI installed
- Docker and Docker CLI installed locally

## Local Testing

1. Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key (optional)
GROQ_API_KEY=your_groq_api_key (optional)
JWT_SECRET=your_jwt_secret_key
```

2. Build and run locally with Docker Compose:
```bash
docker-compose up --build
```

3. Access the application at http://localhost

## Deploy to Azure Container Apps

1. Log in to Azure:
```bash
az login
```

2. Build and deploy the backend:
```bash
cd ragbot-backend

# Build and deploy to Azure Container Apps
az containerapp up \
  --resource-group ragbot-rg \
  --name ragbot-backend \
  --ingress external \
  --target-port 50505 \
  --env-vars OPENAI_API_KEY=your_openai_api_key ANTHROPIC_API_KEY=your_anthropic_api_key GROQ_API_KEY=your_groq_api_key JWT_SECRET=your_jwt_secret_key \
  --source .
```

3. Get the backend URL:
```bash
az containerapp show -n ragbot-backend -g ragbot-rg --query properties.configuration.ingress.fqdn -o tsv
```

4. Update the frontend Nginx configuration with the backend URL. Edit `ragbot-frontend/nginx.conf` and update the proxy_pass line:
```
proxy_pass https://your-backend-url;
```

5. Deploy the frontend:
```bash
cd ../ragbot-frontend

# Build and deploy to Azure Container Apps
az containerapp up \
  --resource-group ragbot-rg \
  --name ragbot-frontend \
  --ingress external \
  --target-port 80 \
  --source .
```

6. Access your application at the frontend URL:
```bash
az containerapp show -n ragbot-frontend -g ragbot-rg --query properties.configuration.ingress.fqdn -o tsv
```

## Persistent Storage

For production deployments, consider using:
- Azure Blob Storage for uploads and data
- Azure Database for PostgreSQL or Azure Cosmos DB to replace file-based storage

## Environment Variables

Set these environment variables in Azure Container Apps:
- OPENAI_API_KEY: Your OpenAI API key
- ANTHROPIC_API_KEY: Your Anthropic API key (optional)
- GROQ_API_KEY: Your Groq API key (optional)
- JWT_SECRET: Secret for JWT token generation

## Monitoring and Logging

Azure Container Apps automatically sends logs to Azure Log Analytics. Use the Azure portal to:
- Monitor application performance
- View container logs
- Set up alerts

## Scaling

Configure scaling rules in Azure Container Apps:
- Scale based on HTTP traffic
- Set minimum and maximum replicas
- Configure CPU and memory limits 