FROM node:16-alpine as build

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci

# Copy source files
COPY . .

# Update the API URL to point to the backend service
ENV REACT_APP_API_URL=http://localhost:50505/api

# Build the application
RUN npm run build

# Production environment
FROM nginx:alpine

# Copy build files from the build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"] 