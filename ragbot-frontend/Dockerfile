FROM node:16-alpine as build

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci

# Copy source files
COPY . .

# Update the proxy setting in package.json to point to the backend
RUN sed -i 's|"proxy": "http://192.168.17.64:5000"|"proxy": "http://backend:50505"|g' package.json

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