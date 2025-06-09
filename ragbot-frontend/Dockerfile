FROM node:16-alpine as build

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy source files
COPY . .

# Set the API URL to use a relative path - this inherits the protocol (https://)
ENV REACT_APP_API_URL=/api

# Build the application
RUN npm run build

# The build output will be in /app/build
WORKDIR /app/build

# Expose the port the app runs on
EXPOSE 3000

# Start the app
CMD ["npx", "serve", "-s", ".", "-l", "3000"] 