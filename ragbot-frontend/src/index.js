import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import 'bootstrap/dist/css/bootstrap.min.css';

// Display loading message while app is initializing
const rootElement = document.getElementById('root');
if (rootElement) {
  rootElement.innerHTML = '<div style="display: flex; justify-content: center; align-items: center; height: 100vh; font-family: sans-serif;">Loading application...</div>';
}

const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
); 