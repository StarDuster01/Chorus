import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Helper function to get auth header
const getAuthHeader = () => {
  const token = localStorage.getItem('token');
  return token ? { Authorization: `Bearer ${token}` } : {};
};

// Datasets
const getDatasets = async () => {
  try {
    const response = await axios.get(`${API_URL}/datasets`, {
      headers: getAuthHeader()
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching datasets:', error);
    throw error;
  }
};

const createDataset = async (dataset) => {
  try {
    const response = await axios.post(`${API_URL}/datasets`, dataset, {
      headers: getAuthHeader()
    });
    return response.data;
  } catch (error) {
    console.error('Error creating dataset:', error);
    throw error;
  }
};

const uploadDocument = async (datasetId, file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await axios.post(
      `${API_URL}/datasets/${datasetId}/documents`,
      formData,
      {
        headers: {
          ...getAuthHeader(),
          'Content-Type': 'multipart/form-data'
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error uploading document:', error);
    throw error;
  }
};

const checkDatasetStatus = async (datasetId) => {
  try {
    const response = await axios.get(
      `${API_URL}/datasets/${datasetId}/status`,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error checking dataset status:', error);
    throw error;
  }
};

const getDatasetDocuments = async (datasetId) => {
  try {
    const response = await axios.get(
      `${API_URL}/datasets/${datasetId}/documents`,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching dataset documents:', error);
    throw error;
  }
};

const removeDocument = async (datasetId, documentId) => {
  try {
    const response = await axios.delete(
      `${API_URL}/datasets/${datasetId}/documents/${documentId}`,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error removing document:', error);
    throw error;
  }
};

const rebuildDataset = async (datasetId) => {
  try {
    const response = await axios.post(
      `${API_URL}/admin/rebuild_dataset/${datasetId}`,
      {},
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error rebuilding dataset:', error);
    throw error;
  }
};

// Bots
const getBots = async () => {
  try {
    const response = await axios.get(`${API_URL}/bots`, {
      headers: getAuthHeader()
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching bots:', error);
    throw error;
  }
};

const createBot = async (bot) => {
  try {
    const response = await axios.post(`${API_URL}/bots`, bot, {
      headers: getAuthHeader()
    });
    return response.data;
  } catch (error) {
    console.error('Error creating bot:', error);
    throw error;
  }
};

const chatWithBot = async (botId, message, debugMode = false) => {
  try {
    const response = await axios.post(
      `${API_URL}/bots/${botId}/chat`,
      { message, debug_mode: debugMode },
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error chatting with bot:', error);
    throw error;
  }
};

const chatWithImage = async (botId, formData) => {
  try {
    const response = await axios.post(
      `${API_URL}/bots/${botId}/chat-with-image`,
      formData,
      {
        headers: {
          ...getAuthHeader(),
          'Content-Type': 'multipart/form-data'
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error chatting with bot using image:', error);
    throw error;
  }
};

const generateDebugFlowchart = async (botId, message) => {
  try {
    console.log(`Generating flowchart for bot ${botId} with message: ${message}`);
    const response = await axios.post(
      `${API_URL}/bots/${botId}/debug-flowchart`,
      { message },
      {
        headers: getAuthHeader(),
        timeout: 30000 // Allow up to 30 seconds for the request to complete
      }
    );
    
    console.log('Flowchart generation successful', response.data);
    
    // Validate response structure
    if (!response.data.mermaid_code) {
      console.error('Invalid flowchart response format:', response.data);
      throw new Error('Flowchart data missing from server response');
    }
    
    return response.data;
  } catch (error) {
    console.error('Error generating debug flowchart:', error);
    
    // Provide more detailed error information
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      console.error('Error response data:', error.response.data);
      console.error('Error response status:', error.response.status);
    } else if (error.request) {
      // The request was made but no response was received
      console.error('No response received:', error.request);
    }
    
    throw error;
  }
};

// Image generation
const generateImage = async (prompt, options = {}) => {
  try {
    const response = await axios.post(
      `${API_URL}/images/generate`,
      { prompt, ...options },
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error generating image:', error);
    throw error;
  }
};

const editImage = async (image, prompt, mask = null) => {
  try {
    const formData = new FormData();
    formData.append('image', image);
    formData.append('prompt', prompt);
    if (mask) {
      formData.append('mask', mask);
    }
    
    const response = await axios.post(
      `${API_URL}/images/edit`,
      formData,
      {
        headers: {
          ...getAuthHeader(),
          'Content-Type': 'multipart/form-data'
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error editing image:', error);
    throw error;
  }
};

const botService = {
  getDatasets,
  createDataset,
  uploadDocument,
  checkDatasetStatus,
  getDatasetDocuments,
  removeDocument,
  rebuildDataset,
  getBots,
  createBot,
  chatWithBot,
  chatWithImage,
  generateDebugFlowchart,
  generateImage,
  editImage
};

export default botService; 