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

const chatWithBot = async (botId, message, debugMode = false, useModelChorus = false, chorusId = '') => {
  try {
    const response = await axios.post(
      `${API_URL}/bots/${botId}/chat`,
      { message, debug_mode: debugMode, use_model_chorus: useModelChorus, chorus_id: chorusId },
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

// Image generation
const generateImage = async (prompt, options = {}) => {
  try {
    const response = await axios.post(
      `${API_URL}/images/generate`,
      { 
        prompt, 
        ...options 
      },
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

const editImage = async (formData) => {
  try {
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

// Model chorus functions
const getChorusConfig = async (botId) => {
  try {
    const response = await axios.get(
      `${API_URL}/bots/${botId}/chorus`,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching chorus configuration:', error);
    throw error;
  }
};

const saveChorusConfig = async (botId, chorusConfig) => {
  try {
    const response = await axios.post(
      `${API_URL}/choruses`,
      chorusConfig,
      {
        headers: getAuthHeader()
      }
    );
    await setBotChorus(botId, response.data.id);
    return response.data;
  } catch (error) {
    console.error('Error saving chorus configuration:', error);
    throw error;
  }
};

const setBotChorus = async (botId, chorusId) => {
  try {
    const response = await axios.post(
      `${API_URL}/bots/${botId}/set-chorus`,
      { chorus_id: chorusId },
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error setting bot chorus:', error);
    throw error;
  }
};

const deleteBot = async (botId) => {
  try {
    const response = await axios.delete(
      `${API_URL}/bots/${botId}`,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error deleting bot:', error);
    throw error;
  }
};

const getAllChoruses = async () => {
  try {
    const response = await axios.get(
      `${API_URL}/choruses`,
      {
        headers: getAuthHeader()
      }
    );
    
    // Add response and evaluator model counts for each chorus
    const choruses = response.data.map(chorus => ({
      ...chorus,
      response_model_count: chorus.response_models?.length || 0,
      evaluator_model_count: chorus.evaluator_models?.length || 0
    }));
    
    return choruses;
  } catch (error) {
    console.error('Error fetching choruses:', error);
    throw error;
  }
};

const getChorus = async (chorusId) => {
  try {
    const response = await axios.get(
      `${API_URL}/choruses/${chorusId}`,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching chorus:', error);
    throw error;
  }
};

const createChorus = async (chorusData) => {
  try {
    const response = await axios.post(
      `${API_URL}/choruses`,
      chorusData,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error creating chorus:', error);
    throw error;
  }
};

const updateChorus = async (chorusId, chorusData) => {
  try {
    const response = await axios.put(
      `${API_URL}/choruses/${chorusId}`,
      chorusData,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error updating chorus:', error);
    throw error;
  }
};

const deleteChorus = async (chorusId) => {
  try {
    const response = await axios.delete(
      `${API_URL}/choruses/${chorusId}`,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error deleting chorus:', error);
    throw error;
  }
};

const deleteDataset = async (datasetId) => {
  try {
    const response = await axios.delete(
      `${API_URL}/datasets/${datasetId}`,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error deleting dataset:', error);
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
  generateImage,
  editImage,
  getChorusConfig,
  saveChorusConfig,
  setBotChorus,
  deleteBot,
  getAllChoruses,
  getChorus,
  createChorus,
  updateChorus,
  deleteChorus,
  deleteDataset
};

export default botService; 