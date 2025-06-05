import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL;

// Helper function to get auth header
const getAuthHeader = () => {
  const token = localStorage.getItem('token');
  if (!token) {
    throw new Error('No authentication token found. Please log in again.');
  }
  return { Authorization: `Bearer ${token}` };
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

const downloadDocument = async (datasetId, documentId, filename) => {
  try {
    const response = await axios.get(
      `${API_URL}/datasets/${datasetId}/documents/${documentId}/download/${encodeURIComponent(filename)}`,
      {
        headers: getAuthHeader(),
        responseType: 'blob'
      }
    );
    
    // Create a URL for the blob
    const url = window.URL.createObjectURL(new Blob([response.data]));
    
    // Create a temporary link element
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename);
    
    // Append to body, click, and remove
    document.body.appendChild(link);
    link.click();
    link.parentNode.removeChild(link);
    
    // Clean up the URL
    window.URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Error downloading document:', error);
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

const chatWithBot = async (botId, message, debugMode = false, useModelChorus = false, chorusId = '', conversationId = '') => {
  try {
    // First get the bot's datasets to ensure they exist and are accessible
    const botDatasets = await getBotDatasets(botId);
    if (!botDatasets || botDatasets.length === 0) {
      console.warn('Bot has no datasets configured');
    }

    const response = await axios.post(
      `${API_URL}/bots/${botId}/chat`,
      { 
        message, 
        debug_mode: debugMode, 
        use_model_chorus: useModelChorus, 
        chorus_id: chorusId,
        conversation_id: conversationId,
        response_format: "markdown"
      },
      {
        headers: getAuthHeader()
      }
    );

    // Ensure image_details is always an array
    if (!response.data.image_details) {
      response.data.image_details = [];
    }

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

const enhancePrompt = async (prompt) => {
  try {
    console.log('Sending enhance prompt request:', prompt);
    const response = await axios.post(
      `${API_URL}/images/enhance-prompt`,
      { prompt },
      {
        headers: getAuthHeader()
      }
    );
    
    console.log('Enhance prompt response:', response.data);
    
    // The backend sends success, original_prompt, and enhanced_prompt
    if (response.data && response.data.enhanced_prompt) {
      return response.data;
    } else {
      throw new Error('Invalid response format from server');
    }
  } catch (error) {
    console.error('Error enhancing prompt:', error);
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

// Conversation Management
const getConversations = async (botId) => {
  try {
    const response = await axios.get(
      `${API_URL}/bots/${botId}/conversations`,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching conversations:', error);
    throw error;
  }
};

const getConversation = async (botId, conversationId) => {
  try {
    const response = await axios.get(
      `${API_URL}/bots/${botId}/conversations/${conversationId}`,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching conversation:', error);
    throw error;
  }
};

const deleteConversation = async (botId, conversationId) => {
  try {
    const response = await axios.delete(
      `${API_URL}/bots/${botId}/conversations/${conversationId}`,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error deleting conversation:', error);
    throw error;
  }
};

const deleteAllConversations = async (botId) => {
  try {
    const response = await axios.delete(
      `${API_URL}/bots/${botId}/conversations`,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error deleting all conversations:', error);
    throw error;
  }
};

const renameConversation = async (botId, conversationId, title) => {
  try {
    const response = await axios.post(
      `${API_URL}/bots/${botId}/conversations/${conversationId}/rename`,
      { title },
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error renaming conversation:', error);
    throw error;
  }
};

// Add these new functions to handle images
const getDatasetImages = async (datasetId) => {
  try {
    const response = await axios.get(
      `${API_URL}/datasets/${datasetId}/images`,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching dataset images:', error);
    throw error;
  }
};

const uploadImage = async (datasetId, file) => {
  try {
    const formData = new FormData();
    formData.append('image', file);
    
    const response = await axios.post(
      `${API_URL}/datasets/${datasetId}/images`,
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
    console.error('Error uploading image:', error);
    throw error;
  }
};

const removeImage = async (datasetId, imageId) => {
  try {
    const response = await axios.delete(
      `${API_URL}/datasets/${datasetId}/images/${imageId}`,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error removing image:', error);
    throw error;
  }
};

// After the getAllChoruses function, add these new functions:

const getBotDatasets = async (botId) => {
  try {
    const response = await axios.get(
      `${API_URL}/bots/${botId}/datasets`,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching bot datasets:', error);
    throw error;
  }
};

const addDatasetToBot = async (botId, datasetId) => {
  try {
    const response = await axios.post(
      `${API_URL}/bots/${botId}/datasets`,
      { dataset_id: datasetId },
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error adding dataset to bot:', error);
    throw error;
  }
};

const removeDatasetFromBot = async (botId, datasetId) => {
  try {
    const response = await axios.delete(
      `${API_URL}/bots/${botId}/datasets/${datasetId}`,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error removing dataset from bot:', error);
    throw error;
  }
};

const setBotDatasets = async (botId, datasetIds) => {
  try {
    const response = await axios.post(
      `${API_URL}/bots/${botId}/set-datasets`,
      { dataset_ids: datasetIds },
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error setting bot datasets:', error);
    throw error;
  }
};

const checkUploadStatus = async (datasetId, statusId) => {
  try {
    const response = await axios.get(
      `${API_URL}/datasets/${datasetId}/upload-status/${statusId}`,
      {
        headers: getAuthHeader()
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error checking upload status:', error);
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
  downloadDocument,
  getBots,
  createBot,
  chatWithBot,
  chatWithImage,
  generateImage,
  enhancePrompt,
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
  deleteDataset,
  getConversations,
  getConversation,
  deleteConversation,
  deleteAllConversations,
  renameConversation,
  getDatasetImages,
  uploadImage,
  removeImage,
  getBotDatasets,
  addDatasetToBot,
  removeDatasetFromBot,
  setBotDatasets,
  checkUploadStatus
};

export default botService; 