import React, { useState, useEffect } from 'react';
import { Container, Card, Button, Form, Row, Col, ListGroup, Badge, Alert, Spinner, Modal, ProgressBar, Toast, ToastContainer } from 'react-bootstrap';
import botService from '../services/botService';
import TesseractLoader from './TesseractLoader';
import { FaDatabase, FaPlus, FaFileUpload, FaFile, FaTimes, FaFileAlt, FaTrash, FaList, FaExclamationTriangle, FaImage, FaChartBar, FaInfo, FaCheckCircle, FaExclamationCircle } from 'react-icons/fa';
import axios from 'axios';

const DatasetPanel = () => {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [detailedStatus, setDetailedStatus] = useState('');
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [createMode, setCreateMode] = useState(false);
  const [uploadMode, setUploadMode] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [file, setFile] = useState(null);
  const [newDataset, setNewDataset] = useState({
    name: '',
    description: '',
    type: 'mixed'
  });
  const [documents, setDocuments] = useState([]);
  const [viewDocumentsMode, setViewDocumentsMode] = useState(false);
  const [documentListLoading, setDocumentListLoading] = useState(false);
  const [showConfirmDelete, setShowConfirmDelete] = useState(false);
  const [documentToDelete, setDocumentToDelete] = useState(null);
  const [showConfirmDatasetDelete, setShowConfirmDatasetDelete] = useState(false);
  const [datasetToDelete, setDatasetToDelete] = useState(null);
  const [images, setImages] = useState([]);
  const [imageListLoading, setImageListLoading] = useState(false);
  const [viewImagesMode, setViewImagesMode] = useState(false);
  const [imageToDelete, setImageToDelete] = useState(null);
  const [showConfirmImageDelete, setShowConfirmImageDelete] = useState(false);
  const [acceptedFileTypes, setAcceptedFileTypes] = useState('.pdf,.txt,.docx,.pptx,.jpg,.jpeg,.png,.gif,.webp,.bmp');
  const [fileTypeDescription, setFileTypeDescription] = useState('Supported formats: PDF, TXT, DOCX, PPTX, JPG, JPEG, PNG, GIF, WEBP, BMP');
  const [bulkUploadMode, setBulkUploadMode] = useState(false);
  const [bulkZipFile, setBulkZipFile] = useState(null);
  const [bulkResult, setBulkResult] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [statusCheckInterval, setStatusCheckInterval] = useState(null);
  const [operationLogs, setOperationLogs] = useState([]);
  const [showSystemHealth, setShowSystemHealth] = useState(false);
  const [systemHealth, setSystemHealth] = useState(null);

  // Enhanced logging function
  const addOperationLog = (message, type = 'info', details = null) => {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = {
      id: Date.now() + Math.random(),
      timestamp,
      message,
      type, // 'info', 'success', 'warning', 'error'
      details
    };
    
    setOperationLogs(prev => [logEntry, ...prev].slice(0, 50)); // Keep last 50 logs
    console.log(`[Dataset Panel ${type.toUpperCase()}] ${message}`, details || '');
    
    // Auto-clear success/info logs after 5 seconds
    if (type === 'success' || type === 'info') {
      setTimeout(() => {
        setOperationLogs(prev => prev.filter(log => log.id !== logEntry.id));
      }, 5000);
    }
  };

  // System health check
  const checkSystemHealth = async () => {
    try {
      addOperationLog('Checking system health...', 'info');
      const healthResponse = await fetch('/health/models');
      const healthData = await healthResponse.json();
      setSystemHealth(healthData);
      
      if (healthData.status === 'healthy') {
        addOperationLog('✅ System is healthy and ready', 'success');
      } else if (healthData.status === 'loading') {
        addOperationLog('⏳ System is still loading AI models, performance may be slower', 'warning');
      } else {
        addOperationLog('⚠️ System health check failed', 'error', healthData.error);
      }
    } catch (error) {
      addOperationLog('❌ Could not check system health', 'error', error.message);
    }
  };

  useEffect(() => {
    loadDatasets();
    checkSystemHealth();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      const getAcceptedTypes = () => {
        const datasetType = selectedDataset.type || 'text';
        switch(datasetType) {
          case 'text':
            return '.pdf,.txt,.docx,.pptx';
          case 'mixed':
          default:
            return '.pdf,.txt,.docx,.pptx,.jpg,.jpeg,.png,.gif,.webp,.bmp';
        }
      };
      
      setAcceptedFileTypes(getAcceptedTypes());
      
      const getFileTypeDescription = () => {
        const datasetType = selectedDataset.type || 'text';
        switch(datasetType) {
          case 'text':
            return 'Supported formats: PDF, TXT, DOCX, PPTX';
          case 'mixed':
          default:
            return 'Supported formats: PDF, TXT, DOCX, PPTX, JPG, JPEG, PNG, GIF, WEBP, BMP';
        }
      };
      
      setFileTypeDescription(getFileTypeDescription());
    }
  }, [selectedDataset]);

  useEffect(() => {
    return () => {
      if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
      }
    };
  }, [statusCheckInterval]);

  const loadDatasets = async () => {
    setLoading(true);
    setLoadingMessage("Initializing dataset loading...");
    setDetailedStatus("Checking authentication...");
    addOperationLog('Starting dataset loading process', 'info');
    
    try {
      // Check if we have a valid token
      const token = localStorage.getItem('token');
      if (!token) {
        throw new Error('No authentication token found. Please log in again.');
      }
      
      setDetailedStatus("Contacting server...");
      addOperationLog('Authentication verified, contacting server', 'info');
      
      const startTime = Date.now();
      const response = await botService.getDatasets();
      const loadTime = Date.now() - startTime;
      
      addOperationLog(`Server responded in ${loadTime}ms`, loadTime > 3000 ? 'warning' : 'success');
      
      // Safety check: ensure datasets is always an array
      let datasetsArray = [];
      
      // Handle the enhanced response format
      if (response && response.datasets && response.status) {
        datasetsArray = Array.isArray(response.datasets) ? response.datasets : [];
        
        // Show detailed status message
        const status = response.status;
        if (status.cache_hit) {
          setLoadingMessage("✅ Retrieved datasets from cache");
          setDetailedStatus("Using cached data for faster loading");
          addOperationLog('Datasets loaded from cache', 'success');
        } else {
          setLoadingMessage(status.message);
          setDetailedStatus("Processing fresh dataset information");
          addOperationLog(`Loaded ${datasetsArray.length} datasets from database`, 'success');
          
          // Show additional details if available
          if (status.details) {
            const details = status.details;
            const detailMessage = `Found ${details.total_datasets} datasets with ${details.total_documents} documents and ${details.total_images} images`;
            setTimeout(() => {
              setLoadingMessage(detailMessage);
              setDetailedStatus("Dataset summary computed");
              addOperationLog(detailMessage, 'info');
            }, 500);
          }
        }
        
        // Keep the status message visible for a moment
        setTimeout(() => {
          setLoading(false);
          setLoadingMessage("");
          setDetailedStatus("");
          addOperationLog(`Dataset loading completed - ${datasetsArray.length} datasets available`, 'success');
        }, 1500);
      } else {
        // Fallback for old response format (response is direct array)
        datasetsArray = Array.isArray(response) ? response : [];
        setLoadingMessage("✅ Datasets loaded successfully");
        setDetailedStatus("Using fallback response format");
        addOperationLog(`Loaded ${datasetsArray.length} datasets (legacy format)`, 'success');
        
        setTimeout(() => {
          setLoading(false);
          setLoadingMessage("");
          setDetailedStatus("");
        }, 1000);
      }
      
      setDatasets(datasetsArray);
    } catch (error) {
      console.error('Error fetching datasets:', error);
      setDatasets([]); // Ensure datasets is always an array
      setLoadingMessage("❌ Error loading datasets");
      setDetailedStatus(`Error: ${error.message}`);
      addOperationLog('Failed to load datasets', 'error', error.message);
      
      // Handle specific error types
      if (error.message.includes('authentication') || error.message.includes('401')) {
        addOperationLog('Authentication failed - please log in again', 'error');
        setError('Authentication failed. Please refresh the page and log in again.');
      } else if (error.message.includes('Network Error') || error.message.includes('ERR_NETWORK')) {
        addOperationLog('Network connection failed', 'error');
        setError('Cannot connect to server. Please check your internet connection.');
      } else {
        setError(`Failed to load datasets: ${error.message}`);
      }
      
      setTimeout(() => {
        setLoading(false);
        setLoadingMessage("");
        setDetailedStatus("");
      }, 3000);
    }
  };

  const handleCreateDataset = async (e) => {
    e.preventDefault();
    setLoading(true);
    setLoadingMessage('🔧 Preparing to create dataset...');
    setDetailedStatus('Validating dataset configuration...');
    addOperationLog(`Creating new ${newDataset.type} dataset: ${newDataset.name}`, 'info');
    
    try {
      // Show initial progress
      setLoadingMessage('📝 Validating dataset parameters...');
      setDetailedStatus(`Type: ${newDataset.type}, Name: ${newDataset.name}`);
      addOperationLog('Dataset parameters validated', 'info');
      await new Promise(resolve => setTimeout(resolve, 300));
      
      setLoadingMessage('🗄️ Creating vector database collection...');
      setDetailedStatus('Initializing ChromaDB collection for embeddings...');
      addOperationLog('Initializing vector database collection', 'info');
      
      const startTime = Date.now();
      const response = await botService.createDataset(newDataset);
      const createTime = Date.now() - startTime;
      
      addOperationLog(`Dataset created in ${createTime}ms`, 'success');
      
      // Handle enhanced response format with detailed status
      if (response.dataset && response.status) {
        const status = response.status;
        
        // Show the detailed creation message
        setLoadingMessage(`✅ ${status.message}`);
        setDetailedStatus(`Dataset ID: ${response.dataset.id}`);
        addOperationLog(status.message, 'success', response.dataset);
        
        // Show additional details if available
        if (status.details && status.details.ready_for_uploads) {
          setTimeout(() => {
            setLoadingMessage("🚀 Dataset ready for file uploads!");
            setDetailedStatus('Vector database initialized and ready');
            addOperationLog('Dataset is ready for file uploads', 'success');
          }, 800);
        }
        
        // Keep status visible then refresh
        setTimeout(async () => {
          setLoadingMessage('🔄 Refreshing dataset list...');
          setDetailedStatus('Updating interface...');
          addOperationLog('Refreshing dataset list', 'info');
          
          const data = await botService.getDatasets();
          setDatasets(data.datasets || data);
          
          setCreateMode(false);
          setNewDataset({
            name: '',
            description: '',
            type: 'mixed'
          });
          setSuccess(`Dataset '${newDataset.name}' created successfully!`);
          setTimeout(() => setSuccess(null), 3000);
          setLoading(false);
          setLoadingMessage('');
          setDetailedStatus('');
          addOperationLog(`Dataset creation process completed successfully`, 'success');
        }, 1500);
      } else {
        // Fallback for old response format
        setLoadingMessage('🔄 Refreshing dataset list...');
        setDetailedStatus('Using legacy response format');
        const data = await botService.getDatasets();
        setDatasets(data.datasets || data);
        setCreateMode(false);
        setNewDataset({
          name: '',
          description: '',
          type: 'mixed'
        });
        setSuccess('Dataset created successfully');
        setTimeout(() => setSuccess(null), 3000);
        setLoading(false);
        setLoadingMessage('');
        setDetailedStatus('');
        addOperationLog('Dataset created (legacy format)', 'success');
      }
    } catch (err) {
      console.error('Error creating dataset:', err);
      addOperationLog('Dataset creation failed', 'error', err.message);
      setError(err.response?.data?.error || 'Failed to create dataset');
      if (err.response?.data?.details) {
        setLoadingMessage(`❌ Error: ${err.response.data.details}`);
        setDetailedStatus('Creation failed - see details above');
        setTimeout(() => {
          setLoading(false);
          setLoadingMessage('');
          setDetailedStatus('');
        }, 3000);
      } else {
        setLoading(false);
        setLoadingMessage('');
        setDetailedStatus('');
      }
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    if (selectedFile) {
      addOperationLog(`File selected: ${selectedFile.name} (${(selectedFile.size / 1024).toFixed(2)} KB)`, 'info');
    }
  };

  const handleBulkZipChange = (e) => {
    const selectedFile = e.target.files[0];
    setBulkZipFile(selectedFile);
    if (selectedFile) {
      addOperationLog(`Zip file selected: ${selectedFile.name} (${(selectedFile.size / 1024 / 1024).toFixed(2)} MB)`, 'info');
    }
  };

  const handleUploadDocument = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file to upload');
      addOperationLog('Upload failed: No file selected', 'error');
      return;
    }

    setLoading(true);
    const fileExtension = file.name.split('.').pop().toLowerCase();
    const isImage = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'].includes(fileExtension);
    
    setLoadingMessage(isImage ? `📸 Processing image: ${file.name}...` : `📄 Processing document: ${file.name}...`);
    setDetailedStatus(`File type: ${fileExtension.toUpperCase()}, Size: ${(file.size / 1024).toFixed(2)} KB`);
    addOperationLog(`Starting upload: ${file.name} (${isImage ? 'image' : 'document'})`, 'info');
    
    try {
      if (isImage) {
        setLoadingMessage('🧠 Generating image embeddings with CLIP...');
        setDetailedStatus('Computing visual features and generating caption...');
        addOperationLog('Processing image with AI models (CLIP + BLIP)', 'info');
        await botService.uploadImage(selectedDataset.id, file);
        addOperationLog('Image processed and added to vector database', 'success');
      } else {
        setLoadingMessage('📝 Extracting text and creating semantic chunks...');
        setDetailedStatus('Parsing document content and splitting into chunks...');
        addOperationLog('Extracting text and creating embeddings', 'info');
        await botService.uploadDocument(selectedDataset.id, file);
        addOperationLog('Document processed and chunked into vector database', 'success');
      }
      
      setLoadingMessage('🔄 Updating dataset information...');
      setDetailedStatus('Refreshing dataset metadata...');
      addOperationLog('Refreshing dataset list', 'info');
      
      const data = await botService.getDatasets();
      
      // Handle both response formats safely
      let datasetsArray = [];
      if (data && data.datasets && Array.isArray(data.datasets)) {
        datasetsArray = data.datasets;
      } else if (Array.isArray(data)) {
        datasetsArray = data;
      } else {
        console.warn('Unexpected datasets response format:', data);
        datasetsArray = [];
        addOperationLog('Warning: Unexpected response format from server', 'warning');
      }
      
      setDatasets(datasetsArray);
      setUploadMode(false);
      setSelectedDataset(null);
      setFile(null);
      setSuccess(isImage ? 'Image uploaded successfully' : 'Document uploaded successfully');
      setTimeout(() => setSuccess(null), 3000);
      setLoading(false);
      setLoadingMessage('');
      setDetailedStatus('');
      addOperationLog(`Upload completed successfully: ${file.name}`, 'success');
    } catch (err) {
      addOperationLog(`Upload failed: ${file.name}`, 'error', err.message);
      setError(`Failed to upload ${file.name}`);
      setLoading(false);
      setLoadingMessage('');
      setDetailedStatus('');
    }
  };

  const handleBulkUpload = async (e) => {
    e.preventDefault();
    if (!bulkZipFile) {
      setError('Please select a zip file to upload');
      addOperationLog('Bulk upload failed: No zip file selected', 'error');
      return;
    }
    setLoading(true);
    setLoadingMessage('📦 Uploading zip file...');
    setDetailedStatus(`Processing ${bulkZipFile.name} (${(bulkZipFile.size / 1024 / 1024).toFixed(2)} MB)`);
    setBulkResult(null);
    setUploadStatus(null);
    addOperationLog(`Starting bulk upload: ${bulkZipFile.name}`, 'info');
    
    try {
      const formData = new FormData();
      formData.append('file', bulkZipFile);
      
      setLoadingMessage('🚀 Starting bulk upload process...');
      setDetailedStatus('Transferring file to server...');
      addOperationLog('Transferring zip file to server', 'info');
      
      const response = await fetch(`/api/datasets/${selectedDataset.id}/bulk-upload`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: formData
      });
      const result = await response.json();
      if (!response.ok) {
        addOperationLog('Bulk upload failed', 'error', result.error);
        setError(result.error || 'Bulk upload failed');
        setLoading(false);
        setLoadingMessage('');
        setDetailedStatus('');
        return;
      }
      
      setLoadingMessage('⚙️ Processing files in background...');
      setDetailedStatus('Server is extracting and processing files...');
      addOperationLog('Bulk processing started on server', 'info');
      
      const statusId = result.status_file.split('/').pop();
      let statusCheckCount = 0;
      
      const checkStatus = async () => {
        try {
          statusCheckCount++;
          const status = await botService.checkUploadStatus(selectedDataset.id, statusId);
          setUploadStatus(status);
          
          if (status.current_file) {
            setDetailedStatus(`Processing: ${status.current_file}`);
            addOperationLog(`Processing file ${status.processed_files}/${status.total_files}: ${status.current_file}`, 'info');
          } else if (status.message) {
            setDetailedStatus(status.message);
          }
          
          if (status.status === 'completed' || status.status === 'error') {
            clearInterval(statusCheckInterval);
            setStatusCheckInterval(null);
            setLoading(false);
            setLoadingMessage('');
            setDetailedStatus('');
            
            if (status.status === 'completed') {
              addOperationLog(`Bulk upload completed: ${status.successes?.length || 0} files processed`, 'success');
              setSuccess('Bulk upload completed');
              setTimeout(() => setSuccess(null), 3000);
              loadDatasets();
              setBulkZipFile(null);
            } else {
              addOperationLog('Bulk upload failed', 'error', status.message);
              setError(status.message || 'Bulk upload failed');
            }
          } else if (statusCheckCount > 300) { // 10 minutes timeout
            clearInterval(statusCheckInterval);
            setStatusCheckInterval(null);
            setLoading(false);
            setLoadingMessage('');
            setDetailedStatus('');
            addOperationLog('Bulk upload timed out', 'error');
            setError('Upload timed out. Please try again with smaller files.');
          }
        } catch (err) {
          console.error('Error checking status:', err);
          addOperationLog('Error checking upload status', 'error', err.message);
        }
      };
      
      checkStatus();
      const interval = setInterval(checkStatus, 2000);
      setStatusCheckInterval(interval);
      
    } catch (err) {
      addOperationLog('Bulk upload failed', 'error', err.message);
      setError('Bulk upload failed');
      setLoading(false);
      setLoadingMessage('');
      setDetailedStatus('');
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewDataset({
      ...newDataset,
      [name]: value
    });
  };

  const startUpload = (dataset) => {
    setSelectedDataset(dataset);
    setUploadMode(true);
    setCreateMode(false);
    addOperationLog(`Starting upload to dataset: ${dataset.name}`, 'info');
  };

  const startViewDocuments = async (dataset) => {
    setSelectedDataset(dataset);
    setViewDocumentsMode(true);
    setViewImagesMode(false);
    setCreateMode(false);
    setUploadMode(false);
    setDocumentListLoading(true);
    
    try {
      const data = await botService.getDatasetDocuments(dataset.id);
      setDocuments(data.documents || []);
      setDocumentListLoading(false);
    } catch (err) {
      setError('Failed to load documents');
      setDocumentListLoading(false);
    }
  };

  const startViewImages = async (dataset) => {
    setSelectedDataset(dataset);
    setViewImagesMode(true);
    setViewDocumentsMode(false);
    setCreateMode(false);
    setUploadMode(false);
    setImageListLoading(true);
    
    try {
      const data = await botService.getDatasetImages(dataset.id);
      setImages(data.images || []);
      setImageListLoading(false);
    } catch (err) {
      setError('Failed to load images');
      setImageListLoading(false);
    }
  };

  const confirmRemoveDocument = (document) => {
    setDocumentToDelete(document);
    setShowConfirmDelete(true);
  };

  const confirmRemoveImage = (image) => {
    setImageToDelete(image);
    setShowConfirmImageDelete(true);
  };

  const confirmRemoveDataset = (dataset) => {
    setDatasetToDelete(dataset);
    setShowConfirmDatasetDelete(true);
  };

  const handleRemoveDocument = async () => {
    if (!documentToDelete) return;
    
    setLoading(true);
    setLoadingMessage(`Removing document: ${documentToDelete.filename}...`);
    
    try {
      await botService.removeDocument(selectedDataset.id, documentToDelete.id);
      
      setLoadingMessage('Updating document list...');
      const data = await botService.getDatasetDocuments(selectedDataset.id);
      setDocuments(data.documents || []);
      
      setLoadingMessage('Refreshing datasets...');
      const datasets = await botService.getDatasets();
      
      // Handle both response formats safely
      let datasetsArray = [];
      if (datasets && datasets.datasets && Array.isArray(datasets.datasets)) {
        datasetsArray = datasets.datasets;
      } else if (Array.isArray(datasets)) {
        datasetsArray = datasets;
      } else {
        console.warn('Unexpected datasets response format:', datasets);
        datasetsArray = [];
      }
      
      setDatasets(datasetsArray);
      
      setShowConfirmDelete(false);
      setDocumentToDelete(null);
      setSuccess('Document removed successfully');
      setTimeout(() => setSuccess(null), 3000);
      setLoading(false);
      setLoadingMessage('');
    } catch (err) {
      setError('Failed to remove document');
      setLoading(false);
      setLoadingMessage('');
    }
  };

  const handleRemoveImage = async () => {
    if (!imageToDelete) return;
    
    setLoading(true);
    setLoadingMessage(`Removing image...`);
    
    try {
      await botService.removeImage(selectedDataset.id, imageToDelete.id);
      
      setLoadingMessage('Updating image list...');
      const data = await botService.getDatasetImages(selectedDataset.id);
      setImages(data.images || []);
      
      setLoadingMessage('Refreshing datasets...');
      const datasets = await botService.getDatasets();
      
      // Handle both response formats safely
      let datasetsArray = [];
      if (datasets && datasets.datasets && Array.isArray(datasets.datasets)) {
        datasetsArray = datasets.datasets;
      } else if (Array.isArray(datasets)) {
        datasetsArray = datasets;
      } else {
        console.warn('Unexpected datasets response format:', datasets);
        datasetsArray = [];
      }
      
      setDatasets(datasetsArray);
      
      setShowConfirmImageDelete(false);
      setImageToDelete(null);
      setSuccess('Image removed successfully');
      setTimeout(() => setSuccess(null), 3000);
      setLoading(false);
      setLoadingMessage('');
    } catch (err) {
      setError('Failed to remove image');
      setLoading(false);
      setLoadingMessage('');
    }
  };

  const handleRemoveDataset = async () => {
    if (!datasetToDelete) return;
    
    setLoading(true);
    setLoadingMessage(`Removing dataset: ${datasetToDelete.name}...`);
    
    try {
      setLoadingMessage('Deleting all files and embeddings...');
      await botService.removeDataset(datasetToDelete.id);
      
      setLoadingMessage('Updating dataset list...');
      const data = await botService.getDatasets();
      
      // Handle both response formats safely
      let datasetsArray = [];
      if (data && data.datasets && Array.isArray(data.datasets)) {
        datasetsArray = data.datasets;
      } else if (Array.isArray(data)) {
        datasetsArray = data;
      } else {
        console.warn('Unexpected datasets response format:', data);
        datasetsArray = [];
      }
      
      setDatasets(datasetsArray);
      
      setShowConfirmDatasetDelete(false);
      setDatasetToDelete(null);
      setSuccess('Dataset removed successfully');
      setTimeout(() => setSuccess(null), 3000);
      setLoading(false);
      setLoadingMessage('');
    } catch (err) {
      setError('Failed to remove dataset');
      setLoading(false);
      setLoadingMessage('');
    }
  };

  const getFileIcon = (filename) => {
    const extension = filename.split('.').pop().toLowerCase();
    const color = '#6c757d';
    
    if (['pdf'].includes(extension)) {
      return <FaFileAlt className="me-2" style={{ color: '#dc3545' }} />;
    } else if (['docx', 'doc'].includes(extension)) {
      return <FaFileAlt className="me-2" style={{ color: '#0d6efd' }} />;
    } else if (['txt'].includes(extension)) {
      return <FaFile className="me-2" style={{ color: color }} />;
    } else if (['pptx', 'ppt'].includes(extension)) {
      return <FaFileAlt className="me-2" style={{ color: '#fd7e14' }} />;
    } else if (['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'].includes(extension)) {
      return <FaImage className="me-2" style={{ color: '#198754' }} />;
    }
    
    return <FaFile className="me-2" style={{ color: color }} />;
  };

  return (
    <Container className="mt-4">
      {/* System Health and Operation Logs Panel */}
      {(operationLogs.length > 0 || systemHealth) && (
        <Card className="mb-4 border-0 shadow-sm">
          <Card.Body>
            <div className="d-flex justify-content-between align-items-center mb-3">
              <h6 className="mb-0">
                <FaInfo className="me-2" />
                System Status & Activity Log
              </h6>
              <Button 
                variant="outline-secondary" 
                size="sm" 
                onClick={() => setShowSystemHealth(!showSystemHealth)}
                className="rounded-pill"
              >
                {showSystemHealth ? 'Hide Details' : 'Show Details'}
              </Button>
            </div>
            
            {systemHealth && (
              <div className="mb-3">
                <Badge 
                  bg={systemHealth.status === 'healthy' ? 'success' : systemHealth.status === 'loading' ? 'warning' : 'danger'} 
                  className="me-2"
                >
                  AI Models: {systemHealth.status === 'healthy' ? '✅ Ready' : systemHealth.status === 'loading' ? '⏳ Loading' : '❌ Error'}
                </Badge>
                {systemHealth.models_loaded && (
                  <Badge bg="info" className="me-2">
                    🧠 CLIP + BLIP Models Loaded
                  </Badge>
                )}
              </div>
            )}
            
            {showSystemHealth && operationLogs.length > 0 && (
              <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
                {operationLogs.slice(0, 10).map(log => (
                  <div key={log.id} className={`d-flex align-items-start mb-2 p-2 rounded ${
                    log.type === 'error' ? 'bg-danger bg-opacity-10' :
                    log.type === 'warning' ? 'bg-warning bg-opacity-10' :
                    log.type === 'success' ? 'bg-success bg-opacity-10' :
                    'bg-light'
                  }`}>
                    <div className="me-2">
                      {log.type === 'error' ? <FaExclamationCircle className="text-danger" /> :
                       log.type === 'warning' ? <FaExclamationTriangle className="text-warning" /> :
                       log.type === 'success' ? <FaCheckCircle className="text-success" /> :
                       <FaInfo className="text-info" />}
                    </div>
                    <div>
                      <div className="small text-muted">{log.timestamp}</div>
                      <div className="small">{log.message}</div>
                      {log.details && (
                        <div className="small text-muted">{JSON.stringify(log.details)}</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card.Body>
        </Card>
      )}

      <div className="d-flex justify-content-between align-items-center mb-4">
        <h1 className="mb-0">
          <FaDatabase className="me-2" />
          Dataset Management
        </h1>
        {!createMode && !uploadMode && !viewDocumentsMode && !viewImagesMode && (
          <Button 
            variant="primary" 
            onClick={() => setCreateMode(true)}
            className="rounded-pill"
          >
            <FaPlus className="me-1" /> Create Dataset
          </Button>
        )}
      </div>

      {error && <Alert variant="danger">{error}</Alert>}
      {success && <Alert variant="success">{success}</Alert>}

      {createMode && (
        <Card className="mb-4 border-0 shadow-sm">
          <Card.Body>
            <Card.Title className="mb-3">
              <FaPlus className="me-2" />
              Create New Dataset
            </Card.Title>
            <Form onSubmit={handleCreateDataset}>
              <Row>
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>Dataset Name</Form.Label>
                    <Form.Control
                      type="text"
                      name="name"
                      value={newDataset.name}
                      onChange={handleInputChange}
                      placeholder="Enter dataset name"
                      required
                      className="rounded-pill"
                    />
                  </Form.Group>
                </Col>
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>Description (Optional)</Form.Label>
                    <Form.Control
                      type="text"
                      name="description"
                      value={newDataset.description}
                      onChange={handleInputChange}
                      placeholder="Brief description"
                      className="rounded-pill"
                    />
                  </Form.Group>
                </Col>
              </Row>

              <Row>
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>Dataset Type</Form.Label>
                    <Form.Select
                      name="type"
                      value={newDataset.type}
                      onChange={handleInputChange}
                      className="rounded-pill"
                    >
                      <option value="mixed">Mixed (Documents & Images)</option>
                      <option value="text">Text Only (Documents)</option>
                    </Form.Select>
                    <Form.Text className="text-muted">
                      {newDataset.type === 'text' ? 'Only document files will be accepted (PDF, DOCX, etc.)' :
                       'Both document and image files will be accepted'}
                    </Form.Text>
                  </Form.Group>
                </Col>
              </Row>

              <div className="d-flex justify-content-end">
                <Button variant="success" type="submit" disabled={loading} className="px-4 rounded-pill">
                  {loading ? (
                    <>
                      <TesseractLoader size={20} /> 
                      <span className="ms-2">{loadingMessage || 'Creating...'}</span>
                    </>
                  ) : 'Create Dataset'}
                </Button>
              </div>
            </Form>
          </Card.Body>
        </Card>
      )}

      {uploadMode && selectedDataset && (
        <Card className="mb-4 border-0 shadow-sm">
          <Card.Body>
            <Card.Title className="mb-3">
              <FaFileUpload className="me-2" />
              Upload to {selectedDataset.name}
              {selectedDataset.type && (
                <Badge bg="light" text="dark" className="ms-2">
                  {selectedDataset.type === 'text' ? 'Text Only' : 'Mixed Content'}
                </Badge>
              )}
            </Card.Title>
            <div className="mb-3 d-flex gap-2">
              <Button
                variant={!bulkUploadMode ? 'primary' : 'outline-primary'}
                onClick={() => setBulkUploadMode(false)}
                className="rounded-pill"
              >
                Single File Upload
              </Button>
              <Button
                variant={bulkUploadMode ? 'primary' : 'outline-primary'}
                onClick={() => setBulkUploadMode(true)}
                className="rounded-pill"
              >
                Bulk Upload Zip
              </Button>
            </div>
            {!bulkUploadMode ? (
              <Form onSubmit={handleUploadDocument}>
                <Form.Group className="mb-4">
                  <Form.Label>Select File</Form.Label>
                  <div className="custom-file-upload">
                    <Form.Control
                      type="file"
                      onChange={handleFileChange}
                      required
                      accept={acceptedFileTypes}
                      className="rounded-pill"
                    />
                    {file && (
                      <div className="selected-file mt-2">
                        <Card className="p-2 border-0 bg-light">
                          <div className="d-flex align-items-center">
                            {getFileIcon(file.name)}
                            <div className="ms-2">
                              <strong>{file.name}</strong>
                              <div className="text-muted small">{(file.size / 1024).toFixed(2)} KB</div>
                            </div>
                          </div>
                        </Card>
                      </div>
                    )}
                  </div>
                  <Form.Text className="text-muted">
                    {fileTypeDescription}
                  </Form.Text>
                </Form.Group>
                <div className="d-flex">
                  <Button variant="primary" type="submit" disabled={loading} className="me-2 rounded-pill">
                    {loading ? (
                      <>
                        <TesseractLoader size={20} /> 
                        <span className="ms-2">{loadingMessage || 'Uploading...'}</span>
                      </>
                    ) : 'Upload'}
                  </Button>
                  <Button 
                    variant="outline-secondary" 
                    onClick={() => {
                      setUploadMode(false);
                      setSelectedDataset(null);
                    }} 
                    disabled={loading}
                    className="rounded-pill"
                  >
                    Cancel
                  </Button>
                </div>
              </Form>
            ) : (
              <Form onSubmit={handleBulkUpload}>
                <Form.Group className="mb-4">
                  <Form.Label>Select Zip File</Form.Label>
                  <Form.Control
                    type="file"
                    accept=".zip"
                    onChange={handleBulkZipChange}
                    required
                    className="rounded-pill"
                    disabled={loading}
                  />
                  <Form.Text className="text-muted">
                    Upload a .zip file containing documents and images. Supported: PDF, DOCX, TXT, PPTX, JPG, PNG, etc.
                  </Form.Text>
                </Form.Group>
                
                {uploadStatus && (
                  <div className="mb-4">
                    <ProgressBar 
                      now={uploadStatus.total_files ? (uploadStatus.processed_files / uploadStatus.total_files) * 100 : 0} 
                      label={`${Math.round((uploadStatus.processed_files / uploadStatus.total_files) * 100)}%`}
                      className="mb-2"
                    />
                    <div className="text-muted">
                      <small>
                        Status: {uploadStatus.status}<br />
                        {uploadStatus.current_file && `Processing: ${uploadStatus.current_file}`}<br />
                        {uploadStatus.message}
                      </small>
                    </div>
                  </div>
                )}
                
                <div className="d-flex">
                  <Button variant="primary" type="submit" disabled={loading} className="me-2 rounded-pill">
                    {loading ? (
                      <>
                        <TesseractLoader size={20} /> 
                        <span className="ms-2">{loadingMessage || 'Uploading...'}</span>
                      </>
                    ) : 'Upload'}
                  </Button>
                  <Button 
                    variant="outline-secondary" 
                    onClick={() => {
                      setUploadMode(false);
                      setSelectedDataset(null);
                      if (statusCheckInterval) {
                        clearInterval(statusCheckInterval);
                        setStatusCheckInterval(null);
                      }
                    }} 
                    disabled={loading}
                    className="rounded-pill"
                  >
                    Cancel
                  </Button>
                </div>
              </Form>
            )}
            {bulkResult && (
              <div className="mt-4">
                <h5>Bulk Upload Summary</h5>
                <div className="mb-2 text-success">{bulkResult.message}</div>
                {bulkResult.successes && bulkResult.successes.length > 0 && (
                  <div className="mb-2">
                    <strong>Added:</strong>
                    <ul>
                      {bulkResult.successes.map((s, i) => (
                        <li key={i}>{s.file} ({s.type}{s.chunks ? `, ${s.chunks} chunks` : ''})</li>
                      ))}
                    </ul>
                  </div>
                )}
                {bulkResult.errors && bulkResult.errors.length > 0 && (
                  <div className="mb-2 text-danger">
                    <strong>Errors:</strong>
                    <ul>
                      {bulkResult.errors.map((e, i) => (
                        <li key={i}>{e.file}: {e.error}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </Card.Body>
        </Card>
      )}

      {viewDocumentsMode && selectedDataset && (
        <Card className="mb-4 border-0 shadow-sm">
          <Card.Body>
            <Card.Title className="mb-3">
              <FaList className="me-2" />
              Documents in {selectedDataset.name}
            </Card.Title>
            
            {documentListLoading ? (
              <div className="text-center py-4">
                <TesseractLoader />
                <p className="mt-2 text-muted">Loading documents...</p>
              </div>
            ) : documents.length === 0 ? (
              <Alert variant="info">No documents in this dataset yet.</Alert>
            ) : (
              <ListGroup variant="flush">
                {documents.map(doc => (
                  <ListGroup.Item key={doc.id} className="border-bottom py-3">
                    <div className="d-flex justify-content-between align-items-center">
                      <div className="d-flex align-items-center">
                        {getFileIcon(doc.filename)}
                        <div className="ms-3">
                          <h5 className="mb-1">{doc.filename}</h5>
                          <div className="d-flex gap-2">
                            <Badge bg="light" text="dark" className="border">
                              {doc.chunk_count} {doc.chunk_count === 1 ? 'chunk' : 'chunks'}
                            </Badge>
                            {doc.file_type && (
                              <Badge bg="secondary" className="text-white">
                                {doc.file_type}
                              </Badge>
                            )}
                          </div>
                        </div>
                      </div>
                      <div>
                        <Button 
                          variant="danger" 
                          size="sm"
                          onClick={() => confirmRemoveDocument(doc)}
                          className="rounded-pill"
                        >
                          <FaTrash className="me-1" /> Remove
                        </Button>
                      </div>
                    </div>
                  </ListGroup.Item>
                ))}
              </ListGroup>
            )}
            
            <div className="d-flex mt-3">
              <Button 
                variant="outline-secondary" 
                onClick={() => {
                  setViewDocumentsMode(false);
                  setSelectedDataset(null);
                }} 
                className="rounded-pill"
              >
                Back to Datasets
              </Button>
            </div>
          </Card.Body>
        </Card>
      )}

      {viewImagesMode && selectedDataset && (
        <Card className="mb-4 border-0 shadow-sm">
          <Card.Body>
            <Card.Title className="mb-3">
              <FaImage className="me-2" />
              Images in {selectedDataset.name}
            </Card.Title>
            
            {imageListLoading ? (
              <div className="text-center py-4">
                <TesseractLoader />
                <p className="mt-2 text-muted">Loading images...</p>
              </div>
            ) : images.length === 0 ? (
              <Alert variant="info">No images in this dataset yet.</Alert>
            ) : (
              <div>
                <Row className="mt-3">
                  {images.map(img => (
                    <Col key={img.id} xs={12} sm={6} md={4} lg={3} className="mb-4">
                      <Card className="h-100 border-0 shadow-sm image-card">
                        <div className="image-container" style={{ height: '160px', overflow: 'hidden' }}>
                          <img 
                            src={img.url} 
                            alt={img.caption || 'Dataset image'} 
                            className="card-img-top" 
                            style={{ objectFit: 'cover', width: '100%', height: '100%' }}
                          />
                        </div>
                        <Card.Body>
                          <Card.Title className="text-truncate fs-6">
                            {img.caption || 'Untitled Image'}
                          </Card.Title>
                          <div className="mt-2 d-flex justify-content-between">
                            <Button 
                              variant="outline-secondary" 
                              size="sm"
                              onClick={() => window.open(img.url, '_blank')}
                              className="rounded-pill"
                            >
                              <FaImage className="me-1" /> View
                            </Button>
                            <Button 
                              variant="outline-danger" 
                              size="sm"
                              onClick={() => confirmRemoveImage(img)}
                              className="rounded-pill"
                            >
                              <FaTrash className="me-1" /> Remove
                            </Button>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                  ))}
                </Row>
                <div className="mt-3 d-flex justify-content-between">
                  <Button 
                    variant="outline-secondary" 
                    onClick={() => {
                      setViewImagesMode(false);
                      setSelectedDataset(null);
                    }} 
                    className="rounded-pill"
                  >
                    Back to Datasets
                  </Button>
                  <Button 
                    variant="outline-primary" 
                    onClick={() => startUpload(selectedDataset)}
                    className="rounded-pill"
                  >
                    <FaFileUpload className="me-1" /> Upload More
                  </Button>
                </div>
              </div>
            )}
          </Card.Body>
        </Card>
      )}

      {loading && !createMode && !uploadMode && !viewDocumentsMode && !viewImagesMode ? (
        <div className="text-center my-5">
          <TesseractLoader size={80} />
          <p className="mt-3 text-muted font-weight-bold">{loadingMessage || 'Loading your datasets...'}</p>
          {detailedStatus && (
            <p className="mt-2 text-muted small">{detailedStatus}</p>
          )}
        </div>
      ) : (
        <Row className="mt-4">
          {!createMode && !uploadMode && !viewDocumentsMode && !viewImagesMode && datasets.length === 0 ? (
            <Col>
              <Card className="border-0 shadow-sm text-center p-5">
                <div className="py-5">
                  <FaDatabase size={48} className="text-muted mb-3" />
                  <h4>No Datasets Yet</h4>
                  <p className="text-muted">Create your first dataset to get started.</p>
                  <Button 
                    variant="primary" 
                    onClick={() => setCreateMode(true)}
                    className="mt-2 rounded-pill"
                  >
                    <FaPlus className="me-1" /> Create Dataset
                  </Button>
                </div>
              </Card>
            </Col>
          ) : !createMode && !uploadMode && !viewDocumentsMode && !viewImagesMode && (
            <Col>
              <Card className="border-0 shadow-sm">
                <ListGroup variant="flush">
                  {datasets.map(dataset => (
                    <ListGroup.Item 
                      key={dataset.id}
                      className="border-bottom py-3"
                    >
                      <div className="d-flex justify-content-between align-items-center">
                        <div>
                          <div className="d-flex align-items-center">
                            <div className="dataset-icon me-3">
                              <FaDatabase size={24} className="text-primary" />
                            </div>
                            <div>
                              <h5 className="mb-1">{dataset.name}</h5>
                              <p className="mb-1 text-muted">{dataset.description}</p>
                              <div className="d-flex flex-wrap gap-2 mb-2">
                                <Badge bg="light" text="dark" className="border">
                                  <FaFileAlt className="me-1" /> {dataset.document_count || 0} {dataset.document_count === 1 ? 'Document' : 'Documents'}
                                </Badge>
                                <Badge bg="light" text="dark" className="border">
                                  <FaChartBar className="me-1" /> {dataset.chunk_count || 0} {dataset.chunk_count === 1 ? 'Chunk' : 'Chunks'}
                                </Badge>
                                {dataset.type !== 'text' && (
                                  <Badge bg="light" text="dark" className="border">
                                    <FaImage className="me-1" /> {dataset.image_count || 0} {dataset.image_count === 1 ? 'Image' : 'Images'}
                                  </Badge>
                                )}
                              </div>
                              
                              {dataset.image_previews && dataset.image_previews.length > 0 && (
                                <div className="d-flex mt-2 flex-wrap">
                                  {dataset.image_previews.map((img, index) => (
                                    <div key={img.id || index} className="me-2 mb-2" style={{ maxWidth: '80px' }}>
                                      <img 
                                        src={img.url} 
                                        alt={img.caption || 'Dataset image'} 
                                        className="img-thumbnail" 
                                        style={{ width: '100%', height: '60px', objectFit: 'cover' }}
                                        title={img.caption || 'Dataset image'}
                                      />
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                        <div>
                          <Button 
                            variant="outline-primary" 
                            size="sm"
                            onClick={() => startUpload(dataset)}
                            className="rounded-pill me-2"
                          >
                            <FaFileUpload className="me-1" /> Upload
                          </Button>
                          {dataset.document_count > 0 && (
                            <Button 
                              variant="outline-secondary" 
                              size="sm"
                              onClick={() => startViewDocuments(dataset)}
                              className="rounded-pill me-2"
                            >
                              <FaFileAlt className="me-1" /> Documents
                            </Button>
                          )}
                          {dataset.image_count > 0 && dataset.type !== 'text' && (
                            <Button 
                              variant="outline-secondary" 
                              size="sm"
                              onClick={() => startViewImages(dataset)}
                              className="rounded-pill me-2"
                            >
                              <FaImage className="me-1" /> Images
                            </Button>
                          )}
                          <Button 
                            variant="outline-danger" 
                            size="sm"
                            onClick={() => confirmRemoveDataset(dataset)}
                            className="rounded-pill"
                          >
                            <FaTrash className="me-1" /> Delete
                          </Button>
                        </div>
                      </div>
                    </ListGroup.Item>
                  ))}
                </ListGroup>
              </Card>
            </Col>
          )}
        </Row>
      )}

      <Modal 
        show={showConfirmDelete} 
        onHide={() => setShowConfirmDelete(false)}
        centered
      >
        <Modal.Header closeButton>
          <Modal.Title>Confirm Document Removal</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p>Are you sure you want to remove this document?</p>
          {documentToDelete && (
            <div className="my-3 p-3 bg-light rounded">
              <div className="d-flex align-items-center">
                {getFileIcon(documentToDelete.filename)}
                <div className="ms-2">
                  <strong>{documentToDelete.filename}</strong>
                </div>
              </div>
            </div>
          )}
          <p className="mb-0 text-danger">This action cannot be undone.</p>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowConfirmDelete(false)}>
            Cancel
          </Button>
          <Button variant="danger" onClick={handleRemoveDocument} disabled={loading}>
            {loading ? (
              <>
                <TesseractLoader size={20} /> 
                <span className="ms-2">{loadingMessage || 'Removing...'}</span>
              </>
            ) : 'Remove Document'}
          </Button>
        </Modal.Footer>
      </Modal>

      <Modal 
        show={showConfirmImageDelete} 
        onHide={() => setShowConfirmImageDelete(false)}
        centered
      >
        <Modal.Header closeButton>
          <Modal.Title>Confirm Image Removal</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p>Are you sure you want to remove this image?</p>
          {imageToDelete && (
            <div className="my-3 p-3 bg-light rounded">
              <div className="d-flex align-items-center">
                <img 
                  src={imageToDelete.url} 
                  alt={imageToDelete.caption || 'Dataset image'} 
                  className="img-thumbnail" 
                  style={{ width: '80px', height: '60px', objectFit: 'cover' }}
                />
                <div className="ms-2">
                  <strong>{imageToDelete.caption || 'Untitled Image'}</strong>
                </div>
              </div>
            </div>
          )}
          <p className="mb-0 text-danger">This action cannot be undone.</p>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowConfirmImageDelete(false)}>
            Cancel
          </Button>
          <Button variant="danger" onClick={handleRemoveImage} disabled={loading}>
            {loading ? (
              <>
                <TesseractLoader size={20} /> 
                <span className="ms-2">{loadingMessage || 'Removing...'}</span>
              </>
            ) : 'Remove Image'}
          </Button>
        </Modal.Footer>
      </Modal>

      <Modal 
        show={showConfirmDatasetDelete} 
        onHide={() => setShowConfirmDatasetDelete(false)}
        centered
      >
        <Modal.Header closeButton>
          <Modal.Title>Confirm Dataset Removal</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p>Are you sure you want to remove this dataset?</p>
          {datasetToDelete && (
            <div className="my-3 p-3 bg-light rounded">
              <div className="d-flex align-items-center">
                <div className="dataset-icon me-3">
                  <FaDatabase size={24} className="text-primary" />
                </div>
                <div>
                  <strong>{datasetToDelete.name}</strong>
                </div>
              </div>
            </div>
          )}
          <p className="mb-0 text-danger">This action cannot be undone.</p>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowConfirmDatasetDelete(false)}>
            Cancel
          </Button>
          <Button variant="danger" onClick={handleRemoveDataset} disabled={loading}>
            {loading ? (
              <>
                <TesseractLoader size={20} /> 
                <span className="ms-2">{loadingMessage || 'Removing...'}</span>
              </>
            ) : 'Remove Dataset'}
          </Button>
        </Modal.Footer>
      </Modal>
    </Container>
  );
};

export default DatasetPanel; 