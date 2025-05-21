import React, { useState, useEffect } from 'react';
import { Container, Card, Button, Form, Row, Col, ListGroup, Badge, Alert, Spinner, Modal } from 'react-bootstrap';
import botService from '../services/botService';
import { FaDatabase, FaPlus, FaFileUpload, FaFile, FaTimes, FaFileAlt, FaTrash, FaList, FaExclamationTriangle, FaImage, FaChartBar } from 'react-icons/fa';

const DatasetPanel = () => {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
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

  useEffect(() => {
    loadDatasets();
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

  const loadDatasets = async () => {
    setLoading(true);
    try {
      const data = await botService.getDatasets();
      setDatasets(data);
      setLoading(false);
    } catch (err) {
      setError('Failed to load datasets');
      setLoading(false);
    }
  };

  const handleCreateDataset = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const data = await botService.createDataset(newDataset);
      setDatasets([...datasets, data]);
      setCreateMode(false);
      setNewDataset({
        name: '',
        description: '',
        type: 'mixed'
      });
      setSuccess('Dataset created successfully');
      setTimeout(() => setSuccess(null), 3000);
      setLoading(false);
    } catch (err) {
      setError('Failed to create dataset');
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleBulkZipChange = (e) => {
    setBulkZipFile(e.target.files[0]);
  };

  const handleUploadDocument = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file to upload');
      return;
    }

    setLoading(true);
    try {
      const fileExtension = file.name.split('.').pop().toLowerCase();
      const isImage = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'].includes(fileExtension);
      
      if (isImage) {
        await botService.uploadImage(selectedDataset.id, file);
      } else {
        await botService.uploadDocument(selectedDataset.id, file);
      }
      
      const data = await botService.getDatasets();
      setDatasets(data);
      setUploadMode(false);
      setSelectedDataset(null);
      setFile(null);
      setSuccess(isImage ? 'Image uploaded successfully' : 'Document uploaded successfully');
      setTimeout(() => setSuccess(null), 3000);
      setLoading(false);
    } catch (err) {
      setError(`Failed to upload ${file.name}`);
      setLoading(false);
    }
  };

  const handleBulkUpload = async (e) => {
    e.preventDefault();
    if (!bulkZipFile) {
      setError('Please select a zip file to upload');
      return;
    }
    setLoading(true);
    setBulkResult(null);
    try {
      const formData = new FormData();
      formData.append('file', bulkZipFile);
      const response = await fetch(`/api/datasets/${selectedDataset.id}/bulk-upload`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        },
        body: formData
      });
      const result = await response.json();
      if (!response.ok) {
        setError(result.error || 'Bulk upload failed');
        setLoading(false);
        return;
      }
      setBulkResult(result);
      setSuccess('Bulk upload completed');
      setTimeout(() => setSuccess(null), 3000);
      setLoading(false);
      // Refresh datasets
      loadDatasets();
      setBulkZipFile(null);
    } catch (err) {
      setError('Bulk upload failed');
      setLoading(false);
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

  const handleRemoveDocument = async () => {
    if (!documentToDelete || !selectedDataset) return;
    
    setLoading(true);
    try {
      await botService.removeDocument(selectedDataset.id, documentToDelete.id);
      
      setDocuments(documents.filter(doc => doc.id !== documentToDelete.id));
      
      const data = await botService.getDatasets();
      setDatasets(data);
      
      setSuccess('Document removed successfully');
      setTimeout(() => setSuccess(null), 3000);
      setShowConfirmDelete(false);
      setDocumentToDelete(null);
      setLoading(false);
    } catch (err) {
      setError('Failed to remove document');
      setShowConfirmDelete(false);
      setLoading(false);
    }
  };

  const confirmRemoveImage = (image) => {
    setImageToDelete(image);
    setShowConfirmImageDelete(true);
  };

  const handleRemoveImage = async () => {
    if (!imageToDelete || !selectedDataset) return;
    
    setLoading(true);
    try {
      await botService.removeImage(selectedDataset.id, imageToDelete.id);
      
      setImages(images.filter(img => img.id !== imageToDelete.id));
      
      const data = await botService.getDatasets();
      setDatasets(data);
      
      setSuccess('Image removed successfully');
      setTimeout(() => setSuccess(null), 3000);
      setShowConfirmImageDelete(false);
      setImageToDelete(null);
      setLoading(false);
    } catch (err) {
      setError('Failed to remove image');
      setShowConfirmImageDelete(false);
      setLoading(false);
    }
  };

  const getFileIcon = (fileName) => {
    if (!fileName) return <FaFile />;
    const extension = fileName.split('.').pop().toLowerCase();
    switch(extension) {
      case 'pdf':
        return <FaFileAlt style={{color: '#e74c3c'}} />;
      case 'docx':
      case 'doc':
        return <FaFileAlt style={{color: '#3498db'}} />;
      case 'pptx':
      case 'ppt':
        return <FaFileAlt style={{color: '#e67e22'}} />;
      case 'txt':
        return <FaFileAlt style={{color: '#7f8c8d'}} />;
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif':
      case 'webp':
      case 'bmp':
        return <FaImage style={{color: '#9b59b6'}} />;
      default:
        return <FaFile />;
    }
  };

  const confirmRemoveDataset = (dataset) => {
    setDatasetToDelete(dataset);
    setShowConfirmDatasetDelete(true);
  };

  const handleRemoveDataset = async () => {
    if (!datasetToDelete) return;
    
    setLoading(true);
    try {
      await botService.deleteDataset(datasetToDelete.id);
      
      setDatasets(datasets.filter(ds => ds.id !== datasetToDelete.id));
      
      setSuccess('Dataset removed successfully');
      setTimeout(() => setSuccess(null), 3000);
      setShowConfirmDatasetDelete(false);
      setDatasetToDelete(null);
      setLoading(false);
    } catch (err) {
      setError('Failed to remove dataset');
      setShowConfirmDatasetDelete(false);
      setLoading(false);
    }
  };

  return (
    <Container className="my-5">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h2><FaDatabase className="me-2" />Your Datasets</h2>
        <Button 
          variant={createMode ? "outline-secondary" : "primary"} 
          onClick={() => {
            setCreateMode(!createMode);
            setUploadMode(false);
            setSelectedDataset(null);
          }}
          className="rounded-pill"
        >
          {createMode ? <><FaTimes className="me-1" /> Cancel</> : <><FaPlus className="me-1" /> Create New Dataset</>}
        </Button>
      </div>
      
      {error && <Alert variant="danger" onClose={() => setError(null)} dismissible>{error}</Alert>}
      {success && <Alert variant="success" onClose={() => setSuccess(null)} dismissible>{success}</Alert>}

      {createMode && (
        <Card className="mb-4 border-0 shadow-sm">
          <Card.Body>
            <Card.Title className="mb-3">Create a New Dataset</Card.Title>
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
                      required
                      placeholder="Give your dataset a name"
                      className="rounded-pill"
                    />
                  </Form.Group>
                </Col>
                
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>Description</Form.Label>
                    <Form.Control
                      type="text"
                      name="description"
                      value={newDataset.description}
                      onChange={handleInputChange}
                      placeholder="What kind of documents will this dataset contain?"
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
                  {loading ? <><Spinner as="span" animation="border" size="sm" /> Creating...</> : 'Create Dataset'}
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
                    {loading ? <><Spinner as="span" animation="border" size="sm" /> Uploading...</> : 'Upload'}
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
                  />
                  <Form.Text className="text-muted">
                    Upload a .zip file containing documents and images. Supported: PDF, DOCX, TXT, PPTX, JPG, PNG, etc.
                  </Form.Text>
                </Form.Group>
                <div className="d-flex">
                  <Button variant="primary" type="submit" disabled={loading} className="me-2 rounded-pill">
                    {loading ? <><Spinner as="span" animation="border" size="sm" /> Uploading...</> : 'Bulk Upload'}
                  </Button>
                  <Button 
                    variant="outline-secondary" 
                    onClick={() => {
                      setUploadMode(false);
                      setSelectedDataset(null);
                      setBulkUploadMode(false);
                      setBulkZipFile(null);
                      setBulkResult(null);
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
                <Spinner animation="border" variant="primary" />
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
                <Spinner animation="border" variant="primary" />
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
                            src={`${process.env.REACT_APP_API_URL || 'http://localhost:50505'}${img.url}`} 
                            alt={img.caption || 'Dataset image'} 
                            className="card-img-top" 
                            style={{ objectFit: 'cover', width: '100%', height: '100%' }}
                            onError={(e) => {
                              e.target.onerror = null;
                              e.target.src = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4MCIgaGVpZ2h0PSI2MCIgdmlld0JveD0iMCAwIDgwIDYwIj48cmVjdCB3aWR0aD0iODAiIGhlaWdodD0iNjAiIGZpbGw9IiNlZWUiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZG9taW5hbnQtYmFzZWxpbmU9Im1pZGRsZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIiIGZpbGw9IiNhYWEiPk5vIEltYWdlPC90ZXh0Pjwvc3ZnPg==';
                            }}
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
                              onClick={() => window.open(`${process.env.REACT_APP_API_URL || 'http://localhost:50505'}${img.url}`, '_blank')}
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
          <Spinner animation="border" variant="primary" />
          <p className="mt-2 text-muted">Loading your datasets...</p>
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
                                        src={`${process.env.REACT_APP_API_URL || 'http://localhost:50505'}${img.url}`} 
                                        alt={img.caption || 'Dataset image'} 
                                        className="img-thumbnail" 
                                        style={{ width: '100%', height: '60px', objectFit: 'cover' }}
                                        title={img.caption || 'Dataset image'}
                                        onError={(e) => {
                                          e.target.onerror = null;
                                          e.target.src = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4MCIgaGVpZ2h0PSI2MCIgdmlld0JveD0iMCAwIDgwIDYwIj48cmVjdCB3aWR0aD0iODAiIGhlaWdodD0iNjAiIGZpbGw9IiNlZWUiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZG9taW5hbnQtYmFzZWxpbmU9Im1pZGRsZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIiIGZpbGw9IiNhYWEiPk5vIEltYWdlPC90ZXh0Pjwvc3ZnPg==';
                                        }}
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
            {loading ? <><Spinner as="span" animation="border" size="sm" /> Removing...</> : 'Remove Document'}
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
                  src={`${process.env.REACT_APP_API_URL || 'http://localhost:50505'}${imageToDelete.url}`} 
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
            {loading ? <><Spinner as="span" animation="border" size="sm" /> Removing...</> : 'Remove Image'}
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
            {loading ? <><Spinner as="span" animation="border" size="sm" /> Removing...</> : 'Remove Dataset'}
          </Button>
        </Modal.Footer>
      </Modal>
    </Container>
  );
};

export default DatasetPanel; 