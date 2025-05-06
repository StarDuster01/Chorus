import React, { useState, useEffect } from 'react';
import { Container, Card, Button, Form, Row, Col, ListGroup, Badge, Alert, Spinner, Modal } from 'react-bootstrap';
import botService from '../services/botService';
import { FaDatabase, FaPlus, FaFileUpload, FaFile, FaTimes, FaFileAlt, FaTrash, FaList } from 'react-icons/fa';

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
    description: ''
  });
  const [documents, setDocuments] = useState([]);
  const [viewDocumentsMode, setViewDocumentsMode] = useState(false);
  const [documentListLoading, setDocumentListLoading] = useState(false);
  const [showConfirmDelete, setShowConfirmDelete] = useState(false);
  const [documentToDelete, setDocumentToDelete] = useState(null);

  useEffect(() => {
    loadDatasets();
  }, []);

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
        description: ''
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

  const handleUploadDocument = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file to upload');
      return;
    }

    setLoading(true);
    try {
      await botService.uploadDocument(selectedDataset.id, file);
      // Refresh the datasets to update document count
      const data = await botService.getDatasets();
      setDatasets(data);
      setUploadMode(false);
      setSelectedDataset(null);
      setFile(null);
      setSuccess('Document uploaded successfully');
      setTimeout(() => setSuccess(null), 3000);
      setLoading(false);
    } catch (err) {
      setError('Failed to upload document');
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

  const confirmRemoveDocument = (document) => {
    setDocumentToDelete(document);
    setShowConfirmDelete(true);
  };

  const handleRemoveDocument = async () => {
    if (!documentToDelete || !selectedDataset) return;
    
    setLoading(true);
    try {
      await botService.removeDocument(selectedDataset.id, documentToDelete.id);
      
      // Remove document from the list
      setDocuments(documents.filter(doc => doc.id !== documentToDelete.id));
      
      // Refresh the datasets to update document count
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
      default:
        return <FaFile />;
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
              Upload Document to {selectedDataset.name}
            </Card.Title>
            <Form onSubmit={handleUploadDocument}>
              <Form.Group className="mb-4">
                <Form.Label>Select Document</Form.Label>
                <div className="custom-file-upload">
                  <Form.Control
                    type="file"
                    onChange={handleFileChange}
                    required
                    accept=".pdf,.txt,.docx,.pptx"
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
                  Supported formats: PDF, TXT, DOCX, PPTX
                </Form.Text>
              </Form.Group>

              <div className="d-flex">
                <Button variant="primary" type="submit" disabled={loading} className="me-2 rounded-pill">
                  {loading ? <><Spinner as="span" animation="border" size="sm" /> Uploading...</> : 'Upload Document'}
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
                          <Badge bg="light" text="dark" className="border">
                            {doc.chunk_count} chunks
                          </Badge>
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

      {loading && !createMode && !uploadMode && !viewDocumentsMode ? (
        <div className="text-center my-5">
          <Spinner animation="border" variant="primary" />
          <p className="mt-2 text-muted">Loading your datasets...</p>
        </div>
      ) : (
        <Row className="mt-4">
          {!createMode && !uploadMode && !viewDocumentsMode && datasets.length === 0 ? (
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
          ) : !createMode && !uploadMode && !viewDocumentsMode && (
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
                              <Badge bg="light" text="dark" className="border">
                                <FaFileAlt className="me-1" /> {dataset.document_count} {dataset.document_count === 1 ? 'Document' : 'Documents'}
                              </Badge>
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
                              className="rounded-pill"
                            >
                              <FaList className="me-1" /> Documents
                            </Button>
                          )}
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
    </Container>
  );
};

export default DatasetPanel; 