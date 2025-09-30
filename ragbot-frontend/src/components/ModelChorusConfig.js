import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  Container, Card, Button, Form, Row, Col, 
  ListGroup, Alert, Spinner, Badge, Modal,
  OverlayTrigger, Popover
} from 'react-bootstrap';
import { 
  FaChevronLeft, FaSave, FaPlus, FaTrash, 
  FaRobot, FaVoteYea, FaGavel, FaUsers,
  FaInfoCircle
} from 'react-icons/fa';
import botService from '../services/botService';

// Safe UUID generator for older environments without crypto.randomUUID
const generateId = () => {
  try {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID();
    }
  } catch (_) {}
  const rand = Math.random().toString(16).slice(2);
  return `id_${Date.now()}_${rand}`;
};

const MODEL_OPTIONS = [
  { provider: 'OpenAI', models: ['gpt-5-2025-08-07', 'gpt-4.1-nano-2025-04-14'] },
  { provider: 'Anthropic', models: ['claude-3-7-sonnet-latest'] },
  { provider: 'Groq', models: ['llama3-70b-8192', 'llama3-8b-8192'] },
  { provider: 'Mistral', models: ['mistral-large', 'mistral-medium', 'mistral-small'] },
];

const ModelChorusConfig = () => {
  const { botId, chorusId: chorusIdParam } = useParams();
  const navigate = useNavigate();
  
  const [bot, setBot] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [isStandalone, setIsStandalone] = useState(true); // All choruses are standalone now
  
  // Chorus configuration state
  const [chorusName, setChorusName] = useState('');
  const [chorusDescription, setChorusDescription] = useState('');
  const [isActive, setIsActive] = useState(false);
  const [useDiverseRag, setUseDiverseRag] = useState(false);
  const [chorusId, setChorusId] = useState('');
  
  // Models configuration
  const [responseModels, setResponseModels] = useState([]);
  const [evaluatorModels, setEvaluatorModels] = useState([]);
  
  // Modal state for adding new models
  const [showAddModal, setShowAddModal] = useState(false);
  const [modelType, setModelType] = useState('response'); // 'response' or 'evaluator'
  const [newModelProvider, setNewModelProvider] = useState('OpenAI');
  const [newModelName, setNewModelName] = useState('gpt-4.1-nano-2025-04-14');
  const [newModelTemperature, setNewModelTemperature] = useState(0.7);
  const [newModelWeight, setNewModelWeight] = useState(1);
  
  // Load chorus data if editing an existing one
  useEffect(() => {
    const loadChorus = async () => {
      try {
        setLoading(true);
        
        // Check if we're editing an existing chorus
        if (chorusIdParam && chorusIdParam !== 'new') {
          // Load the chorus by ID
          const chorusData = await botService.getChorus(chorusIdParam);
          
          // Populate the form with existing configuration
          setChorusId(chorusData.id);
          setChorusName(chorusData.name || '');
          setChorusDescription(chorusData.description || '');
          setIsActive(chorusData.is_active || false);
          setUseDiverseRag(chorusData.use_diverse_rag || false);
          setResponseModels(chorusData.response_models || []);
          setEvaluatorModels(chorusData.evaluator_models || []);
        } else {
          // Creating a new chorus
          setChorusName('New Model Chorus');
          setChorusDescription('Model chorus configuration');
          setIsActive(false);
          setUseDiverseRag(false);
          setResponseModels([
            { id: generateId(), provider: 'OpenAI', model: 'gpt-5-2025-08-07', temperature: 0.6, weight: 1 },
            { id: generateId(), provider: 'OpenAI', model: 'gpt-4.1-nano-2025-04-14', temperature: 0.7, weight: 1 }
          ]);
          setEvaluatorModels([
            { id: generateId(), provider: 'OpenAI', model: 'gpt-5-2025-08-07', temperature: 0.2, weight: 1 }
          ]);
        }
      } catch (err) {
        setError(`Failed to load chorus details: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };
    
    loadChorus();
  }, [chorusIdParam]);
  
  const handleAddModel = () => {
    const newModel = {
      id: generateId(),
      provider: newModelProvider,
      model: newModelName,
      temperature: parseFloat(newModelTemperature),
      weight: parseInt(newModelWeight, 10)
    };
    
    if (modelType === 'response') {
      setResponseModels([...responseModels, newModel]);
    } else {
      setEvaluatorModels([...evaluatorModels, newModel]);
    }
    
    // Reset form and close modal
    setShowAddModal(false);
  };
  
  const handleRemoveModel = (type, modelId) => {
    if (type === 'response') {
      setResponseModels(responseModels.filter(model => model.id !== modelId));
    } else {
      setEvaluatorModels(evaluatorModels.filter(model => model.id !== modelId));
    }
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate form
    if (!chorusName.trim()) {
      setError('Chorus name is required');
      return;
    }
    
    if (responseModels.length === 0) {
      setError('At least one response model is required');
      return;
    }
    
    if (evaluatorModels.length === 0) {
      setError('At least one evaluator model is required');
      return;
    }
    
    try {
      setSaving(true);
      
      const chorusConfig = {
        name: chorusName,
        description: chorusDescription,
        is_active: isActive,
        use_diverse_rag: useDiverseRag,
        response_models: responseModels,
        evaluator_models: evaluatorModels
      };
      
      // If we have a chorus ID, we're updating an existing chorus
      if (chorusIdParam && chorusIdParam !== 'new') {
        // Update existing chorus
        await botService.updateChorus(chorusIdParam, chorusConfig);
        setSuccess('Chorus configuration updated successfully');
      } else {
        // Create a new chorus
        const result = await botService.createChorus(chorusConfig);
        setSuccess('Chorus configuration created successfully');
      }
      
      // Redirect to chorus management after successful save
      setTimeout(() => navigate('/chorus'), 1000);
      
    } catch (err) {
      setError(`Failed to save chorus configuration: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };
  
  // Open modal to add a new model
  const openAddModelModal = (type) => {
    setModelType(type);
    setNewModelProvider('OpenAI');
    setNewModelName('gpt-4.1-nano-2025-04-14');
    setNewModelTemperature(type === 'response' ? 0.7 : 0.2);
    setNewModelWeight(1);
    setShowAddModal(true);
  };
  
  if (loading) {
    return (
      <Container className="d-flex justify-content-center align-items-center" style={{ minHeight: '50vh' }}>
        <Spinner animation="border" />
      </Container>
    );
  }

  return (
    <Container className="py-4">
      <Card className="shadow-sm">
        <Card.Header className="d-flex justify-content-between align-items-center">
          <div>
            <Button 
              variant="link" 
              className="p-0 text-decoration-none" 
              onClick={() => navigate('/chorus')}
            >
              <FaChevronLeft className="me-2" />
              Back to Chorus Management
            </Button>
            <h3 className="mt-2 mb-0">
              <FaUsers className="me-2" />
              {chorusIdParam && chorusIdParam !== 'new' ? 'Edit Model Chorus' : 'Create New Model Chorus'}
            </h3>
          </div>
        </Card.Header>
        
        <Card.Body>
          {error && <Alert variant="danger" className="mb-3">{error}</Alert>}
          {success && <Alert variant="success" className="mb-3">{success}</Alert>}
          
          <Form onSubmit={handleSubmit}>
            <Row className="mb-4">
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Chorus Name</Form.Label>
                  <Form.Control 
                    type="text" 
                    value={chorusName} 
                    onChange={(e) => setChorusName(e.target.value)}
                    placeholder="Enter a name for this chorus"
                    required
                  />
                </Form.Group>
              </Col>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Description</Form.Label>
                  <Form.Control 
                    type="text" 
                    value={chorusDescription} 
                    onChange={(e) => setChorusDescription(e.target.value)}
                    placeholder="Describe this chorus configuration"
                  />
                </Form.Group>
              </Col>
              <Col md={12}>
                <Form.Check 
                  type="switch"
                  id="active-switch"
                  label="Activate this chorus (will be used for all chat interactions)"
                  checked={isActive}
                  onChange={(e) => setIsActive(e.target.checked)}
                  className="mb-3"
                />
                <Form.Check 
                  type="switch"
                  id="diverse-rag-switch"
                  label={
                    <>
                      Use Diverse RAG (each model will get different retrieval results, covering more information)
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Popover id="diverse-rag-popover">
                            <Popover.Header as="h3">Diverse RAG</Popover.Header>
                            <Popover.Body>
                              <p>When enabled, each response model will receive a unique set of retrieved contexts.</p>
                              <p>Benefits:</p>
                              <ul>
                                <li>Broader coverage of relevant information</li>
                                <li>Higher chance of finding critical information</li>
                                <li>Models can focus on different aspects of the question</li>
                              </ul>
                              <p>Recommended for complex queries that might benefit from exploring multiple angles or when working with large document collections.</p>
                            </Popover.Body>
                          </Popover>
                        }
                      >
                        <FaInfoCircle className="ms-2 text-primary" style={{ cursor: 'pointer' }} />
                      </OverlayTrigger>
                    </>
                  }
                  checked={useDiverseRag}
                  onChange={(e) => setUseDiverseRag(e.target.checked)}
                  className="mb-3"
                />
              </Col>
            </Row>
            
            <Row className="mb-4">
              <Col md={6}>
                <div className="d-flex justify-content-between align-items-center mb-3">
                  <h5 className="mb-0"><FaRobot className="me-2" /> Response Models</h5>
                  <Button 
                    variant="outline-primary" 
                    size="sm"
                    onClick={() => openAddModelModal('response')}
                  >
                    <FaPlus className="me-1" /> Add Model
                  </Button>
                </div>
                
                <ListGroup>
                  {responseModels.length === 0 ? (
                    <ListGroup.Item className="text-center text-muted py-3">
                      No response models configured
                    </ListGroup.Item>
                  ) : (
                    responseModels.map(model => (
                      <ListGroup.Item 
                        key={model.id}
                        className="d-flex justify-content-between align-items-center"
                      >
                        <div>
                          <Badge 
                            bg={
                              model.provider === 'OpenAI' ? 'success' : 
                              model.provider === 'Anthropic' ? 'primary' :
                              model.provider === 'Groq' ? 'warning' : 
                              'secondary'
                            }
                            className="me-2"
                          >
                            {model.provider}
                          </Badge>
                          <strong>{model.model}</strong>
                          <div className="small text-muted">
                            Temperature: {model.temperature}, Weight: {model.weight}
                          </div>
                        </div>
                        <Button 
                          variant="link" 
                          className="text-danger p-0" 
                          onClick={() => handleRemoveModel('response', model.id)}
                        >
                          <FaTrash />
                        </Button>
                      </ListGroup.Item>
                    ))
                  )}
                </ListGroup>
              </Col>
              
              <Col md={6}>
                <div className="d-flex justify-content-between align-items-center mb-3">
                  <h5 className="mb-0"><FaGavel className="me-2" /> Evaluator Models</h5>
                  <Button 
                    variant="outline-primary" 
                    size="sm"
                    onClick={() => openAddModelModal('evaluator')}
                  >
                    <FaPlus className="me-1" /> Add Model
                  </Button>
                </div>
                
                <ListGroup>
                  {evaluatorModels.length === 0 ? (
                    <ListGroup.Item className="text-center text-muted py-3">
                      No evaluator models configured
                    </ListGroup.Item>
                  ) : (
                    evaluatorModels.map(model => (
                      <ListGroup.Item 
                        key={model.id}
                        className="d-flex justify-content-between align-items-center"
                      >
                        <div>
                          <Badge 
                            bg={
                              model.provider === 'OpenAI' ? 'success' : 
                              model.provider === 'Anthropic' ? 'primary' :
                              model.provider === 'Groq' ? 'warning' : 
                              'secondary'
                            }
                            className="me-2"
                          >
                            {model.provider}
                          </Badge>
                          <strong>{model.model}</strong>
                          <div className="small text-muted">
                            Temperature: {model.temperature}, Weight: {model.weight}
                          </div>
                        </div>
                        <Button 
                          variant="link" 
                          className="text-danger p-0" 
                          onClick={() => handleRemoveModel('evaluator', model.id)}
                        >
                          <FaTrash />
                        </Button>
                      </ListGroup.Item>
                    ))
                  )}
                </ListGroup>
              </Col>
            </Row>
            
            <div className="d-flex justify-content-between">
              <Button 
                variant="outline-secondary"
                onClick={() => navigate('/chorus')}
              >
                Cancel
              </Button>
              <Button 
                type="submit" 
                variant="primary"
                disabled={saving}
              >
                {saving ? (
                  <>
                    <Spinner as="span" animation="border" size="sm" className="me-2" />
                    Saving...
                  </>
                ) : (
                  <>
                    <FaSave className="me-2" /> Save Configuration
                  </>
                )}
              </Button>
            </div>
          </Form>
        </Card.Body>
      </Card>
      
      {/* Add Model Modal */}
      <Modal show={showAddModal} onHide={() => setShowAddModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>
            Add {modelType === 'response' ? 'Response' : 'Evaluator'} Model
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Form>
            <Form.Group className="mb-3">
              <Form.Label>Provider</Form.Label>
              <Form.Select 
                value={newModelProvider} 
                onChange={(e) => {
                  setNewModelProvider(e.target.value);
                  // Set the first model of the selected provider as default
                  const providerModels = MODEL_OPTIONS.find(p => p.provider === e.target.value)?.models || [];
                  if (providerModels.length > 0) {
                    setNewModelName(providerModels[0]);
                  }
                }}
              >
                {MODEL_OPTIONS.map(provider => (
                  <option key={provider.provider} value={provider.provider}>
                    {provider.provider}
                  </option>
                ))}
              </Form.Select>
            </Form.Group>
            
            <Form.Group className="mb-3">
              <Form.Label>Model</Form.Label>
              <Form.Select 
                value={newModelName} 
                onChange={(e) => setNewModelName(e.target.value)}
              >
                {MODEL_OPTIONS.find(p => p.provider === newModelProvider)?.models.map(model => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </Form.Select>
            </Form.Group>
            
            <Form.Group className="mb-3">
              <Form.Label>Temperature (0.0 - 1.0)</Form.Label>
              <Form.Control 
                type="number" 
                min="0" 
                max="1" 
                step="0.1" 
                value={newModelTemperature} 
                onChange={(e) => setNewModelTemperature(e.target.value)}
              />
              <Form.Text className="text-muted">
                {modelType === 'response' ? 
                  'Higher values produce more diverse responses. For response models, 0.7 is a good default.' : 
                  'For evaluators, lower values (0.1-0.3) are recommended for more consistent judgments.'}
              </Form.Text>
            </Form.Group>
            
            <Form.Group className="mb-3">
              <Form.Label>Weight</Form.Label>
              <Form.Control 
                type="number" 
                min="1" 
                max="10" 
                step="1" 
                value={newModelWeight} 
                onChange={(e) => setNewModelWeight(e.target.value)}
              />
              <Form.Text className="text-muted">
                {modelType === 'response' ? 
                  'How many instances of this model to include in the chorus. Higher weight means more influence.' : 
                  'Weight determines how many votes this evaluator gets. Higher weight gives this model more influence.'}
              </Form.Text>
            </Form.Group>
          </Form>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowAddModal(false)}>
            Cancel
          </Button>
          <Button variant="primary" onClick={handleAddModel}>
            Add Model
          </Button>
        </Modal.Footer>
      </Modal>
      
      {/* Info area to explain the configuration */}
      <Card className="mt-4 shadow-sm">
        <Card.Header>
          <h5 className="mb-0"><FaVoteYea className="me-2" /> How Model Chorus Works</h5>
        </Card.Header>
        <Card.Body>
          <p>
            A Model Chorus consists of two groups of models that work together to produce high-quality responses:
          </p>
          
          <h6><FaRobot className="me-2" />Response Models</h6>
          <p>
            These models generate candidate responses to the user's question. Each model produces its own unique response based on its capabilities and training. The more diverse your selection of models, the greater the variety of perspectives and information in your candidate responses.
          </p>
          
          <h6><FaGavel className="me-2" />Evaluator Models</h6>
          <p>
            Evaluator models judge the quality of responses generated by the response models. They vote on which response is most accurate, helpful and relevant. The response with the most votes wins and is presented to the user.
          </p>
          
          <h6>Best Practices</h6>
          <ul>
            <li>Include a diverse set of response models from different providers</li>
            <li>Use low temperature (0.1-0.3) for evaluator models to ensure consistent judgments</li>
            <li>For critical applications, use more powerful models (like GPT-4 or Claude) as evaluators</li>
            <li>Experiment with different combinations to find what works best for your specific use case</li>
          </ul>
        </Card.Body>
      </Card>
    </Container>
  );
};

export default ModelChorusConfig; 