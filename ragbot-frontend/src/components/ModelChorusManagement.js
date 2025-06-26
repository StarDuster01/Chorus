import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Container, Card, Button, Table, Badge, 
  Spinner, Alert, Modal, Form
} from 'react-bootstrap';
import { 
  FaPlus, FaEdit, FaTrash, FaCheck, 
  FaTimes, FaUsers, FaRobot, FaLink, FaExclamationTriangle
} from 'react-icons/fa';
import botService from '../services/botService';

const ModelChorusManagement = () => {
  const navigate = useNavigate();
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [bots, setBots] = useState([]);
  const [chorusConfigs, setChorusConfigs] = useState([]);
  
  // State for apply to bot modal
  const [showApplyModal, setShowApplyModal] = useState(false);
  const [selectedChorusId, setSelectedChorusId] = useState('');
  const [selectedBotId, setSelectedBotId] = useState('');
  const [applyLoading, setApplyLoading] = useState(false);
  
  // Delete chorus confirmation modal
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [chorusToDelete, setChorusToDelete] = useState(null);
  
  // Load all bots and their chorus configurations
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        
        // First, load all bots
        const botsData = await botService.getBots();
        setBots(botsData);
        
        // Then load all available choruses
        const chorusesData = await botService.getAllChoruses();
        
        // Enhance the choruses with bot information
        const enhancedChoruses = chorusesData.map(chorus => {
          // Find the bot that uses this chorus (if any)
          const associatedBot = botsData.find(bot => bot.chorus_id === chorus.id);
          
          return {
            ...chorus,
            botId: associatedBot?.id || '',
            botName: associatedBot?.name || 'Not linked'
          };
        });
        
        setChorusConfigs(enhancedChoruses);
      } catch (err) {
        setError(`Failed to load data: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, []);
  
  const handleCreateNew = () => {
    // Navigate to create new chorus page directly
    navigate('/chorus/new');
  };
  
  const handleEdit = (chorusId) => {
    navigate(`/chorus/${chorusId}`);
  };
  
  const handleApplyToBot = (chorusId) => {
    setSelectedChorusId(chorusId);
    setSelectedBotId('');
    setShowApplyModal(true);
  };
  
  const handleConfirmApply = async () => {
    if (!selectedBotId) {
      return;
    }
    
    setApplyLoading(true);
    try {
      await botService.setBotChorus(selectedBotId, selectedChorusId);
      
      setSuccess('Model chorus applied to bot successfully!');
      setShowApplyModal(false);
      
      // Refresh data after applying
      const botsData = await botService.getBots();
      setBots(botsData);
      
      // Update the chorus configs to reflect the new association
      setChorusConfigs(prev => 
        prev.map(chorus => {
          if (chorus.id === selectedChorusId) {
            const associatedBot = botsData.find(bot => bot.id === selectedBotId);
            return {
              ...chorus,
              botId: selectedBotId,
              botName: associatedBot?.name || 'Unknown'
            };
          }
          return chorus;
        })
      );
      
      // Clear success message after 3 seconds
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(`Failed to apply chorus to bot: ${err.message}`);
    } finally {
      setApplyLoading(false);
    }
  };
  
  const handleDeleteChorus = async (chorusId) => {
    // Find the chorus config to get its name
    const chorus = chorusConfigs.find(c => c.id === chorusId);
    setChorusToDelete(chorus);
    setShowDeleteModal(true);
  };
  
  const handleConfirmDelete = async () => {
    if (!chorusToDelete) {
      return;
    }
    
    try {
      await botService.deleteChorus(chorusToDelete.id);
      
      // Remove from local state
      setChorusConfigs(prev => prev.filter(c => c.id !== chorusToDelete.id));
      
      setSuccess('Chorus deleted successfully');
      setTimeout(() => setSuccess(null), 3000);
      setShowDeleteModal(false);
      setChorusToDelete(null);
    } catch (err) {
      setError(`Failed to delete chorus: ${err.message}`);
      setShowDeleteModal(false);
    }
  };
  
  const renderModelCount = (models) => {
    if (!models || models.length === 0) return '0';
    
    // Count by provider
    const counts = {};
    models.forEach(model => {
      const provider = model.provider || 'Unknown';
      counts[provider] = (counts[provider] || 0) + 1;
    });
    
    return Object.entries(counts).map(([provider, count]) => (
      <Badge 
        key={provider} 
        bg={
          provider === 'OpenAI' ? 'success' : 
          provider === 'Anthropic' ? 'primary' :
          provider === 'Groq' ? 'warning' : 
          provider === 'Mistral' ? 'info' :
          'secondary'
        }
        className="me-1"
      >
        {provider}: {count}
      </Badge>
    ));
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
            <h3 className="mb-0"><FaUsers className="me-2" />Model Chorus Management</h3>
            <p className="mb-0 text-muted small">Configure and manage model choruses for your bots</p>
          </div>
          <Button 
            variant="primary" 
            onClick={handleCreateNew}
          >
            <FaPlus className="me-2" />Create New
          </Button>
        </Card.Header>
        
        <Card.Body>
          {error && <Alert variant="danger" onClose={() => setError(null)} dismissible>{error}</Alert>}
          {success && <Alert variant="success" onClose={() => setSuccess(null)} dismissible>{success}</Alert>}
          
          {chorusConfigs.length === 0 ? (
            <div className="text-center py-5">
              <FaUsers size={48} className="mb-3 text-muted" />
              <h4>No Model Chorus Configurations Yet</h4>
              <p className="text-muted">
                Create your first model chorus to leverage multiple AI models to generate better responses.
              </p>
              <Button variant="primary" onClick={handleCreateNew}>
                Create Your First Model Chorus
              </Button>
            </div>
          ) : (
            <Table responsive>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Bot</th>
                  <th>Response Models</th>
                  <th>Evaluator Models</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {chorusConfigs.map(config => (
                  <tr key={config.id}>
                    <td>
                      <strong>{config.name}</strong>
                      {config.description && <div className="small text-muted">{config.description}</div>}
                    </td>
                    <td>{config.botName}</td>
                    <td>{renderModelCount(config.response_models)}</td>
                    <td>{renderModelCount(config.evaluator_models)}</td>
                    <td>
                      {config.is_active ? (
                        <Badge bg="success">Active</Badge>
                      ) : (
                        <Badge bg="secondary">Inactive</Badge>
                      )}
                    </td>
                    <td>
                      <Button 
                        variant="outline-primary" 
                        size="sm" 
                        onClick={() => handleEdit(config.id)}
                        className="me-2"
                        title="Edit Configuration"
                      >
                        <FaEdit />
                      </Button>
                      <Button 
                        variant="outline-success" 
                        size="sm"
                        onClick={() => navigate(`/chat/${config.botId}`)}
                        className="me-2"
                        title="Chat with Bot"
                      >
                        <FaRobot />
                      </Button>
                      <Button
                        variant="outline-info"
                        size="sm"
                        onClick={() => handleApplyToBot(config.id)}
                        className="me-2"
                        title="Apply to Another Bot"
                      >
                        <FaLink />
                      </Button>
                      <Button
                        variant="outline-danger"
                        size="sm"
                        onClick={() => handleDeleteChorus(config.id)}
                        title="Delete Chorus"
                      >
                        <FaTrash />
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Card.Body>
      </Card>
      
      {/* Info section */}
      <Card className="mt-4 shadow-sm">
        <Card.Header>
          <h5 className="mb-0">About Model Chorus</h5>
        </Card.Header>
        <Card.Body>
          <p>
            A Model Chorus is a system that combines multiple AI models to produce higher quality responses than any single model could produce alone:
          </p>
          
          <ul>
            <li><strong>Response Models:</strong> Generate candidate answers to user queries</li>
            <li><strong>Evaluator Models:</strong> Vote on which response is most accurate and helpful</li>
            <li><strong>Multi-provider:</strong> Mix and match models from OpenAI, Anthropic, Groq, and Mistral</li>
                            <li><strong>Configurable Weights:</strong> Control how many times each model is queried (response models) or votes (evaluator models)</li>
          </ul>
          
          <p>
            To use a model chorus, first create a configuration, then activate it when chatting with your bot.
          </p>
        </Card.Body>
      </Card>
      
      {/* Apply to Bot Modal */}
      <Modal show={showApplyModal} onHide={() => setShowApplyModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Apply Model Chorus to Bot</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Form>
            <Form.Group className="mb-3">
              <Form.Label>Select Bot</Form.Label>
              <Form.Select
                value={selectedBotId}
                onChange={(e) => setSelectedBotId(e.target.value)}
                required
              >
                <option value="">Choose a bot</option>
                {bots.filter(bot => bot.id !== selectedChorusId).map(bot => (
                  <option key={bot.id} value={bot.id}>
                    {bot.name}
                  </option>
                ))}
              </Form.Select>
              <Form.Text className="text-muted">
                This will set the selected model chorus as the default for the chosen bot.
              </Form.Text>
            </Form.Group>
          </Form>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowApplyModal(false)}>
            Cancel
          </Button>
          <Button 
            variant="primary" 
            onClick={handleConfirmApply}
            disabled={!selectedBotId || applyLoading}
          >
            {applyLoading ? (
              <>
                <Spinner as="span" animation="border" size="sm" className="me-2" />
                Applying...
              </>
            ) : (
              'Apply Chorus to Bot'
            )}
          </Button>
        </Modal.Footer>
      </Modal>
      
      {/* Delete Chorus Confirmation Modal */}
      <Modal show={showDeleteModal} onHide={() => setShowDeleteModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Confirm Chorus Deletion</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <div className="d-flex align-items-center mb-3">
            <FaExclamationTriangle className="text-warning me-2" size={24} />
            <p className="mb-0">
              Are you sure you want to delete this chorus? This action cannot be undone.
            </p>
          </div>
          
          {chorusToDelete && (
            <div className="p-3 bg-light rounded">
              <strong>Name:</strong> {chorusToDelete.name}
              {chorusToDelete.botName !== 'Not linked' && (
                <div>
                  <strong>Used by bot:</strong> {chorusToDelete.botName}
                </div>
              )}
            </div>
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowDeleteModal(false)}>
            Cancel
          </Button>
          <Button 
            variant="danger" 
            onClick={handleConfirmDelete}
          >
            Delete Chorus
          </Button>
        </Modal.Footer>
      </Modal>
    </Container>
  );
};

export default ModelChorusManagement; 