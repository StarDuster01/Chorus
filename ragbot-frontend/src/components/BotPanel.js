import React, { useState, useEffect } from 'react';
import { Card, Button, Container, Row, Col, Form, ListGroup, Badge, Alert, Spinner, Modal, Dropdown } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import botService from '../services/botService';
import { FaRobot, FaPlus, FaTimes, FaDatabase, FaComment, FaUsers, FaTrash, FaExchangeAlt } from 'react-icons/fa';

const BotPanel = () => {
  const [bots, setBots] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [chorusConfigs, setChorusConfigs] = useState([]);
  const [availableChoruses, setAvailableChoruses] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [selectedBot, setSelectedBot] = useState(null);
  const [createMode, setCreateMode] = useState(false);
  const [newBot, setNewBot] = useState({
    name: '',
    dataset_id: '',
    chorus_id: '',
    system_instruction: 'You are a helpful AI assistant. Answer questions based on the provided context.'
  });
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [botToDelete, setBotToDelete] = useState(null);
  const [showChorusModal, setShowChorusModal] = useState(false);
  const [botToChangeChorus, setBotToChangeChorus] = useState(null);
  const [selectedChorusId, setSelectedChorusId] = useState('');
  
  const navigate = useNavigate();

  useEffect(() => {
    loadBots();
    loadDatasets();
    loadChorusConfigs();
    loadAvailableChoruses();
  }, []);

  const loadAvailableChoruses = async () => {
    try {
      const choruses = await botService.getAllChoruses();
      setAvailableChoruses(choruses);
    } catch (err) {
      console.error('Failed to load available choruses:', err);
    }
  };

  const loadBots = async () => {
    setLoading(true);
    try {
      const data = await botService.getBots();
      setBots(data);
      setLoading(false);
    } catch (err) {
      setError('Failed to load bots');
      setLoading(false);
    }
  };

  const loadDatasets = async () => {
    try {
      const data = await botService.getDatasets();
      setDatasets(data);
    } catch (err) {
      setError('Failed to load datasets');
    }
  };

  const loadChorusConfigs = async () => {
    try {
      // Load all bots to get their IDs
      const botsData = await botService.getBots();
      const configs = [];
      
      // Try to load chorus configs for each bot
      for (const bot of botsData) {
        try {
          const config = await botService.getChorusConfig(bot.id);
          configs.push({
            ...config,
            id: bot.id, // Use the bot ID as the chorus config ID
            botId: bot.id,
            botName: bot.name
          });
        } catch (err) {
          // No chorus config for this bot - just skip it
        }
      }
      
      setChorusConfigs(configs);
    } catch (err) {
      console.error('Failed to load chorus configs:', err);
    }
  };

  const handleDeleteBot = async () => {
    if (!botToDelete) return;
    
    setLoading(true);
    try {
      await botService.deleteBot(botToDelete.id);
      setBots(bots.filter(bot => bot.id !== botToDelete.id));
      setShowDeleteModal(false);
      setBotToDelete(null);
      setSuccess('Bot deleted successfully!');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError('Failed to delete bot');
      console.error('Error deleting bot:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleChangeChorus = async () => {
    if (!botToChangeChorus) return;
    
    setLoading(true);
    try {
      await botService.setBotChorus(botToChangeChorus.id, selectedChorusId);
      
      // Update the bot in the local state
      const updatedBots = bots.map(bot => {
        if (bot.id === botToChangeChorus.id) {
          return { ...bot, chorus_id: selectedChorusId };
        }
        return bot;
      });
      
      setBots(updatedBots);
      setShowChorusModal(false);
      setBotToChangeChorus(null);
      setSelectedChorusId('');
      setSuccess('Chorus updated successfully!');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError('Failed to update chorus');
      console.error('Error updating chorus:', err);
    } finally {
      setLoading(false);
    }
  };

  const confirmDeleteBot = (bot, e) => {
    e.stopPropagation();
    setBotToDelete(bot);
    setShowDeleteModal(true);
  };

  const showChangeChorusModal = (bot, e) => {
    e.stopPropagation();
    setBotToChangeChorus(bot);
    setSelectedChorusId(bot.chorus_id || '');
    setShowChorusModal(true);
  };

  const handleCreateBot = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const data = await botService.createBot(newBot);
      setBots([...bots, data]);
      setCreateMode(false);
      setNewBot({
        name: '',
        dataset_id: '',
        chorus_id: '',
        system_instruction: 'You are a helpful AI assistant. Answer questions based on the provided context.'
      });
      setSuccess('Bot created successfully!');
      setTimeout(() => setSuccess(null), 3000);
      setLoading(false);
    } catch (err) {
      setError('Failed to create bot');
      setLoading(false);
    }
  };

  const handleSelectBot = (bot) => {
    setSelectedBot(bot);
    navigate(`/chat/${bot.id}`);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewBot({
      ...newBot,
      [name]: value
    });
  };

  const handleCreateNewChorus = () => {
    // First save the bot, then redirect to chorus creation
    if (newBot.name && newBot.dataset_id) {
      handleCreateBot({ preventDefault: () => {} });
      
      // Navigate to chorus configuration after bot is created
      // This needs to be improved to wait for bot creation
      setTimeout(() => {
        // Get the list of bots again to ensure we have the newly created one
        botService.getBots().then(updatedBots => {
          // Find the created bot (should be the most recently created one)
          const createdBot = updatedBots[updatedBots.length - 1];
          if (createdBot) {
            // Show the change chorus modal directly
            setBotToChangeChorus(createdBot);
            setSelectedChorusId(createdBot.chorus_id || '');
            setShowChorusModal(true);
          }
        }).catch(err => {
          console.error('Error getting updated bots:', err);
        });
      }, 1000);
    } else {
      setError('Please fill in the bot name and select a dataset first');
    }
  };

  return (
    <Container className="my-5">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h2><FaRobot className="me-2" />Your Bots</h2>
        <Button 
          variant={createMode ? "outline-secondary" : "primary"} 
          onClick={() => setCreateMode(!createMode)}
          className="rounded-pill"
        >
          {createMode ? <><FaTimes className="me-1" /> Cancel</> : <><FaPlus className="me-1" /> Create New Bot</>}
        </Button>
      </div>
      
      {error && <Alert variant="danger" onClose={() => setError(null)} dismissible>{error}</Alert>}
      {success && <Alert variant="success" onClose={() => setSuccess(null)} dismissible>{success}</Alert>}

      <Modal show={showDeleteModal} onHide={() => setShowDeleteModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Confirm Deletion</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          Are you sure you want to delete the bot "{botToDelete?.name}"? This action cannot be undone.
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowDeleteModal(false)}>
            Cancel
          </Button>
          <Button variant="danger" onClick={handleDeleteBot} disabled={loading}>
            {loading ? <Spinner animation="border" size="sm" /> : <><FaTrash className="me-1" /> Delete</>}
          </Button>
        </Modal.Footer>
      </Modal>

      <Modal show={showChorusModal} onHide={() => setShowChorusModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Change Chorus for {botToChangeChorus?.name}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Form.Group>
            <Form.Label>Select a Chorus</Form.Label>
            <Form.Select
              value={selectedChorusId}
              onChange={(e) => setSelectedChorusId(e.target.value)}
            >
              <option value="">No Chorus (Standard Mode)</option>
              {availableChoruses.map(chorus => (
                <option key={chorus.id} value={chorus.id}>
                  {chorus.name} ({chorus.response_model_count} models, {chorus.evaluator_model_count} evaluators)
                </option>
              ))}
            </Form.Select>
          </Form.Group>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowChorusModal(false)}>
            Cancel
          </Button>
          <Button variant="primary" onClick={handleChangeChorus} disabled={loading}>
            {loading ? <Spinner animation="border" size="sm" /> : 'Save Changes'}
          </Button>
        </Modal.Footer>
      </Modal>

      {createMode && (
        <Card className="mb-4 border-0 shadow-sm">
          <Card.Body>
            <Card.Title className="mb-3">Create a New Bot</Card.Title>
            <Form onSubmit={handleCreateBot}>
              <Row>
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>Bot Name</Form.Label>
                    <Form.Control
                      type="text"
                      name="name"
                      value={newBot.name}
                      onChange={handleInputChange}
                      required
                      placeholder="Give your bot a name"
                      className="rounded-pill"
                    />
                  </Form.Group>
                </Col>
                
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>Select Dataset</Form.Label>
                    <Form.Select
                      name="dataset_id"
                      value={newBot.dataset_id}
                      onChange={handleInputChange}
                      required
                      className="rounded-pill"
                    >
                      <option value="">Choose a dataset</option>
                      {datasets.map(dataset => (
                        <option key={dataset.id} value={dataset.id}>
                          {dataset.name} ({dataset.document_count} documents)
                        </option>
                      ))}
                    </Form.Select>
                  </Form.Group>
                </Col>
              </Row>

              <Row>
                <Col md={12}>
                  <Form.Group className="mb-3">
                    <Form.Label>
                      <FaUsers className="me-1" /> Model Chorus Configuration
                    </Form.Label>
                    <div className="d-flex">
                      <Form.Select
                        name="chorus_id"
                        value={newBot.chorus_id}
                        onChange={handleInputChange}
                        className="rounded-pill me-2"
                      >
                        <option value="">No Model Chorus (optional)</option>
                        {chorusConfigs.map(config => (
                          <option key={config.id} value={config.id}>
                            {config.name} - {config.botName}
                          </option>
                        ))}
                      </Form.Select>
                      <Button 
                        variant="outline-primary" 
                        className="rounded-pill"
                        onClick={() => navigate('/chorus')}
                      >
                        Browse
                      </Button>
                    </div>
                    <Form.Text className="text-muted">
                      A Model Chorus combines multiple AI models to produce higher quality responses.
                      You can select an existing configuration or create a new one after saving your bot.
                    </Form.Text>
                  </Form.Group>
                </Col>
              </Row>

              <Form.Group className="mb-3">
                <Form.Label>System Instructions</Form.Label>
                <Form.Control
                  as="textarea"
                  rows={4}
                  name="system_instruction"
                  value={newBot.system_instruction}
                  onChange={handleInputChange}
                  placeholder="Customize how your bot should behave"
                  className="rounded"
                />
                <Form.Text className="text-muted">
                  This defines your bot's personality and behavior. Be specific about how it should respond to questions.
                </Form.Text>
              </Form.Group>

              <div className="d-flex justify-content-end">
                <Button 
                  variant="outline-primary" 
                  className="px-4 rounded-pill me-2"
                  onClick={handleCreateNewChorus}
                >
                  Create Bot & Change Chorus
                </Button>
                <Button variant="success" type="submit" disabled={loading} className="px-4 rounded-pill">
                  {loading ? <><Spinner as="span" animation="border" size="sm" /> Creating...</> : 'Create Bot'}
                </Button>
              </div>
            </Form>
          </Card.Body>
        </Card>
      )}

      {loading && !createMode ? (
        <div className="text-center my-5">
          <Spinner animation="border" variant="primary" />
          <p className="mt-2 text-muted">Loading your bots...</p>
        </div>
      ) : (
        <Row className="mt-4">
          {bots.length === 0 ? (
            <Col>
              <div className="text-center my-5 py-5 bg-light rounded">
                <FaRobot size={50} className="mb-3 text-muted" />
                <h4>No bots yet</h4>
                <p className="text-muted">Create your first bot to start chatting with your data!</p>
                <Button 
                  variant="primary" 
                  onClick={() => setCreateMode(true)}
                  className="mt-2 rounded-pill"
                >
                  <FaPlus className="me-1" /> Create New Bot
                </Button>
              </div>
            </Col>
          ) : (
            <Col>
              <Row xs={1} md={2} lg={3} className="g-4">
                {bots.map(bot => (
                  <Col key={bot.id}>
                    <Card 
                      className="h-100 border-0 shadow-sm" 
                      onClick={() => handleSelectBot(bot)}
                      style={{ cursor: 'pointer' }}
                    >
                      <Card.Body>
                        <div className="d-flex justify-content-between align-items-start">
                          <div>
                            <h4 className="mb-2">{bot.name}</h4>
                            <p className="mb-3 text-muted small">
                              <FaDatabase className="me-1" />
                              {datasets.find(d => d.id === bot.dataset_id)?.name || 'Unknown dataset'}
                              {bot.chorus_id && (
                                <Badge className="ms-2" bg="info">
                                  <FaUsers className="me-1" />
                                  Model Chorus
                                </Badge>
                              )}
                            </p>
                          </div>
                          <div className="bot-icon">
                            <FaRobot size={24} />
                          </div>
                        </div>
                        <Card.Text className="small text-truncate">
                          {typeof bot.system_instruction === 'string' 
                            ? bot.system_instruction.substring(0, 100) 
                            : (bot.system_instruction 
                              ? String(bot.system_instruction).substring(0, 100) 
                              : 'No system instruction')}...
                        </Card.Text>
                      </Card.Body>
                      <Card.Footer className="bg-white border-top-0">
                        <div className="d-flex justify-content-between">
                          <Button 
                            variant="primary" 
                            className="rounded-pill flex-grow-1 me-1"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSelectBot(bot);
                            }}
                          >
                            <FaComment className="me-1" /> Chat Now
                          </Button>
                          <Button 
                            variant="outline-danger" 
                            className="rounded-pill me-1"
                            onClick={(e) => confirmDeleteBot(bot, e)}
                          >
                            <FaTrash className="me-1" />
                          </Button>
                          <Button 
                            variant="outline-secondary" 
                            className="rounded-pill"
                            onClick={(e) => showChangeChorusModal(bot, e)}
                          >
                            <FaExchangeAlt className="me-1" />
                          </Button>
                        </div>
                      </Card.Footer>
                    </Card>
                  </Col>
                ))}
              </Row>
            </Col>
          )}
        </Row>
      )}
    </Container>
  );
};

export default BotPanel; 