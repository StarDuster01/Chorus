import React, { useState, useEffect } from 'react';
import { Card, Button, Container, Row, Col, Form, ListGroup, Badge, Alert, Spinner } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import botService from '../services/botService';
import { FaRobot, FaPlus, FaTimes, FaDatabase, FaComment } from 'react-icons/fa';

const BotPanel = () => {
  const [bots, setBots] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [selectedBot, setSelectedBot] = useState(null);
  const [createMode, setCreateMode] = useState(false);
  const [newBot, setNewBot] = useState({
    name: '',
    dataset_id: '',
    system_instruction: 'You are a helpful AI assistant. Answer questions based on the provided context.'
  });
  
  const navigate = useNavigate();

  useEffect(() => {
    loadBots();
    loadDatasets();
  }, []);

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
                            </p>
                          </div>
                          <div className="bot-icon">
                            <FaRobot size={24} />
                          </div>
                        </div>
                        <Card.Text className="small text-truncate">
                          {bot.system_instruction.substring(0, 100)}...
                        </Card.Text>
                      </Card.Body>
                      <Card.Footer className="bg-white border-top-0">
                        <Button 
                          variant="primary" 
                          className="w-100 rounded-pill"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleSelectBot(bot);
                          }}
                        >
                          <FaComment className="me-1" /> Chat Now
                        </Button>
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