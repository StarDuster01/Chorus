import React, { useState, useEffect, useRef } from 'react';
import { Container, Row, Col, Form, Button, Card, Spinner, Alert, Dropdown, Modal, Badge } from 'react-bootstrap';
import { useParams, useNavigate } from 'react-router-dom';
import botService from '../services/botService';
import { FaRobot, FaBug, FaChevronLeft, FaCode, FaListAlt, FaTerminal, FaVoteYea, FaImage, FaTimes, FaMagic, FaEdit, FaUsers, FaCog, FaExchangeAlt, FaCheck } from 'react-icons/fa';

// Add some custom styles for the chorus UI
const chorusStyles = {
  activeChorus: {
    background: '#e8f4fe',
    borderColor: '#3498db',
    boxShadow: '0 0 0 2px rgba(52, 152, 219, 0.25)',
    fontWeight: 'bold',
    padding: '0.5rem 0.8rem',
  },
  chorusName: {
    display: 'inline-block',
    maxWidth: '150px',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    verticalAlign: 'middle',
  },
  chorusBadge: {
    position: 'absolute',
    top: '-8px',
    right: '-8px',
    fontSize: '0.7rem',
    padding: '0.2rem 0.4rem',
    borderRadius: '10px',
  },
  selectedChorusInfo: {
    background: '#f8f9fa',
    border: '1px solid #e9ecef',
    borderRadius: '8px',
    padding: '16px',
    marginTop: '16px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
  },
  modelCounts: {
    display: 'flex',
    gap: '10px',
    marginTop: '10px',
  },
  modelCountBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '5px',
    padding: '5px 10px',
    borderRadius: '50px',
    fontSize: '0.8rem',
    fontWeight: 'bold'
  }
};

const ChatInterface = () => {
  const { botId } = useParams();
  const navigate = useNavigate();
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  
  const [bot, setBot] = useState(null);
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [debugMode, setDebugMode] = useState(false);
  const [debugInfo, setDebugInfo] = useState(null);
  const [imageAttachment, setImageAttachment] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [lastMessageContent, setLastMessageContent] = useState('');
  
  // Add state for image generation mode
  const [imageGenMode, setImageGenMode] = useState(false);
  const [generatingImage, setGeneratingImage] = useState(false);
  const [imageGenPrompt, setImageGenPrompt] = useState('');
  const [, setGeneratedImage] = useState(null);
  
  // Add state for model chorus
  const [useModelChorus, setUseModelChorus] = useState(false);
  const [availableChorus, setAvailableChorus] = useState([]);
  const [selectedChorusId, setSelectedChorusId] = useState('');
  const [showChorusDropdown, setShowChorusDropdown] = useState(false);

  useEffect(() => {
    // Fetch bot details
    const loadBot = async () => {
      try {
        const bots = await botService.getBots();
        const foundBot = bots.find(b => b.id === botId);
        if (foundBot) {
          setBot(foundBot);
          
          // If bot has a chorus associated with it, enable model chorus by default
          if (foundBot.chorus_id) {
            setUseModelChorus(true);
            setSelectedChorusId(foundBot.chorus_id);
          }
          
          // Check dataset status
          try {
            const statusResult = await botService.checkDatasetStatus(foundBot.dataset_id);
            if (!statusResult.collection_exists || statusResult.document_count === 0) {
              setError(`Dataset issue detected: ${!statusResult.collection_exists ? 
                'Collection does not exist' : 'No documents in collection'}. Try rebuilding the dataset.`);
            }
          } catch (statusErr) {
            console.error('Error checking dataset status:', statusErr);
          }

          // Load available chorus configurations
          loadChorusConfigurations();
        } else {
          setError('Bot not found');
          setTimeout(() => navigate('/bots'), 3000);
        }
      } catch (err) {
        setError('Failed to load bot details');
      }
    };
    
    loadBot();
  }, [botId, navigate]);

  const loadChorusConfigurations = async () => {
    try {
      // Load all available choruses directly
      const choruses = await botService.getAllChoruses();
      
      // Make sure we never set to undefined or null
      setAvailableChorus(Array.isArray(choruses) ? choruses : []);
      
      if (!Array.isArray(choruses) || choruses.length === 0) {
        console.log('No choruses found or chorus data is invalid');
      }
    } catch (err) {
      console.error('Failed to load chorus configs:', err);
      setAvailableChorus([]);
    }
  };

  useEffect(() => {
    // Scroll to bottom whenever messages change
    // Use a small delay to account for image loading
    const scrollTimer = setTimeout(() => {
      scrollToBottom();
    }, 100);
    
    return () => clearTimeout(scrollTimer);
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleImageSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageAttachment(file);
      
      // Create a preview URL
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const removeImage = () => {
    setImageAttachment(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSelectChorus = async (chorusId) => {
    setSelectedChorusId(chorusId);
    setUseModelChorus(true);
    setShowChorusDropdown(false);
    
    // Optionally update the bot's associated chorus
    try {
      await botService.setBotChorus(botId, chorusId);
    } catch (err) {
      console.error('Error setting bot chorus:', err);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    
    if (loading) return;
    
    // Handle image generation mode
    if (imageGenMode) {
      if (!imageGenPrompt.trim()) return;
      await handleGenerateImage();
      return;
    }
    
    if (!message.trim() && !imageAttachment) return;
    
    // Store the message for later use if needed
    const currentMessage = message.trim();
    setLastMessageContent(currentMessage);
    
    setLoading(true);
    
    // Add user message to chat
    const newUserMessage = {
      id: Date.now(),
      sender: 'user',
      content: message,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, newUserMessage]);
    setMessage('');
    
    try {
      let response;
      
      if (imageAttachment) {
        // Create form data for image upload
        const formData = new FormData();
        formData.append('image', imageAttachment);
        formData.append('message', message);
        formData.append('debug_mode', debugMode);
        
        // Send message with image
        response = await botService.chatWithImage(botId, formData);
        
        // Add image preview to UI
        const userMessageWithImage = {
          ...newUserMessage,
          content: (
            <div>
              <div className="mb-2">{message}</div>
              <img 
                src={imagePreview} 
                alt="Uploaded" 
                style={{ maxHeight: '200px', maxWidth: '100%', borderRadius: '8px' }} 
              />
            </div>
          )
        };
        
        // Replace the text-only message with one containing the image
        setMessages(prev => [...prev.slice(0, prev.length - 1), userMessageWithImage]);
        
        // Clear image attachment
        removeImage();
      } else {
        // Use selected chorus ID if available
        const effectiveChorusId = selectedChorusId || (bot && bot.chorus_id) || '';
        
        // Send text-only message
        response = await botService.chatWithBot(botId, currentMessage, debugMode, useModelChorus, effectiveChorusId);
      }
      
      // Add bot response to chat
      const botMessage = {
        id: Date.now(),
        sender: 'bot',
        content: response.response,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, botMessage]);
      
      // Store debug info if in debug mode
      if (debugMode && response.debug) {
        setDebugInfo(response.debug);
      }
      
      // Update lastMessageContent
      setLastMessageContent(currentMessage);
      
    } catch (err) {
      console.error('Error in chat:', err);
      setError(`Error: ${err.message || 'Failed to send message'}`);
    } finally {
      setLoading(false);
      scrollToBottom();
    }
  };

  // Add function to handle image generation
  const handleGenerateImage = async () => {
    if (!imageGenPrompt.trim() || generatingImage) return;
    
    setGeneratingImage(true);
    
    try {
      // Add user prompt to chat
      const newUserMessage = {
        id: Date.now(),
        sender: 'user',
        content: `Generate image: ${imageGenPrompt}`,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, newUserMessage]);
      
      // Call image generation API
      const result = await botService.generateImage(imageGenPrompt, {
        model: "gpt-image-1",
        size: "1024x1024",
        quality: "medium"
      });
      
      if (result.image_url) {
        const fullUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000' + result.image_url;
        setGeneratedImage(fullUrl);
        
        // Add bot response with the generated image
        const botMessage = {
          id: Date.now(),
          sender: 'bot',
          content: (
            <div>
              <p>Here's the generated image:</p>
              <img 
                src={fullUrl} 
                alt="AI Generated" 
                style={{ maxHeight: '300px', maxWidth: '100%', borderRadius: '8px' }} 
              />
              <div className="mt-2">
                <a 
                  href={fullUrl} 
                  target="_blank" 
                  rel="noreferrer" 
                  className="btn btn-sm btn-outline-primary"
                >
                  Open Full Size
                </a>
                <Button 
                  variant="outline-secondary" 
                  size="sm" 
                  className="ms-2"
                  onClick={() => {
                    // Clear prompt but stay in image gen mode
                    setImageGenPrompt('');
                  }}
                >
                  Generate Another
                </Button>
              </div>
            </div>
          ),
          timestamp: new Date()
        };
        
        setMessages(prev => [...prev, botMessage]);
      }
    } catch (err) {
      console.error('Error generating image:', err);
      
      // Add error message
      const errorMessage = {
        id: Date.now(),
        sender: 'bot',
        content: `Sorry, I couldn't generate that image: ${err.message || 'Unknown error'}`,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setGeneratingImage(false);
      setImageGenPrompt('');
      scrollToBottom();
    }
  };
  
  // Toggle between chat and image generation mode
  const toggleImageGenMode = () => {
    setImageGenMode(!imageGenMode);
    if (imageGenMode) {
      setImageGenPrompt('');
    }
  };

  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <Container fluid className="p-0">
      {error && (
        <Alert variant="danger" className="m-3">
          {error}
          {bot && (
            <Button 
              variant="outline-danger" 
              size="sm" 
              className="ms-3"
              onClick={async () => {
                try {
                  setError("Rebuilding dataset collection...");
                  await botService.rebuildDataset(bot.dataset_id);
                  setError("Dataset rebuilt successfully. Please upload documents to this dataset.");
                } catch (err) {
                  setError(`Failed to rebuild dataset: ${err.message}`);
                }
              }}
            >
              Rebuild Dataset
            </Button>
          )}
        </Alert>
      )}
      
      {bot ? (
        <div className="chat-wrapper">
          {/* Debug Panel - Left Side (when in debug mode) */}
          {debugMode && (
            <div className="debug-panel">
              <Card className="border-0 shadow-sm h-100">
                <Card.Header className="d-flex justify-content-between align-items-center">
                  <h5 className="mb-0"><FaBug className="me-2" />Debug Mode</h5>
                  <Button 
                    variant="link" 
                    className="p-0 text-muted" 
                    onClick={() => navigate('/bots')}
                    title="Back to Bots"
                  >
                    <FaChevronLeft />
                  </Button>
                </Card.Header>
                <Card.Body className="p-0 overflow-auto">
                  {debugInfo ? (
                    <>
                      <div className="debug-section">
                        <div className="debug-header">
                          <span><FaCode className="me-2" />Context Documents</span>
                        </div>
                        <div className="debug-content">
                          {debugInfo.contexts.map((ctx, idx) => (
                            <div key={idx} className="debug-context-item">
                              <strong>Context {idx + 1}:</strong>
                              <p className="mb-0">{typeof ctx === 'string' ? ctx.substring(0, 150) : JSON.stringify(ctx).substring(0, 150)}...</p>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      {debugInfo.all_responses && (
                        <div className="debug-section">
                          <div className="debug-header">
                            <span><FaVoteYea className="me-2" />Response Voting</span>
                          </div>
                          <div className="debug-content">
                            {debugInfo.all_responses.map((resp, idx) => (
                              <div key={idx} className="debug-response-item">
                                <strong>Response {idx + 1} (Votes: {debugInfo.votes[idx]}) - {resp.provider} {resp.model}</strong>
                                <p className="mb-0">{typeof resp === 'string' ? resp.substring(0, 100) : (resp.response && typeof resp.response === 'string' ? resp.response.substring(0, 100) : JSON.stringify(resp).substring(0, 100))}...</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {debugInfo.logs && (
                        <div className="debug-section">
                          <div className="debug-header">
                            <span><FaTerminal className="me-2" />Processing Logs</span>
                          </div>
                          <div className="debug-content">
                            <div className="debug-logs">
                              {debugInfo.logs.map((log, idx) => (
                                <div key={idx}>{log}</div>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="p-4 text-center text-muted">
                      <p>Send a message to see debug information</p>
                    </div>
                  )}
                </Card.Body>
              </Card>
            </div>
          )}
          
          {/* Chat Main Area */}
          <div className="chat-main">
            <Card className="chat-container border-0 shadow-sm h-100">
              <Card.Header className="d-flex justify-content-between align-items-center">
                <div className="d-flex align-items-center">
                  <Button 
                    variant="link" 
                    className="p-0 text-dark me-3" 
                    onClick={() => navigate('/bots')}
                    title="Back to Bots"
                  >
                    <FaChevronLeft />
                  </Button>
                  <div>
                    <h3 className="mb-0">{bot.name}</h3>
                    <p className="mb-0 text-muted small">RAG-powered chat bot</p>
                  </div>
                </div>
                <div>
                  <Button 
                    variant={debugMode ? "primary" : "outline-primary"} 
                    className="me-2"
                    onClick={() => setDebugMode(!debugMode)}
                    title={debugMode ? "Disable Debug Mode" : "Enable Debug Mode"}
                  >
                    <FaBug />
                  </Button>
                  <Button
                    variant="outline-primary"
                    onClick={() => setShowChorusDropdown(true)}
                    title="Change Model Chorus"
                  >
                    <FaExchangeAlt className="me-1" /><FaUsers />
                  </Button>
                </div>
              </Card.Header>
              
              <Card.Body className="chat-messages">
                {messages.length === 0 ? (
                  <div className="text-center my-5 text-muted">
                    <FaRobot size={50} className="mb-3" />
                    <h4>Start chatting with {bot.name}!</h4>
                    <p className="small">This bot uses RAG (Retrieval Augmented Generation) to answer your questions based on specific documents.</p>
                  </div>
                ) : (
                  messages.map(msg => (
                    <div 
                      key={msg.id} 
                      className={`message ${msg.sender === 'user' ? 'user-message' : 'bot-message'}`}
                    >
                      <div className="message-content">
                        {typeof msg.content === 'object' ? msg.content : msg.content}
                      </div>
                      <div className="message-time">
                        {formatTime(msg.timestamp)}
                      </div>
                    </div>
                  ))
                )}
                
                {loading && (
                  <div className="bot-message">
                    <div className="message-content thinking-indicator">
                      <Spinner animation="border" size="sm" className="me-2" /> 
                      Thinking...
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </Card.Body>
              
              <Card.Footer className="p-3">
                {/* Mode toggle buttons */}
                <div className="mb-2 d-flex justify-content-between align-items-center">
                  <div>
                    <Button 
                      variant={imageGenMode ? "outline-primary" : "primary"} 
                      size="sm" 
                      className="me-2" 
                      onClick={() => {
                        setImageGenMode(false);
                        removeImage();
                      }}
                      title="Text Chat Mode"
                    >
                      <FaRobot className="me-1" /> Chat
                    </Button>
                    <Button 
                      variant={imageGenMode ? "primary" : "outline-primary"} 
                      size="sm" 
                      className="me-2"
                      onClick={toggleImageGenMode}
                      title="Image Generation Mode"
                    >
                      <FaMagic className="me-1" /> Generate Images
                    </Button>
                    <Dropdown className="d-inline-block me-2 position-relative">
                      {useModelChorus && (
                        <Badge 
                          bg="info" 
                          style={chorusStyles.chorusBadge}
                        >
                          CHORUS
                        </Badge>
                      )}
                      <Dropdown.Toggle
                        variant={useModelChorus ? "primary" : "outline-primary"}
                        size="sm"
                        id="dropdown-chorus"
                        style={useModelChorus ? chorusStyles.activeChorus : {}}
                        title={useModelChorus ? `Using Model Chorus: ${availableChorus?.find(c => c.id === selectedChorusId)?.name || 'Active'}` : "Enable Model Chorus"}
                      >
                        <FaUsers className="me-1" /> 
                        {useModelChorus ? (
                          <span style={chorusStyles.chorusName}>
                            {availableChorus?.find(c => c.id === selectedChorusId)?.name || 'Model Chorus'}
                          </span>
                        ) : (
                          'Model Chorus'
                        )}
                      </Dropdown.Toggle>

                      <Dropdown.Menu>
                        <Dropdown.Header className="fw-bold">Select Model Chorus</Dropdown.Header>
                        {!availableChorus || availableChorus.length === 0 ? (
                          <Dropdown.Item disabled>No chorus configurations found</Dropdown.Item>
                        ) : (
                          availableChorus.map(chorus => (
                            <Dropdown.Item 
                              key={chorus.id} 
                              active={selectedChorusId === chorus.id}
                              onClick={() => handleSelectChorus(chorus.id)}
                              className={selectedChorusId === chorus.id ? "bg-primary text-white" : ""}
                            >
                              {selectedChorusId === chorus.id && <FaCheck className="me-2" />}
                              {chorus.name} 
                              <Badge bg="info" className="ms-2">
                                {chorus.response_model_count || 0} models
                              </Badge>
                            </Dropdown.Item>
                          ))
                        )}
                        <Dropdown.Divider />
                        <Dropdown.Item 
                          onClick={() => setUseModelChorus(false)}
                          className={!useModelChorus ? "bg-light fw-bold" : ""}
                        >
                          <FaTimes className="me-1" /> Disable Model Chorus
                        </Dropdown.Item>
                        <Dropdown.Item onClick={() => setShowChorusDropdown(true)}>
                          <FaCog className="me-1" /> Manage Choruses
                        </Dropdown.Item>
                      </Dropdown.Menu>
                    </Dropdown>
                  </div>
                  <div>
                    <Button
                      variant={debugMode ? "danger" : "outline-danger"}
                      size="sm"
                      onClick={() => setDebugMode(!debugMode)}
                      title={debugMode ? "Disable Debug Mode" : "Enable Debug Mode"}
                    >
                      <FaBug />
                    </Button>
                  </div>
                </div>
                
                <Form onSubmit={handleSendMessage}>
                  {imageGenMode ? (
                    <Form.Group className="mb-3">
                      <Form.Label>Describe the image you want to generate</Form.Label>
                      <Form.Control
                        as="textarea"
                        rows={3}
                        value={imageGenPrompt}
                        onChange={(e) => setImageGenPrompt(e.target.value)}
                        placeholder="Describe the image you want to create..."
                        disabled={generatingImage}
                      />
                    </Form.Group>
                  ) : (
                    <>
                      {imageAttachment && (
                        <div className="mb-3 position-relative d-inline-block">
                          <img 
                            src={imagePreview} 
                            alt="Preview" 
                            style={{ 
                              maxHeight: '100px', 
                              maxWidth: '200px', 
                              borderRadius: '8px',
                              border: '1px solid #ddd' 
                            }} 
                          />
                          <Button 
                            variant="danger" 
                            size="sm" 
                            className="position-absolute top-0 end-0" 
                            style={{ margin: '5px' }}
                            onClick={removeImage}
                          >
                            <FaTimes />
                          </Button>
                        </div>
                      )}
                    
                      <div className="d-flex gap-2">
                        <Form.Control
                          type="text"
                          value={message}
                          onChange={(e) => setMessage(e.target.value)}
                          placeholder="Type your message..."
                          disabled={loading}
                        />
                        
                        <Button 
                          variant="outline-secondary"
                          onClick={() => fileInputRef.current?.click()}
                          title="Attach Image"
                        >
                          <FaImage />
                        </Button>
                        
                        <Form.Control
                          type="file"
                          ref={fileInputRef}
                          accept="image/*"
                          className="d-none"
                          onChange={handleImageSelect}
                        />
                      </div>
                    </>
                  )}
                  
                  <div className="mt-3 d-flex justify-content-end">
                    <Button 
                      variant="primary" 
                      type="submit" 
                      disabled={loading || generatingImage || (imageGenMode && !imageGenPrompt.trim()) || (!imageGenMode && !message.trim() && !imageAttachment)}
                    >
                      {loading || generatingImage ? (
                        <>
                          <Spinner as="span" animation="border" size="sm" className="me-2" />
                          {imageGenMode ? 'Generating...' : 'Sending...'}
                        </>
                      ) : (
                        imageGenMode ? 'Generate Image' : 'Send'
                      )}
                    </Button>
                  </div>
                </Form>
              </Card.Footer>
            </Card>
          </div>
        </div>
      ) : (
        <div className="text-center my-5">
          {!error && <Spinner animation="border" variant="primary" />}
        </div>
      )}

      {/* Add the Chorus Selection Modal */}
      {bot && (
        <Modal show={showChorusDropdown} onHide={() => setShowChorusDropdown(false)}>
          <Modal.Header closeButton className="bg-light">
            <Modal.Title>
              <FaUsers className="me-2" />
              Change Chorus for {bot.name}
            </Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <Form.Group>
              <Form.Label>Select a Chorus</Form.Label>
              <Form.Select
                value={selectedChorusId}
                onChange={(e) => setSelectedChorusId(e.target.value)}
                className="mb-3"
                size="lg"
              >
                <option value="">No Chorus (Standard Mode)</option>
                {availableChorus && availableChorus.length > 0 ? (
                  availableChorus.map(chorus => (
                    <option key={chorus.id} value={chorus.id}>
                      {chorus.name} ({chorus.response_model_count || 0} models, {chorus.evaluator_model_count || 0} evaluators)
                    </option>
                  ))
                ) : (
                  <option disabled>No available choruses</option>
                )}
              </Form.Select>
              
              {selectedChorusId && availableChorus && (
                <div style={chorusStyles.selectedChorusInfo}>
                  <h5 className="mb-1">{availableChorus.find(c => c.id === selectedChorusId)?.name}</h5>
                  <p className="text-muted mb-2">
                    {availableChorus.find(c => c.id === selectedChorusId)?.description || 'No description available'}
                  </p>
                  <div style={chorusStyles.modelCounts}>
                    <div style={{...chorusStyles.modelCountBadge, background: '#e8f4fe', color: '#2980b9'}}>
                      <FaRobot size={14} /> {availableChorus.find(c => c.id === selectedChorusId)?.response_model_count || 0} Response Models
                    </div>
                    <div style={{...chorusStyles.modelCountBadge, background: '#f8e9fb', color: '#8e44ad'}}>
                      <FaVoteYea size={14} /> {availableChorus.find(c => c.id === selectedChorusId)?.evaluator_model_count || 0} Evaluators
                    </div>
                  </div>
                </div>
              )}
              
              {!selectedChorusId && (
                <Alert variant="info" className="mt-3">
                  <FaRobot className="me-2" />
                  Without a chorus, the bot will use a single model to answer questions.
                </Alert>
              )}
            </Form.Group>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setShowChorusDropdown(false)}>
              Cancel
            </Button>
            <Button 
              variant="primary" 
              onClick={async () => {
                try {
                  await botService.setBotChorus(botId, selectedChorusId);
                  setUseModelChorus(selectedChorusId !== '');
                  setShowChorusDropdown(false);
                } catch (err) {
                  console.error('Error setting chorus:', err);
                  setError('Error setting chorus configuration');
                }
              }}
            >
              Apply Chorus
            </Button>
          </Modal.Footer>
        </Modal>
      )}
    </Container>
  );
};

export default ChatInterface; 