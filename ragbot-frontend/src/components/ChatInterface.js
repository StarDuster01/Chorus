import React, { useState, useEffect, useRef } from 'react';
import { Container, Row, Col, Form, Button, Card, Spinner, Alert } from 'react-bootstrap';
import { useParams, useNavigate } from 'react-router-dom';
import botService from '../services/botService';
import { FaRobot, FaBug, FaChevronLeft, FaCode, FaListAlt, FaTerminal, FaVoteYea, FaImage, FaTimes, FaProjectDiagram } from 'react-icons/fa';
import MermaidDiagram from './MermaidDiagram';

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
  const [flowchartData, setFlowchartData] = useState(null);
  const [loadingFlowchart, setLoadingFlowchart] = useState(false);
  const [lastMessageContent, setLastMessageContent] = useState('');

  useEffect(() => {
    // Fetch bot details
    const loadBot = async () => {
      try {
        const bots = await botService.getBots();
        const foundBot = bots.find(b => b.id === botId);
        if (foundBot) {
          setBot(foundBot);
          
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

  const handleSendMessage = async (e) => {
    e.preventDefault();
    
    if (loading) return;
    if (!message.trim() && !imageAttachment) return;
    
    // Store the message for flowchart generation if needed later
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
        // Send text-only message
        response = await botService.chatWithBot(botId, currentMessage, debugMode);
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
        
        // Generate flowchart for the debug visualization
        if (currentMessage) {
          generateFlowchart(currentMessage);
        }
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

  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const generateFlowchart = async (messageToAnalyze) => {
    if (!messageToAnalyze || !messageToAnalyze.trim() || loadingFlowchart) return;
    
    console.log("Generating flowchart for message:", messageToAnalyze);
    setLoadingFlowchart(true);
    setFlowchartData(null); // Clear any existing flowchart
    
    try {
      const response = await botService.generateDebugFlowchart(botId, messageToAnalyze);
      
      // Validate response structure
      if (!response || !response.mermaid_code || !response.data) {
        console.error("Invalid flowchart response format:", response);
        throw new Error("Received invalid flowchart data from server");
      }
      
      console.log("Successfully generated flowchart");
      
      // Ensure the flowchart data is properly formatted with all necessary components
      const responseData = {
        ...response,
        data: {
          ...response.data,
          // Default any missing data to prevent UI errors
          votes: response.data.votes || [0, 0, 0, 0, 0],
          responses: response.data.responses || [],
          logs: response.data.logs || []
        }
      };
      
      setFlowchartData(responseData);
    } catch (err) {
      console.error('Failed to generate flowchart:', err);
      // Don't display the error in the UI since this is a secondary feature
      // Just log it to console for debugging
    } finally {
      setLoadingFlowchart(false);
    }
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
                              <p className="mb-0">{ctx.substring(0, 150)}...</p>
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
                                <strong>Response {idx + 1} (Votes: {debugInfo.votes[idx]})</strong>
                                <p className="mb-0">{resp.substring(0, 100)}...</p>
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
                      
                      {/* Add the flowchart section */}
                      <div className="debug-section">
                        <div className="debug-header">
                          <span><FaProjectDiagram className="me-2" />Decision Process Flowchart</span>
                        </div>
                        <div className="debug-content">
                          {loadingFlowchart ? (
                            <div className="text-center p-4">
                              <Spinner animation="border" size="sm" className="me-2" />
                              <span>Generating flowchart...</span>
                            </div>
                          ) : flowchartData ? (
                            <div className="flowchart-container">
                              <MermaidDiagram chart={flowchartData.mermaid_code} />
                              
                              {/* Add response comparison */}
                              <div className="mt-4">
                                <h6>Response Comparison</h6>
                                <div className="response-comparison">
                                  {flowchartData.data.responses.map((response, idx) => {
                                    // Find winner index - handle possible array emptiness
                                    const votes = flowchartData.data.votes || [0, 0, 0, 0, 0];
                                    const maxVotes = Math.max(...votes);
                                    const winnerIndices = votes.map((v, i) => v === maxVotes ? i : -1).filter(i => i !== -1);
                                    const isWinner = winnerIndices.includes(idx);
                                    
                                    return (
                                      <div 
                                        key={idx} 
                                        className={`response-item ${isWinner ? 'winner' : ''}`}
                                      >
                                        <h6>Response {idx + 1} (Votes: {votes[idx] || 0})</h6>
                                        <p>{response.substring(0, 150)}...</p>
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                              
                              {/* Add button to regenerate flowchart if needed */}
                              <div className="text-center mt-3">
                                <Button 
                                  variant="outline-secondary" 
                                  size="sm" 
                                  onClick={() => generateFlowchart(lastMessageContent)}
                                  disabled={loadingFlowchart}
                                >
                                  {loadingFlowchart ? (
                                    <>
                                      <Spinner animation="border" size="sm" className="me-1" />
                                      Regenerating...
                                    </>
                                  ) : (
                                    <>Regenerate Flowchart</>
                                  )}
                                </Button>
                              </div>
                            </div>
                          ) : (
                            <div className="text-center p-4 text-muted">
                              <p>Flowchart will appear here after a response</p>
                            </div>
                          )}
                        </div>
                      </div>
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
                <Form onSubmit={handleSendMessage}>
                  {imagePreview && (
                    <div className="image-preview mb-3 position-relative">
                      <img 
                        src={imagePreview} 
                        alt="Preview" 
                        style={{ maxHeight: '150px', maxWidth: '100%', borderRadius: '8px' }} 
                      />
                      <Button 
                        variant="danger" 
                        size="sm" 
                        className="position-absolute" 
                        style={{ top: '5px', right: '5px', borderRadius: '50%', padding: '4px 8px' }}
                        onClick={removeImage}
                      >
                        <FaTimes />
                      </Button>
                    </div>
                  )}
                  <Row className="align-items-center">
                    <Col xs={9}>
                      <Form.Control
                        type="text"
                        placeholder="Type your message..."
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        disabled={loading}
                        className="rounded-pill"
                      />
                    </Col>
                    <Col xs={1} className="p-0 text-center">
                      <Button 
                        variant="outline-secondary" 
                        className="rounded-circle p-2" 
                        onClick={() => fileInputRef.current.click()}
                        disabled={loading}
                        title="Attach image"
                      >
                        <FaImage />
                      </Button>
                      <Form.Control
                        type="file"
                        accept="image/*"
                        ref={fileInputRef}
                        style={{ display: 'none' }}
                        onChange={handleImageSelect}
                      />
                    </Col>
                    <Col xs={2}>
                      <Button 
                        variant="primary" 
                        type="submit" 
                        className="w-100 rounded-pill" 
                        disabled={loading || (!message.trim() && !imageAttachment)}
                      >
                        Send
                      </Button>
                    </Col>
                  </Row>
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
    </Container>
  );
};

export default ChatInterface; 