import React, { useState, useEffect, useRef } from 'react';
import { Container, Row, Col, Form, Button, Card, Alert, Dropdown, Modal, Badge, OverlayTrigger, Tooltip } from 'react-bootstrap';
import { useParams, useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import botService from '../services/botService';
import LoadingAnimation from './LoadingAnimation';
import TesseractLoader from './TesseractLoader';
import '../markdown-styles.css';
import { FaRobot, FaBug, FaChevronLeft, FaCode, FaListAlt, FaTerminal, FaVoteYea, FaImage, FaTimes, FaMagic, FaEdit, FaUsers, FaCog, FaExchangeAlt, FaCheck, FaLightbulb, FaFileAlt, FaDatabase, FaPlus } from 'react-icons/fa';
import { v4 as uuid } from 'uuid';

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

// Add styles for markdown content
const markdownStyles = {
  container: {
    fontFamily: 'inherit',
  },
  code: {
    backgroundColor: '#f6f8fa',
    padding: '0.2em 0.4em',
    borderRadius: '3px',
    fontFamily: 'SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace',
    fontSize: '85%',
  },
  pre: {
    backgroundColor: '#f6f8fa',
    padding: '1em',
    borderRadius: '6px',
    overflow: 'auto',
    fontSize: '85%',
  },
  blockquote: {
    borderLeft: '4px solid #dfe2e5',
    color: '#6a737d',
    paddingLeft: '1em',
    margin: '0',
  },
  table: {
    borderCollapse: 'collapse',
    width: '100%',
    marginBottom: '1em',
  },
  th: {
    border: '1px solid #dfe2e5',
    padding: '6px 13px',
    backgroundColor: '#f6f8fa',
    fontWeight: '600',
  },
  td: {
    border: '1px solid #dfe2e5',
    padding: '6px 13px',
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
  
  // Add state for model chorus
  const [useModelChorus, setUseModelChorus] = useState(false);
  const [availableChorus, setAvailableChorus] = useState([]);
  const [selectedChorusId, setSelectedChorusId] = useState('');
  const [showChorusDropdown, setShowChorusDropdown] = useState(false);

  // Add state for conversations
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState('');
  const [showConversations, setShowConversations] = useState(false);
  const [loadingConversations, setLoadingConversations] = useState(false);
  const [conversationRenameMode, setConversationRenameMode] = useState(false);
  const [newConversationTitle, setNewConversationTitle] = useState('');

  // Add state for enhance prompt functionality
  const [enhancingPrompt, setEnhancingPrompt] = useState(false);

  // Add state for datasets
  const [availableDatasets, setAvailableDatasets] = useState([]);
  const [botDatasets, setBotDatasets] = useState([]);
  const [showDatasetsDropdown, setShowDatasetsDropdown] = useState(false);
  const [loadingDatasets, setLoadingDatasets] = useState(false);

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
          
          // Check dataset status for each dataset associated with the bot
          try {
            const botDatasets = await botService.getBotDatasets(botId);
            if (botDatasets && botDatasets.length > 0) {
              for (const dataset of botDatasets) {
                const statusResult = await botService.checkDatasetStatus(dataset.id);
                if (!statusResult.collection_exists || statusResult.document_count === 0) {
                  console.warn(`Dataset ${dataset.id} issue: ${!statusResult.collection_exists ? 
                    'Collection does not exist' : 'No documents in collection'}`);
                }
              }
            }
          } catch (statusErr) {
            console.error('Error checking dataset status:', statusErr);
          }

          // Load available chorus configurations
          loadChorusConfigurations();
          
          // Load conversation history
          loadConversations();
          
          // Load bot datasets
          loadBotDatasets();
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

  // Load conversations for this bot
  const loadConversations = async () => {
    try {
      setLoadingConversations(true);
      const conversationList = await botService.getConversations(botId);
      setConversations(conversationList);
      setLoadingConversations(false);
    } catch (err) {
      console.error('Error loading conversations:', err);
      setLoadingConversations(false);
    }
  };
  
  // Load a specific conversation
  const loadConversation = async (conversationId) => {
    try {
      setLoading(true);
      const conversation = await botService.getConversation(botId, conversationId);
      
      // Set the current conversation ID
      setCurrentConversationId(conversationId);
      
      // Clear any existing messages and load the conversation messages
      setMessages(conversation.messages || []);
      
      // Hide the conversation list
      setShowConversations(false);
      setLoading(false);
    } catch (err) {
      console.error('Error loading conversation:', err);
      setError('Failed to load conversation');
      setLoading(false);
    }
  };
  
  // Delete a conversation
  const deleteConversation = async (conversationId) => {
    if (!window.confirm('Are you sure you want to delete this conversation?')) {
      return;
    }
    
    try {
      await botService.deleteConversation(botId, conversationId);
      
      // If this was the current conversation, clear it
      if (conversationId === currentConversationId) {
        setCurrentConversationId('');
        setMessages([]);
      }
      
      // Reload conversations
      loadConversations();
    } catch (err) {
      console.error('Error deleting conversation:', err);
      setError('Failed to delete conversation');
    }
  };
  
  // Start a new conversation
  const startNewConversation = () => {
    setCurrentConversationId('');
    setMessages([]);
    setShowConversations(false);
  };

  // Rename conversation
  const handleRenameConversation = async (e) => {
    e.preventDefault();
    
    if (!newConversationTitle.trim()) {
      return;
    }
    
    try {
      await botService.renameConversation(botId, currentConversationId, newConversationTitle);
      
      // Exit rename mode
      setConversationRenameMode(false);
      
      // Reload conversations to show the new title
      loadConversations();
    } catch (err) {
      console.error('Error renaming conversation:', err);
      setError('Failed to rename conversation');
    }
  };

  // Delete all conversations
  const handleDeleteAllConversations = async () => {
    if (!window.confirm('Are you sure you want to delete ALL conversations? This cannot be undone.')) {
      return;
    }
    
    try {
      await botService.deleteAllConversations(botId);
      
      // Clear current conversation
      setCurrentConversationId('');
      setMessages([]);
      
      // Reload (now empty) conversations
      loadConversations();
    } catch (err) {
      console.error('Error deleting all conversations:', err);
      setError('Failed to delete all conversations');
    }
  };

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

  const handleDownloadDocument = async (doc) => {
    try {
      await botService.downloadDocument(doc.dataset_id, doc.id, doc.filename);
    } catch (err) {
      setError('Failed to download document');
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!message.trim() && !imageAttachment) return;

    const currentMessage = message;
    setMessage('');
    setLastMessageContent(currentMessage);

    // Create and add user message
    const newUserMessage = {
      id: uuid(),
      role: 'user',
      content: currentMessage,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, newUserMessage]);
    setLoading(true);
    setError(null);

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
        setMessages(prev => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1] = userMessageWithImage;
          return newMessages;
        });
        
        // Clear the image attachment
        removeImage();
      } else {
        // Determine which chorus ID to use (if any)
        const effectiveChorusId = selectedChorusId || (bot && bot.chorus_id ? bot.chorus_id : '');
        
        // Send message to bot
        response = await botService.chatWithBot(
          botId, 
          currentMessage, 
          debugMode, 
          useModelChorus, 
          effectiveChorusId,
          currentConversationId
        );
      }
      
      // Create bot response message with source documents if available
      const botResponseContent = response.source_documents ? (
        <div>
          <div className="mb-3">{response.response}</div>
          {response.source_documents.length > 0 && (
            <div className="source-documents mt-2 p-2 border-top">
              <div className="text-muted mb-2">Source Documents:</div>
              {response.source_documents.map((doc, index) => (
                <div key={index} className="source-document">
                  <Button
                    variant="link"
                    className="p-0 text-decoration-none"
                    onClick={() => handleDownloadDocument(doc)}
                  >
                    <FaFileAlt className="me-1" />
                    {doc.filename}
                  </Button>
                </div>
              ))}
            </div>
          )}
          
          {/* Add support for displaying referenced images */}
          {response.image_details && response.image_details.length > 0 && (
            <div className="source-images mt-3 p-2 border-top">
              <div className="text-muted mb-2">Referenced Images:</div>
              <div className="d-flex flex-wrap gap-3">
                {response.image_details.map((img, index) => (
                  <div key={index} className="source-image">
                    <div className="mb-1 text-center small fw-bold">
                      {img.index}
                    </div>
                    <a 
                      href={img.download_url || img.url} 
                      target="_blank" 
                      rel="noreferrer"
                      title={img.caption || "View full image"}
                    >
                      <img 
                        src={img.url} 
                        alt={img.caption || "Referenced image"} 
                        style={{ 
                          maxWidth: '100%', 
                          maxHeight: '200px', 
                          borderRadius: '8px', 
                          boxShadow: '0 2px 4px rgba(0,0,0,0.1)' 
                        }}
                      />
                    </a>
                    {img.caption && (
                      <div className="mt-1 small text-muted">
                        {img.caption}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : response.response;

      const botMessage = {
        id: uuid(),
        role: 'assistant',
        content: botResponseContent,
        timestamp: new Date().toISOString(),
        source_documents: response.source_documents,
        image_details: response.image_details // Store image details in the message
      };

      setMessages(prev => [...prev, botMessage]);
      
      // Update the current conversation ID if a new one was created
      if (response.conversation_id && !currentConversationId) {
        setCurrentConversationId(response.conversation_id);
      }
      
      // If debug mode is enabled and we have debug info, set it
      if (debugMode && response.debug) {
        setDebugInfo(response.debug);
      }
      
      // Reload conversations to reflect the new or updated conversation
      loadConversations();
    } catch (err) {
      console.error('Error in chat:', err);
      setError(`Error: ${err.message || 'Failed to send message'}`);
    } finally {
      setLoading(false);
    }
  };

  // Add function to handle enhance prompt
  const handleEnhancePrompt = async () => {
    // Check if we're in image generation mode or regular chat mode
    const promptText = message;
    
    if (!promptText.trim() || enhancingPrompt) return;
    
    try {
      setEnhancingPrompt(true);
      console.log('Starting prompt enhancement for:', promptText);
      
      // Call API to enhance the prompt
      const result = await botService.enhancePrompt(promptText);
      console.log('Enhancement result:', result);
      
      if (result && result.success && result.enhanced_prompt) {
        console.log('Setting enhanced prompt:', result.enhanced_prompt);
        
        // Update the appropriate state variable based on the current mode
        setMessage(result.enhanced_prompt);
        
        // Add a temporary success message
        setError('Prompt enhanced successfully!');
        setTimeout(() => setError(''), 2000);
      } else {
        console.warn('Enhance prompt response missing enhanced_prompt property:', result);
        throw new Error('Invalid response format');
      }
    } catch (err) {
      console.error('Error enhancing prompt:', err);
      setError('Failed to enhance prompt');
      setTimeout(() => setError(''), 2000);
    } finally {
      setEnhancingPrompt(false);
    }
  };

  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Load datasets for this bot
  const loadBotDatasets = async () => {
    try {
      setLoadingDatasets(true);
      const result = await botService.getBotDatasets(botId);
      setBotDatasets(result.datasets);
      setAvailableDatasets(result.available_datasets);
      setLoadingDatasets(false);
    } catch (err) {
      console.error('Error loading bot datasets:', err);
      setLoadingDatasets(false);
    }
  };

  // Add a function to handle dataset selection
  const handleSelectDataset = async (datasetId) => {
    try {
      setLoading(true);
      const currentDatasetIds = botDatasets.map(dataset => dataset.id);
      
      // Add the selected dataset if it's not already in the list
      if (!currentDatasetIds.includes(datasetId)) {
        currentDatasetIds.push(datasetId);
      }
      
      // Update bot with the selected datasets
      await botService.setBotDatasets(botId, currentDatasetIds);
      
      // Reload datasets to update UI
      await loadBotDatasets();
      
      setLoading(false);
      setShowDatasetsDropdown(false);
    } catch (err) {
      console.error('Error setting dataset:', err);
      setError('Failed to update datasets');
      setLoading(false);
    }
  };

  // Add a function to handle removing a dataset
  const handleRemoveDataset = async (datasetId) => {
    try {
      setLoading(true);
      const currentDatasetIds = botDatasets.map(dataset => dataset.id);
      
      // Remove the selected dataset
      const updatedDatasetIds = currentDatasetIds.filter(id => id !== datasetId);
      
      // Don't allow removing the last dataset
      if (updatedDatasetIds.length === 0) {
        setError('Cannot remove the last dataset. A bot must have at least one dataset.');
        setLoading(false);
        return;
      }
      
      // Update bot with the remaining datasets
      await botService.setBotDatasets(botId, updatedDatasetIds);
      
      // Reload datasets to update UI
      await loadBotDatasets();
      
      setLoading(false);
    } catch (err) {
      console.error('Error removing dataset:', err);
      setError('Failed to update datasets');
      setLoading(false);
    }
  };

  return (
    <Container fluid className="h-100 d-flex flex-column p-0">
      <div className="chat-header py-2 px-3 border-bottom d-flex justify-content-between align-items-center">
        <div className="d-flex align-items-center">
          <Button 
            variant="link" 
            className="p-0 me-2 text-dark" 
            onClick={() => navigate('/bots')}
          >
            <FaChevronLeft />
          </Button>
          <h5 className="mb-0">
            {bot ? bot.name : 'Chat'}
            {currentConversationId && conversations.find(c => c.id === currentConversationId) && (
              <span className="ms-2 text-muted" style={{ fontSize: '0.9rem' }}>
                {conversationRenameMode ? (
                  <form onSubmit={handleRenameConversation} className="d-inline-flex ms-2">
                    <input
                      type="text"
                      className="form-control form-control-sm"
                      value={newConversationTitle}
                      onChange={(e) => setNewConversationTitle(e.target.value)}
                      autoFocus
                      placeholder="Conversation title"
                      style={{ width: '200px' }}
                    />
                    <Button 
                      variant="outline-secondary" 
                      size="sm" 
                      type="submit" 
                      className="ms-1"
                    >
                      <FaCheck />
                    </Button>
                    <Button 
                      variant="outline-secondary" 
                      size="sm" 
                      onClick={() => setConversationRenameMode(false)} 
                      className="ms-1"
                    >
                      <FaTimes />
                    </Button>
                  </form>
                ) : (
                  <>
                    {conversations.find(c => c.id === currentConversationId)?.title || 'Conversation'}
                    <Button 
                      variant="link" 
                      className="p-0 ms-1" 
                      onClick={() => {
                        const currentConversation = conversations.find(c => c.id === currentConversationId);
                        if (currentConversation) {
                          setNewConversationTitle(currentConversation.title);
                          setConversationRenameMode(true);
                        }
                      }}
                    >
                      <FaEdit size={12} />
                    </Button>
                  </>
                )}
              </span>
            )}
          </h5>
        </div>
        
        <div>
          {/* Toggle conversations dropdown button */}
          <Button 
            variant="outline-secondary" 
            size="sm" 
            className="me-2"
            onClick={() => setShowConversations(!showConversations)}
          >
            <FaListAlt className="me-1" />
            Conversations
          </Button>
          
          {/* Other buttons... */}
        </div>
      </div>

      {/* Conversations Sidebar */}
      {showConversations && (
        <div className="conversation-sidebar border-end" style={{ 
          position: 'absolute', 
          top: '50px', 
          right: '0', 
          bottom: '60px', 
          width: '300px', 
          zIndex: 1000, 
          backgroundColor: 'white',
          overflowY: 'auto',
          padding: '1rem',
          boxShadow: '-2px 0 10px rgba(0,0,0,0.1)'
        }}>
          <div className="d-flex justify-content-between align-items-center mb-3">
            <h6 className="mb-0">Conversations</h6>
            <div>
              <Button 
                variant="primary" 
                size="sm" 
                className="me-2"
                onClick={startNewConversation}
              >
                New
              </Button>
              {conversations.length > 0 && (
                <Button 
                  variant="outline-danger" 
                  size="sm"
                  onClick={handleDeleteAllConversations}
                >
                  Delete All
                </Button>
              )}
            </div>
          </div>
          
          {loadingConversations ? (
            <div className="text-center py-4">
              <LoadingAnimation />
            </div>
          ) : conversations.length === 0 ? (
            <div className="text-center py-4 text-muted">
              No conversations yet
            </div>
          ) : (
            <div>
              {conversations.map(convo => (
                <div 
                  key={convo.id} 
                  className={`conversation-item p-2 mb-2 rounded ${currentConversationId === convo.id ? 'bg-light border' : 'border'}`}
                  style={{ cursor: 'pointer' }}
                >
                  <div 
                    className="d-flex justify-content-between"
                    onClick={() => loadConversation(convo.id)}
                  >
                    <div>
                      <div className="fw-bold text-truncate" style={{ maxWidth: '200px' }}>
                        {convo.title || 'Untitled Conversation'}
                      </div>
                      <div className="text-muted small">
                        {new Date(convo.updated_at).toLocaleString()} · {convo.message_count} messages
                      </div>
                      <div className="text-truncate small" style={{ maxWidth: '240px' }}>
                        {convo.preview}
                      </div>
                    </div>
                    <div>
                      <Button 
                        variant="link" 
                        className="p-0 text-danger" 
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteConversation(convo.id);
                        }}
                      >
                        <FaTimes />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

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
                      className={`message ${msg.role === 'user' ? 'user-message' : 'bot-message'}`}
                    >
                      <div className="message-content">
                        {msg.role === 'user' ? (
                          // Display user messages normally
                          typeof msg.content === 'object' ? msg.content : msg.content
                        ) : (
                          // Display bot messages with markdown rendering
                          typeof msg.content === 'object' ? (
                            msg.content
                          ) : (
                            <div className="markdown-content">
                              <ReactMarkdown 
                                components={{
                                  // Apply custom styles to markdown elements
                                  code: ({node, inline, className, children, ...props}) => {
                                    const style = inline ? markdownStyles.code : markdownStyles.pre;
                                    return (
                                      <code
                                        className={className}
                                        style={style}
                                        {...props}
                                      >
                                        {children}
                                      </code>
                                    );
                                  },
                                  pre: ({node, children, ...props}) => (
                                    <pre style={markdownStyles.pre} {...props}>
                                      {children}
                                    </pre>
                                  ),
                                  blockquote: ({node, children, ...props}) => (
                                    <blockquote style={markdownStyles.blockquote} {...props}>
                                      {children}
                                    </blockquote>
                                  ),
                                  table: ({node, children, ...props}) => (
                                    <table style={markdownStyles.table} {...props}>
                                      {children}
                                    </table>
                                  ),
                                  th: ({node, children, ...props}) => (
                                    <th style={markdownStyles.th} {...props}>
                                      {children}
                                    </th>
                                  ),
                                  td: ({node, children, ...props}) => (
                                    <td style={markdownStyles.td} {...props}>
                                      {children}
                                    </td>
                                  )
                                }}
                              >
                                {msg.content}
                              </ReactMarkdown>
                              
                              {/* Display images from bot response if any */}
                              {msg.role === 'assistant' && msg.image_details && msg.image_details.length > 0 && (
                                <div className="source-images mt-3 p-2 border-top">
                                  <div className="d-flex flex-wrap gap-3">
                                    {msg.image_details.map((img, index) => (
                                      <div key={index} className="source-image">
                                        <div className="mb-1 text-center small fw-bold">
                                          {img.index}
                                        </div>
                                        <a 
                                          href={img.download_url || img.url} 
                                          target="_blank" 
                                          rel="noreferrer"
                                          title={img.caption || "View full image"}
                                        >
                                          <img 
                                            src={img.url} 
                                            alt={img.caption || "Referenced image"} 
                                            style={{ 
                                              maxWidth: '100%', 
                                              maxHeight: '200px', 
                                              borderRadius: '8px', 
                                              boxShadow: '0 2px 4px rgba(0,0,0,0.1)' 
                                            }}
                                          />
                                        </a>
                                        {img.caption && (
                                          <div className="mt-1 small text-muted">
                                            {img.caption}
                                          </div>
                                        )}
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          )
                        )}
                        {msg.role === 'assistant' && msg.source_documents && msg.source_documents.length > 0 && (
                          <div className="source-documents mt-2">
                            <small className="text-muted">Sources used:</small>
                            {msg.source_documents.map((doc, index) => (
                              <div key={doc.id} className="source-document">
                                <button
                                  className="btn btn-link btn-sm p-0"
                                  onClick={() => handleDownloadDocument(doc)}
                                >
                                  {doc.filename}
                                </button>
                              </div>
                            ))}
                          </div>
                        )}
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
                      <TesseractLoader size={80} />
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
                      variant="primary" 
                      size="sm" 
                      className="me-2" 
                      title="Text Chat Mode"
                    >
                      <FaRobot className="me-1" /> Chat
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
                    
                    {/* Dataset dropdown moved here from bottom */}
                    <Dropdown className="d-inline-block me-2 position-relative" show={showDatasetsDropdown} onToggle={setShowDatasetsDropdown}>
                      <Dropdown.Toggle
                        variant="outline-primary"
                        size="sm"
                        id="dropdown-datasets"
                        title="Select datasets for this bot"
                      >
                        <FaDatabase className="me-1" />
                        Datasets
                        {botDatasets && botDatasets.length > 0 && (
                          <Badge 
                            bg="primary" 
                            pill 
                            className="ms-1"
                          >
                            {botDatasets.length}
                          </Badge>
                        )}
                      </Dropdown.Toggle>
                      <Dropdown.Menu>
                        <Dropdown.Header>Current Datasets</Dropdown.Header>
                        {botDatasets && botDatasets.map((dataset) => (
                          <Dropdown.Item key={dataset.id} className="d-flex justify-content-between align-items-center">
                            <span>
                              {dataset.name}
                              {dataset.missing && <Badge bg="danger" className="ms-1">Missing</Badge>}
                            </span>
                            {botDatasets.length > 1 && (
                              <Button
                                size="sm"
                                variant="outline-danger"
                                className="btn-sm py-0 px-1"
                                onClick={() => handleRemoveDataset(dataset.id)}
                                disabled={loading}
                              >
                                <FaTimes />
                              </Button>
                            )}
                          </Dropdown.Item>
                        ))}
                        <Dropdown.Divider />
                        <Dropdown.Header>Add Dataset</Dropdown.Header>
                        {loadingDatasets ? (
                          <div className="text-center p-2">
                            <LoadingAnimation />
                          </div>
                        ) : (
                          availableDatasets && availableDatasets
                            .filter(dataset => !botDatasets.some(bd => bd.id === dataset.id))
                            .map((dataset) => (
                              <Dropdown.Item 
                                key={dataset.id} 
                                onClick={() => handleSelectDataset(dataset.id)}
                                disabled={loading}
                              >
                                <FaPlus className="me-1" /> {dataset.name}
                              </Dropdown.Item>
                            ))
                        )}
                        {availableDatasets && availableDatasets.length === botDatasets.length && (
                          <Dropdown.Item disabled>No more datasets available</Dropdown.Item>
                        )}
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
                      disabled={loading || enhancingPrompt}
                    />
                    
                    <OverlayTrigger
                      placement="top"
                      overlay={<Tooltip>Enhance your prompt with AI</Tooltip>}
                    >
                      <Button 
                        variant="outline-primary"
                        onClick={(e) => {
                          e.preventDefault();
                          try {
                            handleEnhancePrompt();
                          } catch (err) {
                            console.error('Error in enhance prompt button click:', err);
                            setError('Failed to start prompt enhancement');
                            setEnhancingPrompt(false);
                          }
                        }}
                        onContextMenu={(e) => {
                          e.preventDefault();
                          if (debugMode) {
                            // Display API details in console for debugging
                            console.log('Debug info for enhance prompt:');
                            console.log('Primary API URL:', `${process.env.REACT_APP_API_URL || '/api'}/api/bots/enhance-prompt`);
                            console.log('Fallback API URL:', `${process.env.REACT_APP_API_URL || '/api'}/api/bots/[botId]/chat`);
                            console.log('Current message:', message);
                            alert('Check console for API debugging info');
                          }
                        }}
                        disabled={!message.trim() || loading || enhancingPrompt}
                        title="Enhance Prompt"
                      >
                        {enhancingPrompt ? <TesseractLoader /> : <FaLightbulb />}
                      </Button>
                    </OverlayTrigger>
                    
                    <Button 
                      variant="outline-secondary"
                      onClick={() => fileInputRef.current?.click()}
                      title="Attach Image"
                      disabled={loading}
                    >
                      {loading ? <TesseractLoader /> : <FaImage />}
                    </Button>
                    
                    <Form.Control
                      type="file"
                      ref={fileInputRef}
                      accept="image/*"
                      className="d-none"
                      onChange={handleImageSelect}
                    />
                  </div>
                  
                  <div className="mt-3 d-flex justify-content-end">
                    <Button 
                      variant="primary" 
                      type="submit" 
                      disabled={loading || (imageAttachment && !message.trim()) || (!imageAttachment && !message.trim())}
                    >
                      {loading ? (
                        <>
                          <TesseractLoader />
                          Sending...
                        </>
                      ) : (
                        imageAttachment ? 'Send' : 'Send'
                      )}
                    </Button>
                  </div>
                </Form>

                {/* Image Retrieval Instruction Card */}

              </Card.Footer>
            </Card>
          </div>
        </div>
      ) : (
        <div className="text-center my-5">
          {!error && <TesseractLoader />}
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