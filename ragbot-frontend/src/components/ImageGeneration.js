import React, { useState } from 'react';
import { Container, Row, Col, Form, Button, Card, Tabs, Tab, Alert, Spinner } from 'react-bootstrap';
import botService from '../services/botService';

const ImageGeneration = () => {
  // State for generation
  const [prompt, setPrompt] = useState('');
  const [size, setSize] = useState('1024x1024');
  const [quality, setQuality] = useState('auto');
  const [outputFormat, setOutputFormat] = useState('png');
  const [generatedImage, setGeneratedImage] = useState(null);
  const [moderation, setModeration] = useState('auto');
  
  // State for editing
  const [editPrompt, setEditPrompt] = useState('');
  const [editQuality, setEditQuality] = useState('auto');
  const [editSize, setEditSize] = useState('auto');
  const [editOutputFormat, setEditOutputFormat] = useState('png');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [maskFile, setMaskFile] = useState(null);
  const [editedImage, setEditedImage] = useState(null);
  
  // Status states
  const [loading, setLoading] = useState(false);
  const [enhancingPrompt, setEnhancingPrompt] = useState(false);
  const [error, setError] = useState(null);
  const [message, setMessage] = useState(null);
  
  // Handle image generation
  const handleGenerateImage = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setMessage(null);
    setGeneratedImage(null);
    
    try {
      // Prepare options for gpt-image-1
      const options = {
        model: "gpt-image-1",
        size,
        quality,
        prompt,
        output_format: outputFormat,
        moderation
      };
      
      const result = await botService.generateImage(prompt, options);
      
      if (result.images && result.images.length > 0) {
        const imageData = result.images[0];
        // Make sure the default URL and API_URL are consistent
        const baseUrl = process.env.REACT_APP_API_URL || '/api';
        // Remove any duplicate /api prefixes if present
        const imageUrl = imageData.image_url.startsWith('/api/') 
          ? imageData.image_url 
          : `/api${imageData.image_url}`;
        // Construct full URL correctly
        const fullUrl = baseUrl.endsWith('/api') 
          ? `${baseUrl.substring(0, baseUrl.length - 4)}${imageUrl}`
          : `${baseUrl}${imageUrl}`;
        setGeneratedImage(fullUrl);
        setMessage('Image generated successfully!');
      }
    } catch (err) {
      console.error('Error generating image:', err);
      setError(err.response?.data?.error || err.message || 'Failed to generate image');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle image editing
  const handleEditImage = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setMessage(null);
    setEditedImage(null);
    
    if (selectedFiles.length === 0) {
      setError('Please select at least one image');
      setLoading(false);
      return;
    }
    
    try {
      const formData = new FormData();
      
      // Add all selected files to formData
      selectedFiles.forEach(file => {
        formData.append('image', file);
      });
      
      // Add mask if provided
      if (maskFile) {
        formData.append('mask', maskFile);
      }
      
      // Add other parameters
      formData.append('prompt', editPrompt);
      formData.append('model', 'gpt-image-1');
      formData.append('quality', editQuality);
      formData.append('size', editSize);
      formData.append('output_format', editOutputFormat);
      
      const result = await botService.editImage(formData);
      
      if (result.image_url) {
        // Make sure the default URL and API_URL are consistent
        const baseUrl = process.env.REACT_APP_API_URL || '/api';
        // Remove any duplicate /api prefixes if present
        const imageUrl = result.image_url.startsWith('/api/') 
          ? result.image_url 
          : `/api${result.image_url}`;
        // Construct full URL correctly
        const fullUrl = baseUrl.endsWith('/api') 
          ? `${baseUrl.substring(0, baseUrl.length - 4)}${imageUrl}`
          : `${baseUrl}${imageUrl}`;
        setEditedImage(fullUrl);
        setMessage('Image edited successfully!');
      }
    } catch (err) {
      console.error('Error editing image:', err);
      setError(err.response?.data?.error || err.message || 'Failed to edit image');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle prompt enhancement
  const handleEnhancePrompt = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt to enhance');
      return;
    }
    
    setEnhancingPrompt(true);
    setError(null);
    setMessage(null);
    
    try {
      const result = await botService.enhancePrompt(prompt);
      
      if (result.success && result.enhanced_prompt) {
        setPrompt(result.enhanced_prompt);
        setMessage('Prompt enhanced successfully!');
        
        // Clear message after 3 seconds
        setTimeout(() => {
          setMessage(null);
        }, 3000);
      }
    } catch (err) {
      console.error('Error enhancing prompt:', err);
      setError(err.response?.data?.error || err.message || 'Failed to enhance prompt');
    } finally {
      setEnhancingPrompt(false);
    }
  };
  
  // Handle file selection for editing
  const handleFileSelect = (e) => {
    if (e.target.files) {
      setSelectedFiles(Array.from(e.target.files));
    }
  };
  
  // Handle mask file selection
  const handleMaskSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      setMaskFile(e.target.files[0]);
    }
  };
  
  return (
    <Container className="my-4">
      <h1 className="mb-4">Image Generation</h1>
      
      {error && <Alert variant="danger">{error}</Alert>}
      {message && <Alert variant="success">{message}</Alert>}
      
      <Tabs defaultActiveKey="generate" id="image-tabs" className="mb-3">
        <Tab eventKey="generate" title="Generate Image">
          <Row>
            <Col md={6}>
              <Card className="p-4">
                <Form onSubmit={handleGenerateImage}>
                  <Form.Group className="mb-3">
                    <Form.Label>Prompt</Form.Label>
                    <Form.Control
                      as="textarea"
                      rows={4}
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder="Describe the image you want to generate..."
                      required
                    />
                    <div className="d-flex justify-content-end mt-1">
                      <Button 
                        variant="outline-primary"
                        size="sm"
                        onClick={handleEnhancePrompt}
                        disabled={enhancingPrompt || !prompt.trim()}
                      >
                        {enhancingPrompt ? (
                          <>
                            <Spinner as="span" size="sm" animation="border" className="me-1" />
                            Enhancing...
                          </>
                        ) : 'Enhance Prompt'}
                      </Button>
                    </div>
                  </Form.Group>
                  
                  <Row>
                    <Col md={6}>
                      <Form.Group className="mb-3">
                        <Form.Label>Size</Form.Label>
                        <Form.Select 
                          value={size}
                          onChange={(e) => setSize(e.target.value)}
                        >
                          <option value="1024x1024">1024x1024 (Square)</option>
                          <option value="1536x1024">1536x1024 (Landscape)</option>
                          <option value="1024x1536">1024x1536 (Portrait)</option>
                          <option value="auto">Auto</option>
                        </Form.Select>
                      </Form.Group>
                    </Col>
                    <Col md={6}>
                      <Form.Group className="mb-3">
                        <Form.Label>Quality</Form.Label>
                        <Form.Select 
                          value={quality}
                          onChange={(e) => setQuality(e.target.value)}
                        >
                          <option value="low">Low</option>
                          <option value="medium">Medium</option>
                          <option value="high">High</option>
                          <option value="auto">Auto</option>
                        </Form.Select>
                      </Form.Group>
                    </Col>
                  </Row>
                  
                  <Row>
                    <Col md={6}>
                      <Form.Group className="mb-3">
                        <Form.Label>Output Format</Form.Label>
                        <Form.Select 
                          value={outputFormat}
                          onChange={(e) => setOutputFormat(e.target.value)}
                        >
                          <option value="png">PNG</option>
                          <option value="jpeg">JPEG</option>
                          <option value="webp">WebP</option>
                        </Form.Select>
                      </Form.Group>
                    </Col>
                    <Col md={6}>
                      <Form.Group className="mb-3">
                        <Form.Label>Moderation</Form.Label>
                        <Form.Select 
                          value={moderation}
                          onChange={(e) => setModeration(e.target.value)}
                        >
                          <option value="auto">Auto (Standard filtering)</option>
                          <option value="low">Low (Less restrictive)</option>
                        </Form.Select>
                        <Form.Text className="text-muted">
                          Controls content policy filtering strictness
                        </Form.Text>
                      </Form.Group>
                    </Col>
                  </Row>
                  
                  <Button 
                    variant="primary" 
                    type="submit" 
                    disabled={loading || !prompt.trim()}
                    className="w-100"
                  >
                    {loading ? (
                      <>
                        <Spinner as="span" size="sm" animation="border" className="me-2" />
                        Generating...
                      </>
                    ) : 'Generate Image'}
                  </Button>
                </Form>
              </Card>
            </Col>
            
            <Col md={6}>
              <Card className="h-100">
                <Card.Body className="d-flex flex-column align-items-center justify-content-center">
                  {loading ? (
                    <div className="text-center">
                      <Spinner animation="border" role="status" />
                      <p className="mt-3">Generating your image...</p>
                    </div>
                  ) : generatedImage ? (
                    <div className="text-center">
                      <img 
                        src={generatedImage} 
                        alt="Generated" 
                        className="img-fluid mb-3" 
                        style={{ maxHeight: '400px' }}
                      />
                      <div>
                        <Button 
                          variant="outline-secondary" 
                          as="a" 
                          href={generatedImage} 
                          target="_blank" 
                          className="me-2"
                        >
                          Open Full Size
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center text-muted">
                      <i className="bi bi-image" style={{ fontSize: '3rem' }}></i>
                      <p>Your generated image will appear here</p>
                    </div>
                  )}
                </Card.Body>
              </Card>
            </Col>
          </Row>
        </Tab>
        
        <Tab eventKey="edit" title="Edit Image">
          <Row>
            <Col md={6}>
              <Card className="p-4">
                <Form onSubmit={handleEditImage}>
                  <Form.Group className="mb-3">
                    <Form.Label>Images to Edit</Form.Label>
                    <Form.Control
                      type="file"
                      multiple={true}
                      onChange={handleFileSelect}
                      accept="image/*"
                      required
                    />
                    <Form.Text className="text-muted">
                      You can select multiple images for reference
                    </Form.Text>
                  </Form.Group>
                  
                  <Form.Group className="mb-3">
                    <Form.Label>Mask Image (Optional)</Form.Label>
                    <Form.Control
                      type="file"
                      onChange={handleMaskSelect}
                      accept="image/png"
                    />
                    <Form.Text className="text-muted">
                      Upload a PNG with transparency to specify which areas to edit
                    </Form.Text>
                  </Form.Group>
                  
                  <Form.Group className="mb-3">
                    <Form.Label>Edit Prompt</Form.Label>
                    <Form.Control
                      as="textarea"
                      rows={4}
                      value={editPrompt}
                      onChange={(e) => setEditPrompt(e.target.value)}
                      placeholder="Describe how you want to edit the image..."
                      required
                    />
                  </Form.Group>
                  
                  <Row>
                    <Col md={6}>
                      <Form.Group className="mb-3">
                        <Form.Label>Size</Form.Label>
                        <Form.Select 
                          value={editSize}
                          onChange={(e) => setEditSize(e.target.value)}
                        >
                          <option value="auto">Auto</option>
                          <option value="1024x1024">1024x1024 (Square)</option>
                          <option value="1536x1024">1536x1024 (Landscape)</option>
                          <option value="1024x1536">1024x1536 (Portrait)</option>
                        </Form.Select>
                      </Form.Group>
                    </Col>
                    <Col md={6}>
                      <Form.Group className="mb-3">
                        <Form.Label>Quality</Form.Label>
                        <Form.Select 
                          value={editQuality}
                          onChange={(e) => setEditQuality(e.target.value)}
                        >
                          <option value="auto">Auto</option>
                          <option value="low">Low</option>
                          <option value="medium">Medium</option>
                          <option value="high">High</option>
                        </Form.Select>
                      </Form.Group>
                    </Col>
                  </Row>
                  
                  <Form.Group className="mb-3">
                    <Form.Label>Output Format</Form.Label>
                    <Form.Select 
                      value={editOutputFormat}
                      onChange={(e) => setEditOutputFormat(e.target.value)}
                    >
                      <option value="png">PNG</option>
                      <option value="jpeg">JPEG</option>
                      <option value="webp">WebP</option>
                    </Form.Select>
                  </Form.Group>
                  
                  <Button 
                    variant="primary" 
                    type="submit" 
                    disabled={loading || !editPrompt.trim() || selectedFiles.length === 0}
                    className="w-100"
                  >
                    {loading ? (
                      <>
                        <Spinner as="span" size="sm" animation="border" className="me-2" />
                        Editing...
                      </>
                    ) : 'Edit Image'}
                  </Button>
                </Form>
              </Card>
            </Col>
            
            <Col md={6}>
              <Card className="h-100">
                <Card.Body className="d-flex flex-column align-items-center justify-content-center">
                  {loading ? (
                    <div className="text-center">
                      <Spinner animation="border" role="status" />
                      <p className="mt-3">Editing your image...</p>
                    </div>
                  ) : editedImage ? (
                    <div className="text-center">
                      <img 
                        src={editedImage} 
                        alt="Edited" 
                        className="img-fluid mb-3" 
                        style={{ maxHeight: '400px' }}
                      />
                      <div>
                        <Button 
                          variant="outline-secondary" 
                          as="a" 
                          href={editedImage} 
                          target="_blank" 
                          className="me-2"
                        >
                          Open Full Size
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center text-muted">
                      <i className="bi bi-image" style={{ fontSize: '3rem' }}></i>
                      <p>Your edited image will appear here</p>
                    </div>
                  )}
                </Card.Body>
              </Card>
            </Col>
          </Row>
        </Tab>
      </Tabs>
      
      <Card className="mt-4 p-3">
        <h5>About GPT Image Generation</h5>
        <p>This tool uses OpenAI's gpt-image-1 model to generate images based on text prompts.</p>
        <ul>
          <li><strong>Size options:</strong> 1024x1024 (square), 1536x1024 (landscape), or 1024x1536 (portrait)</li>
          <li><strong>Quality options:</strong> low (faster), medium, high (more detailed), or auto</li>
          <li><strong>Format options:</strong> png, jpeg, or webp</li>
          <li><strong>Moderation options:</strong> auto (standard filtering), low (less restrictive filtering)</li>
        </ul>
        <p>For best results, provide detailed descriptions in your prompts.</p>
        <p className="small text-muted">Note: All prompts and images are filtered against OpenAI's content policy regardless of moderation setting.</p>
      </Card>
    </Container>
  );
};

export default ImageGeneration; 