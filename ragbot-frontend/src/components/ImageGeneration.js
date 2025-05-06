import React, { useState } from 'react';
import { Container, Row, Col, Form, Button, Card, Tabs, Tab, Alert, Spinner } from 'react-bootstrap';
import botService from '../services/botService';

const ImageGeneration = () => {
  // State for generation
  const [prompt, setPrompt] = useState('');
  const [model, setModel] = useState('gpt-image-1');
  const [size, setSize] = useState('1024x1024');
  const [quality, setQuality] = useState('standard');
  const [outputFormat, setOutputFormat] = useState('png');
  const [background, setBackground] = useState('auto');
  const [outputCompression, setOutputCompression] = useState(80);
  const [moderation, setModeration] = useState('auto');
  const [generatedImage, setGeneratedImage] = useState(null);
  
  // State for editing
  const [editPrompt, setEditPrompt] = useState('');
  const [editModel, setEditModel] = useState('gpt-image-1');
  const [editQuality, setEditQuality] = useState('auto');
  const [editSize, setEditSize] = useState('auto');
  const [editOutputFormat, setEditOutputFormat] = useState('png');
  const [editBackground, setEditBackground] = useState('auto');
  const [editCompression, setEditCompression] = useState(80);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [maskFile, setMaskFile] = useState(null);
  const [editedImage, setEditedImage] = useState(null);
  
  // Status states
  const [loading, setLoading] = useState(false);
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
      // Prepare options based on selected model
      const options = {
        model,
        size,
        prompt
      };
      
      if (model === 'gpt-image-1') {
        options.quality = quality;
        options.output_format = outputFormat;
        options.background = background !== 'auto' ? background : undefined;
        options.moderation = moderation !== 'auto' ? moderation : undefined;
        
        if (outputFormat !== 'png' && outputCompression) {
          options.output_compression = parseInt(outputCompression, 10);
        }
      } else {
        // DALL-E options
        options.quality = quality;
        options.style = 'vivid'; // Default for DALL-E 3
      }
      
      const result = await botService.generateImage(prompt, options);
      
      if (result.image_url) {
        const fullUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000' + result.image_url;
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
      formData.append('model', editModel);
      formData.append('quality', editQuality);
      formData.append('size', editSize);
      formData.append('output_format', editOutputFormat);
      formData.append('background', editBackground);
      
      if (editOutputFormat !== 'png' && editCompression) {
        formData.append('output_compression', editCompression);
      }
      
      const result = await botService.editImage(formData);
      
      if (result.image_url) {
        const fullUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000' + result.image_url;
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
                  </Form.Group>
                  
                  <Form.Group className="mb-3">
                    <Form.Label>Model</Form.Label>
                    <Form.Select 
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                    >
                      <option value="gpt-image-1">GPT-Image-1 (Latest)</option>
                      <option value="dall-e-3">DALL-E 3</option>
                      <option value="dall-e-2">DALL-E 2</option>
                    </Form.Select>
                  </Form.Group>
                  
                  <Row>
                    <Col md={6}>
                      <Form.Group className="mb-3">
                        <Form.Label>Size</Form.Label>
                        <Form.Select 
                          value={size}
                          onChange={(e) => setSize(e.target.value)}
                        >
                          {model === 'gpt-image-1' ? (
                            <>
                              <option value="1024x1024">1024x1024 (Square)</option>
                              <option value="1536x1024">1536x1024 (Landscape)</option>
                              <option value="1024x1536">1024x1536 (Portrait)</option>
                              <option value="auto">Auto</option>
                            </>
                          ) : model === 'dall-e-3' ? (
                            <>
                              <option value="1024x1024">1024x1024</option>
                              <option value="1792x1024">1792x1024</option>
                              <option value="1024x1792">1024x1792</option>
                            </>
                          ) : (
                            <>
                              <option value="256x256">256x256</option>
                              <option value="512x512">512x512</option>
                              <option value="1024x1024">1024x1024</option>
                            </>
                          )}
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
                          {model === 'gpt-image-1' ? (
                            <>
                              <option value="low">Low</option>
                              <option value="medium">Medium</option>
                              <option value="high">High</option>
                              <option value="auto">Auto</option>
                            </>
                          ) : (
                            <>
                              <option value="standard">Standard</option>
                              <option value="hd">HD</option>
                            </>
                          )}
                        </Form.Select>
                      </Form.Group>
                    </Col>
                  </Row>
                  
                  {model === 'gpt-image-1' && (
                    <>
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
                            <Form.Label>Background</Form.Label>
                            <Form.Select 
                              value={background}
                              onChange={(e) => setBackground(e.target.value)}
                              disabled={outputFormat === 'jpeg'}
                            >
                              <option value="auto">Auto</option>
                              <option value="transparent">Transparent</option>
                            </Form.Select>
                          </Form.Group>
                        </Col>
                      </Row>
                      
                      {(outputFormat === 'jpeg' || outputFormat === 'webp') && (
                        <Form.Group className="mb-3">
                          <Form.Label>Compression ({outputCompression}%)</Form.Label>
                          <Form.Range
                            min="0"
                            max="100"
                            step="1"
                            value={outputCompression}
                            onChange={(e) => setOutputCompression(e.target.value)}
                          />
                        </Form.Group>
                      )}
                      
                      <Form.Group className="mb-3">
                        <Form.Label>Moderation</Form.Label>
                        <Form.Select 
                          value={moderation}
                          onChange={(e) => setModeration(e.target.value)}
                        >
                          <option value="auto">Auto (Standard)</option>
                          <option value="low">Low (Less strict)</option>
                        </Form.Select>
                      </Form.Group>
                    </>
                  )}
                  
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
                      multiple={editModel === 'gpt-image-1'}
                      onChange={handleFileSelect}
                      accept="image/*"
                      required
                    />
                    <Form.Text className="text-muted">
                      {editModel === 'gpt-image-1' 
                        ? 'You can select multiple images for reference' 
                        : 'Select an image to edit'}
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
                  
                  <Form.Group className="mb-3">
                    <Form.Label>Model</Form.Label>
                    <Form.Select 
                      value={editModel}
                      onChange={(e) => setEditModel(e.target.value)}
                    >
                      <option value="gpt-image-1">GPT-Image-1 (Latest)</option>
                      <option value="dall-e-2">DALL-E 2</option>
                    </Form.Select>
                    <Form.Text className="text-muted">
                      Note: DALL-E 3 doesn't support editing
                    </Form.Text>
                  </Form.Group>
                  
                  {editModel === 'gpt-image-1' && (
                    <>
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
                      
                      <Row>
                        <Col md={6}>
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
                        </Col>
                        <Col md={6}>
                          <Form.Group className="mb-3">
                            <Form.Label>Background</Form.Label>
                            <Form.Select 
                              value={editBackground}
                              onChange={(e) => setEditBackground(e.target.value)}
                              disabled={editOutputFormat === 'jpeg'}
                            >
                              <option value="auto">Auto</option>
                              <option value="transparent">Transparent</option>
                            </Form.Select>
                          </Form.Group>
                        </Col>
                      </Row>
                      
                      {(editOutputFormat === 'jpeg' || editOutputFormat === 'webp') && (
                        <Form.Group className="mb-3">
                          <Form.Label>Compression ({editCompression}%)</Form.Label>
                          <Form.Range
                            min="0"
                            max="100"
                            step="1"
                            value={editCompression}
                            onChange={(e) => setEditCompression(e.target.value)}
                          />
                        </Form.Group>
                      )}
                    </>
                  )}
                  
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
    </Container>
  );
};

export default ImageGeneration; 