import React, { useState, useRef } from 'react';

const UploadXray = ({ onUpload, isLoading }) => {
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const processFile = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target.result);
      };
      reader.readAsDataURL(file);
    } else {
      alert("Please upload an image file.");
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    inputRef.current.click();
  };

  const handleSubmit = () => {
    if (selectedFile) {
      onUpload(selectedFile, preview);
    }
  };

  return (
    <div className="card">
      <h2>Upload Patient X-Ray</h2>
      
      {!preview ? (
        <div 
          className={`upload-area ${dragActive ? 'dragging' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={onButtonClick}
        >
          <div className="upload-icon">📁</div>
          <p>Drag and drop image here or <strong>click to browse</strong></p>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            onChange={handleChange}
            style={{ display: 'none' }}
          />
        </div>
      ) : (
        <div className="preview-container">
          <div style={{ textAlign: 'center' }}>
            <img src={preview} alt="X-ray preview" className="image-preview" />
          </div>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginTop: '15px' }}>
            <button 
              className="btn-primary" 
              onClick={() => { setPreview(null); setSelectedFile(null); }}
              style={{ background: '#7f8c8d' }}
              disabled={isLoading}
            >
              Clear
            </button>
            <button 
              className="btn-primary" 
              onClick={handleSubmit}
              disabled={isLoading}
            >
              {isLoading ? 'Processing...' : 'Analyze Image'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default UploadXray;
