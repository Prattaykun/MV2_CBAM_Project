import { useState, useRef } from 'react';
import Head from 'next/head';

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [camImage, setCamImage] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setPrediction(null);
      setCamImage(null);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      let apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/predict';
      
      // Smart check: If the user provided a root URL (e.g. onrender.com) without /predict, add it.
      // We check if it DOES NOT end with '/predict' and is not just a root slash.
      if (!apiUrl.endsWith('/predict')) {
          // Remove trailing slash if present then append /predict
          apiUrl = apiUrl.replace(/\/$/, '') + '/predict';
      }

      const res = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Prediction failed');

      const data = await res.json();
      setPrediction(data);
      if (data.cam_image_base64) {
        setCamImage(`data:image/jpeg;base64,${data.cam_image_base64}`);
      }
    } catch (error) {
      console.error(error);
      alert('Error extracting prediction');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white font-sans">
      <Head>
        <title>Fire Detection Dashboard | MV2-CBAM</title>
        <meta name="description" content="UAV-based Fire Detection using MV2-CBAM" />
      </Head>

      <main className="container mx-auto px-4 py-10">
        <header className="text-center mb-12">
          <h1 className="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-orange-400 to-red-600 mb-4">
            Wildfire Detection System
          </h1>
          <p className="text-xl text-gray-400">
            Powered by MobileNetV2 + CBAM Attention Mechanism
          </p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-10 max-w-5xl mx-auto">
          {/* Upload Section */}
          <div className="bg-gray-800 p-8 rounded-2xl shadow-xl border border-gray-700">
            <h2 className="text-2xl font-semibold mb-6 flex items-center">
              <span className="bg-gray-700 p-2 rounded-lg mr-3">📷</span> 
              Upload Image
            </h2>
            
            <div 
              className="border-2 border-dashed border-gray-600 rounded-xl p-10 text-center cursor-pointer hover:border-orange-500 transition-colors"
              onClick={() => fileInputRef.current.click()}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                className="hidden" 
                accept="image/*"
              />
              {preview ? (
                <img 
                  src={preview} 
                  alt="Preview" 
                  className="max-h-64 mx-auto rounded-lg shadow-md"
                />
              ) : (
                <div className="text-gray-500">
                  <p className="text-4xl mb-2">☁️</p>
                  <p>Click or Drag to Upload Image</p>
                </div>
              )}
            </div>

            <button
              onClick={handleSubmit}
              disabled={!file || loading}
              className={`w-full mt-6 py-3 rounded-xl font-bold text-lg transition-all transform hover:scale-105 ${
                loading 
                  ? 'bg-gray-600 cursor-not-allowed' 
                  : 'bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700 shadow-lg shadow-orange-500/30'
              }`}
            >
              {loading ? 'Analyzing...' : 'Detect Fire'}
            </button>
          </div>

          {/* Results Section */}
          <div className="bg-gray-800 p-8 rounded-2xl shadow-xl border border-gray-700 relative overflow-hidden">
            {loading && (
              <div className="absolute inset-0 bg-gray-900/80 flex items-center justify-center z-10 backdrop-blur-sm">
                <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-orange-500"></div>
              </div>
            )}
            
            <h2 className="text-2xl font-semibold mb-6 flex items-center">
              <span className="bg-gray-700 p-2 rounded-lg mr-3">📊</span> 
              Analysis Results
            </h2>

            {prediction ? (
              <div className="space-y-6">
                <div className={`p-4 rounded-xl border ${prediction.prediction === 'Fire' ? 'bg-red-900/30 border-red-500' : 'bg-green-900/30 border-green-500'}`}>
                  <p className="text-sm text-gray-400 uppercase tracking-wider mb-1">Prediction</p>
                  <p className={`text-3xl font-bold ${prediction.prediction === 'Fire' ? 'text-red-400' : 'text-green-400'}`}>
                    {prediction.prediction}
                  </p>
                </div>

                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-gray-400">Confidence Score</span>
                    <span className="font-mono text-white">{(prediction.confidence.toFixed(4) * 100).toFixed(2)}%</span>
                  </div>
                  <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className={`h-full ${prediction.prediction === 'Fire' ? 'bg-red-500' : 'bg-green-500'}`} 
                      style={{ width: `${prediction.confidence * 100}%` }}
                    ></div>
                  </div>
                </div>

                {camImage && (
                  <div className="mt-6">
                    <p className="text-sm text-gray-400 uppercase tracking-wider mb-3">Attention Map (Grad-CAM)</p>
                    <div className="relative rounded-xl overflow-hidden shadow-2xl border border-gray-600">
                      <img src={camImage} alt="Grad-CAM" className="w-full" />
                      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
                        <p className="text-xs text-gray-300">Visualizing regions influencing the decision</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-gray-600 opacity-50">
                <p className="text-6xl mb-4">🔍</p>
                <p className="text-xl">Waiting for detailed analysis...</p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Global basic styles for dark mode reset if Tailwind not fully loaded */}
      <style jsx global>{`
        body {
          margin: 0;
          background-color: #111827;
          color: white;
          font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
        }
      `}</style>
      
      {/* CDN for Tailwind (Quickest way to ensure styles work without config) */}
      <script src="https://cdn.tailwindcss.com"></script>
    </div>
  );
}
