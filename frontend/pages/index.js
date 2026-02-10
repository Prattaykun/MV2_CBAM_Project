
import { useState, useRef } from 'react';
import Head from 'next/head';

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [camImage, setCamImage] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  // --- Drag & Drop Handlers ---
  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('image/')) {
        handleFileSelect(droppedFile);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      handleFileSelect(selectedFile);
    }
  };

  const handleFileSelect = (selectedFile) => {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setPrediction(null);
      setCamImage(null);
  };

  const handleSubmit = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      let apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/predict';
      
      if (!apiUrl.endsWith('/predict')) {
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
    <div className="min-h-screen bg-[#0f172a] text-white font-sans selection:bg-orange-500 selection:text-white">
      <Head>
        <title>FireGuard AI | UAV Surveillance Dashboard</title>
        <meta name="description" content="Next-Gen Aerial Fire Detection System" />
        <link rel="icon" href="/fireguard_favicon.png" />
      </Head>

      {/* Navbar */}
      <nav className="absolute top-0 w-full p-6 flex justify-between items-center z-50">
          <div className="flex items-center gap-3">
              <div className="h-10 w-10 bg-gradient-to-br from-orange-500 to-red-600 rounded-lg flex items-center justify-center shadow-lg shadow-orange-500/20">
                  <span className="text-2xl">🔥</span>
              </div>
              <span className="text-xl font-bold tracking-tight">FireGuard<span className="text-orange-500">AI</span></span>
          </div>
          <div className="hidden md:flex gap-6 text-sm font-medium text-gray-400">
              <span 
              onClick={() => window.open('https://github.com/prattaykun/MV2_CBAM_Project', '_blank')}
              className="hover:text-white cursor-pointer transition-colors">Documentation</span>
              {/* <span className="hover:text-white cursor-pointer transition-colors">Model Specs</span>
              <span className="hover:text-white cursor-pointer transition-colors">About</span> */}
          </div>
      </nav>

      <main className="container mx-auto px-4 min-h-screen flex flex-col justify-center py-20 relative">
        {/* Background Elements */}
        <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
            <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-orange-600/10 rounded-full blur-[120px]"></div>
            <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[120px]"></div>
        </div>

        <header className="text-center mb-16 max-w-3xl mx-auto">
          <div className="inline-block px-4 py-1.5 mb-6 rounded-full bg-slate-800/50 border border-slate-700 backdrop-blur-md">
             <span className="text-xs font-semibold text-orange-400 uppercase tracking-widest">v2.0 Beta Release</span>
          </div>
          <h1 className="text-5xl md:text-7xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-200 to-slate-400 mb-6 tracking-tight leading-tight">
            Aerial Wildfire <br />
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-orange-400 to-red-600">Surveillance System</span>
          </h1>
          <p className="text-lg md:text-xl text-slate-400 leading-relaxed">
            Advanced real-time fire detection powered by 
            <span className="text-slate-200 font-semibold mx-1">MobileNetV2</span> 
            and 
            <span className="text-slate-200 font-semibold mx-1">CBAM Attention</span>.
            Upload aerial imagery to instantly pinpoint hazards.
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-6xl mx-auto w-full">
          {/* Left Column: Upload */}
          <div className="flex flex-col gap-6">
            <div 
               className={`relative group bg-slate-900/50 backdrop-blur-xl border-2 border-dashed rounded-3xl p-10 text-center cursor-pointer transition-all duration-300 ease-out h-full flex flex-col justify-center items-center overflow-hidden ${
                   isDragging 
                   ? 'border-orange-500 bg-orange-500/10 shadow-[0_0_40px_-10px_rgba(249,115,22,0.3)] scale-[1.02]' 
                   : 'border-slate-700 hover:border-slate-500 hover:bg-slate-800/50'
               }`}
               onDragOver={onDragOver}
               onDragLeave={onDragLeave}
               onDrop={onDrop}
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
                <div className="relative w-full h-full min-h-[300px] flex items-center justify-center">
                    <img 
                      src={preview} 
                      alt="Preview" 
                      className="max-h-[400px] w-auto mx-auto rounded-xl shadow-2xl object-contain z-10"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-slate-900/80 to-transparent rounded-xl z-20 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-6">
                        <p className="text-white font-medium">Click to change image</p>
                    </div>
                </div>
              ) : (
                <div className="space-y-4 pointer-events-none">
                  <div className={`w-20 h-20 mx-auto rounded-2xl flex items-center justify-center transition-all duration-300 ${isDragging ? 'bg-orange-500 text-white scale-110' : 'bg-slate-800 text-slate-400 group-hover:bg-slate-700 group-hover:text-white'}`}>
                      <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
                  </div>
                  <div>
                      <p className="text-xl font-semibold text-slate-200">Drag & Drop Image Here</p>
                      <p className="text-slate-500 text-sm mt-1">or click to browse from your device</p>
                  </div>
                  <div className="flex justify-center gap-3 pt-4">
                      <span className="px-3 py-1 rounded-full bg-slate-800 border border-slate-700 text-xs text-slate-400">JPG</span>
                      <span className="px-3 py-1 rounded-full bg-slate-800 border border-slate-700 text-xs text-slate-400">PNG</span>
                  </div>
                </div>
              )}
            </div>

            <button
              onClick={handleSubmit}
              disabled={!file || loading}
              className={`w-full py-4 rounded-2xl font-bold text-lg tracking-wide shadow-xl transition-all duration-300 transform active:scale-[0.98] ${
                loading 
                  ? 'bg-slate-700 text-slate-400 cursor-not-allowed' 
                  : 'bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-400 hover:to-red-500 text-white shadow-orange-900/20 hover:shadow-orange-700/40 ring-1 ring-orange-500/50'
              }`}
            >
              <span className="flex items-center justify-center gap-2">
                  {loading && <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>}
                  {loading ? 'Processing Analysis...' : 'Run Analysis System'}
              </span>
            </button>
          </div>

          {/* Right Column: Results */}
          <div className="relative bg-slate-900/50 backdrop-blur-xl border border-slate-800 rounded-3xl p-8 lg:p-10 flex flex-col shadow-2xl overflow-hidden min-h-[500px]">
            {/* Decorative Grid */}
            <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-5"></div>
            
            {loading ? (
              <div className="flex flex-col items-center justify-center flex-grow z-10">
                <div className="relative w-24 h-24">
                    <div className="absolute top-0 left-0 w-full h-full border-4 border-slate-700 rounded-full"></div>
                    <div className="absolute top-0 left-0 w-full h-full border-4 border-t-orange-500 rounded-full animate-spin"></div>
                </div>
                <p className="mt-8 text-xl font-medium text-slate-300 animate-pulse">Running Neural Inference...</p>
                <div className="flex gap-2 mt-2">
                    <span className="w-2 h-2 rounded-full bg-slate-600 animate-bounce delay-75"></span>
                    <span className="w-2 h-2 rounded-full bg-slate-600 animate-bounce delay-150"></span>
                    <span className="w-2 h-2 rounded-full bg-slate-600 animate-bounce delay-300"></span>
                </div>
              </div>
            ) : prediction ? (
              <div className="flex flex-col h-full z-10 animate-fade-in-up">
                  <div className="flex items-center justify-between mb-8 pb-6 border-b border-slate-800">
                     <div>
                         <h3 className="text-slate-400 text-sm font-semibold uppercase tracking-wider mb-1">Status Report</h3>
                         <div className="flex items-center gap-3">
                             <div className={`w-3 h-3 rounded-full ${prediction.prediction === 'Fire' ? 'bg-red-500 animate-pulse shadow-[0_0_10px_red]' : 'bg-green-500 shadow-[0_0_10px_green]'}`}></div>
                             <span className="text-white font-mono text-xs">ID: {Math.random().toString(36).substr(2, 9).toUpperCase()}</span>
                         </div>
                     </div>
                     <div className="text-right">
                         <p className="text-slate-400 text-sm font-semibold uppercase tracking-wider mb-1">Confidence</p>
                         <p className="text-2xl font-bold font-mono text-white">{(prediction.confidence.toFixed(4) * 100).toFixed(1)}%</p>
                     </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 mb-8">
                     <div className={`p-6 rounded-2xl border flex flex-col justify-center items-center text-center ${prediction.prediction === 'Fire' ? 'bg-red-500/10 border-red-500/30 text-red-400' : 'bg-slate-800 border-slate-700 text-slate-400 opacity-50'}`}>
                         <span className="text-3xl mb-2">🚨</span>
                         <span className="font-bold text-lg">Fire Detected</span>
                     </div>
                     <div className={`p-6 rounded-2xl border flex flex-col justify-center items-center text-center ${prediction.prediction === 'No_Fire' ? 'bg-green-500/10 border-green-500/30 text-green-400' : 'bg-slate-800 border-slate-700 text-slate-400 opacity-50'}`}>
                         <span className="text-3xl mb-2">🌲</span>
                         <span className="font-bold text-lg">Safe Conditions</span>
                     </div>
                  </div>

                  {camImage && (
                    <div className="mt-auto">
                        <div className="flex items-center justify-between mb-4">
                            <h4 className="text-slate-300 font-semibold flex items-center gap-2">
                                <span className="w-5 h-5 rounded bg-slate-800 flex items-center justify-center text-xs">👁️</span>
                                Vision Attention Map
                            </h4>
                            <span className="text-xs text-slate-500 bg-slate-800 px-2 py-1 rounded">Grad-CAM</span>
                        </div>
                        <div className="relative rounded-2xl overflow-hidden border border-slate-700 shadow-2xl group">
                            <img src={camImage} alt="Grad-CAM" className="w-full h-48 object-cover transform group-hover:scale-105 transition-transform duration-700" />
                            <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent"></div>
                            <div className="absolute bottom-3 left-4 right-4 flex justify-between items-end">
                                <p className="text-xs text-slate-300 font-mono">Layer: features.18</p>
                            </div>
                        </div>
                    </div>
                  )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center flex-grow z-10 text-slate-600">
                <div className="w-32 h-32 rounded-full bg-slate-800/50 flex items-center justify-center mb-6 border border-slate-700">
                   <span className="text-5xl opacity-50">📊</span>
                </div>
                <h3 className="text-xl font-semibold text-slate-400 mb-2">Ready for Analysis</h3>
                <p className="text-slate-500 max-w-xs text-center leading-relaxed">
                  Upload an aerial image to generate real-time classification and attention heatmaps.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="w-full text-center py-8 text-slate-600 text-sm">
        <p>&copy; 2026 FireGuard AI • Developed by Prattay Roy Chowdhury</p>
      </footer>
      
      {/* Styles */}
      <style jsx global>{`
        body { margin: 0; background-color: #0f172a; }
        @keyframes fade-in-up {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in-up {
            animation: fade-in-up 0.5s ease-out forwards;
        }
      `}</style>
      
      {/* Tailwind CDN */}
      <script src="https://cdn.tailwindcss.com"></script>
    </div>
  );
}
