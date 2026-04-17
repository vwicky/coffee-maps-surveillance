import { Card, CardContent } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { useState, useRef } from "react";

interface CameraData {
  id: number;
  streamUrl: string; 
  status: "idle" | "streaming" | "finished";
}

function CameraView({ 
  cam, 
  onRemove, 
  onReplace 
}: { 
  cam: CameraData, 
  onRemove: (id: number) => void,
  onReplace: (id: number, file: File) => void
}) {
  const replaceInputRef = useRef<HTMLInputElement>(null);
  // Додаємо стан для відстеження: чи відкрито відео на весь екран?
  const [isExpanded, setIsExpanded] = useState(false);

  const handleReplace = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onReplace(cam.id, file);
    }
    if (event.target) event.target.value = '';
  };

  return (
    <>
      {/* Темний напівпрозорий фон, який з'являється тільки якщо відео розгорнуто */}
      {isExpanded && (
        <div 
          className="fixed inset-0 bg-slate-900/80 z-40 backdrop-blur-sm transition-opacity"
          onClick={() => setIsExpanded(false)} // Якщо клікнути повз відео — воно згорнеться
        />
      )}

      {/* Сама картка камери. Її класи динамічно змінюються залежно від isExpanded */}
      <Card className={
        isExpanded 
          ? "fixed inset-4 md:inset-10 z-50 rounded-2xl shadow-2xl border-slate-200 overflow-hidden group flex flex-col bg-white transition-all duration-300"
          : "rounded-2xl shadow-lg border-slate-200 overflow-hidden group flex flex-col relative transition-all duration-300"
      }>
        <CardContent className={`p-4 relative flex-1 flex flex-col ${isExpanded ? 'h-full' : ''}`}>
          
          {/* ВЕРХНІЙ ПРАВИЙ КУТОК (Хрестик та Заміна) */}
          <div className={`absolute top-6 right-6 flex gap-2 z-10 transition-opacity duration-200 ${isExpanded ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}>
            <input
              type="file"
              accept="video/mp4,video/x-m4v,video/*"
              className="hidden"
              ref={replaceInputRef}
              onChange={handleReplace}
            />
            
            <button 
              onClick={() => replaceInputRef.current?.click()}
              className="p-2 bg-slate-900/70 hover:bg-blue-500 text-white rounded-lg backdrop-blur-sm transition-colors"
              title="Замінити відео"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
            </button>

            <button 
              onClick={() => {
                setIsExpanded(false); // Спочатку згортаємо
                setTimeout(() => onRemove(cam.id), 100); // Потім видаляємо
              }}
              className="p-2 bg-slate-900/70 hover:bg-red-500 text-white rounded-lg backdrop-blur-sm transition-colors"
              title="Закрити камеру"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
            </button>
          </div>

          {/* КОНТЕЙНЕР ВІДЕО */}
          <div className="relative flex-1 w-full bg-slate-900 rounded-xl overflow-hidden flex items-center justify-center min-h-[200px]">
            {cam.status === "streaming" ? (
              <img
                src={cam.streamUrl}
                // Якщо на весь екран - object-contain (щоб не обрізало), інакше object-cover
                className={`w-full h-full ${isExpanded ? 'object-contain' : 'object-cover'}`}
                alt={`Live Stream ${cam.id}`}
              />
            ) : (
              <div className="text-slate-400 text-sm animate-pulse">Очікування відео...</div>
            )}
            
            <div className="absolute top-2 left-2 bg-black/60 text-white px-2 py-1 rounded text-xs font-medium pointer-events-none z-10">
              🎥 Камера {cam.id}
            </div>

            {/* НИЖНІЙ ПРАВИЙ КУТОК (Розгорнути / Згорнути) */}
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className={`absolute bottom-4 right-4 p-2 bg-slate-900/70 hover:bg-slate-700 text-white rounded-lg backdrop-blur-sm transition-all duration-200 z-10 ${isExpanded ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}
              title={isExpanded ? "Згорнути відео" : "Відкрити на весь екран"}
            >
              {isExpanded ? (
                // Іконка "Згорнути" (Стрілочки всередину)
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M8 3v3a2 2 0 0 1-2 2H3"/><path d="M21 8h-3a2 2 0 0 1-2-2V3"/><path d="M3 16h3a2 2 0 0 1 2 2v3"/><path d="M16 21v-3a2 2 0 0 1 2-2h3"/></svg>
              ) : (
                // Іконка "Розгорнути" (Стрілочки назовні)
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M15 3h6v6"/><path d="M9 21H3v-6"/><path d="M21 3l-7 7"/><path d="M3 21l7-7"/></svg>
              )}
            </button>
          </div>

        </CardContent>
      </Card>
    </>
  );
}

function AddCameraCard({ onUpload }: { onUpload: (file: File) => void }) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onUpload(file);
    }
    if (event.target) event.target.value = '';
  };

  return (
    <>
      <input
        type="file"
        accept="video/mp4,video/x-m4v,video/*"
        className="hidden"
        ref={fileInputRef}
        onChange={handleFileChange}
      />
      <div
        onClick={() => fileInputRef.current?.click()}
        className="aspect-video border-2 border-dashed border-slate-300 rounded-2xl flex flex-col items-center justify-center cursor-pointer hover:bg-slate-50 transition"
      >
        <span className="text-4xl text-slate-400">＋</span>
        <span className="text-sm text-slate-500 mt-2">Додати камеру</span>
      </div>
    </>
  );
}

export default function Dashboard() {
  const [cameras, setCameras] = useState<CameraData[]>([]);

  const uploadVideoToServer = async (file: File) => {
    const maxId = cameras.length > 0 ? Math.max(...cameras.map(c => c.id)) : 0;
    const newCamId = maxId + 1;
    
    setCameras((prev) => [...prev, { id: newCamId, streamUrl: "", status: "idle" }]);
    await sendVideoToBackend(file, newCamId);
  };

  const replaceCameraVideo = async (id: number, file: File) => {
    setCameras((prev) => prev.map(c => c.id === id ? { ...c, streamUrl: "", status: "idle" } : c));
    await sendVideoToBackend(file, id);
  };

  const sendVideoToBackend = async (file: File, camId: number) => {
    const formData = new FormData();
    formData.append("video", file);

    try {
      const response = await fetch("http://localhost:8000/upload_video", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      
      if (data.stream_url) {
        setCameras((prev) => 
          prev.map(c => c.id === camId ? { ...c, streamUrl: data.stream_url, status: "streaming" } : c)
        );
      }
    } catch (error) {
      console.error("Бекенд не відповідає:", error);
      alert("Помилка сервера! Переконайтеся, що Python FastAPI працює.");
    }
  };

  const removeCamera = (id: number) => {
    setCameras((prev) => prev.filter(c => c.id !== id));
  };

  return (
    <div className="min-h-screen bg-slate-50 p-6 text-slate-900 font-sans">
      <div className="max-w-6xl mx-auto grid gap-6">
        
        {/* Header */}
        <div className="flex justify-between items-center bg-white p-4 rounded-2xl shadow-sm border border-slate-200">
          <h1 className="text-2xl font-bold">☕ Cafe Monitor</h1>
          <div className="flex gap-2">
            <Button className="bg-slate-900 text-white hover:bg-slate-800">Live</Button>
            <Button variant="outline">Analytics</Button>
          </div>
        </div>

        {/* Cameras Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {cameras.map((cam) => (
            <CameraView 
              key={cam.id} 
              cam={cam} 
              onRemove={removeCamera}
              onReplace={replaceCameraVideo}
            />
          ))}
          <AddCameraCard onUpload={uploadVideoToServer} />
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="rounded-2xl shadow-sm border-slate-200">
            <CardContent className="p-6">
              <h2 className="font-semibold mb-4 text-lg border-b pb-2">🧍 Аналітика сесії</h2>
              <ul className="space-y-2 text-sm text-slate-600">
                {cameras.length > 0 ? (
                  <li className="text-green-600 font-medium animate-pulse">Система активна. Стрімів: {cameras.length}</li>
                ) : (
                  <li className="text-slate-400 italic">Очікування підключення камер...</li>
                )}
              </ul>
            </CardContent>
          </Card>

          <Card className="rounded-2xl shadow-sm border-slate-200">
            <CardContent className="p-6">
              <h2 className="font-semibold mb-4 text-lg border-b pb-2">📊 Статистика YOLO</h2>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-slate-500">Активних стрімів:</span>
                  <span className="font-bold text-xl">{cameras.length}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
        
      </div>
    </div>
  );
}