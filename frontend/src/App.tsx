import { Card, CardContent } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, LogOut, User } from "lucide-react";
import Analytics from "./pages/Analytics";
import Login from "./pages/Login";
import Register from "./pages/Register";

type SessionStatus = "idle" | "running" | "finished";
type ActiveTab = "live" | "analytics";

interface CameraData {
  id: number;
  streamUrl: string;
  thumbnailUrl?: string;
  filename?: string;
}

interface Point {
  x: number;
  y: number;
}

interface OccupancyPoint {
  x: number;
  y: number;
  is_staff?: boolean;
}

interface OccupancyHistoryPoint extends OccupancyPoint {
  timestamp: number;
}

interface MapZone {
  id: string;
  name: string;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

interface MyCafe {
  name: string;
  address?: string;
  floor_plan_url: string;
}

const GENERIC_THUMBNAIL_URL =
  "data:image/svg+xml;utf8," +
  encodeURIComponent(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">
      <defs>
        <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#0f172a" />
          <stop offset="100%" stop-color="#1e293b" />
        </linearGradient>
      </defs>
      <rect width="1280" height="720" fill="url(#bg)" />
      <rect x="140" y="100" width="1000" height="520" rx="26" fill="none" stroke="#475569" stroke-width="10" stroke-dasharray="18 14" />
      <text x="640" y="360" text-anchor="middle" font-family="Arial, sans-serif" font-size="44" fill="#e2e8f0">
        Preview unavailable
      </text>
    </svg>
  `);

const API_ROOT = "http://localhost:8000";
const CALIBRATION_POINT_COLORS = ["#22c55e", "#38bdf8", "#f59e0b", "#f43f5e"];

const getAuthToken = () => localStorage.getItem("cafe_token") || "";

const getAuthHeaders = (includeJson = false) => {
  const token = getAuthToken();
  return {
    ...(includeJson ? { "Content-Type": "application/json" } : {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
};

const normalizeMapZone = (zone: Partial<MapZone>, index: number): MapZone => {
  const xMin = Number(zone.xMin ?? 0);
  const xMax = Number(zone.xMax ?? xMin);
  const yMin = Number(zone.yMin ?? 0);
  const yMax = Number(zone.yMax ?? yMin);

  return {
    id: String(zone.id || `zone_${index + 1}`),
    name: String(zone.name || `Zone ${index + 1}`),
    xMin: Math.min(xMin, xMax),
    xMax: Math.max(xMin, xMax),
    yMin: Math.min(yMin, yMax),
    yMax: Math.max(yMin, yMax),
  };
};

async function loadFloorPlanZones(): Promise<MapZone[]> {
  const response = await fetch(`${API_ROOT}/api/floor_plan/zones`, {
    headers: getAuthHeaders(),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Не вдалося завантажити зони floor plan.");
  }

  return Array.isArray(data.zones)
    ? data.zones.map((zone: Partial<MapZone>, index: number) => normalizeMapZone(zone, index))
    : [];
}

async function saveFloorPlanZones(zones: MapZone[]): Promise<MapZone[]> {
  const response = await fetch(`${API_ROOT}/api/floor_plan/zones`, {
    method: "POST",
    headers: getAuthHeaders(true),
    body: JSON.stringify(zones),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Не вдалося зберегти зони floor plan.");
  }

  return Array.isArray(data.zones)
    ? data.zones.map((zone: Partial<MapZone>, index: number) => normalizeMapZone(zone, index))
    : [];
}

async function uploadFloorPlanFile(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_ROOT}/api/cafes/me/floor-plan`, {
    method: "POST",
    headers: getAuthHeaders(),
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Не вдалося завантажити floor plan.");
  }

  const data = await response.json();
  return data.floor_plan_url as string;
}

function CalibrationModal({
  cam,
  onClose,
  floorPlanImg,
  setFloorPlanImg,
}: {
  cam: CameraData;
  onClose: () => void;
  floorPlanImg: string;
  setFloorPlanImg: React.Dispatch<React.SetStateAction<string>>;
}) {
  const [srcPts, setSrcPts] = useState<Point[]>([]);
  const [dstPts, setDstPts] = useState<Point[]>([]);
  const [showFloorPlanUpload, setShowFloorPlanUpload] = useState(false);
  const [isUploadingFloorPlan, setIsUploadingFloorPlan] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    setShowFloorPlanUpload(false);
  }, []);

  useEffect(() => {
    if (floorPlanImg) {
      setShowFloorPlanUpload(false);
    }
  }, [floorPlanImg]);

  useEffect(() => {
    let cancelled = false;

    const loadCalibration = async () => {
      if (!cam.filename) {
        setSrcPts([]);
        setDstPts([]);
        return;
      }

      try {
        const response = await fetch(`${API_ROOT}/calibration/${encodeURIComponent(cam.filename)}`, {
          headers: getAuthHeaders(),
        });
        const data = await response.json();
        if (!cancelled) {
          setSrcPts(Array.isArray(data.src_pts) ? data.src_pts.slice(0, 4) : []);
          setDstPts(Array.isArray(data.dst_pts) ? data.dst_pts.slice(0, 4) : []);
        }
      } catch (error) {
        if (!cancelled) {
          console.error("Помилка завантаження калібрування:", error);
          setSrcPts([]);
          setDstPts([]);
        }
      }
    };

    loadCalibration();
    return () => {
      cancelled = true;
    };
  }, [cam.filename]);

  const addPoint =
    (setPoints: React.Dispatch<React.SetStateAction<Point[]>>) =>
    (event: React.MouseEvent<HTMLDivElement>) => {
      const rect = event.currentTarget.getBoundingClientRect();
      setPoints((prev) => {
        if (prev.length >= 4) {
          return prev;
        }
        return [
          ...prev,
          {
            x: (event.clientX - rect.left) / rect.width,
            y: (event.clientY - rect.top) / rect.height,
          },
        ];
      });
    };

  const uploadFloorPlan = async (file: File) => {
    setIsUploadingFloorPlan(true);

    try {
      const floorPlanUrl = await uploadFloorPlanFile(file);
      setShowFloorPlanUpload(false);
      setFloorPlanImg(floorPlanUrl);
    } catch (error) {
      console.error("Помилка завантаження floor plan:", error);
      alert("Не вдалося завантажити floor plan.");
    } finally {
      setIsUploadingFloorPlan(false);
    }
  };

  const saveCalibration = async () => {
    if (!cam.filename) {
      alert("Файл відео ще не готовий для калібрування.");
      return;
    }

    setIsSaving(true);
    try {
      const response = await fetch(`${API_ROOT}/calibrate_camera/${encodeURIComponent(cam.filename)}`, {
        method: "POST",
        headers: getAuthHeaders(true),
        body: JSON.stringify({ src_pts: srcPts, dst_pts: dstPts }),
      });

      if (!response.ok) {
        throw new Error("Не вдалося зберегти калібрування.");
      }

      onClose();
    } catch (error) {
      console.error("Помилка збереження калібрування:", error);
      alert("Не вдалося зберегти калібрування.");
    } finally {
      setIsSaving(false);
    }
  };

  const renderPointOverlay = (points: Point[]) => (
    <svg className="pointer-events-none absolute inset-0 h-full w-full">
      {points.map((point, index) => {
        const color = CALIBRATION_POINT_COLORS[index % CALIBRATION_POINT_COLORS.length];
        return (
          <g key={`${point.x}-${point.y}-${index}`}>
            <circle
              cx={`${point.x * 100}%`}
              cy={`${point.y * 100}%`}
              r="14"
              fill={color}
              stroke="#ffffff"
              strokeWidth="2"
            />
            <text
              x={`${point.x * 100}%`}
              y={`${point.y * 100}%`}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize="12"
              fontWeight="700"
              fill="#0f172a"
            >
              {index + 1}
            </text>
          </g>
        );
      })}
    </svg>
  );

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-slate-900/95 p-6">
      <div className="mb-6 flex items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">Калібрування: {cam.filename || "Нова камера"}</h2>
          <p className="text-sm text-slate-300">Оберіть 4 відповідні точки на кадрі та на floor plan.</p>
        </div>
        <Button variant="outline" className="border-slate-600 bg-slate-900 text-white hover:bg-slate-800" onClick={onClose}>
          Close
        </Button>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-2 gap-6">
        <div className="flex min-h-0 flex-col rounded-2xl border border-slate-700 bg-slate-950/60 p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-white">Камера</h3>
            <span className="text-sm text-slate-400">{srcPts.length}/4 точок</span>
          </div>
          <div
            className="relative flex min-h-0 flex-1 cursor-crosshair items-center justify-center overflow-hidden rounded-xl bg-black"
            onClick={addPoint(setSrcPts)}
          >
            <img
              src={cam.thumbnailUrl || GENERIC_THUMBNAIL_URL}
              alt="Camera thumbnail"
              className="h-full w-full object-contain"
              onError={(event) => {
                event.currentTarget.src = GENERIC_THUMBNAIL_URL;
              }}
            />
            {renderPointOverlay(srcPts)}
          </div>
        </div>

        <div className="flex min-h-0 flex-col rounded-2xl border border-slate-700 bg-slate-950/60 p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-white">Floor Plan</h3>
            <span className="text-sm text-slate-400">{dstPts.length}/4 точок</span>
          </div>
          <div
            className="relative flex min-h-0 flex-1 cursor-crosshair items-center justify-center overflow-hidden rounded-xl bg-black"
            onClick={!showFloorPlanUpload ? addPoint(setDstPts) : undefined}
          >
            {showFloorPlanUpload ? (
              <div className="flex h-full w-full flex-col items-center justify-center gap-4 border-2 border-dashed border-slate-600 px-6 text-center">
                <div className="space-y-2">
                  <h4 className="text-lg font-semibold text-white">Floor plan не знайдено</h4>
                  <p className="text-sm text-slate-300">Завантажте зображення плану приміщення, щоб продовжити калібрування.</p>
                </div>
                <input
                  type="file"
                  accept="image/*"
                  className="block w-full max-w-xs cursor-pointer text-sm text-slate-200 file:mr-4 file:rounded-lg file:border-0 file:bg-emerald-500 file:px-4 file:py-2 file:font-medium file:text-white hover:file:bg-emerald-600"
                  disabled={isUploadingFloorPlan}
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) {
                      void uploadFloorPlan(file);
                    }
                    event.target.value = "";
                  }}
                />
                {isUploadingFloorPlan && <p className="text-sm text-slate-400">Завантаження...</p>}
              </div>
            ) : (
              <>
                <label className="absolute top-3 right-3 z-20 cursor-pointer rounded-lg bg-slate-900/70 px-3 py-2 text-xs font-medium text-white shadow-sm backdrop-blur-sm hover:bg-slate-800">
                  Змінити план
                  <input
                    type="file"
                    accept="image/*"
                    className="hidden"
                    disabled={isUploadingFloorPlan}
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) {
                        void uploadFloorPlan(file);
                      }
                      event.target.value = "";
                    }}
                  />
                </label>
                <img
                  src={floorPlanImg}
                  alt="Floor plan"
                  className="h-full w-full object-contain"
                  onError={() => setShowFloorPlanUpload(true)}
                />
                {renderPointOverlay(dstPts)}
              </>
            )}
          </div>
        </div>
      </div>

      <div className="mt-6 flex items-center justify-between rounded-2xl border border-slate-700 bg-slate-950/70 p-4">
        <p className="text-sm text-slate-300">
          Натисніть по зображеннях у порядку 1-4. Точки на камері та floor plan мають відповідати одна одній.
        </p>
        <div className="flex gap-3">
          <Button
            variant="outline"
            className="border-slate-600 bg-slate-900 text-white hover:bg-slate-800"
            onClick={() => {
              setSrcPts([]);
              setDstPts([]);
            }}
          >
            Очистити точки
          </Button>
          <Button
            className="bg-emerald-500 text-white hover:bg-emerald-600"
            onClick={saveCalibration}
            disabled={srcPts.length !== 4 || dstPts.length !== 4 || isSaving}
          >
            Зберегти
          </Button>
        </div>
      </div>
    </div>
  );
}

function CameraView({
  cam,
  cameraIndex,
  sessionStatus,
  onRemove,
  onReplace,
  floorPlanImg,
  setFloorPlanImg,
}: {
  cam: CameraData;
  cameraIndex: number;
  sessionStatus: SessionStatus;
  onRemove: (id: number) => void;
  onReplace: (id: number, file: File) => void;
  floorPlanImg: string;
  setFloorPlanImg: React.Dispatch<React.SetStateAction<string>>;
}) {
  const replaceInputRef = useRef<HTMLInputElement>(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  const [savedZonePolygon, setSavedZonePolygon] = useState<Point[]>([]);
  const [zonePoints, setZonePoints] = useState<Point[]>([]);
  const [thumbnailSrc, setThumbnailSrc] = useState(cam.thumbnailUrl || GENERIC_THUMBNAIL_URL);
  const [streamSrc, setStreamSrc] = useState("");
  const [streamEnded, setStreamEnded] = useState(false);
  const [showCalibration, setShowCalibration] = useState(false);

  useEffect(() => {
    setThumbnailSrc(cam.thumbnailUrl || GENERIC_THUMBNAIL_URL);
  }, [cam.thumbnailUrl]);

  useEffect(() => {
    if (sessionStatus === "running" && cam.streamUrl) {
      setStreamEnded(false);
      setStreamSrc(cam.streamUrl);
      return;
    }
    setStreamSrc("");
  }, [cam.streamUrl, sessionStatus]);

  const saveZone = async () => {
    if (zonePoints.length < 3) {
      alert("Намалюйте мінімум 3 точки!");
      return;
    }
    if (!cam.filename) {
      alert("Файл відео ще не готовий для збереження зони.");
      return;
    }

    const pointsToSave = [...zonePoints];
    setSavedZonePolygon(pointsToSave);
    setIsDrawing(false);
    setZonePoints([]);
    try {
      await fetch(`http://localhost:8000/set_zone/${encodeURIComponent(cam.filename)}`, {
        method: "POST",
        headers: getAuthHeaders(true),
        body: JSON.stringify(pointsToSave),
      });
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <>
      {isExpanded && (
        <div
          className="fixed inset-0 z-40 bg-slate-900/80 backdrop-blur-sm"
          onClick={() => !isDrawing && setIsExpanded(false)}
        />
      )}

      {showConfirm && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-slate-900/50 p-4 backdrop-blur-sm">
          <div className="w-full max-w-sm rounded-2xl bg-white p-6 shadow-2xl">
            <h3 className="mb-4 text-lg font-bold text-[#411e03]">Видалити камеру?</h3>
            <p className="mb-6 text-sm text-[#846851]">
              Ви впевнені, що хочете видалити цей стрім? Це розірве з&apos;єднання для поточної сесії.
            </p>
            <div className="flex justify-end gap-3">
              <Button variant="outline" onClick={() => setShowConfirm(false)}>
                Скасувати
              </Button>
              <Button
                className="bg-red-500 text-white hover:bg-red-600"
                onClick={() => {
                  setShowConfirm(false);
                  onRemove(cam.id);
                }}
              >
                Видалити
              </Button>
            </div>
          </div>
        </div>
      )}

      {showCalibration && (
        <CalibrationModal
          cam={cam}
          onClose={() => setShowCalibration(false)}
          floorPlanImg={floorPlanImg}
          setFloorPlanImg={setFloorPlanImg}
        />
      )}

      <Card
        className={
          isExpanded
            ? "fixed inset-4 z-50 flex flex-col overflow-hidden rounded-2xl bg-white shadow-2xl transition-all duration-300 md:inset-10"
            : "group relative flex flex-col overflow-hidden rounded-2xl shadow-lg transition-all duration-300"
        }
      >
        <CardContent className="relative flex min-h-0 flex-1 flex-col p-4">
          <div
            className={`pointer-events-none absolute top-6 left-6 z-10 flex items-center rounded-lg border border-slate-700/50 bg-slate-900/80 px-3 py-1.5 text-sm font-medium tracking-wide text-white shadow-sm backdrop-blur-md transition-all duration-300 ${
              isExpanded ? "opacity-100" : "opacity-0 group-hover:opacity-100"
            }`}
          >
            Camera {cameraIndex + 1}
          </div>

          {sessionStatus === "idle" && !isDrawing && (
            <div
              className={`absolute top-6 right-6 z-20 flex gap-2 transition-opacity duration-200 ${
                isExpanded ? "opacity-100" : "opacity-0 group-hover:opacity-100"
              }`}
            >
              <input
                type="file"
                accept="video/mp4,video/x-m4v,video/*"
                className="hidden"
                ref={replaceInputRef}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    onReplace(cam.id, file);
                  }
                  e.target.value = "";
                }}
              />
              <button
                onClick={() => {
                  setIsDrawing(true);
                  setZonePoints([]);
                }}
                className="rounded-lg bg-slate-900/70 p-2 text-white shadow-sm backdrop-blur-sm hover:bg-emerald-500"
                title="Визначити зону стаффу"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>
              </button>
              <button
                onClick={() => setShowCalibration(true)}
                className="rounded-lg bg-slate-900/70 p-2 text-white shadow-sm backdrop-blur-sm hover:bg-cyan-500 disabled:cursor-not-allowed disabled:opacity-50"
                title="Калібрування камери"
                disabled={!cam.filename}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2v4"/><path d="M12 18v4"/><path d="M2 12h4"/><path d="M18 12h4"/><circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="8"/></svg>
              </button>
              <button
                onClick={() => replaceInputRef.current?.click()}
                className="rounded-lg bg-slate-900/70 p-2 text-white shadow-sm backdrop-blur-sm hover:bg-blue-500"
                title="Замінити відео"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
              </button>
              <button
                onClick={() => setShowConfirm(true)}
                className="rounded-lg bg-slate-900/70 p-2 text-white shadow-sm backdrop-blur-sm hover:bg-red-500"
                title="Видалити камеру"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
              </button>
            </div>
          )}

          <div
            className={`relative aspect-video min-h-[250px] flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-xl bg-slate-950 ${
              isDrawing ? "cursor-crosshair ring-4 ring-emerald-500" : ""
            }`}
            onClick={(e) => {
              if (!isDrawing) return;
              const rect = e.currentTarget.getBoundingClientRect();
              setZonePoints((prev) => [
                ...prev,
                {
                  x: (e.clientX - rect.left) / rect.width,
                  y: (e.clientY - rect.top) / rect.height,
                },
              ]);
            }}
          >
            {sessionStatus === "running" && streamSrc && !streamEnded ? (
              <img
                src={streamSrc}
                className="h-full w-full object-contain"
                alt="Stream"
                onError={() => setStreamEnded(true)}
              />
            ) : (
              <img
                src={thumbnailSrc}
                onError={() => setThumbnailSrc(GENERIC_THUMBNAIL_URL)}
                className="h-full w-full animate-in object-contain fade-in"
                alt="Thumbnail for Zone Setup"
              />
            )}

            {streamEnded && (
              <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/60">
                <span className="text-lg font-bold text-white">🏁 Трансляція завершена</span>
              </div>
            )}

            <svg
              className="pointer-events-none absolute inset-0 z-10 h-full w-full"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
            >
              {!isDrawing && savedZonePolygon.length > 0 && (
                <polygon
                  points={savedZonePolygon.map((p) => `${p.x * 100},${p.y * 100}`).join(" ")}
                  fill="rgba(255, 0, 0, 0.08)"
                  stroke="red"
                  strokeOpacity="0.7"
                  strokeWidth="2"
                />
              )}
              {isDrawing && zonePoints.length > 1 && (
                <polyline
                  points={zonePoints.map((p) => `${p.x * 100},${p.y * 100}`).join(" ")}
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="1.5"
                  strokeDasharray="2 2"
                />
              )}
              {isDrawing &&
                zonePoints.map((p, index) => (
                  <g key={index}>
                    <circle
                      cx={p.x * 100}
                      cy={p.y * 100}
                      r="1"
                      fill="#10b981"
                      stroke="#ffffff"
                      strokeWidth="0.6"
                    />
                    <text
                      x={p.x * 100 + 1.5}
                      y={p.y * 100 - 1.5}
                      fontSize="4"
                      fontWeight="700"
                      fill="#10b981"
                      stroke="#0f172a"
                      strokeWidth="0.3"
                      paintOrder="stroke"
                    >
                      {index + 1}
                    </text>
                  </g>
                ))}
            </svg>

            {isDrawing && (
              <div className="absolute bottom-4 left-1/2 z-40 flex -translate-x-1/2 gap-2 rounded-full border border-slate-700 bg-slate-900/95 p-2 shadow-2xl backdrop-blur-md">
                {zonePoints.length > 0 && (
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-8 border-slate-600 text-white"
                    onClick={(e) => {
                      e.stopPropagation();
                      setZonePoints((prev) => prev.slice(0, -1));
                    }}
                  >
                    Крок назад
                  </Button>
                )}
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8 border-slate-600 text-white hover:bg-slate-800"
                  onClick={(e) => {
                    e.stopPropagation();
                    setZonePoints([]);
                  }}
                >
                  Очистити
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8 border-red-400/60 text-red-200 hover:bg-red-500/20"
                  onClick={(e) => {
                    e.stopPropagation();
                    setIsDrawing(false);
                    setZonePoints([]);
                  }}
                >
                  Скасувати
                </Button>
                <Button
                  size="sm"
                  className="h-8 bg-emerald-500 text-white hover:bg-emerald-600"
                  onClick={(e) => {
                    e.stopPropagation();
                    saveZone();
                  }}
                >
                  Зберегти
                </Button>
              </div>
            )}

            {!isDrawing && (
              <div
                className={`absolute right-4 bottom-4 z-40 transition-opacity duration-200 ${
                  isExpanded ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                }`}
              >
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setIsExpanded(!isExpanded);
                  }}
                  className="rounded-lg bg-slate-900/70 p-2 text-white shadow-sm backdrop-blur-sm hover:bg-slate-700"
                  title={isExpanded ? "Зменшити екран" : "Збільшити екран"}
                >
                  {isExpanded ? (
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M8 3v3a2 2 0 0 1-2 2H3"/><path d="M21 8h-3a2 2 0 0 1-2-2V3"/><path d="M3 16h3a2 2 0 0 1 2 2v3"/><path d="M16 21v-3a2 2 0 0 1 2-2h3"/></svg>
                  ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M15 3h6v6"/><path d="M9 21H3v-6"/><path d="M21 3l-7 7"/><path d="M3 21l7-7"/></svg>
                  )}
                </button>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </>
  );
}

function LiveMonitorSidePanel({
  sessionStatus,
  camerasCount,
  totalCustomers,
  floorPlanImg,
  setFloorPlanImg,
  mapZones,
  setMapZones,
}: {
  sessionStatus: SessionStatus;
  camerasCount: number;
  totalCustomers: number;
  floorPlanImg: string;
  setFloorPlanImg: React.Dispatch<React.SetStateAction<string>>;
  mapZones: MapZone[];
  setMapZones: React.Dispatch<React.SetStateAction<MapZone[]>>;
}) {
  const [occupancy, setOccupancy] = useState<OccupancyPoint[]>([]);
  const [pointHistory, setPointHistory] = useState<OccupancyHistoryPoint[]>([]);
  const [floorPlanError, setFloorPlanError] = useState(false);
  const [floorPlanSize, setFloorPlanSize] = useState({ width: 1000, height: 1000 });
  const [mapScale, setMapScale] = useState(1);
  const [isAnalyzingZones, setIsAnalyzingZones] = useState(false);
  const floorPlanUploadRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (sessionStatus !== "running") {
      setOccupancy([]);
      setPointHistory([]);
      return;
    }

    let cancelled = false;

    const loadOccupancy = async () => {
      try {
        const response = await fetch(`${API_ROOT}/api/occupancy`, {
          headers: getAuthHeaders(),
        });
        const data = await response.json();
        if (!cancelled) {
          const nextPoints = Array.isArray(data.points) ? data.points : [];
          const now = Date.now();
          setOccupancy(nextPoints);
          setPointHistory((prev) => {
            const validOld = prev.filter((point) => now - point.timestamp < 8000);
            const newPoints = nextPoints.map((point: OccupancyPoint) => ({
              ...point,
              timestamp: now,
            }));
            return [...validOld, ...newPoints];
          });
        }
      } catch (error) {
        if (!cancelled) {
          console.error("Не вдалося завантажити live points:", error);
          setOccupancy([]);
        }
      }
    };

    void loadOccupancy();
    const intervalId = window.setInterval(loadOccupancy, 1500);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [sessionStatus]);

  const activeCustomers = occupancy.filter((point) => !point.is_staff).length;
  const activeStaff = occupancy.filter((point) => point.is_staff).length;
  const zoneCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    mapZones.forEach((zone) => {
      counts[zone.id] = 0;
    });

    let otherCount = 0;

    occupancy.forEach((point) => {
      if (point.is_staff) {
        return;
      }

      let foundZone = false;
      for (const zone of mapZones) {
        if (point.x >= zone.xMin && point.x <= zone.xMax && point.y >= zone.yMin && point.y <= zone.yMax) {
          counts[zone.id] += 1;
          foundZone = true;
          break;
        }
      }

      if (!foundZone) {
        otherCount += 1;
      }
    });

    return { counts, otherCount };
  }, [mapZones, occupancy]);
  const handleZoomIn = () => setMapScale((s) => Math.min(s + 0.2, 3));
  const handleZoomOut = () => setMapScale((s) => Math.max(s - 0.2, 0.5));

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    try {
      const floorPlanUrl = await uploadFloorPlanFile(file);
      setFloorPlanImg(floorPlanUrl);
      setMapZones([]);
      setFloorPlanError(false);
    } catch (error) {
      console.error("Помилка завантаження floor plan:", error);
      alert("Не вдалося завантажити floor plan.");
    } finally {
      event.target.value = "";
    }
  };

  const handleAutoDetectZones = async () => {
    if (!floorPlanImg) {
      return;
    }

    setIsAnalyzingZones(true);
    try {
      const response = await fetch(`${API_ROOT}/api/analyze_floor_plan`, {
        method: "POST",
        headers: getAuthHeaders(),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Не вдалося проаналізувати floor plan.");
      }

      const detectedZones = Array.isArray(data.zones)
        ? data.zones.map((zone: Partial<MapZone>, index: number) => normalizeMapZone(zone, index))
        : [];
      const savedZones = await saveFloorPlanZones(detectedZones);
      setMapZones(savedZones);
    } catch (error) {
      console.error("Помилка auto-detect зон:", error);
      alert("Не вдалося автоматично визначити зони.");
    } finally {
      setIsAnalyzingZones(false);
    }
  };

  return (
    <div className="flex h-fit self-start flex-col gap-6">
      <div className="grid grid-cols-2 gap-4">
        <Card className="rounded-2xl border-slate-200 bg-white shadow-sm">
          <CardContent className="p-4">
            <div className="mb-3 border-b pb-2">
              <h2 className="text-base font-semibold text-[#411e03]">📊 Live Stats</h2>
              <p className="mt-1 text-xs text-[#846851]">Поточні дані з live occupancy та активної сесії.</p>
            </div>
            <div className="space-y-1.5 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-[#846851]">Статус сесії</span>
                <span className="font-semibold text-[#411e03]">
                  {sessionStatus === "idle" ? "Очікує запуску" : sessionStatus}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[#846851]">Активних стрімів</span>
                <span className="font-bold text-[#411e03]">{sessionStatus === "running" ? camerasCount : 0}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[#846851]">Точок на карті</span>
                <span className="font-bold text-[#411e03]">{occupancy.length}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[#846851]">Клієнтів на карті</span>
                <span className="font-semibold text-blue-600">{activeCustomers}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[#846851]">Staff на карті</span>
                <span className="font-semibold text-emerald-600">{activeStaff}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[#846851]">Клієнтів у закладі</span>
                <span className="font-semibold text-[#411e03]">{totalCustomers}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="rounded-2xl border-slate-200 bg-white shadow-sm">
          <CardContent className="p-4">
            <h2 className="mb-2 text-base font-semibold text-[#411e03]">🧍 Розподіл по зонах</h2>
            <ul className="space-y-1 text-sm text-slate-600">
              {mapZones.map((zone) => (
                <li key={zone.id} className="flex items-center justify-between border-b border-slate-100 pb-0.5 last:border-0">
                  <span>{zone.name}</span>
                  <span className="font-medium text-[#411e03]">
                    {zoneCounts.counts[zone.id]} {zoneCounts.counts[zone.id] === 1 ? "людина" : "людей"}
                  </span>
                </li>
              ))}
              <li className="flex items-center justify-between pt-0.5">
                <span className="text-[#846851]">В русі / Інше</span>
                <span className="font-medium text-[#846851]">{zoneCounts.otherCount}</span>
              </li>
            </ul>
          </CardContent>
        </Card>
      </div>

      <Card className="h-fit min-h-[420px] rounded-2xl border-slate-200 bg-white shadow-sm">
        <CardContent className="p-6">
          <div className="mb-4 border-b pb-2">
            <h2 className="text-lg font-semibold text-[#411e03]">Live 2D Map (Дедупліковано)</h2>
          </div>
          <div className="relative group/map h-auto w-full overflow-hidden rounded-lg">
            <input
              ref={floorPlanUploadRef}
              type="file"
              accept="image/*"
              className="hidden"
              id="floorPlanUpload"
              onChange={handleImageUpload}
            />
            {floorPlanImg && !floorPlanError && (
              <div className="absolute top-4 right-4 z-10 flex flex-col items-end gap-2 opacity-0 transition-all duration-300 group-hover/map:opacity-100">
                <button
                  type="button"
                  onClick={() => void handleAutoDetectZones()}
                  className="group flex items-center overflow-hidden rounded-full border border-slate-200 bg-white/90 p-2 shadow-md backdrop-blur transition-all duration-300 hover:pl-4 disabled:cursor-not-allowed disabled:opacity-60"
                  title="Авто-розпізнавання зон"
                  disabled={isAnalyzingZones}
                >
                  <span className="max-w-0 overflow-hidden whitespace-nowrap text-sm font-medium text-slate-700 opacity-0 transition-all duration-300 group-hover:mr-2 group-hover:max-w-[160px] group-hover:opacity-100">
                    Авто-розпізнавання
                  </span>
                  <span className="text-lg flex-shrink-0">{isAnalyzingZones ? "…" : "🔍"}</span>
                </button>

                <button
                  type="button"
                  onClick={() => document.getElementById("floorPlanUpload")?.click()}
                  className="group flex items-center overflow-hidden rounded-full border border-slate-200 bg-white/90 p-2 shadow-md backdrop-blur transition-all duration-300 hover:pl-4"
                  title="Змінити план"
                >
                  <span className="max-w-0 overflow-hidden whitespace-nowrap text-sm font-medium text-slate-700 opacity-0 transition-all duration-300 group-hover:mr-2 group-hover:max-w-[120px] group-hover:opacity-100">
                    Змінити план
                  </span>
                  <span className="text-lg flex-shrink-0">🗺️</span>
                </button>
              </div>
            )}
            {floorPlanImg && !floorPlanError ? (
              <div className="h-auto w-full overflow-auto">
                <div
                  className="relative h-auto w-full"
                  style={{
                    transform: `scale(${mapScale})`,
                    transformOrigin: "top left",
                    transition: "transform 0.2s ease-in-out",
                  }}
                >
                  <img
                    src={floorPlanImg}
                    alt="Floor plan"
                    className="block h-auto w-full"
                    onLoad={(event) => {
                      setFloorPlanError(false);
                      setFloorPlanSize({
                        width: event.currentTarget.naturalWidth || 1000,
                        height: event.currentTarget.naturalHeight || 1000,
                      });
                    }}
                    onError={() => setFloorPlanError(true)}
                  />
                  <svg
                    className="pointer-events-none absolute inset-0 h-full w-full"
                    viewBox={`0 0 ${floorPlanSize.width} ${floorPlanSize.height}`}
                    preserveAspectRatio="xMidYMid meet"
                  >
                    {mapZones.map((zone) => (
                      <g key={zone.id}>
                        <rect
                          x={zone.xMin}
                          y={zone.yMin}
                          width={Math.max(0, zone.xMax - zone.xMin)}
                          height={Math.max(0, zone.yMax - zone.yMin)}
                          fill="rgba(59, 130, 246, 0.2)"
                          stroke="#2563eb"
                          strokeWidth="2"
                        />
                        <text
                          x={zone.xMin + 8}
                          y={zone.yMin + 20}
                          fill="#dbeafe"
                          fontSize="16"
                          fontWeight="700"
                          paintOrder="stroke"
                          stroke="#1e3a8a"
                          strokeWidth="0.6"
                        >
                          {zone.name}
                        </text>
                      </g>
                    ))}
                    {pointHistory.map((pt, i) => {
                      const age = Date.now() - pt.timestamp;
                      const fadeRatio = 1 - age / 8000;
                      if (fadeRatio <= 0) {
                        return null;
                      }
                      return (
                        <circle
                          key={`${pt.timestamp}-${pt.x}-${pt.y}-${i}`}
                          cx={pt.x}
                          cy={pt.y}
                          r={4 * fadeRatio}
                          fill={pt.is_staff ? "#22c55e" : "#3b82f6"}
                          opacity={fadeRatio * 0.5}
                        />
                      );
                    })}
                    {occupancy.map((point, index) => (
                      <g key={`${point.x}-${point.y}-${index}`}>
                        <circle
                          cx={point.x}
                          cy={point.y}
                          r="22"
                          fill={point.is_staff ? "#22c55e" : "#3b82f6"}
                          opacity="0.18"
                          className="animate-ping"
                        />
                        <circle cx={point.x} cy={point.y} r="8" fill={point.is_staff ? "#22c55e" : "#3b82f6"} opacity="0.95" />
                        <circle
                          cx={point.x}
                          cy={point.y}
                          r="12"
                          fill="none"
                          stroke={point.is_staff ? "#86efac" : "#93c5fd"}
                          strokeWidth="2"
                          opacity="0.9"
                        />
                      </g>
                    ))}
                  </svg>
                </div>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => floorPlanUploadRef.current?.click()}
                className="flex h-[320px] w-full flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-300 bg-white/60 px-6 text-center transition hover:bg-white"
              >
                <span className="mb-3 flex h-14 w-14 items-center justify-center rounded-full bg-[#f5f3f0] text-3xl text-[#846851]">
                  +
                </span>
                <span className="text-base font-semibold text-[#411e03]">Завантажити план приміщення</span>
                <span className="mt-2 text-sm text-[#846851]">Додайте floor plan для вашого кафе</span>
              </button>
            )}
            <div className="absolute right-4 bottom-4 z-10 flex flex-col gap-2 rounded-lg bg-white/80 p-1 opacity-0 shadow backdrop-blur transition-all duration-300 group-hover/map:opacity-100">
              <button
                type="button"
                className="flex h-9 w-9 items-center justify-center rounded-md text-lg font-semibold text-[#411e03] hover:bg-white"
                onClick={handleZoomIn}
              >
                +
              </button>
              <button
                type="button"
                className="flex h-9 w-9 items-center justify-center rounded-md text-lg font-semibold text-[#411e03] hover:bg-white"
                onClick={handleZoomOut}
              >
                -
              </button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function AddCameraCard({ onUpload }: { onUpload: (file: File) => void }) {
  const ref = useRef<HTMLInputElement>(null);

  return (
    <div
      onClick={() => ref.current?.click()}
      className="group aspect-video cursor-pointer rounded-2xl border-2 border-dashed border-slate-300 transition-all duration-300 hover:bg-white"
    >
      <input
        type="file"
        className="hidden"
        ref={ref}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) {
            onUpload(file);
          }
          e.target.value = "";
        }}
      />
      <div className="flex h-full flex-col items-center justify-center">
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-slate-100 text-slate-400 transition-transform group-hover:scale-110">
          ＋
        </div>
        <span className="mt-3 text-sm font-medium text-[#846851] group-hover:text-emerald-600">
          Додати нову камеру
        </span>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [isAuthenticated, setIsAuthenticated] = useState(() => !!localStorage.getItem("cafe_token"));
  const [authView, setAuthView] = useState<"login" | "register">("login");
  const [currentUser, setCurrentUser] = useState<{ full_name: string; email: string } | null>(null);
  const [myCafe, setMyCafe] = useState<MyCafe | null>(null);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [cameras, setCameras] = useState<CameraData[]>([]);
  const [sessionStatus, setSessionStatus] = useState<SessionStatus>("idle");
  const [activeTab, setActiveTab] = useState<ActiveTab>("live");
  const [totalCustomers, setTotalCustomers] = useState(0);
  const [floorPlanImg, setFloorPlanImg] = useState("");
  const [mapZones, setMapZones] = useState<MapZone[]>([]);
  const [isEndingSession, setIsEndingSession] = useState(false);
  const nextCameraIdRef = useRef(1);
  const profileDropdownRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (!profileDropdownRef.current?.contains(event.target as Node)) {
        setIsProfileOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  useEffect(() => {
    const fetchUser = async () => {
      const token = localStorage.getItem("cafe_token");
      if (!token) return;

      try {
        const response = await fetch("http://127.0.0.1:8000/api/auth/me", {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (response.ok) {
          const data = await response.json();
          setCurrentUser(data);
        } else {
          localStorage.removeItem("cafe_token");
          setCurrentUser(null);
          setIsAuthenticated(false);
        }
      } catch (error) {
        console.error("Error fetching user", error);
      }
    };

    if (isAuthenticated) {
      void fetchUser();
    } else {
      setCurrentUser(null);
    }
  }, [isAuthenticated]);

  useEffect(() => {
    const fetchCafe = async () => {
      const token = localStorage.getItem("cafe_token");
      if (!token) return;

      try {
        const response = await fetch("http://127.0.0.1:8000/api/cafes/me", {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (!response.ok) {
          throw new Error("Не вдалося завантажити дані кафе.");
        }

        const data = await response.json();
        setMyCafe(data);
        setFloorPlanImg(data.floor_plan_url || "");
      } catch (error) {
        console.error("Помилка завантаження кафе:", error);
        setMyCafe(null);
        setFloorPlanImg("");
      }
    };

    if (isAuthenticated) {
      void fetchCafe();
    } else {
      setMyCafe(null);
      setFloorPlanImg("");
    }
  }, [isAuthenticated]);

  useEffect(() => {
    let cancelled = false;

    const syncFloorPlanZones = async () => {
      try {
        const zones = await loadFloorPlanZones();
        if (!cancelled) {
          setMapZones(zones);
        }
      } catch (error) {
        if (!cancelled) {
          console.error("Помилка завантаження зон floor plan:", error);
          setMapZones([]);
        }
      }
    };

    void syncFloorPlanZones();
    return () => {
      cancelled = true;
    };
  }, [floorPlanImg]);

  useEffect(() => {
    if (!floorPlanImg) {
      setMyCafe((prev) => (prev ? { ...prev, floor_plan_url: "" } : prev));
      return;
    }

    setMyCafe((prev) =>
      prev
        ? { ...prev, floor_plan_url: floorPlanImg }
        : { name: "", floor_plan_url: floorPlanImg },
    );
  }, [floorPlanImg]);

  useEffect(() => {
    if (sessionStatus !== "running") {
      setTotalCustomers(0);
      return;
    }

    let cancelled = false;

    const loadOccupancy = async () => {
      try {
        const response = await fetch("http://localhost:8000/api/occupancy", {
          headers: getAuthHeaders(),
        });
        const data = await response.json();
        if (!cancelled) {
          setTotalCustomers(Number(data.total_customers ?? 0));
        }
      } catch (error) {
        if (!cancelled) {
          console.error("Помилка отримання occupancy:", error);
        }
      }
    };

    loadOccupancy();
    const intervalId = window.setInterval(loadOccupancy, 1500);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [sessionStatus]);

  const addCamera = async (file: File) => {
    const id = nextCameraIdRef.current;
    nextCameraIdRef.current += 1;
    setCameras((prev) => [...prev, { id, streamUrl: "", thumbnailUrl: "", filename: "" }]);

    const formData = new FormData();
    formData.append("video", file);
    formData.append("camera_id", id.toString());

    try {
      const res = await fetch("http://localhost:8000/upload_video", {
        method: "POST",
        headers: getAuthHeaders(),
        body: formData,
      });
      const data = await res.json();

      setCameras((prev) =>
        prev.map((c) =>
          c.id === id
            ? {
                ...c,
                streamUrl: data.stream_url,
                thumbnailUrl: data.thumbnail_url,
                filename: data.filename,
              }
            : c,
        ),
      );
    } catch (error) {
      console.error("Помилка при додаванні камери:", error);
      setCameras((prev) => prev.filter((c) => c.id !== id));
      alert("Помилка сервера! Переконайтеся, що Python працює.");
    }
  };

  const replaceCamera = async (id: number, file: File) => {
    setCameras((prev) =>
      prev.map((c) => (c.id === id ? { ...c, streamUrl: "", thumbnailUrl: "", filename: "" } : c)),
    );

    const formData = new FormData();
    formData.append("video", file);
    formData.append("camera_id", id.toString());

    try {
      const res = await fetch("http://localhost:8000/upload_video", {
        method: "POST",
        headers: getAuthHeaders(),
        body: formData,
      });
      const data = await res.json();
      setCameras((prev) =>
        prev.map((c) =>
          c.id === id
            ? {
                ...c,
                streamUrl: data.stream_url,
                thumbnailUrl: data.thumbnail_url,
                filename: data.filename,
              }
            : c,
        ),
      );
    } catch (error) {
      console.error("Помилка при заміні камери:", error);
      alert("Помилка сервера! Переконайтеся, що Python працює.");
    }
  };

  const endSession = async (clearCameras: boolean) => {
    if (isEndingSession) {
      return;
    }

    setIsEndingSession(true);
    try {
      const response = await fetch("http://localhost:8000/api/end_session", {
        method: "POST",
        headers: getAuthHeaders(),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      console.log("Session finalization started:", data.session_id);

      setSessionStatus("idle");
      if (clearCameras) {
        setCameras([]);
      }
    } catch (error) {
      console.error("Помилка завершення сесії:", error);
      alert("Не вдалося завершити сесію та запустити пост-обробку.");
    } finally {
      setIsEndingSession(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("cafe_token");
    setCurrentUser(null);
    setMyCafe(null);
    setFloorPlanImg("");
    setIsProfileOpen(false);
    setIsAuthenticated(false);
    setAuthView("login");
  };

  if (!isAuthenticated) {
    return authView === "login" ? (
      <Login
        onLogin={() => setIsAuthenticated(true)}
        onNavigateToRegister={() => setAuthView("register")}
      />
    ) : (
      <Register
        onLogin={() => setIsAuthenticated(true)}
        onNavigateToLogin={() => setAuthView("login")}
      />
    );
  }

  return (
    <div className="min-h-screen bg-[#f5f3f0] font-sans">
      <div className="grid w-full gap-6 px-[30px] py-6">
        <header className="flex flex-col justify-between gap-4 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition-all duration-300 lg:flex-row lg:items-center">
          <div className="flex items-center gap-6">
            <div>
              <h1 className="text-xl font-bold text-[#411e03]">☕ Cafe Monitor</h1>
              <p className="text-xs font-semibold uppercase tracking-wider text-[#846851]">
                {activeTab === "live" ? (
                  <>
                    Статус сесії:{" "}
                    <span className={sessionStatus === "running" ? "text-emerald-500" : "text-slate-400"}>
                      {sessionStatus === "idle" ? "Очікує запуску" : sessionStatus}
                    </span>
                  </>
                ) : (
                  "MongoDB analytics dashboard"
                )}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            {activeTab === "live" && sessionStatus === "idle" && (
              <Button
                className="rounded-xl bg-emerald-600 px-6 text-white shadow-lg shadow-emerald-200 transition-all active:scale-95 hover:bg-emerald-700"
                onClick={() => setSessionStatus("running")}
                disabled={cameras.length === 0}
              >
                ▶ Почати сесію
              </Button>
            )}

            <div className="flex gap-2">
              <Button
                onClick={() => setActiveTab("live")}
                className={activeTab === "live" ? "bg-slate-900 text-white hover:bg-slate-800" : ""}
                variant={activeTab === "live" ? "default" : "outline"}
              >
                Live
              </Button>
              <Button
                onClick={() => setActiveTab("analytics")}
                className={activeTab === "analytics" ? "bg-slate-900 text-white hover:bg-slate-800" : ""}
                variant={activeTab === "analytics" ? "default" : "outline"}
              >
                Analytics
              </Button>
            </div>

            <div className="relative" ref={profileDropdownRef}>
              <button
                onClick={() => setIsProfileOpen(!isProfileOpen)}
                className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white py-1.5 pl-2 pr-3 shadow-sm transition-colors hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-[#8a895f]"
              >
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#411e03] text-sm font-bold text-white">
                  {currentUser?.full_name ? currentUser.full_name.charAt(0).toUpperCase() : "U"}
                </div>
                <span className="hidden text-sm font-medium text-[#411e03] sm:block">Профіль</span>
                <ChevronDown
                  size={16}
                  className={`text-[#846851] transition-transform duration-200 ${isProfileOpen ? "rotate-180" : ""}`}
                />
              </button>

              {isProfileOpen && (
                <div className="absolute right-0 z-50 mt-2 w-56 rounded-xl border border-slate-100 bg-white py-2 shadow-xl">
                  <div className="mb-1 border-b border-slate-100 px-4 py-3">
                    <div className="flex items-center gap-2">
                      <User size={14} className="text-[#846851]" />
                      <p className="text-sm font-bold text-[#411e03]">{currentUser?.full_name || "User"}</p>
                    </div>
                    <p className="mt-0.5 truncate text-xs text-[#846851]">{currentUser?.email || ""}</p>
                  </div>

                  <div className="px-2">
                    <button
                      onClick={handleLogout}
                      className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-red-600 transition-colors hover:bg-red-50 hover:text-red-700"
                    >
                      <LogOut size={16} />
                      Вийти з акаунта
                    </button>
                  </div>
                </div>
              )}
            </div>

            {activeTab === "live" &&
              (sessionStatus === "idle" ? null : (
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    className="h-10 border-slate-300 px-5 text-slate-700 hover:bg-slate-100"
                    onClick={() => void endSession(false)}
                    disabled={isEndingSession}
                  >
                    {isEndingSession ? "Завершення..." : "⏸ Зупинити"}
                  </Button>
                  <Button
                    className="h-10 rounded-xl bg-slate-900 px-5 text-white hover:bg-slate-800"
                    onClick={() => void endSession(true)}
                    disabled={isEndingSession}
                  >
                    {isEndingSession ? "Завершення..." : "⏹ Завершити"}
                  </Button>
                </div>
              ))}
          </div>
        </header>

        <div className={activeTab === "live" ? "block" : "hidden"}>
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
            <div className="xl:col-span-2">
              <div className="grid auto-rows-fr grid-cols-1 gap-6 md:grid-cols-2">
                {cameras.map((cam, index) => (
                  <CameraView
                    key={cam.id}
                    cam={cam}
                    cameraIndex={index}
                    sessionStatus={sessionStatus}
                    onRemove={(id) => setCameras((prev) => prev.filter((c) => c.id !== id))}
                    onReplace={replaceCamera}
                    floorPlanImg={floorPlanImg}
                    setFloorPlanImg={setFloorPlanImg}
                  />
                ))}
                {sessionStatus === "idle" && <AddCameraCard onUpload={addCamera} />}
              </div>
            </div>

            <div className="xl:col-span-1 self-start h-fit">
              <LiveMonitorSidePanel
                sessionStatus={sessionStatus}
                camerasCount={cameras.length}
                totalCustomers={totalCustomers}
                floorPlanImg={floorPlanImg}
                setFloorPlanImg={setFloorPlanImg}
                mapZones={mapZones}
                setMapZones={setMapZones}
              />
            </div>
          </div>
        </div>

        <div className={activeTab === "analytics" ? "block" : "hidden"}>
          <Analytics
            externalCafe={myCafe}
            onCafeUpdate={(cafe) => {
              setMyCafe(cafe);
              setFloorPlanImg(cafe.floor_plan_url || "");
            }}
          />
        </div>
      </div>
    </div>
  );
}