import { Card, CardContent } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { useEffect, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

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

interface ChartDatum {
  label: string;
  count: number;
}

interface StatsResponse {
  gender_distribution: ChartDatum[];
  age_groups: ChartDatum[];
  total_customers: number;
  error?: string;
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

const PIE_COLORS = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#64748b"];
const API_ROOT = "http://localhost:8000";
const FLOOR_PLAN_URL = `${API_ROOT}/floor_plan`;
const CALIBRATION_POINT_COLORS = ["#22c55e", "#38bdf8", "#f59e0b", "#f43f5e"];

function CalibrationModal({
  cam,
  onClose,
}: {
  cam: CameraData;
  onClose: () => void;
}) {
  const [srcPts, setSrcPts] = useState<Point[]>([]);
  const [dstPts, setDstPts] = useState<Point[]>([]);
  const [floorPlanSrc, setFloorPlanSrc] = useState(`${FLOOR_PLAN_URL}?t=${Date.now()}`);
  const [showFloorPlanUpload, setShowFloorPlanUpload] = useState(false);
  const [isUploadingFloorPlan, setIsUploadingFloorPlan] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    setFloorPlanSrc(`${FLOOR_PLAN_URL}?t=${Date.now()}`);
    setShowFloorPlanUpload(false);
  }, []);

  useEffect(() => {
    let cancelled = false;

    const loadCalibration = async () => {
      if (!cam.filename) {
        setSrcPts([]);
        setDstPts([]);
        return;
      }

      try {
        const response = await fetch(`${API_ROOT}/calibration/${encodeURIComponent(cam.filename)}`);
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
    const formData = new FormData();
    formData.append("file", file);
    setIsUploadingFloorPlan(true);

    try {
      const response = await fetch(`${API_ROOT}/upload_floor_plan`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Не вдалося завантажити floor plan.");
      }

      setShowFloorPlanUpload(false);
      setFloorPlanSrc(`${FLOOR_PLAN_URL}?t=${Date.now()}`);
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
        headers: { "Content-Type": "application/json" },
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
                  src={floorPlanSrc}
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
  sessionStatus,
  onRemove,
  onReplace,
}: {
  cam: CameraData;
  sessionStatus: SessionStatus;
  onRemove: (id: number) => void;
  onReplace: (id: number, file: File) => void;
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
        headers: { "Content-Type": "application/json" },
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
            <h3 className="mb-4 text-lg font-bold text-slate-900">Видалити камеру?</h3>
            <p className="mb-6 text-sm text-slate-500">
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

      {showCalibration && <CalibrationModal cam={cam} onClose={() => setShowCalibration(false)} />}

      <Card
        className={
          isExpanded
            ? "fixed inset-4 z-50 flex flex-col overflow-hidden rounded-2xl bg-white shadow-2xl transition-all duration-300 md:inset-10"
            : "group relative flex flex-col overflow-hidden rounded-2xl shadow-lg transition-all duration-300"
        }
      >
        <CardContent className="relative flex min-h-0 flex-1 flex-col p-4">
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

function AnalyticsDashboard() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [livePoints, setLivePoints] = useState<{ x: number; y: number }[]>([]);
  const [floorPlanError, setFloorPlanError] = useState(false);
  const [floorPlanSize, setFloorPlanSize] = useState({ width: 1000, height: 1000 });

  useEffect(() => {
    let cancelled = false;

    const loadStats = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`${API_ROOT}/api/stats`);
        const data: StatsResponse = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Не вдалося завантажити аналітику.");
        }
        if (!cancelled) {
          setStats(data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Не вдалося завантажити аналітику.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    loadStats();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    const loadLivePoints = async () => {
      try {
        const response = await fetch(`${API_ROOT}/api/occupancy`);
        const data = await response.json();
        if (!cancelled) {
          setLivePoints(Array.isArray(data.points) ? data.points : []);
        }
      } catch (err) {
        if (!cancelled) {
          console.error("Не вдалося завантажити live points:", err);
          setLivePoints([]);
        }
      }
    };

    loadLivePoints();
    const intervalId = window.setInterval(loadLivePoints, 1500);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, []);

  const genderData = stats?.gender_distribution ?? [];
  const ageData = stats?.age_groups ?? [];

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      <Card className="rounded-2xl border-slate-200 shadow-sm">
        <CardContent className="p-6">
          <div className="mb-4 flex items-center justify-between border-b pb-2">
            <h2 className="text-lg font-semibold text-slate-900">Gender Distribution</h2>
            <span className="text-sm text-slate-500">Всього: {stats?.total_customers ?? 0}</span>
          </div>
          {loading ? (
            <p className="text-sm text-slate-500">Завантаження аналітики...</p>
          ) : error ? (
            <p className="text-sm text-red-500">{error}</p>
          ) : genderData.length > 0 ? (
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={genderData}
                    dataKey="count"
                    nameKey="label"
                    innerRadius={70}
                    outerRadius={110}
                    paddingAngle={3}
                  >
                    {genderData.map((entry, index) => (
                      <Cell key={entry.label} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="text-sm text-slate-500">Ще немає збережених customer-записів.</p>
          )}
        </CardContent>
      </Card>

      <Card className="rounded-2xl border-slate-200 shadow-sm">
        <CardContent className="p-6">
          <div className="mb-4 border-b pb-2">
            <h2 className="text-lg font-semibold text-slate-900">Age Groups</h2>
          </div>
          {loading ? (
            <p className="text-sm text-slate-500">Завантаження аналітики...</p>
          ) : error ? (
            <p className="text-sm text-red-500">{error}</p>
          ) : (
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={ageData}>
                  <XAxis dataKey="label" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#10b981" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="rounded-2xl border-slate-200 shadow-sm lg:col-span-2">
        <CardContent className="p-6">
          <div className="mb-4 flex items-center justify-between border-b pb-2">
            <h2 className="text-lg font-semibold text-slate-900">Live 2D Map (Дедупліковано)</h2>
            <span className="text-sm text-slate-500">Активних точок: {livePoints.length}</span>
          </div>
          <div className="relative w-full aspect-video overflow-hidden rounded-lg bg-slate-900">
            {!floorPlanError ? (
              <>
                <img
                  src={FLOOR_PLAN_URL}
                  alt="Floor plan"
                  className="absolute inset-0 h-full w-full object-contain"
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
                  {livePoints.map((point, index) => (
                    <g key={`${point.x}-${point.y}-${index}`}>
                      <circle
                        cx={point.x}
                        cy={point.y}
                        r="22"
                        fill="#10b981"
                        opacity="0.18"
                        className="animate-ping"
                      />
                      <circle cx={point.x} cy={point.y} r="8" fill="#10b981" opacity="0.95" />
                      <circle cx={point.x} cy={point.y} r="12" fill="none" stroke="#6ee7b7" strokeWidth="2" opacity="0.9" />
                    </g>
                  ))}
                </svg>
              </>
            ) : (
              <div className="flex h-full w-full items-center justify-center px-6 text-center text-sm text-slate-300">
                Floor plan ще не завантажено. Додайте його в калібруванні камери, щоб побачити live 2D map.
              </div>
            )}
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
        <span className="mt-3 text-sm font-medium text-slate-500 group-hover:text-emerald-600">
          Додати нову камеру
        </span>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [cameras, setCameras] = useState<CameraData[]>([]);
  const [sessionStatus, setSessionStatus] = useState<SessionStatus>("idle");
  const [activeTab, setActiveTab] = useState<ActiveTab>("live");
  const [totalCustomers, setTotalCustomers] = useState(0);
  const nextCameraIdRef = useRef(1);

  useEffect(() => {
    if (sessionStatus !== "running") {
      setTotalCustomers(0);
      return;
    }

    let cancelled = false;

    const loadOccupancy = async () => {
      try {
        const response = await fetch("http://localhost:8000/api/occupancy");
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
      const res = await fetch("http://localhost:8000/upload_video", { method: "POST", body: formData });
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
      const res = await fetch("http://localhost:8000/upload_video", { method: "POST", body: formData });
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

  return (
    <div className="min-h-screen bg-slate-50 p-6 font-sans">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="flex flex-col justify-between gap-4 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition-all duration-300 lg:flex-row lg:items-center">
          <div>
            <h1 className="text-xl font-bold text-slate-900">☕ Cafe Monitor</h1>
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
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

          <div className="flex flex-wrap items-center gap-3">
            {sessionStatus === "running" && (
              <div className="rounded-full bg-red-50 px-4 py-2 text-sm font-semibold text-red-600 border border-red-200">
                🔴 Live: {totalCustomers} клієнтів у закладі
              </div>
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

            {activeTab === "live" &&
              (sessionStatus === "idle" ? (
                <Button
                  className="rounded-xl bg-emerald-600 px-6 text-white shadow-lg shadow-emerald-200 transition-all active:scale-95 hover:bg-emerald-700"
                  onClick={() => setSessionStatus("running")}
                  disabled={cameras.length === 0}
                >
                  ▶ Почати сесію
                </Button>
              ) : (
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    className="h-10 border-slate-300 px-5 text-slate-700 hover:bg-slate-100"
                    onClick={() => setSessionStatus("idle")}
                  >
                    ⏸ Зупинити
                  </Button>
                  <Button
                    className="h-10 rounded-xl bg-slate-900 px-5 text-white hover:bg-slate-800"
                    onClick={() => {
                      setSessionStatus("idle");
                      setCameras([]);
                    }}
                  >
                    ⏹ Завершити
                  </Button>
                </div>
              ))}
          </div>
        </header>

        <div className={activeTab === "live" ? "block" : "hidden"}>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            {cameras.map((cam) => (
              <CameraView
                key={cam.id}
                cam={cam}
                sessionStatus={sessionStatus}
                onRemove={(id) => setCameras((prev) => prev.filter((c) => c.id !== id))}
                onReplace={replaceCamera}
              />
            ))}
            {sessionStatus === "idle" && <AddCameraCard onUpload={addCamera} />}
          </div>

          <div className="grid grid-cols-1 gap-6 pt-10 md:grid-cols-2">
            <Card className="rounded-2xl border-slate-200 bg-white shadow-sm">
              <CardContent className="p-6">
                <h2 className="mb-4 border-b pb-2 text-lg font-semibold text-slate-900">
                  🧍 Аналітика сесії
                </h2>
                <ul className="space-y-2 text-sm text-slate-600">
                  {cameras.length > 0 && sessionStatus === "running" ? (
                    <li className="font-medium text-green-600 animate-pulse">
                      Система активна. Стрімів: {cameras.length}. Йде трекінг клієнтів...
                    </li>
                  ) : cameras.length > 0 ? (
                    <li className="font-medium text-slate-500">
                      Підключено камер: {cameras.length}. Статус: Очікує запуску.
                    </li>
                  ) : (
                    <li className="italic text-slate-400">Очікування підключення камер...</li>
                  )}
                </ul>
              </CardContent>
            </Card>

            <Card className="rounded-2xl border-slate-200 bg-white shadow-sm">
              <CardContent className="p-6">
                <h2 className="mb-4 border-b pb-2 text-lg font-semibold text-slate-900">
                  📊 Статистика YOLO
                </h2>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-500">Активних стрімів:</span>
                    <span className="text-xl font-bold text-slate-950">
                      {sessionStatus === "running" ? cameras.length : 0}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        <div className={activeTab === "analytics" ? "block" : "hidden"}>
          <AnalyticsDashboard />
        </div>
      </div>
    </div>
  );
}