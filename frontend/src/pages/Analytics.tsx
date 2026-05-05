import { useEffect, useRef, useState } from "react";
import jsPDF from "jspdf";
import { toPng } from "html-to-image";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "../components/ui/tabs";
import {
  Calendar,
  Clock,
  Download,
  MapPinned,
  RefreshCw,
  TrendingUp,
  Users,
  Zap,
} from "lucide-react";

interface ChartDatum {
  label: string;
  count: number;
}

interface StatsGroup {
  total_customers: number;
  gender_distribution: ChartDatum[];
  age_groups: ChartDatum[];
  average_duration_sec?: number;
  peak_hour?: string;
}

interface ZoneStat {
  name: string;
  avg_time_sec: number;
  percent: number;
}

interface HeatmapPoint {
  x: number;
  y: number;
  weight: number;
}

interface StatsResponse {
  overall: StatsGroup;
  by_date: Record<string, StatsGroup>;
  zone_stats: ZoneStat[];
  heatmap_points: HeatmapPoint[];
  floor_plan_url?: string | null;
}

const API_ROOT = "http://localhost:8000";
const FLOOR_PLAN_URL = `${API_ROOT}/floor_plan`;
const PIE_COLORS = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#64748b"];

const buildFloorPlanSrc = (url?: string | null, cacheKey?: number) => {
  const rawUrl = (url || "").trim();
  const baseUrl = rawUrl
    ? rawUrl.startsWith("http://") || rawUrl.startsWith("https://") || rawUrl.startsWith("data:")
      ? rawUrl
      : `${API_ROOT}${rawUrl.startsWith("/") ? "" : "/"}${rawUrl}`
    : FLOOR_PLAN_URL;

  if (!cacheKey || baseUrl.startsWith("data:")) {
    return baseUrl;
  }

  const separator = baseUrl.includes("?") ? "&" : "?";
  return `${baseUrl}${separator}t=${cacheKey}`;
};

const formatDuration = (seconds: number | undefined) => {
  if (!seconds) return "0 хв";
  const safeSeconds = Number.isFinite(seconds) ? Math.max(0, Math.round(seconds)) : 0;
  const minutes = Math.floor(safeSeconds / 60);
  const remainingSeconds = safeSeconds % 60;
  return `${minutes} хв ${remainingSeconds} с`;
};

function AnalyticsHeatmap({
  floorPlanUrl,
  points,
}: {
  floorPlanUrl?: string | null;
  points: HeatmapPoint[];
}) {
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageReady, setImageReady] = useState(false);
  const [imageError, setImageError] = useState(false);
  const [cacheKey, setCacheKey] = useState(() => Date.now());
  const resolvedFloorPlanUrl = buildFloorPlanSrc(floorPlanUrl, cacheKey);

  useEffect(() => {
    setCacheKey(Date.now());
    setImageReady(false);
    setImageError(false);
  }, [floorPlanUrl]);

  const drawHeatmap = () => {
    const image = imageRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!image || !canvas || !ctx) return;

    const canvasWidth = image.clientWidth;
    const canvasHeight = image.clientHeight;
    if (canvasWidth <= 0 || canvasHeight <= 0) return;

    if (canvas.width !== canvasWidth) canvas.width = canvasWidth;
    if (canvas.height !== canvasHeight) canvas.height = canvasHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!points.length || !image.naturalWidth || !image.naturalHeight) return;

    const scaleX = canvas.width / image.naturalWidth;
    const scaleY = canvas.height / image.naturalHeight;

    points.forEach((point) => {
      const x = point.x * scaleX;
      const y = point.y * scaleY;
      const radius = 25;
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
      gradient.addColorStop(0, "rgba(255, 0, 0, 0.2)");
      gradient.addColorStop(0.5, "rgba(255, 165, 0, 0.1)");
      gradient.addColorStop(1, "rgba(255, 255, 0, 0)");

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = gradient;
      ctx.fill();
    });
  };

  useEffect(() => {
    drawHeatmap();
  }, [imageReady, points, resolvedFloorPlanUrl]);

  useEffect(() => {
    const image = imageRef.current;
    if (!image || typeof ResizeObserver === "undefined") return;

    const resizeObserver = new ResizeObserver(() => {
      drawHeatmap();
    });
    resizeObserver.observe(image);

    return () => {
      resizeObserver.disconnect();
    };
  }, [imageReady, points, resolvedFloorPlanUrl]);

  return (
    <div className="relative overflow-hidden rounded-xl border border-slate-200 bg-slate-100">
      <img
        ref={imageRef}
        src={resolvedFloorPlanUrl}
        alt="Floor plan"
        crossOrigin="anonymous"
        className="w-full h-auto object-contain rounded-lg"
        onLoad={() => {
          setImageError(false);
          setImageReady(true);
        }}
        onError={() => {
          setImageReady(false);
          setImageError(true);
        }}
      />
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
      {imageError && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/90 text-sm text-slate-500">
          Не вдалося завантажити план приміщення.
        </div>
      )}
      {!imageError && points.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900/10 text-sm font-medium text-slate-600">
          Для цього фільтра ще немає теплових точок.
        </div>
      )}
    </div>
  );
}

export default function Analytics() {
  const dashboardRef = useRef<HTMLDivElement>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [dateTab, setDateTab] = useState("today");
  const [customDate, setCustomDate] = useState("");
  const [timeRange, setTimeRange] = useState("00:00 - 23:59");

  const loadStats = async () => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        tab: dateTab,
        date: customDate,
        time: timeRange,
      });
      const response = await fetch(`${API_ROOT}/api/stats?${params.toString()}`);
      const data: StatsResponse = await response.json();
      if (!response.ok) {
        throw new Error("Не вдалося завантажити аналітику.");
      }
      setStats(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не вдалося завантажити аналітику.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      void loadStats();
    }, 250);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [dateTab, customDate, timeRange]);

  const genderData = stats?.overall.gender_distribution ?? [];
  const ageData = stats?.overall.age_groups ?? [];
  const zoneStats = stats?.zone_stats ?? [];
  const heatmapPoints = stats?.heatmap_points ?? [];
  const byDateEntries = Object.entries(stats?.by_date ?? {});
  const trendData = byDateEntries.map(([date, group]) => ({
    date: date.substring(5),
    customers: group.total_customers,
  }));

  const formatBreakdown = (items: ChartDatum[], labelMap?: Record<string, string>) =>
    items.length > 0
      ? items.map((item) => `${labelMap?.[item.label] ?? item.label}: ${item.count}`).join(", ")
      : "—";

  const handleExportPDF = async () => {
    if (!dashboardRef.current) return;
    setIsExporting(true);

    try {
      const node = dashboardRef.current;
      const nodeWidth = node.offsetWidth;
      const nodeHeight = node.offsetHeight;
      const dataUrl = await toPng(node, {
        pixelRatio: 2,
        backgroundColor: "#f8fafc",
        cacheBust: true,
      });

      const pdf = new jsPDF("p", "mm", "a4");
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (nodeHeight * pdfWidth) / nodeWidth;
      pdf.addImage(dataUrl, "PNG", 0, 10, pdfWidth, pdfHeight);
      pdf.save(`cafe-analytics-${new Date().toISOString().split("T")[0]}.pdf`);
    } catch (exportError) {
      console.error("Помилка генерації PDF:", exportError);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="min-h-full w-full space-y-6 bg-slate-50 py-2 font-sans text-slate-900">
      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm font-medium text-rose-600">
          Помилка: {error}
        </div>
      )}

      <div ref={dashboardRef} className="space-y-8 rounded-xl bg-slate-50 py-4">
        <div className="mb-6 flex flex-col items-start justify-between gap-4 xl:flex-row xl:items-center">
          <div className="flex flex-wrap items-center gap-3">
            <Tabs value={dateTab} onValueChange={setDateTab} className="w-auto">
              <TabsList className="h-10 border border-slate-200 bg-white shadow-sm">
                <TabsTrigger value="today" className="data-[state=active]:bg-slate-100">
                  Сьогодні
                </TabsTrigger>
                <TabsTrigger value="7days" className="data-[state=active]:bg-slate-100">
                  7 днів
                </TabsTrigger>
                <TabsTrigger value="30days" className="data-[state=active]:bg-slate-100">
                  30 днів
                </TabsTrigger>
                <TabsTrigger value="all" className="data-[state=active]:bg-slate-100">
                  Весь час
                </TabsTrigger>
              </TabsList>
            </Tabs>

            <div className="flex items-center gap-3">
              <div className="relative">
                <Calendar size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                <Input
                  type="date"
                  value={customDate}
                  onChange={(event) => setCustomDate(event.target.value)}
                  className="h-10 w-40 border-slate-200 bg-white pl-10 text-slate-700 shadow-sm"
                />
              </div>
              <div className="relative">
                <Clock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                <Input
                  type="text"
                  placeholder="00:00 - 23:59"
                  value={timeRange}
                  onChange={(event) => setTimeRange(event.target.value)}
                  className="h-10 w-36 border-slate-200 bg-white pl-10 text-slate-700 shadow-sm"
                />
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              onClick={() => void loadStats()}
              disabled={loading}
              className="gap-2 border-slate-200 bg-white text-slate-600 shadow-sm hover:bg-slate-50"
            >
              <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
              {loading ? "Оновлення..." : "Оновити"}
            </Button>
            <Button
              onClick={handleExportPDF}
              disabled={isExporting}
              className="gap-2 bg-emerald-600 text-white shadow-sm hover:bg-emerald-700"
            >
              {isExporting ? <RefreshCw size={16} className="animate-spin" /> : <Download size={16} />}
              {isExporting ? "Генерація..." : "Експорт"}
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <Card className="border-slate-200 shadow-sm transition-shadow hover:shadow-md">
            <CardContent className="flex items-center gap-4 p-5">
              <div className="rounded-xl bg-blue-50 p-3 text-blue-600">
                <Users size={24} />
              </div>
              <div>
                <p className="text-sm font-medium text-slate-500">Всього клієнтів</p>
                <h3 className="text-2xl font-bold text-slate-800">{stats?.overall?.total_customers ?? 0}</h3>
              </div>
            </CardContent>
          </Card>

          <Card className="border-slate-200 shadow-sm transition-shadow hover:shadow-md">
            <CardContent className="flex items-center gap-4 p-5">
              <div className="rounded-xl bg-emerald-50 p-3 text-emerald-600">
                <Clock size={24} />
              </div>
              <div>
                <p className="text-sm font-medium text-slate-500">Сер. час перебування</p>
                <h3 className="text-2xl font-bold text-slate-800">
                  {formatDuration(stats?.overall?.average_duration_sec)}
                </h3>
              </div>
            </CardContent>
          </Card>

          <Card className="border-slate-200 shadow-sm transition-shadow hover:shadow-md">
            <CardContent className="flex items-center gap-4 p-5">
              <div className="rounded-xl bg-amber-50 p-3 text-amber-600">
                <Zap size={24} />
              </div>
              <div>
                <p className="text-sm font-medium text-slate-500">Година пік</p>
                <h3 className="text-2xl font-bold text-slate-800">{stats?.overall?.peak_hour ?? "—"}</h3>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <Card className="border-slate-200 shadow-sm lg:col-span-2">
            <CardHeader className="pb-4">
              <CardTitle className="flex items-center gap-2 text-lg font-semibold text-slate-800">
                <TrendingUp size={18} className="text-emerald-500" />
                Динаміка відвідуваності по датах
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[280px] w-full">
                {trendData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trendData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" vertical={false} />
                      <XAxis
                        dataKey="date"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: "#64748b", fontSize: 12 }}
                        dy={10}
                      />
                      <YAxis axisLine={false} tickLine={false} tick={{ fill: "#64748b", fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{
                          borderRadius: "8px",
                          border: "none",
                          boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)",
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="customers"
                        name="Клієнти"
                        stroke="#10b981"
                        strokeWidth={3}
                        dot={{ r: 4, fill: "#10b981", strokeWidth: 2 }}
                        activeDot={{ r: 6 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-slate-400">
                    Немає даних для графіка
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card className="border-slate-200 shadow-sm">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg font-semibold text-slate-800">Стать клієнтів</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative h-[280px] w-full">
                {genderData.length > 0 ? (
                  <>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={genderData}
                          dataKey="count"
                          nameKey="label"
                          innerRadius={70}
                          outerRadius={100}
                          paddingAngle={3}
                        >
                          {genderData.map((entry, index) => (
                            <Cell key={entry.label} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{
                            borderRadius: "8px",
                            border: "none",
                            boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)",
                          }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="absolute bottom-0 flex w-full justify-center gap-4 text-sm">
                      {genderData.map((item, index) => (
                        <div key={item.label} className="flex items-center gap-1.5">
                          <span
                            className="h-3 w-3 rounded-full"
                            style={{ backgroundColor: PIE_COLORS[index % PIE_COLORS.length] }}
                          />
                          <span className="text-slate-600">{item.label}</span>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-slate-400">
                    Немає даних про стать
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <Card className="border-slate-200 shadow-sm lg:col-span-2">
            <CardHeader className="pb-4">
              <CardTitle className="flex items-center gap-2 text-lg font-semibold text-slate-800">
                <MapPinned size={18} className="text-rose-500" />
                Теплова карта перебування
              </CardTitle>
            </CardHeader>
            <CardContent>
              <AnalyticsHeatmap floorPlanUrl={stats?.floor_plan_url} points={heatmapPoints} />
            </CardContent>
          </Card>

          <Card className="border-slate-200 shadow-sm">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg font-semibold text-slate-800">Аналіз зон</CardTitle>
            </CardHeader>
            <CardContent>
              {zoneStats.length > 0 ? (
                <div className="space-y-4">
                  {zoneStats.map((zone) => (
                    <div key={zone.name} className="space-y-2 rounded-xl border border-slate-100 bg-slate-50 p-3">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <p className="font-medium text-slate-800">{zone.name}</p>
                          <p className="text-xs text-slate-500">Сер. час: {formatDuration(zone.avg_time_sec)}</p>
                        </div>
                        <span className="rounded-full bg-white px-2.5 py-1 text-xs font-semibold text-slate-600">
                          {zone.percent}%
                        </span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-slate-200">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-yellow-400 via-orange-500 to-red-500"
                          style={{ width: `${Math.min(100, Math.max(0, zone.percent))}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex h-[320px] items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50 text-sm text-slate-400">
                  Немає даних по зонах для цього фільтра.
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <Card className="border-slate-200 shadow-sm">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg font-semibold text-slate-800">Вікові групи</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[250px] w-full">
                {ageData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={ageData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" vertical={false} />
                      <XAxis
                        dataKey="label"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: "#64748b", fontSize: 12 }}
                        dy={10}
                      />
                      <YAxis
                        allowDecimals={false}
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: "#64748b", fontSize: 12 }}
                      />
                      <Tooltip
                        cursor={{ fill: "#f1f5f9" }}
                        contentStyle={{
                          borderRadius: "8px",
                          border: "none",
                          boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)",
                        }}
                      />
                      <Bar dataKey="count" name="Кількість" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-slate-400">
                    Немає даних про вік
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card className="border-slate-200 shadow-sm lg:col-span-2">
            <CardHeader className="border-b border-slate-100 pb-4">
              <CardTitle className="text-lg font-semibold text-slate-800">Детальна статистика по датах</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {byDateEntries.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead className="border-b border-slate-100 bg-slate-50/50 font-medium text-slate-500">
                      <tr>
                        <th className="px-4 py-3">Дата</th>
                        <th className="px-4 py-3 text-center">Клієнти</th>
                        <th className="px-4 py-3">Сер. час</th>
                        <th className="px-4 py-3">Година пік</th>
                        <th className="px-4 py-3">Стать</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {byDateEntries.map(([entryDate, group]) => (
                        <tr key={entryDate} className="transition-colors hover:bg-slate-50/50">
                          <td className="px-4 py-3 font-medium text-slate-900">{entryDate}</td>
                          <td className="px-4 py-3 text-center">
                            <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-bold text-slate-700">
                              {group.total_customers}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-slate-600">{formatDuration(group.average_duration_sec)}</td>
                          <td className="px-4 py-3 text-slate-600">{group.peak_hour ?? "—"}</td>
                          <td className="px-4 py-3 text-xs text-slate-600">
                            {formatBreakdown(group.gender_distribution, {
                              Male: "Ч",
                              Female: "Ж",
                              "Unknown / other": "Інше",
                            })}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="p-8 text-center text-sm text-slate-400">
                  Ще немає збережених записів по датах.
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}