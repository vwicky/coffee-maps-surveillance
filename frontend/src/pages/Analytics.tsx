import { useEffect, useRef, useState } from "react";
import jsPDF from "jspdf";
import { toPng } from "html-to-image";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
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
  ArrowDownRight,
  ArrowUpRight,
  Clock,
  Download,
  Info,
  MapPinned,
  RefreshCw,
  TrendingUp,
  Users,
  X,
  Zap,
} from "lucide-react";

interface HeatmapPoint {
  x: number;
  y: number;
}

interface MyCafe {
  name: string;
  address?: string;
  floor_plan_url: string;
}

interface AnalyticsSummary {
  total_clients: number;
  clients_trend: number | null;
  avg_duration_seconds: number;
  duration_trend: number | null;
  peak_hour: string | null;
  peak_clients: number | null;
}

interface DemographicsDatum {
  age_group: string;
  Чоловіки: number;
  Жінки: number;
}

interface DynamicsDatum {
  time?: string;
  date?: string;
  customers: number;
}

interface HistoryRow {
  date: string;
  total: number;
  avg_time: string;
  peak: string;
  gender: string;
  top_age: string;
}

interface ZoneAnalyticsRow {
  name: string;
  avg_time: string;
  visits: number;
  percentage: number;
  color: string;
}

interface RouteAnalyticsRow {
  path: string;
  percentage: number;
}

const API_ROOT = "http://localhost:8000";

function toYmd(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function buildAnalyticsSummaryQuery(dateTab: string, customDate: string, timeRange: string): string {
  const params = new URLSearchParams();
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const custom = customDate.trim();

  if (custom) {
    params.set("start_date", custom);
    params.set("end_date", custom);
  } else if (dateTab === "today") {
    const y = toYmd(today);
    params.set("start_date", y);
    params.set("end_date", y);
  } else if (dateTab === "7days") {
    const start = new Date(today);
    start.setDate(start.getDate() - 7);
    params.set("start_date", toYmd(start));
    params.set("end_date", toYmd(today));
  } else if (dateTab === "30days") {
    const start = new Date(today);
    start.setDate(start.getDate() - 30);
    params.set("start_date", toYmd(start));
    params.set("end_date", toYmd(today));
  }

  const tr = timeRange.trim();
  if (tr.includes("-")) {
    params.set("time_range", tr);
  }

  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

function formatDurationUk(totalSeconds: number): string {
  const s = Math.max(0, Math.round(Number(totalSeconds) || 0));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m} хв ${r} с`;
}

function summaryComparisonLabel(dateTab: string, customDate: string): string {
  const custom = customDate.trim();
  if (custom) {
    return "до попереднього періоду";
  }
  if (dateTab === "all" || dateTab === "") {
    return "";
  }
  switch (dateTab) {
    case "today":
      return "від учора";
    case "7days":
      return "від минулого тижня";
    case "30days":
      return "від минулого місяця";
    default:
      return "до попереднього періоду";
  }
}

function SummaryTrendFootnote({
  trend,
  comparisonLabel,
}: {
  trend: number | null;
  comparisonLabel: string;
}) {
  if (!comparisonLabel) {
    return null;
  }
  if (trend == null || trend === 0) {
    return <div className="mt-1.5 text-xs font-medium text-[#846851]">Немає змін</div>;
  }
  const up = trend > 0;
  return (
    <div
      className={`mt-1.5 flex flex-wrap items-center gap-1 text-sm font-medium ${
        up ? "text-emerald-600" : "text-red-600"
      }`}
    >
      {up ? <ArrowUpRight size={16} strokeWidth={2.5} aria-hidden /> : <ArrowDownRight size={16} strokeWidth={2.5} aria-hidden />}
      <span>{Math.abs(trend)}%</span>
      <span className="font-normal text-[#846851]">{comparisonLabel}</span>
    </div>
  );
}

const SPLAT_RADIUS_PX = 40;

function heatRgbForNormalizedIntensity(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  const lerp = (a: number, b: number, u: number) => Math.round(a + (b - a) * u);
  if (x <= 0.25) {
    const u = x / 0.25;
    return [0, lerp(0, 255, u), lerp(255, 0, u)];
  }
  if (x <= 0.55) {
    const u = (x - 0.25) / 0.3;
    return [lerp(0, 255, u), 255, 0];
  }
  if (x <= 0.85) {
    const u = (x - 0.55) / 0.3;
    return [255, lerp(255, 0, u), 0];
  }
  return [255, 0, 0];
}

const HEAT_RGB_LUT: ReadonlyArray<readonly [number, number, number]> = (() => {
  const out: [number, number, number][] = [];
  for (let a = 0; a < 256; a++) {
    out.push(heatRgbForNormalizedIntensity(a / 255));
  }
  return out;
})();

const buildFloorPlanSrc = (url?: string | null, cacheKey?: number) => {
  const rawUrl = (url || "").trim();
  if (!rawUrl) {
    return "";
  }

  const baseUrl =
    rawUrl.startsWith("http://") || rawUrl.startsWith("https://") || rawUrl.startsWith("data:")
      ? rawUrl
      : `${API_ROOT}${rawUrl.startsWith("/") ? "" : "/"}${rawUrl}`;

  if (!cacheKey || baseUrl.startsWith("data:")) {
    return baseUrl;
  }

  const separator = baseUrl.includes("?") ? "&" : "?";
  return `${baseUrl}${separator}t=${cacheKey}`;
};

function AnalyticsHeatmap({
  floorPlanUrl,
  points,
}: {
  floorPlanUrl?: string | null;
  points: HeatmapPoint[];
}) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const heatmapContainerRef = useRef<HTMLCanvasElement>(null);
  const [imageError, setImageError] = useState(false);
  const resolvedFloorPlanUrl = buildFloorPlanSrc(floorPlanUrl);
  const [overlayBounds, setOverlayBounds] = useState({ left: 16, top: 16, width: 0, height: 0 });
  const [layoutTick, setLayoutTick] = useState(0);
  const [imageIntrinsicSize, setImageIntrinsicSize] = useState({ width: 1920, height: 1080 });

  useEffect(() => {
    setImageIntrinsicSize({ width: 1920, height: 1080 });
  }, [floorPlanUrl]);

  useEffect(() => {
    const wrapper = wrapperRef.current;
    const image = imageRef.current;
    if (!wrapper || !image) {
      return;
    }

    const updateOverlayBounds = () => {
      const wrapperWidth = wrapper.clientWidth;
      const wrapperHeight = wrapper.clientHeight;
      if (wrapperWidth <= 0 || wrapperHeight <= 0) {
        return;
      }

      const imageStyles = window.getComputedStyle(image);
      const paddingLeft = parseFloat(imageStyles.paddingLeft || "0") || 0;
      const paddingRight = parseFloat(imageStyles.paddingRight || "0") || 0;
      const paddingTop = parseFloat(imageStyles.paddingTop || "0") || 0;
      const paddingBottom = parseFloat(imageStyles.paddingBottom || "0") || 0;
      const contentWidth = Math.max(0, wrapperWidth - paddingLeft - paddingRight);
      const contentHeight = Math.max(0, wrapperHeight - paddingTop - paddingBottom);
      if (contentWidth <= 0 || contentHeight <= 0) {
        return;
      }

      const denomW = imageIntrinsicSize.width > 0 ? imageIntrinsicSize.width : 1;
      const denomH = imageIntrinsicSize.height > 0 ? imageIntrinsicSize.height : 1;
      const scale = Math.min(contentWidth / denomW, contentHeight / denomH);
      if (!Number.isFinite(scale) || scale <= 0) {
        return;
      }
      const renderedWidth = imageIntrinsicSize.width * scale;
      const renderedHeight = imageIntrinsicSize.height * scale;
      const offsetLeft = paddingLeft + (contentWidth - renderedWidth) / 2;
      const offsetTop = paddingTop + (contentHeight - renderedHeight) / 2;

      setOverlayBounds({
        left: offsetLeft,
        top: offsetTop,
        width: renderedWidth,
        height: renderedHeight,
      });
    };

    updateOverlayBounds();
    const resizeObserver = new ResizeObserver(() => {
      updateOverlayBounds();
    });
    resizeObserver.observe(wrapper);

    const onWinResize = () => updateOverlayBounds();
    window.addEventListener("resize", onWinResize);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", onWinResize);
    };
  }, [floorPlanUrl, layoutTick, imageIntrinsicSize]);

  useEffect(() => {
    const canvas = heatmapContainerRef.current;
    if (!canvas) {
      return;
    }

    if (canvas.offsetWidth === 0 || canvas.offsetHeight === 0) {
      return;
    }
    if (overlayBounds.width <= 0 || overlayBounds.height <= 0) {
      return;
    }

    const iw = imageIntrinsicSize.width > 0 ? imageIntrinsicSize.width : 1;
    const ih = imageIntrinsicSize.height > 0 ? imageIntrinsicSize.height : 1;

    const safePoints = Array.isArray(points)
      ? points.filter(
          (p): p is HeatmapPoint =>
            p != null &&
            typeof p === "object" &&
            typeof p.x === "number" &&
            typeof p.y === "number" &&
            Number.isFinite(p.x) &&
            Number.isFinite(p.y),
        )
      : [];

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      return;
    }

    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    canvas.width = w;
    canvas.height = h;

    const scaleX = w / iw;
    const scaleY = h / ih;
    if (!Number.isFinite(scaleX) || !Number.isFinite(scaleY) || scaleX <= 0 || scaleY <= 0) {
      return;
    }

    ctx.clearRect(0, 0, w, h);

    const scaledPoints = safePoints
      .map((point) => ({
        x: Math.round(point.x * scaleX),
        y: Math.round(point.y * scaleY),
      }))
      .filter(
        (point) =>
          Number.isFinite(point.x) &&
          Number.isFinite(point.y) &&
          point.x >= 0 &&
          point.y >= 0 &&
          point.x <= w &&
          point.y <= h,
      );

    if (scaledPoints.length === 0) {
      return;
    }

    try {
      ctx.globalCompositeOperation = "source-over";
      for (const p of scaledPoints) {
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, SPLAT_RADIUS_PX);
        g.addColorStop(0, "rgba(0, 0, 0, 0.15)");
        g.addColorStop(1, "rgba(0, 0, 0, 0)");
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(p.x, p.y, SPLAT_RADIUS_PX, 0, Math.PI * 2);
        ctx.fill();
      }

      const imgData = ctx.getImageData(0, 0, w, h);
      const d = imgData.data;
      let maxA = 0;
      for (let j = 3; j < d.length; j += 4) {
        if (d[j] > maxA) {
          maxA = d[j];
        }
      }
      const norm = maxA > 0 ? 255 / maxA : 1;

      for (let i = 0; i < d.length; i += 4) {
        const aRaw = d[i + 3];
        if (aRaw === 0) {
          continue;
        }
        const t = Math.min(255, Math.round(aRaw * norm));
        const rgb = HEAT_RGB_LUT[t]!;
        d[i] = rgb[0]!;
        d[i + 1] = rgb[1]!;
        d[i + 2] = rgb[2]!;
        d[i + 3] = Math.min(255, t + 35);
      }

      ctx.putImageData(imgData, 0, 0);
    } catch (err) {
      console.error("Heatmap canvas failed:", err);
    }
  }, [points, overlayBounds, layoutTick, imageIntrinsicSize]);

  return (
    <div
      ref={wrapperRef}
      className="relative flex h-full w-full items-center justify-center overflow-hidden rounded-lg bg-white"
    >
      <img
        ref={imageRef}
        src={resolvedFloorPlanUrl}
        alt="Floor plan"
        className="w-full h-full object-contain pointer-events-none border-0 p-4 outline-none ring-0"
        onLoad={(e) => {
          setImageError(false);
          const el = e.currentTarget;
          const nw = el.naturalWidth || 1920;
          const nh = el.naturalHeight || 1080;
          setImageIntrinsicSize({ width: nw, height: nh });
          setLayoutTick((t) => t + 1);
        }}
        onError={() => {
          setImageError(true);
        }}
      />
      <div
        className="pointer-events-none absolute"
        style={{
          left: `${overlayBounds.left}px`,
          top: `${overlayBounds.top}px`,
          width: `${overlayBounds.width}px`,
          height: `${overlayBounds.height}px`,
        }}
      >
        <canvas
          ref={heatmapContainerRef}
          className="pointer-events-none absolute left-0 top-0 block h-full w-full max-w-full border-0 opacity-80 outline-none ring-0"
        />
      </div>
      {imageError && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/90 text-sm text-[#846851]">
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

export default function Analytics({
  externalCafe,
  onCafeUpdate,
}: {
  externalCafe?: MyCafe | null;
  onCafeUpdate?: (cafe: MyCafe) => void;
}) {
  const dashboardRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [dynamicsData, setDynamicsData] = useState<DynamicsDatum[]>([]);
  const [demographics, setDemographics] = useState<DemographicsDatum[]>([]);
  const [historyData, setHistoryData] = useState<HistoryRow[]>([]);
  const [zoneData, setZoneData] = useState<ZoneAnalyticsRow[]>([]);
  const [routesData, setRoutesData] = useState<RouteAnalyticsRow[]>([]);
  const [heatmapPoints, setHeatmapPoints] = useState<HeatmapPoint[]>([]);
  const [myCafe, setMyCafe] = useState<MyCafe | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [isUploadingPlan, setIsUploadingPlan] = useState(false);
  const [dateTab, setDateTab] = useState("today");
  const [customDate, setCustomDate] = useState("");
  const [timeStart, setTimeStart] = useState("");
  const [timeEnd, setTimeEnd] = useState("");
  const [dynamicsView, setDynamicsView] = useState<"hours" | "days">("hours");

  const loadStats = async () => {
    setLoading(true);
    setError(null);

    try {
      const token = localStorage.getItem("cafe_token");
      const headers = token
        ? {
            Authorization: `Bearer ${token}`,
          }
        : undefined;

      const timeRangeForApi = timeStart && timeEnd ? `${timeStart} - ${timeEnd}` : "";

      const filterQs = buildAnalyticsSummaryQuery(dateTab, customDate, timeRangeForApi);
      const filterQuery = filterQs.startsWith("?") ? filterQs.slice(1) : filterQs;
      const withFilter = (path: string) =>
        filterQuery ? `${path}${path.includes("?") ? "&" : "?"}${filterQuery}` : path;

      const [
        summaryResponse,
        dynamicsResponse,
        demographicsResponse,
        historyResponse,
        zonesResponse,
        routesResponse,
        heatmapResponse,
      ] = await Promise.all([
        fetch(`${API_ROOT}/api/analytics/summary${filterQs}`, {
          headers,
        }),
        fetch(withFilter(`${API_ROOT}/api/analytics/dynamics?view=${encodeURIComponent(dynamicsView)}`), {
          headers,
        }),
        fetch(withFilter(`${API_ROOT}/api/analytics/demographics`), {
          headers,
        }),
        fetch(withFilter(`${API_ROOT}/api/analytics/history`), {
          headers,
        }),
        fetch("http://127.0.0.1:8000/api/analytics/zones", {
          headers,
        }),
        fetch("http://127.0.0.1:8000/api/analytics/routes", {
          headers,
        }),
        fetch("http://127.0.0.1:8000/api/analytics/heatmap", {
          headers,
        }),
      ]);

      const summaryData: AnalyticsSummary = await summaryResponse.json();
      const dynamicsRows: DynamicsDatum[] = await dynamicsResponse.json();
      const demographicsData: DemographicsDatum[] = await demographicsResponse.json();
      const historyRows: HistoryRow[] = await historyResponse.json();
      const zonesRows: ZoneAnalyticsRow[] = await zonesResponse.json();
      const routesRows: RouteAnalyticsRow[] = await routesResponse.json();
      const heatmapRows: HeatmapPoint[] = await heatmapResponse.json();

      if (
        !summaryResponse.ok ||
        !dynamicsResponse.ok ||
        !demographicsResponse.ok ||
        !historyResponse.ok ||
        !zonesResponse.ok ||
        !routesResponse.ok ||
        !heatmapResponse.ok
      ) {
        throw new Error("Не вдалося завантажити аналітику.");
      }

      setSummary(summaryData);
      setDynamicsData(dynamicsRows);
      setDemographics(demographicsData);
      setHistoryData(historyRows);
      setZoneData(zonesRows);
      setRoutesData(routesRows);
      setHeatmapPoints(heatmapRows);
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
  }, [dateTab, customDate, timeStart, timeEnd, dynamicsView]);

  useEffect(() => {
    const fetchCafe = async () => {
      const token = localStorage.getItem("cafe_token");
      if (!token) {
        return;
      }

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
        onCafeUpdate?.(data);
      } catch (fetchError) {
        console.error("Помилка завантаження кафе:", fetchError);
      }
    };

    void fetchCafe();
  }, []);

  useEffect(() => {
    if (externalCafe) {
      setMyCafe(externalCafe);
    }
  }, [externalCafe]);

  const isCustomFilterActive =
    dateTab !== "today" || Boolean(customDate) || Boolean(timeStart || timeEnd);

  const resetFilters = () => {
    setDateTab("today");
    setCustomDate("");
    setTimeStart("");
    setTimeEnd("");
  };

  const dynamicsXAxisKey = dynamicsView === "hours" ? "time" : "date";

  const handleFileUpload = async (file: File) => {
    const token = localStorage.getItem("cafe_token");
    if (!token) {
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    setIsUploadingPlan(true);

    try {
      const response = await fetch(`${API_ROOT}/api/cafe/update-floor-plan`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await response.json() as { detail?: string; floor_plan_url?: string };
      if (!response.ok) {
        throw new Error(data.detail || "Не вдалося завантажити план приміщення.");
      }
      const planUrl = data.floor_plan_url;
      if (!planUrl) {
        throw new Error("Сервер не повернув URL плану.");
      }

      const nextCafe = {
        name: myCafe?.name || "",
        address: myCafe?.address || "",
        floor_plan_url: planUrl,
      };
      setMyCafe(nextCafe);
      onCafeUpdate?.(nextCafe);
    } catch (uploadError) {
      console.error("Помилка завантаження плану приміщення:", uploadError);
    } finally {
      setIsUploadingPlan(false);
    }
  };

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

  const trendComparisonLabel = summaryComparisonLabel(dateTab, customDate);

  const hasDynamicsData = dynamicsData.some((d) => Number(d.customers) > 0);
  const hasDemographicData = demographics.some(
    (row) => Number(row.Чоловіки) > 0 || Number(row.Жінки) > 0,
  );

  return (
    <div className="min-h-full w-full bg-[#f5f3f0] pt-6 font-sans text-[#411e03]">
      <div className="max-w-[1600px] mx-auto w-full">
        <div className="flex flex-col gap-6">
          {error && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm font-medium text-rose-600">
              Помилка: {error}
            </div>
          )}

          <div ref={dashboardRef} className="flex flex-col gap-6 rounded-xl bg-[#f5f3f0]">
            <div className="flex flex-col items-start justify-between gap-4 xl:flex-row xl:items-center">
              <div className="flex flex-wrap items-center gap-3">
                <Tabs
                  value={dateTab}
                  onValueChange={(value) => {
                    setDateTab(value);
                    setCustomDate("");
                    setTimeStart("");
                    setTimeEnd("");
                  }}
                  className="w-auto"
                >
                  <TabsList className="h-10 border border-slate-200 bg-white shadow-sm">
                    <TabsTrigger
                      value="today"
                      className="text-slate-600 data-[state=active]:bg-slate-100 data-[state=active]:text-slate-900"
                    >
                      Сьогодні
                    </TabsTrigger>
                    <TabsTrigger
                      value="7days"
                      className="text-slate-600 data-[state=active]:bg-slate-100 data-[state=active]:text-slate-900"
                    >
                      7 днів
                    </TabsTrigger>
                    <TabsTrigger
                      value="30days"
                      className="text-slate-600 data-[state=active]:bg-slate-100 data-[state=active]:text-slate-900"
                    >
                      30 днів
                    </TabsTrigger>
                    <TabsTrigger
                      value="all"
                      className="text-slate-600 data-[state=active]:bg-slate-100 data-[state=active]:text-slate-900"
                    >
                      Весь час
                    </TabsTrigger>
                  </TabsList>
                </Tabs>

                <div className="flex items-center gap-3">
                  <div className="relative">
                    <Input
                      type="date"
                      value={customDate}
                      onChange={(event) => {
                        setCustomDate(event.target.value);
                        setDateTab("");
                      }}
                      className="h-10 w-40 border-[#d7cdc1] bg-white px-3 pr-8 text-[#411e03] shadow-sm focus-visible:ring-[#8a895f]"
                    />
                    {customDate && (
                      <button
                        type="button"
                        onClick={() => setCustomDate("")}
                        className="absolute top-1/2 right-2 -translate-y-1/2 text-[#846851] transition-colors hover:text-[#411e03]"
                      >
                        <X size={14} />
                      </button>
                    )}
                  </div>
                  <div
                    className="flex min-w-[220px] items-center gap-1.5 rounded-md border border-[#d7cdc1] bg-white px-2.5 py-1 shadow-sm focus-within:ring-2 focus-within:ring-[#8a895f]/40"
                  >
                    <Clock size={16} className="shrink-0 text-slate-400" aria-hidden />
                    <input
                      type="time"
                      aria-label="Початок інтервалу часу"
                      value={timeStart}
                      onChange={(event) => {
                        setTimeStart(event.target.value);
                        setDateTab("");
                      }}
                      className="h-9 min-w-[5.75rem] flex-1 border-0 bg-transparent text-sm text-[#411e03] outline-none [&::-webkit-calendar-picker-indicator]:opacity-70"
                    />
                    <span className="shrink-0 select-none text-[#846851]" aria-hidden>
                      –
                    </span>
                    <input
                      type="time"
                      aria-label="Кінець інтервалу часу"
                      value={timeEnd}
                      onChange={(event) => {
                        setTimeEnd(event.target.value);
                        setDateTab("");
                      }}
                      className="h-9 min-w-[5.75rem] flex-1 border-0 bg-transparent text-sm text-[#411e03] outline-none [&::-webkit-calendar-picker-indicator]:opacity-70"
                    />
                    {(timeStart || timeEnd) && (
                      <button
                        type="button"
                        onClick={() => {
                          setTimeStart("");
                          setTimeEnd("");
                        }}
                        className="ml-0.5 shrink-0 rounded p-0.5 text-[#846851] transition-colors hover:bg-[#f5f3f0] hover:text-[#411e03]"
                        aria-label="Очистити час"
                      >
                        <X size={14} />
                      </button>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex shrink-0 flex-wrap items-center gap-3">
                <Button
                  variant="ghost"
                  onClick={resetFilters}
                  disabled={!isCustomFilterActive}
                  className="text-[#846851] hover:bg-[#f5f3f0] hover:text-[#411e03]"
                >
                  Скинути
                </Button>
                <Button
                  variant="outline"
                  onClick={() => void loadStats()}
                  disabled={loading}
                  className="gap-2 border-slate-200 bg-white text-slate-600 shadow-sm hover:bg-[#f5f3f0]"
                >
                  <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
                  {loading ? "Оновлення..." : "Оновити"}
                </Button>
                <Button
                  onClick={handleExportPDF}
                  disabled={isExporting}
                  className="h-10 gap-2 border border-transparent bg-[#411e03]/90 px-4 text-white shadow-sm transition-colors [&_svg]:text-[#d6b38d] hover:bg-[#411e03] active:opacity-70 focus-visible:border focus-visible:border-[#d6b38d]/50 focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0 disabled:opacity-50"
                >
                  {isExporting ? <RefreshCw size={16} className="animate-spin text-[#d6b38d]" /> : <Download size={16} />}
                  {isExporting ? "Генерація..." : "Експорт"}
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <Card className="border-[#d7cdc1] bg-white shadow-sm transition-shadow hover:shadow-md">
                <CardContent className="flex items-center gap-3 p-4">
                  <div className="rounded-xl bg-blue-50 p-2.5 text-blue-600">
                    <Users size={20} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-[#846851]">Всього клієнтів</p>
                    <div className="text-2xl font-bold text-[#411e03]">{summary?.total_clients ?? 0}</div>
                    <SummaryTrendFootnote trend={summary?.clients_trend ?? null} comparisonLabel={trendComparisonLabel} />
                  </div>
                </CardContent>
              </Card>

              <Card className="border-[#d7cdc1] bg-white shadow-sm transition-shadow hover:shadow-md">
                <CardContent className="flex items-center gap-3 p-4">
                  <div className="rounded-xl bg-emerald-50 p-2.5 text-emerald-600">
                    <Clock size={20} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-[#846851]">Сер. час перебування</p>
                    <div className="text-2xl font-bold text-[#411e03]">
                      {formatDurationUk(summary?.avg_duration_seconds ?? 0)}
                    </div>
                    <SummaryTrendFootnote trend={summary?.duration_trend ?? null} comparisonLabel={trendComparisonLabel} />
                  </div>
                </CardContent>
              </Card>

              <Card className="border-[#d7cdc1] bg-white shadow-sm transition-shadow hover:shadow-md">
                <CardContent className="flex items-center gap-3 p-4">
                  <div className="rounded-xl bg-amber-50 p-2.5 text-amber-600">
                    <Zap size={20} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-[#846851]">Година пік</p>
                    <div className="text-2xl font-bold text-[#411e03]">{summary?.peak_hour ?? "Немає даних"}</div>
                    {summary?.peak_hour != null && summary.peak_clients != null ? (
                      <div className="mt-1.5 text-sm text-[#846851]">{summary.peak_clients} клієнтів</div>
                    ) : null}
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
              <Card className="border-[#d7cdc1] bg-white shadow-sm lg:col-span-2">
                <CardHeader className="p-5 pb-4">
                  <CardTitle className="flex items-center gap-2 text-lg font-semibold text-[#411e03]">
                    <MapPinned size={18} className="text-rose-500" />
                    Теплова карта перебування
                  </CardTitle>
                </CardHeader>
                <CardContent className="px-5 pb-5 pt-0">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) {
                        void handleFileUpload(file);
                      }
                      event.target.value = "";
                    }}
                  />
                  {myCafe?.floor_plan_url ? (
                    <div className="mx-auto flex w-full max-w-[880px] flex-row items-center gap-6 mt-4">
                      <div className="flex h-[280px] shrink-0 flex-col items-center justify-between">
                        <span className="w-16 text-center text-xs font-medium text-[#846851]">
                          Більше людей
                        </span>

                        <div className="my-3 w-4 flex-1 rounded-full border border-[#d7cdc1]/50 bg-[linear-gradient(to_bottom,#ef4444,#eab308,#22c55e,#3b82f6)] shadow-inner" />

                        <span className="w-16 text-center text-xs font-medium text-[#846851]">
                          Менше людей
                        </span>
                      </div>

                      <div className="relative flex h-[500px] min-w-0 flex-1 items-center justify-center overflow-hidden rounded-lg bg-white shadow-inner">
                        <AnalyticsHeatmap floorPlanUrl={myCafe.floor_plan_url} points={heatmapPoints} />
                        <button
                          type="button"
                          onClick={() => fileInputRef.current?.click()}
                          className="absolute top-3 right-3 rounded-md bg-white/90 px-3 py-1.5 text-xs font-medium text-[#411e03] shadow-sm transition hover:bg-white"
                        >
                          {isUploadingPlan ? "Завантаження..." : "Змінити план"}
                        </button>
                      </div>
                    </div>
                  ) : (
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="mx-auto flex h-[500px] w-full max-w-[800px] flex-col items-center justify-center rounded-lg border-2 border-dashed border-slate-300 bg-white px-6 text-center transition hover:bg-[#fdfdfd]"
                    >
                      <span className="mb-3 flex h-14 w-14 items-center justify-center rounded-full bg-[#f5f3f0] text-3xl text-[#846851]">
                        +
                      </span>
                      <span className="text-base font-semibold text-[#411e03]">Завантажити план приміщення</span>
                      <span className="mt-2 text-sm text-[#846851]">
                        {isUploadingPlan ? "Завантаження..." : "Додайте floor plan для вашого кафе"}
                      </span>
                    </button>
                  )}
                </CardContent>
              </Card>

              <Card className="border-[#d7cdc1] bg-white shadow-sm lg:col-span-1">
                <CardHeader className="p-5 pb-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-bold text-[#411e03]">Статистика по зонах</h3>
                    <Info size={16} className="cursor-help text-[#846851]/60" />
                  </div>
                </CardHeader>
                <CardContent className="px-5 pb-5 pt-0">
                  <div className="mb-4 grid grid-cols-[1fr_auto_auto_40px] gap-4 border-b border-[#d7cdc1]/30 pb-2 text-xs font-medium text-[#846851]">
                    <div>Зона</div>
                    <div className="text-center">Сер. час</div>
                    <div className="text-right">К-сть</div>
                    <div className="text-right">%</div>
                  </div>

                  <div className="space-y-4">
                    {zoneData.map((zone, idx) => (
                      <div key={idx} className="relative">
                        <div className="relative z-10 mb-1 grid grid-cols-[1fr_auto_auto_40px] items-center gap-4 text-sm">
                          <div className="flex items-center gap-2 font-medium text-[#411e03]">
                            <div className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: zone.color }} />
                            <span className="truncate">{zone.name}</span>
                          </div>
                          <div className="whitespace-nowrap text-center text-[#8a895f]">{zone.avg_time}</div>
                          <div className="text-right font-medium text-[#411e03]">{zone.visits}</div>
                          <div className="text-right text-xs text-[#846851]">{zone.percentage}%</div>
                        </div>
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-[#f5f3f0]">
                          <div
                            className="h-full rounded-full transition-all duration-500 ease-out"
                            style={{ width: `${zone.percentage}%`, backgroundColor: zone.color }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  <button className="mt-6 flex items-center gap-1 text-sm font-bold text-[#8a895f] transition-colors hover:text-[#411e03]">
                    Детальніше про зони &rarr;
                  </button>
                </CardContent>
              </Card>
            </div>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
              <Card className="border-[#d7cdc1] bg-white shadow-sm lg:col-span-2">
                <CardHeader className="flex flex-col gap-4 p-5 pb-4 sm:flex-row sm:items-center sm:justify-between">
                  <CardTitle className="flex items-center gap-2 text-lg font-semibold text-[#411e03]">
                    <TrendingUp size={18} className="text-[#659b4c]" />
                    Динаміка відвідуваності
                  </CardTitle>
                  <Tabs
                    value={dynamicsView}
                    onValueChange={(value) => setDynamicsView(value as "hours" | "days")}
                    className="w-auto"
                  >
                    <TabsList className="h-9 border border-[#d7cdc1] bg-[#f5f3f0] shadow-sm">
                      <TabsTrigger
                        value="hours"
                        className="text-[#846851] data-[state=active]:bg-slate-100 data-[state=active]:text-[#411e03]"
                      >
                        Години
                      </TabsTrigger>
                      <TabsTrigger
                        value="days"
                        className="text-[#846851] data-[state=active]:bg-slate-100 data-[state=active]:text-[#411e03]"
                      >
                        Дні
                      </TabsTrigger>
                    </TabsList>
                  </Tabs>
                </CardHeader>
                <CardContent className="px-5 pb-5 pt-0">
                  <div className="h-64 min-h-[256px] w-full">
                    {hasDynamicsData ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={dynamicsData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                          <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" vertical={false} />
                          <XAxis
                            dataKey={dynamicsXAxisKey}
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
                            stroke="#659b4c"
                            strokeWidth={3}
                            dot={{ r: 4, fill: "#659b4c", stroke: "#ffffff", strokeWidth: 2 }}
                            activeDot={{ r: 6, fill: "#659b4c", stroke: "#ffffff", strokeWidth: 2 }}
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

              <Card className="border-[#d7cdc1] bg-white shadow-sm lg:col-span-1">
                <CardHeader className="p-5 pb-4">
                  <CardTitle className="text-lg font-semibold text-[#411e03]">Розподіл за віком і статтю</CardTitle>
                </CardHeader>
                <CardContent className="px-5 pb-5 pt-0">
                  <div className="h-64 min-h-[256px] w-full">
                    {hasDemographicData ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={demographics} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                          <XAxis
                            dataKey="age_group"
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: "#846851", fontSize: 12 }}
                            dy={10}
                          />
                          <YAxis axisLine={false} tickLine={false} tick={{ fill: "#846851", fontSize: 12 }} />
                          <Tooltip
                            cursor={{ fill: "#f8fafc" }}
                            contentStyle={{
                              borderRadius: "8px",
                              border: "1px solid #d7cdc1",
                              boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)",
                            }}
                          />
                          <Legend
                            iconType="circle"
                            wrapperStyle={{ paddingTop: "20px", fontSize: "12px", color: "#411e03" }}
                          />
                          <Bar dataKey="Чоловіки" fill="#8a895f" radius={[4, 4, 0, 0]} barSize={12} />
                          <Bar dataKey="Жінки" fill="#846851" radius={[4, 4, 0, 0]} barSize={12} />
                        </BarChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex h-full items-center justify-center text-sm text-slate-400">
                        Немає демографічних даних
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
              <Card className="overflow-hidden border-[#d7cdc1] bg-white shadow-sm lg:col-span-2">
                <div className="flex items-center justify-between border-b border-[#d7cdc1]/50 p-5">
                  <h3 className="text-lg font-bold text-[#411e03]">Статистика по датах</h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead className="bg-[#f5f3f0] font-medium text-[#846851]">
                      <tr>
                        <th className="rounded-tl-lg px-6 py-4">Дата</th>
                        <th className="px-6 py-4">Всього клієнтів</th>
                        <th className="px-6 py-4">Сер. час</th>
                        <th className="px-6 py-4">Година пік</th>
                        <th className="px-6 py-4">Стать</th>
                        <th className="rounded-tr-lg px-6 py-4">Осн. вікова група</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#d7cdc1]/30">
                      {historyData.length === 0 ? (
                        <tr>
                          <td colSpan={6} className="px-6 py-10 text-center text-sm text-[#846851]">
                            Немає даних за обраний період
                          </td>
                        </tr>
                      ) : (
                        historyData.map((row, idx) => (
                          <tr key={`${row.date}-${idx}`} className="text-[#411e03] transition-colors hover:bg-slate-50">
                            <td className="px-6 py-4 font-medium">{row.date}</td>
                            <td className="px-6 py-4">{row.total}</td>
                            <td className="px-6 py-4 text-[#8a895f]">{row.avg_time}</td>
                            <td className="px-6 py-4">{row.peak}</td>
                            <td className="px-6 py-4">{row.gender}</td>
                            <td className="px-6 py-4">{row.top_age}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
                <div className="border-t border-[#d7cdc1]/50 bg-white p-4">
                  <button className="flex items-center gap-1 text-sm font-bold text-[#8a895f] transition-colors hover:text-[#411e03]">
                    Переглянути повну історію &rarr;
                  </button>
                </div>
              </Card>

              <Card className="border-[#d7cdc1] bg-white shadow-sm lg:col-span-1">
                <CardHeader className="p-5 pb-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-bold text-[#411e03]">Найпопулярніші маршрути</h3>
                    <Info size={16} className="cursor-help text-[#846851]/60" />
                  </div>
                </CardHeader>
                <CardContent className="px-5 pb-5 pt-0">
                  <div className="space-y-6">
                    {routesData.length === 0 ? (
                      <p className="py-6 text-center text-sm text-[#846851]">Немає даних про маршрути</p>
                    ) : (
                      routesData.map((route, idx) => (
                      <div key={idx} className="group">
                        <div className="mb-2 flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <span className="flex h-6 w-6 items-center justify-center rounded-full border border-[#d7cdc1] bg-[#f5f3f0] text-xs font-bold text-[#411e03]">
                              {idx + 1}
                            </span>
                            <p className="text-sm font-medium text-[#411e03] transition-colors group-hover:text-[#8a895f]">
                              {route.path}
                            </p>
                          </div>
                          <span className="text-sm font-bold text-[#411e03]">{route.percentage}%</span>
                        </div>
                        <div className="ml-9 h-1.5 w-full overflow-hidden rounded-full bg-[#f5f3f0]">
                          <div
                            className="h-full rounded-full bg-[#8a895f] transition-all duration-700 ease-out"
                            style={{ width: `${route.percentage}%` }}
                          />
                        </div>
                      </div>
                      ))
                    )}
                  </div>

                  <button className="mt-8 flex items-center gap-1 text-sm font-bold text-[#8a895f] transition-colors hover:text-[#411e03]">
                    Детальніше про маршрути &rarr;
                  </button>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}