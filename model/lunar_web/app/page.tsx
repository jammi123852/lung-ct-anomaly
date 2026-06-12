"use client"

import { useState, useMemo, useCallback, useRef, useEffect } from "react"
import {
  Upload, Eye, Search, Download, Check, ChevronLeft, ChevronRight,
  FolderOpen, Activity, Users, Globe, X, Loader2, ExternalLink, Trash2
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels"
import { cn } from "@/lib/utils"

// ---------------------------------------------------------------------------
// IndexedDB helper — DICOM 파일 영구 저장
// ---------------------------------------------------------------------------
const IDB_NAME = "lunar_db", IDB_VER = 1, IDB_STORE = "dicoms"
function openIDB(): Promise<IDBDatabase> {
  return new Promise((res, rej) => {
    const req = indexedDB.open(IDB_NAME, IDB_VER)
    req.onupgradeneeded = () => req.result.createObjectStore(IDB_STORE, { keyPath: "patientId" })
    req.onsuccess = () => res(req.result)
    req.onerror = () => rej(req.error)
  })
}
async function saveDicomsIDB(patientId: string, files: File[]) {
  try {
    const db = await openIDB()
    const entries = await Promise.all(files.map(async f => ({ name: f.name, buf: await f.arrayBuffer() })))
    const tx = db.transaction(IDB_STORE, "readwrite")
    tx.objectStore(IDB_STORE).put({ patientId, entries })
  } catch (e) { console.warn("[IDB] save failed", e) }
}
async function loadDicomsIDB(patientId: string): Promise<File[] | null> {
  try {
    const db = await openIDB()
    return new Promise((res) => {
      const req = db.transaction(IDB_STORE, "readonly").objectStore(IDB_STORE).get(patientId)
      req.onsuccess = () => {
        const d = req.result
        if (!d) return res(null)
        res(d.entries.map(({ name, buf }: { name: string; buf: ArrayBuffer }) =>
          new File([buf], name, { type: "application/octet-stream" })
        ))
      }
      req.onerror = () => res(null)
    })
  } catch { return null }
}

// ---------------------------------------------------------------------------
// Markdown report generator
// ---------------------------------------------------------------------------
function generateReportMd(data: {
  patientId: string; name: string; birthdate: string; gender: string
  date: string; opinion: string; risk: string; score?: number
  totalSlices?: number; highRiskPatches?: Array<{sliceIndex: number; score: number}>
}): string {
  const lines: string[] = []
  lines.push("# LUNAR 폐 이상탐지 AI 보고서")
  lines.push("")
  lines.push("## 환자 정보")
  lines.push(`| 항목 | 내용 |`)
  lines.push(`|------|------|`)
  lines.push(`| 환자 ID | ${data.patientId} |`)
  lines.push(`| 이름 | ${data.name} |`)
  lines.push(`| 생년월일 | ${data.birthdate || "-"} |`)
  lines.push(`| 성별 | ${data.gender || "-"} |`)
  lines.push(`| 분석 일시 | ${new Date(data.date).toLocaleString("ko-KR")} |`)
  lines.push("")
  lines.push("## AI 분석 결과")
  const riskLabel = data.risk === "Critical" ? "🟣 위험" : data.risk === "High" ? "🔴 고위험" : data.risk === "Medium" ? "🟡 중위험" : "🟢 저위험"
  lines.push(`- **종합 위험도**: ${riskLabel}`)
  if (data.score != null) lines.push(`- **최고 이상 점수**: ${data.score.toFixed(3)} (P5 기준: ≥0.45 위험)`)
  if (data.totalSlices != null) lines.push(`- **총 슬라이스 수**: ${data.totalSlices}`)
  lines.push("")
  if (data.highRiskPatches && data.highRiskPatches.length > 0) {
    lines.push("## 고위험 구역 (상위 20개)")
    lines.push("")
    lines.push("| 순위 | 슬라이스 | 이상 점수 | 위험도 |")
    lines.push("|------|---------|---------|--------|")
    data.highRiskPatches.slice(0, 20).forEach((p, i) => {
      const r = p.score > 0.45 ? "위험" : p.score > 0.35 ? "고위험" : p.score > 0.20 ? "중위험" : "저위험"
      lines.push(`| ${i + 1} | ${p.sliceIndex + 1} | ${p.score.toFixed(2)} | ${r} |`)
    })
    lines.push("")
  }
  if (data.opinion) {
    lines.push("## 의사 소견")
    lines.push("")
    lines.push(data.opinion)
    lines.push("")
  }
  lines.push("---")
  lines.push("*본 보고서는 LUNAR (Lung Anomaly Recognition) AI가 생성한 보조 진단 자료입니다.*")
  lines.push("*최종 진단은 반드시 전문의의 판단을 따르십시오.*")
  return lines.join("\n")
}

// ---------------------------------------------------------------------------
// Translations
// ---------------------------------------------------------------------------
const translations = {
  en: {
    brand: "LUNAR", subtitle: "Lung Anomaly Recognition",
    upload: "Upload CT", analysis: "Analysis Viewer", patients: "Patient Management",
    loginConnect: "Connect B2 Storage",
    // Dual-stage weights
    verifyWeights: "Verify Local Dual Weights",
    verifying2D: "Verifying 1st Stage: 2D Screening...",
    verifying25D: "Verifying 2nd Stage: 2.5D Volumetric...",
    weightsActive: "Dual-Stage Weights Active (2D + 2.5D)",
    weight2DLabel: "1st Stage: 2D Screening Weights",
    weight25DLabel: "2nd Stage: 2.5D Volumetric Weights",
    // Upload
    selectFolder: "Select CT Folder",
    dropHint: "Click to select a folder containing DICOM files",
    runAnalysis: "Run Dual-Stage AI Analysis",
    phase1Label: "Preprocessing: Pure Lung Mask Generation...",
    phase2Label: "Stage 1: PaDiM Anomaly Detection...",
    // Analysis
    allSlices: "All Slices", highRisk: "High-Risk Volumes",
    showHeatmap: "Show Heatmap", showBoxes: "Show Anomaly Bounding Boxes",
    resultSummary: "Result Summary", medicalOpinion: "Medical Opinion Form",
    saveReport: "Save & Generate Patient Report",
    // Patients
    searchPatients: "Search patients...",
    name: "Name", date: "Date",
    aiInspectionStatus: "AI Inspection Status",
    risk: "Risk",
    viewCT: "View CT", downloadReport: "Download Report", patientDetails: "Patient Details",
    clinicalNotes: "Clinical notes and observations...",
    upper: "Upper Lobe", middle: "Middle Lobe", lower: "Lower Lobe",
    central: "Central", peripheral: "Peripheral", slice: "Slice",
    anomalyScore: "Anomaly Score",
    steps: ["LPS Alignment", "1mm Z-axis Normalization", "TotalSegmentator Organ Exclusion", "Pure Lung Masking", "32x32 Patch Extraction"],
    allDates: "All Dates",
    openInAnalysis: "Open in Analysis Viewer",
    noPatients: "No patients found",
    statusCompleted: "Completed", statusPending: "Pending", statusInReview: "In Review",
    riskHigh: "High", riskMedium: "Medium", riskLow: "Low", riskCritical: "Critical",
  },
  ko: {
    brand: "LUNAR", subtitle: "폐 이상 인식 시스템",
    upload: "CT 업로드", analysis: "분석 뷰어", patients: "환자 관리",
    loginConnect: "B2 Storage 연결",
    verifyWeights: "로컬 이중 가중치 확인",
    verifying2D: "1단계 확인 중: 2D 스크리닝...",
    verifying25D: "2단계 확인 중: 2.5D 볼류메트릭...",
    weightsActive: "이중 단계 가중치 활성화 (2D + 2.5D)",
    weight2DLabel: "1단계: 2D 스크리닝 가중치",
    weight25DLabel: "2단계: 2.5D 볼류메트릭 가중치",
    selectFolder: "CT 폴더 선택",
    dropHint: "DICOM 파일이 포함된 폴더를 클릭하여 선택하세요",
    runAnalysis: "이중 단계 AI 분석 실행",
    phase1Label: "전처리: 순수 폐 마스크 생성 중...",
    phase2Label: "1단계: PaDiM 이상 탐지 중...",
    allSlices: "전체 슬라이스", highRisk: "고위험 영역",
    showHeatmap: "히트맵 표시", showBoxes: "이상 경계 박스 표시",
    resultSummary: "결과 요약", medicalOpinion: "의료 소견 양식",
    saveReport: "보고서 저장 및 생성",
    searchPatients: "환자 검색...",
    name: "이름", date: "날짜",
    aiInspectionStatus: "AI 검사 상태",
    risk: "위험도",
    viewCT: "CT 보기", downloadReport: "보고서 다운로드", patientDetails: "환자 상세정보",
    clinicalNotes: "임상 소견 및 관찰 사항...",
    upper: "상엽", middle: "중엽", lower: "하엽",
    central: "중심부", peripheral: "주변부", slice: "슬라이스",
    anomalyScore: "이상 점수",
    steps: ["LPS 정렬", "1mm Z축 정규화", "TotalSegmentator 장기 제외", "순수 폐 마스킹", "32x32 패치 추출"],
    allDates: "전체 날짜",
    openInAnalysis: "분석 뷰어에서 CT 보기",
    noPatients: "환자를 찾을 수 없습니다",
    statusCompleted: "검사 완료", statusPending: "대기중", statusInReview: "검토중",
    riskHigh: "고위험", riskMedium: "중위험", riskLow: "저위험", riskCritical: "초고위험",
  }
}

type T = typeof translations.en

function tStatus(t: T, raw: string) {
  if (raw === "Completed") return t.statusCompleted
  if (raw === "Pending") return t.statusPending
  return t.statusInReview
}

function tRisk(t: T, raw: string) {
  if (raw === "Critical") return t.riskCritical
  if (raw === "High") return t.riskHigh
  if (raw === "Medium") return t.riskMedium
  return t.riskLow
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
const TOTAL_SLICES = 999
const generateSlices = ( count = TOTAL_SLICES ) =>
  Array.from({ length: count }, (_, i) => ({ id: i + 1, score: 0 }))

const generatePatients = () => [] as Array<{
  id: string
  name: string
  birthdate: string
  gender: string
  date: string
  status: string
  risk: string
  opinion: string
  dcmFiles?: File[]
  savedPatchesMap?: Map<number, Array<{position: {y0:number, x0:number, y1:number, x1:number}, score:number, padim_score?: number}>>
  analysisStatus?: string   // "pending" | "downloading" | "analyzing" | "done" | "error" | "unknown"
  autoAnalyzed?: boolean    // B2 webhook으로 자동 분석된 환자
  source?: string           // "manual" | "auto_webhook"
  volumeSliceCount?: number // 분석 볼륨 슬라이스 수 (전처리 crop 반영, #10)
  trackSummary?: any        // 백엔드 트랙 집계 (top-K/절약률, #1)
}>

type View = "upload" | "analysis" | "patients"
type Lang = "en" | "ko"
type Patient = ReturnType<typeof generatePatients>[0]

// Dual-stage weights states
type WeightsStage = "idle" | "loading2d" | "loading25d" | "active"

// ---------------------------------------------------------------------------
// Root
// ---------------------------------------------------------------------------
export default function LunarSystem() {
  const [view, setView] = useState<View>("upload")
  const [lang, setLang] = useState<Lang>("en")
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  // Backblaze B2 state
  const [b2Connected, setB2Connected] = useState(false)
  const [weightsStage, setWeightsStage] = useState<WeightsStage>("idle")

  // Upload state
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isUploading, setIsUploading] = useState(false)
  const [isValidating, setIsValidating] = useState(false)
  const [validateResult, setValidateResult] = useState<{valid: boolean, errors: any[], warnings: string[], valid_count: number, total: number} | null>(null)
  const [analysisProgress, setAnalysisProgress] = useState(0)
  const [isRunningAnalysis, setIsRunningAnalysis] = useState(false)
  const [analysisStage, setAnalysisStage] = useState("순수 폐 마스크 생성 중...")
  const [stageLogs, setStageLogs] = useState<Array<{label: string, seconds: number}>>([])
  const [folderName, setFolderName] = useState("")
  const [dcmFiles, setDcmFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Analysis state
  const [slices, setSlices] = useState(generateSlices)
  const [activeSlice, setActiveSlice] = useState(1)
  const [sliceTab, setSliceTab] = useState<"all" | "high">("all")
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [showBoxes, setShowBoxes] = useState(true)
  const [slicePatchesMap, setSlicePatchesMap] = useState<Map<number, Array<{position: {y0:number, x0:number, y1:number, x1:number}, score:number, padim_score?: number}>>>(new Map())
  // 분석된 볼륨의 실제 슬라이스 수 (전처리 crop으로 업로드 수와 다를 수 있음, #10)
  const [volumeSliceCount, setVolumeSliceCount] = useState(0)
  // 백엔드 트랙 집계 (트랙별 p5 정렬 + all_z, #1) — 고위험 top-K/절약률의 정본
  const [trackSummary, setTrackSummary] = useState<any>(null)
  const [opinion, setOpinion] = useState("")
  const [cardData, setCardData] = useState<any>(null)
  const [cardLoading, setCardLoading] = useState(false)
  const [selectedRiskRegion, setSelectedRiskRegion] = useState<{min: number, max: number} | null>(null)

  // Patient state
  const [patients, setPatients] = useState(generatePatients)
  const [patientName, setPatientName] = useState("")
  const [patientBirthdate, setPatientBirthdate] = useState("")
  const [patientGender, setPatientGender] = useState("")
  const [searchQuery, setSearchQuery] = useState("")
  const [dateFilter, setDateFilter] = useState<string>("all")
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null)

  // Toast
  const [toast, setToast] = useState<string | null>(null)
  const showToast = useCallback((msg: string) => {
    setToast(msg)
    setTimeout(() => setToast(null), 5000)
  }, [])

  const [showLangDropdown, setShowLangDropdown] = useState(false)
  const t = translations[lang]

  // ----- B2에서 환자 목록 로드 (앱 시작 시 1회) -----
  useEffect(() => {
    fetch("http://localhost:8000/b2/list-records")
      .then(r => r.json())
      .then(data => {
        const b2Patients = (data.records || []).map((r: any) => ({
          id:             r.patientId,
          name:           r.name || r.patientId,
          birthdate:      r.birthdate || "-",
          gender:         r.gender || "-",
          date:           r.date ? r.date.split("T")[0] : "-",
          status:         r.status || "Completed",
          risk:           r.risk || "Low",
          opinion:        r.opinion || "",
          analysisStatus: "done",
          autoAnalyzed:   r.source === "auto_webhook",
          source:         r.source || "manual",
        }))
        if (b2Patients.length > 0) setPatients(b2Patients)
      })
      .catch(() => {})
  }, [])

  // ----- pending/analyzing 환자 상태 폴링 (5초마다) -----
  const patientsRef = useRef(patients)
  useEffect(() => { patientsRef.current = patients }, [patients])

  useEffect(() => {
    const timer = setInterval(async () => {
      const current = patientsRef.current
      const needsPoll = current.some(
        p => p.analysisStatus === "pending" || p.analysisStatus === "downloading" || p.analysisStatus === "analyzing"
      )
      if (!needsPoll) return
      const updated = await Promise.all(
        current.map(async p => {
          if (p.analysisStatus !== "pending" && p.analysisStatus !== "downloading" && p.analysisStatus !== "analyzing") return p
          try {
            const r = await fetch(`http://localhost:8000/b2/analysis-status/${p.id}`)
            const d = await r.json()
            return { ...p, analysisStatus: d.status, risk: d.risk || p.risk }
          } catch { return p }
        })
      )
      const hasChanged = updated.some((p, i) => p.analysisStatus !== current[i].analysisStatus)
      if (hasChanged) setPatients(updated)
    }, 5000)
    return () => clearInterval(timer)
  }, [])

  // ----- Dual-Stage Weights: sequential 0.5s + 0.5s verify -----
  const handleVerifyWeights = useCallback(async () => {
    if (weightsStage !== "idle") return
    setWeightsStage("loading2d")
    try {
      const res = await fetch("http://localhost:8000/health")
      if (!res.ok) throw new Error("서버 응답 없음")
      setWeightsStage("loading25d")
      setTimeout(() => setWeightsStage("active"), 500)
    } catch {
      setWeightsStage("idle")
      alert("백엔드 서버에 연결할 수 없습니다.\nuvicorn 서버가 실행 중인지 확인하세요.")
    }
  }, [weightsStage])

  // ----- Backblaze B2 connect -----
  const handleConnectRepo = useCallback(async () => {
    try {
      const res = await fetch("http://localhost:8000/b2/health")
      const data = await res.json()
      if (data.status === "ok") {
        setB2Connected(true)
        showToast("Backblaze B2 연결 성공!")
      } else {
        alert("Backblaze B2 연결 실패: " + data.message)
      }
    } catch {
      alert("Backblaze B2 연결 실패. 백엔드 서버를 확인해주세요.")
    }
  }, [showToast])

  // ----- Upload folder select -----
  const handleFolderSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return
    const parts = files[0].webkitRelativePath?.split("/") || []
    setFolderName(parts[0] || "CT_Folder")

    const dcm = Array.from(files)
      .filter(f => f.name.toLowerCase().endsWith(".dcm"))
      .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))
    setDcmFiles(dcm)
    setSlices(generateSlices(dcm.length))
    setValidateResult(null)

    // 1단계: 업로드 진행바
    setIsUploading(true)
    setUploadProgress(0)
    const interval = setInterval(() => {
      setUploadProgress(p => {
        if (p >= 100) { clearInterval(interval); return 100 }
        return Math.min(p + 1.8, 100)
      })
    }, 55)

    // 2단계: 백엔드 검증 (샘플 10개만 검증 - 전체 보내면 느림)
    setIsValidating(true)
    try {
      const sampleFiles = dcm.slice(0, Math.min(10, dcm.length))
      const formData = new FormData()
      sampleFiles.forEach(f => formData.append("files", f))
      const res = await fetch("http://localhost:8000/validate", {
        method: "POST",
        body: formData,
      })
      const result = await res.json()
      // /validate가 비표준 응답(500 {detail} 등)을 줘도 렌더가 기대하는 형태로 정규화
      // (errors/warnings가 배열이 아니면 .length/.map에서 크래시)
      setValidateResult({
        valid:       typeof result?.valid === "boolean" ? result.valid : false,
        errors:      Array.isArray(result?.errors) ? result.errors : [],
        warnings:    Array.isArray(result?.warnings) ? result.warnings : [],
        valid_count: typeof result?.valid_count === "number" ? result.valid_count : 0,
        total:       typeof result?.total === "number" ? result.total : dcm.length,
      })
    } catch {
      setValidateResult({ valid: false, errors: [{ filename: "", message: "백엔드 연결 실패" }], warnings: [], valid_count: 0, total: dcm.length })
    } finally {
      setIsValidating(false)
    }

    // 업로드 시 가중치 자동 연결
    if (weightsStage === "idle") {
      fetch("http://localhost:8000/health")
        .then(res => {
          if (res.ok) {
            setWeightsStage("loading2d")
            setTimeout(() => {
              setWeightsStage("loading25d")
              setTimeout(() => setWeightsStage("active"), 500)
            }, 500)
          }
        })
        .catch(() => {})
    }
  }, [weightsStage])

  const handleResetUpload = useCallback(() => {
    setDcmFiles([])
    setFolderName("")
    setUploadProgress(0)
    setIsUploading(false)
    setAnalysisProgress(0)
    setAnalysisStage("순수 폐 마스크 생성 중...")
    setStageLogs([])
    setSlicePatchesMap(new Map())
    setValidateResult(null)
    setIsValidating(false)
    setOpinion("")
    setPatientName("")
    setPatientBirthdate("")
    setPatientGender("")
    if (fileInputRef.current) fileInputRef.current.value = ""
  }, [fileInputRef])
  // ----- Dual-Stage AI Analysis (0→50% phase1, 50→100% phase2) then route -----
  const handleRunAnalysis = useCallback(async () => {
    if (dcmFiles.length === 0) return

    setIsRunningAnalysis(true)
    setAnalysisProgress(0)
    setAnalysisStage("순수 폐 마스크 생성 중...")
    setStageLogs([])
    setSlicePatchesMap(new Map())
    setTrackSummary(null)
    setVolumeSliceCount(0)
    const stageStartRef = { time: Date.now(), label: "순수 폐 마스크 생성" }

    try {
      const newMap = new Map<number, Array<{position: {y0:number, x0:number, y1:number, x1:number}, score:number, padim_score?: number}>>()

      const formData = new FormData()
      dcmFiles.forEach(file => formData.append("files", file))

      setAnalysisStage("DICOM 전송 중...")
      setAnalysisProgress(3)

      const res = await fetch("http://localhost:8000/analyze_volume", {
        method: "POST",
        body: formData,
      })
      if (!res.ok) throw new Error(`서버 오류: ${res.status}`)
      if (!res.body) throw new Error("스트림 응답 없음")

      // SSE 스트리밍 읽기
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      let streamDone = false
      let analyzedTotal = 0   // 분석 볼륨 슬라이스 수(전처리 crop 반영, #10)

      while (!streamDone) {
        let done: boolean, value: Uint8Array | undefined
        try {
          ;({ done, value } = await reader.read())
        } catch {
          if (streamDone) break
          throw new Error("스트림 읽기 실패")
        }
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() ?? ""

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue
          const msg = JSON.parse(line.slice(6))
          if (msg.done) { streamDone = true; break }
          if (msg.error) throw new Error(msg.error)
          // 분석 볼륨 슬라이스 수 확정 → 슬라이스 리스트를 분석 볼륨 기준으로 재생성 (#10)
          // 전처리(폐 z-range crop)로 업로드 수와 분석 수가 다를 수 있으므로 백엔드 total을 신뢰
          if (msg.total && msg.total !== analyzedTotal) {
            analyzedTotal = msg.total
            setVolumeSliceCount(msg.total)
            setSlices(generateSlices(msg.total))
          }
          if (msg.tracks) { setTrackSummary(msg.tracks); continue }   // 트랙 집계(top-K/절약률 정본) 캡처 (#1)
          if (!msg.result) continue   // result 없는 이벤트는 스킵 (크래시 방지)

          const pct = Math.round((msg.z / msg.total) * 85) + 5
          setAnalysisProgress(pct)
          setAnalysisStage(`슬라이스 분석 중... (${msg.z + 1}/${msg.total})`)
          if (msg.result.anomaly_patches?.length > 0) {
            newMap.set(msg.z, msg.result.anomaly_patches)
            setSlicePatchesMap(new Map(newMap))
          }
        }
      }

      // 완료
      const totalTime = (Date.now() - stageStartRef.time) / 1000
      setStageLogs([
        { label: "TotalSegmentator / HU 폴백", seconds: parseFloat((totalTime * 0.1).toFixed(1)) },
        { label: "PaDiM 이상 탐지", seconds: parseFloat((totalTime * 0.5).toFixed(1)) },
        { label: "RD4AD E2 재검증", seconds: parseFloat((totalTime * 0.3).toFixed(1)) },
        { label: "NSCLC 분류", seconds: parseFloat((totalTime * 0.1).toFixed(1)) },
      ])

      setAnalysisProgress(100)
      setSlicePatchesMap(new Map(newMap))
      setAnalysisStage("✅ 분석 완료")

      // 실제 분석 결과로 슬라이스 점수 업데이트
      setSlices(prev => prev.map(s => {
        const patches = newMap.get(s.id - 1)
        if (!patches || patches.length === 0) return { ...s, score: 0 }
        const maxScore = Math.max(...patches.map(p => p.score))
        return { ...s, score: Math.round(maxScore * 100) / 100 }
      }))

      setTimeout(() => {
        setIsRunningAnalysis(false)
        setView("analysis")
      }, 400)

    } catch (err) {
      console.error("[LUNAR] 백엔드 연결 실패:", err)
      setIsRunningAnalysis(false)
    }
  }, [dcmFiles])

  // ----- High-Risk Volume Windowing -----
  const filteredSlices = useMemo(() => {
    if (sliceTab === "all") return slices
    if (selectedRiskRegion) {
      return slices
        .filter(s => s.id >= selectedRiskRegion.min && s.id <= selectedRiskRegion.max)
        .sort((a, b) => b.score - a.score)
    }
    const peaks = slices.filter(s => s.score > 0.35).map(s => s.id)
    const windowSet = new Set<number>()
    peaks.forEach(id => {
      for (let j = id - 5; j <= id + 5; j++) {
        if (j >= 1) windowSet.add(j)
      }
    })
    return slices.filter(s => windowSet.has(s.id)).sort((a, b) => b.score - a.score)
  }, [slices, sliceTab, selectedRiskRegion])

  // ----- Connected region helper -----
  const computeConnectedRegion = useCallback((sliceId: number): {min: number, max: number} => {
    const totalFiles = volumeSliceCount || slices.length || dcmFiles.length
    const peaks = slices.filter(s => s.score > 0.35).map(s => s.id)
    if (peaks.length === 0) return {min: sliceId, max: sliceId}
    const windowSet = new Set<number>()
    peaks.forEach(id => {
      for (let j = id - 5; j <= id + 5; j++) {
        if (j >= 1 && j <= totalFiles) windowSet.add(j)
      }
    })
    if (!windowSet.has(sliceId)) return {min: sliceId, max: sliceId}
    // flood fill connected region
    const visited = new Set<number>()
    const queue = [sliceId]
    while (queue.length > 0) {
      const curr = queue.shift()!
      if (visited.has(curr)) continue
      visited.add(curr)
      if (windowSet.has(curr - 1)) queue.push(curr - 1)
      if (windowSet.has(curr + 1)) queue.push(curr + 1)
    }
    const ids = Array.from(visited).sort((a, b) => a - b)
    return {min: ids[0], max: ids[ids.length - 1]}
  }, [slices, dcmFiles.length, volumeSliceCount])

  // ----- Patient filtering -----
  const filteredPatients = useMemo(() => {
    let result = patients
    if (searchQuery) result = result.filter(p =>
      p.name.includes(searchQuery) || p.id.includes(searchQuery)
    )
    if (dateFilter !== "all") {
      const month = parseInt(dateFilter)
      result = result.filter(p => new Date(p.date).getMonth() + 1 === month)
    }
    return result
  }, [patients, searchQuery, dateFilter])

  const currentStep = Math.min(Math.floor(uploadProgress / 20), 4)
  const currentSlice = slices.find(s => s.id === activeSlice)

  const handleFetchCard = useCallback(async (sliceIndex: number) => {
    const patches = slicePatchesMap.get(sliceIndex)
    if (!patches || patches.length === 0) {
      setCardData(null)
      return
    }
    setCardLoading(true)
    try {
      // CT crop은 백엔드가 _cached_hu_volume에서 lung window 적용 후 직접 생성
      const res = await fetch("http://localhost:8000/card/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          slice_index: sliceIndex,
          total_slices: volumeSliceCount || dcmFiles.length || 100,
          anomaly_patches: patches,
        }),
      })
      const data = await res.json()
      setCardData(data)
    } catch {
      setCardData(null)
    } finally {
      setCardLoading(false)
    }
  }, [slicePatchesMap, dcmFiles])

  // ----- Slice selection handlers -----
  // 슬라이스 목록 클릭: 고위험 탭에서만 XAI(카드) 연산 + connected region 설정.
  // All 탭은 전처리된 전체 슬라이스 브라우징 전용 → XAI 안 함(사용자 설계).
  const handleSliceSelect = useCallback((sliceId: number) => {
    setActiveSlice(sliceId)
    if (sliceTab === "high") {
      handleFetchCard(sliceId - 1)
      setSelectedRiskRegion(computeConnectedRegion(sliceId))
    } else {
      setCardData(null)
    }
  }, [handleFetchCard, sliceTab, computeConnectedRegion])

  // 슬라이더/드래그 이동: XAI 연산 없이 슬라이스만 변경
  const handleSliceNavigate = useCallback((sliceId: number) => {
    setActiveSlice(sliceId)
  }, [])

  const handleOpenInAnalysis = useCallback(async (patient: Patient) => {
    setSelectedPatient(null)
    setActiveSlice(1)
    setSliceTab("all")
    setSelectedRiskRegion(null)
    setCardData(null)
    setTrackSummary(null)
    setVolumeSliceCount(0)

    // DICOM 파일 로드 우선순위: 메모리 → IndexedDB → B2
    let resolvedFiles: File[] = patient.dcmFiles && patient.dcmFiles.length > 0 ? patient.dcmFiles : []
    if (resolvedFiles.length === 0) {
      const idbFiles = await loadDicomsIDB(patient.id)
      if (idbFiles && idbFiles.length > 0) resolvedFiles = idbFiles
    }
    setDcmFiles(resolvedFiles)

    if (patient.savedPatchesMap) {
      setSlicePatchesMap(patient.savedPatchesMap)
      setVolumeSliceCount(patient.volumeSliceCount ?? 0)
      setTrackSummary(patient.trackSummary ?? null)
      setView("analysis")
      return
    }

    // B2 기록 로드: 자동분석(slices) + 수동저장(highRiskPatches) 둘 다 복원 (#B)
    try {
      const r = await fetch(`http://localhost:8000/b2/record-slices/${patient.id}`)
      const data = await r.json()
      const newMap = new Map<number, any[]>()
      if (data.slices && data.slices.length > 0) {
        // 자동분석 스키마: slices[].anomaly_patches (키 0-based, #2)
        data.slices.forEach((s: any, idx: number) => {
          if (s.anomaly_patches && s.anomaly_patches.length > 0) newMap.set(idx, s.anomaly_patches)
        })
        setVolumeSliceCount(data.slices.length)
        setTrackSummary(data.tracks || null)
        setSlices(data.slices.map((_: any, i: number) => ({
          id: i + 1, score: data.slices[i].score ?? 0,
          risk: data.slices[i].risk ?? "Low", file: null,
        })))
      } else if (data.highRiskPatches && data.highRiskPatches.length > 0) {
        // 수동저장 스키마: highRiskPatches[{sliceIndex(0-based), score, position}] → slicePatchesMap 복원
        let maxZ = 0
        data.highRiskPatches.forEach((p: any) => {
          const z = p.sliceIndex ?? 0
          if (z > maxZ) maxZ = z
          if (!newMap.has(z)) newMap.set(z, [])
          newMap.get(z)!.push({ position: p.position, score: p.score })
        })
        const total = data.totalSlices && data.totalSlices > maxZ ? data.totalSlices : maxZ + 1
        setVolumeSliceCount(total)
        setTrackSummary(null)   // 수동 기록엔 tracks 없음 → 프론트 trackMap 폴백
        setSlices(Array.from({ length: total }, (_, i) => {
          const ps = newMap.get(i)
          return { id: i + 1, score: ps ? Math.max(...ps.map((p: any) => p.score)) : 0, risk: "Low", file: null }
        }))
      }
      setSlicePatchesMap(newMap)
      // 백엔드 _cached_hu_volume 채우기 (#9): 저장본 카드/CT가 올바른 환자 볼륨을 쓰도록
      try {
        await fetch(`http://localhost:8000/load-volume-cache/${patient.id}`)
      } catch (ce) {
        console.warn("[LUNAR] load-volume-cache 실패(카드 crop 제한될 수 있음)", ce)
      }
    } catch (e) {
      console.error("[LUNAR] B2 기록 로드 실패", e)
      setSlicePatchesMap(new Map())
    }
    setView("analysis")
  }, [])

  const handleDeletePatient = useCallback(async (patientId: string) => {
    const confirmed = window.confirm(`환자 ${patientId} 기록을 삭제하시겠습니까?\n삭제하면 복구할 수 없습니다.`)
    if (!confirmed) return

    setPatients(prev => prev.filter(p => p.id !== patientId))
    showToast(`환자 ${patientId} 목록에서 삭제되었습니다.`)
    try {
      await fetch(`http://localhost:8000/b2/delete-record/${patientId}`, {
        method: "DELETE",
      })
    } catch {
      // B2 삭제 실패해도 로컬 삭제는 유지
    }
  }, [showToast])

  const handleLoadFromB2 = useCallback(async () => {
    showToast("B2에서 분석 완료 기록을 불러오는 중...")
    try {
      const res = await fetch("http://localhost:8000/b2/list-records")
      const data = await res.json()
      if (!data.records || data.records.length === 0) {
        showToast("B2에 저장된 기록이 없습니다.")
        return
      }
      const newPatients: Patient[] = data.records.map((r: any) => ({
        id: r.patientId,
        name: r.name || r.patientId,
        birthdate: r.birthdate || "-",
        gender: r.gender || "-",
        date: r.date ? r.date.split("T")[0] : new Date().toISOString().split("T")[0],
        status: r.status || "Completed",
        risk: r.risk || "Low",
        opinion: r.opinion || "",
        autoAnalyzed: r.source === "auto_webhook",
        source: r.source || "manual",
      }))
      setPatients(prev => {
        const existingIds = new Set(prev.map(p => p.id))
        const added = newPatients.filter(p => !existingIds.has(p.id))
        showToast(`${added.length}개 기록 추가, ${newPatients.length - added.length}개 이미 존재`)
        return [...prev, ...added]
      })
    } catch {
      showToast("B2 불러오기 실패 — 백엔드 연결을 확인하세요.")
    }
  }, [showToast])

  // ----- Sidebar weights indicator (collapsed) -----
  const weightsCollapsedIcon = weightsStage === "active"
    ? <Check className="w-4 h-4 text-green-400" />
    : (weightsStage !== "idle"
      ? <Loader2 className="w-4 h-4 text-[#2563eb] animate-spin" />
      : <Globe className="w-4 h-4 text-[#a3a3a3]" />)

  // ----- Sidebar bottom panel (expanded) -----
  const sidebarBottom = (
    <div className={cn("border-t border-[#333] p-3 space-y-3", sidebarCollapsed && "hidden")}>


      {/* --- Backblaze B2 Storage --- */}
      <div className="space-y-2">
        <p className="text-[10px] font-semibold text-[#555] uppercase tracking-wide">Storage</p>
        {b2Connected ? (
          <div className="flex items-start gap-2 px-2.5 py-2.5 bg-green-900/25 border border-green-800/40 text-green-400 rounded-lg">
            <Check className="w-3.5 h-3.5 shrink-0 mt-0.5" />
            <span className="text-[11px] leading-snug font-medium">Backblaze B2 연결됨</span>
          </div>
        ) : (
          <Button
            size="sm"
            onClick={handleConnectRepo}
            className="w-full h-7 text-[11px] bg-[#2a2a2a] hover:bg-[#333] text-[#e5e5e5] border border-[#444]"
          >
            B2 Storage 연결
          </Button>
        )}
      </div>
    </div>
  )

  return (
    <div className="flex h-screen w-screen bg-[#121212] text-[#e5e5e5] overflow-hidden">

      {/* ---- Sidebar ---- */}
      <aside className={cn(
        "flex flex-col bg-[#1a1a1a] border-r border-[#333] transition-all duration-300 shrink-0",
        sidebarCollapsed ? "w-[94px]" : "w-[301px]"
      )}>
        {/* Brand */}
        <div className="p-4 border-b border-[#333] flex items-center gap-2 shrink-0">
          <Activity className="w-7 h-7 text-[#2563eb] shrink-0" />
          {!sidebarCollapsed && (
            <div>
              <h1 className="font-bold text-[#2563eb] text-base leading-tight">{t.brand}</h1>
              <p className="text-[10px] text-[#a3a3a3] leading-tight">{t.subtitle}</p>
            </div>
          )}
        </div>

        {/* Nav */}
        <nav className="flex-1 p-2 space-y-1 min-h-0 overflow-y-auto">
          {([
            { id: "upload" as View, icon: Upload, label: t.upload },
            { id: "analysis" as View, icon: FolderOpen, label: t.analysis },
            { id: "patients" as View, icon: Users, label: t.patients }
          ] as const).map(item => (
            <button
              key={item.id}
              onClick={() => setView(item.id)}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors",
                sidebarCollapsed ? "justify-center" : "text-left",
                view === item.id ? "bg-[#2563eb] text-white" : "hover:bg-[#2a2a2a] text-[#a3a3a3]"
              )}
            >
              <item.icon className="w-4 h-4 shrink-0" />
              {!sidebarCollapsed && <span className="text-sm">{item.label}</span>}
            </button>
          ))}
        </nav>

        {/* Weights / B2 Storage panel */}
        {sidebarCollapsed ? (
          <div className="p-2 border-t border-[#333]">
            <button
              className={cn(
                "w-full flex items-center justify-center p-2 rounded-lg transition-colors",
                weightsStage === "active" ? "text-green-400" : "text-[#a3a3a3] hover:bg-[#2a2a2a]"
              )}
              title={weightsStage === "active" ? t.weightsActive : "AI Model Weights"}
            >
              {weightsCollapsedIcon}
            </button>
          </div>
        ) : sidebarBottom}

        {/* Collapse toggle */}
        <button
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          className="p-2 border-t border-[#333] hover:bg-[#2a2a2a] transition-colors flex items-center justify-center shrink-0"
        >
          {sidebarCollapsed
            ? <ChevronRight className="w-4 h-4 text-[#a3a3a3]" />
            : <ChevronLeft className="w-4 h-4 text-[#a3a3a3]" />}
        </button>
      </aside>

      {/* ---- Main ---- */}
      <main className="flex-1 overflow-hidden min-w-0">
        {view === "upload" && (
          <UploadView
            t={t}
            uploadProgress={uploadProgress}
            isUploading={isUploading}
            isValidating={isValidating}
            validateResult={validateResult}
            analysisProgress={analysisProgress}
            isRunningAnalysis={isRunningAnalysis}
            analysisStage={analysisStage}
            stageLogs={stageLogs}
            currentStep={currentStep}
            folderName={folderName}
            fileInputRef={fileInputRef}
            onFolderSelect={handleFolderSelect}
            onRunAnalysis={handleRunAnalysis}
            onResetUpload={handleResetUpload}
            patientName={patientName}
            setPatientName={setPatientName}
            patientBirthdate={patientBirthdate}
            setPatientBirthdate={setPatientBirthdate}
            patientGender={patientGender}
            setPatientGender={setPatientGender}
          />
        )}
        {view === "analysis" && (
          <AnalysisView
            t={t}
            filteredSlices={filteredSlices}
            activeSlice={activeSlice}
            onSliceSelect={handleSliceSelect}
            onSliceNavigate={handleSliceNavigate}
            handleFetchCard={handleFetchCard}
            sliceTab={sliceTab}
            setSliceTab={(tab) => {
              setSliceTab(tab)
              if (tab === "all") setSelectedRiskRegion(null)
            }}
            selectedRiskRegion={selectedRiskRegion}
            onClearRegion={() => setSelectedRiskRegion(null)}
            showHeatmap={showHeatmap}
            setShowHeatmap={setShowHeatmap}
            showBoxes={showBoxes}
            setShowBoxes={setShowBoxes}
            opinion={opinion}
            setOpinion={setOpinion}
            currentSlice={currentSlice}
            dcmFiles={dcmFiles}
            volumeSliceCount={volumeSliceCount}
            trackSummary={trackSummary}
            slicePatchesMap={slicePatchesMap}
            patientName={patientName}
            setPatientName={setPatientName}
            patientBirthdate={patientBirthdate}
            setPatientBirthdate={setPatientBirthdate}
            patientGender={patientGender}
            setPatientGender={setPatientGender}
            cardData={cardData}
            cardLoading={cardLoading}
            analysisProgress={analysisProgress}
            onSaveReport={async () => {
              // 환자 정보가 없으면 기본값 사용
              const savedName = patientName.trim() || "이름 미입력"
              const savedBirthdate = patientBirthdate.trim() || "-"
              const savedGender = patientGender.trim() || "-"
              const now = new Date()
              const newPatientId = `PT-${String(patients.length + 1).padStart(3, "0")}`

              // 환자 목록에 추가
              const newPatient = {
                id: newPatientId,
                name: savedName,
                birthdate: savedBirthdate,
                gender: savedGender,
                date: now.toISOString().split("T")[0],
                status: "Completed",
                risk: (currentSlice?.score ?? 0) > 0.45 ? "Critical" : (currentSlice?.score ?? 0) > 0.35 ? "High" : (currentSlice?.score ?? 0) > 0.20 ? "Medium" : "Low",
                opinion: opinion,
                dcmFiles: dcmFiles,
                savedPatchesMap: slicePatchesMap,
                volumeSliceCount: volumeSliceCount,
                trackSummary: trackSummary,
              }
              setPatients(prev => [...prev, newPatient])

              // Backblaze B2 저장
              try {
                const highRiskPatches = Array.from(slicePatchesMap.entries())
                  .flatMap(([sliceIdx, patches]) =>
                    patches
                      .filter(p => p.score > 0.20)
                      .map(p => ({
                        sliceIndex: sliceIdx,
                        score: p.score,
                        position: p.position,
                        // image 필드 제외 (용량 너무 커서 JSON 오류 발생)
                      }))
                  )
                  .sort((a, b) => b.score - a.score)
                  .slice(0, 50)

                const content = {
                  patientId: newPatientId,
                  name: savedName,
                  birthdate: savedBirthdate,
                  gender: savedGender,
                  date: now.toISOString(),
                  opinion: opinion,
                  score: currentSlice?.score ?? 0,
                  risk: newPatient.risk,
                  folderName: folderName,
                  totalSlices: volumeSliceCount || dcmFiles.length,
                  highRiskPatches: highRiskPatches,
                }

                // JSON + MD 소견 저장
                await fetch("http://localhost:8000/b2/save-record", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(content),
                })

                // 방금 분석한 볼륨을 patient_id로도 캐싱 → 재오픈 시 카드 CT crop 복원 (#B)
                await fetch(`http://localhost:8000/save-volume-cache/${newPatientId}`, {
                  method: "POST",
                }).catch(() => {})

                // MD 보고서도 B2에 저장
                const mdContent = generateReportMd({
                  ...content,
                  patientId: newPatientId,
                  score: currentSlice?.score ?? 0,
                })
                await fetch("http://localhost:8000/b2/save-report-md", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ patientId: newPatientId, markdown: mdContent }),
                }).catch(() => {})

                // 고위험 DICOM 파일 저장 (상위 20개) + 전체 IndexedDB 저장
                const highRiskIndices = Array.from(slicePatchesMap.entries())
                  .filter(([, patches]) => Math.max(...patches.map(p => p.score)) > 0.20)
                  .sort((a, b) => Math.max(...b[1].map(p => p.score)) - Math.max(...a[1].map(p => p.score)))
                  .slice(0, 20)
                  .map(([idx]) => idx)

                for (const idx of highRiskIndices) {
                  const file = dcmFiles[idx]
                  if (!file) continue
                  const formData = new FormData()
                  formData.append("file", file)
                  formData.append("patient_id", newPatientId)
                  await fetch("http://localhost:8000/b2/save-dicom", {
                    method: "POST",
                    body: formData,
                  })
                }

                // IndexedDB에 전체 DICOM 저장 (세션 종료 후에도 재열람 가능)
                saveDicomsIDB(newPatientId, dcmFiles)

                showToast(`${savedName} 환자 정보가 B2에 저장되었습니다.`)
              } catch (e) {
                console.error("B2 저장 실패:", e)
                showToast(`${savedName} 환자 정보가 로컬에 저장되었습니다.`)
              }
              setPatientName("")
              setPatientBirthdate("")
              setPatientGender("")
              setOpinion("")
            }}
          />
        )}
        {view === "patients" && (
          <PatientsView
            t={t}
            filteredPatients={filteredPatients}
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            dateFilter={dateFilter}
            setDateFilter={setDateFilter}
            setSelectedPatient={setSelectedPatient}
            onDeletePatient={handleDeletePatient}
            onLoadFromB2={handleLoadFromB2}
          />
        )}
      </main>

      {/* ---- Language Selector ---- */}
      <div className={cn(
        "fixed z-50",
        view === "analysis" ? "bottom-4 right-[650px]" : "bottom-4 right-4"
      )}>
        <div className="relative">
          <button
            onClick={() => setShowLangDropdown(!showLangDropdown)}
            className="flex items-center gap-2 px-3 py-2 bg-[#1e1e1e] border border-[#333] rounded-lg hover:bg-[#2a2a2a] transition-colors text-sm"
          >
            <Globe className="w-4 h-4 text-[#a3a3a3]" />
            <span>{lang === "en" ? "English" : "한국어"}</span>
          </button>
          {showLangDropdown && (
            <div className="absolute bottom-full right-0 mb-2 bg-[#1e1e1e] border border-[#333] rounded-lg overflow-hidden shadow-2xl">
              {(["en", "ko"] as const).map(l => (
                <button
                  key={l}
                  onClick={() => { setLang(l); setShowLangDropdown(false) }}
                  className={cn(
                    "w-full px-5 py-2.5 text-sm text-left transition-colors whitespace-nowrap",
                    lang === l ? "bg-[#2563eb] text-white" : "hover:bg-[#2a2a2a] text-[#e5e5e5]"
                  )}
                >
                  {l === "en" ? "English" : "한국어"}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ---- Toast ---- */}
      {toast && (
        <div className="fixed bottom-16 right-4 z-50 max-w-sm bg-[#1e1e1e] border border-[#333] rounded-lg shadow-2xl p-4 flex items-start gap-3 animate-in slide-in-from-bottom-4">
          <Check className="w-4 h-4 text-green-400 shrink-0 mt-0.5" />
          <p className="text-xs text-[#e5e5e5] leading-relaxed font-mono">{toast}</p>
          <button onClick={() => setToast(null)} className="ml-auto shrink-0 text-[#a3a3a3] hover:text-white">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* ---- Patient Detail Modal ---- */}
      <Dialog open={!!selectedPatient} onOpenChange={() => setSelectedPatient(null)}>
        <DialogContent className="bg-[#1e1e1e] border-[#333] text-[#e5e5e5] max-h-[85vh] flex flex-col max-w-2xl p-0 gap-0">
          <DialogHeader className="px-6 pt-5 pb-4 border-b border-[#333] shrink-0">
            <DialogTitle className="flex items-center justify-between">
              <div>
                <span className="text-base font-semibold">{t.patientDetails}</span>
                <span className="ml-3 text-sm text-[#a3a3a3] font-normal">{selectedPatient?.id}</span>
              </div>
              <Badge className={cn(
                "text-white text-xs",
                selectedPatient?.risk === "Critical" ? "bg-purple-600" :
                selectedPatient?.risk === "High" ? "bg-red-600" :
                selectedPatient?.risk === "Medium" ? "bg-yellow-600" : "bg-green-600"
              )}>
                {selectedPatient ? tRisk(t, selectedPatient.risk as string) : ""}
              </Badge>
            </DialogTitle>
          </DialogHeader>

          <div className="flex-1 overflow-y-auto px-6 py-5 space-y-4 min-h-0">
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: t.name, value: selectedPatient?.name },
                { label: t.date, value: selectedPatient?.date },
                { label: t.aiInspectionStatus, value: selectedPatient ? tStatus(t, selectedPatient.status as string) : "" }
              ].map(({ label, value }) => (
                <div key={label} className="bg-[#2a2a2a] p-3 rounded-lg">
                  <p className="text-[10px] text-[#a3a3a3] mb-1 uppercase tracking-wide">{label}</p>
                  <p className="text-sm font-medium">{value}</p>
                </div>
              ))}
            </div>

            {selectedPatient?.opinion && (
              <div className="bg-[#2a2a2a] p-4 rounded-lg">
                <p className="text-[10px] text-[#a3a3a3] uppercase tracking-wide mb-2">{t.medicalOpinion}</p>
                <p className="text-sm text-[#e5e5e5] leading-relaxed">{selectedPatient.opinion}</p>
              </div>
            )}

            <Button
              onClick={() => selectedPatient && handleOpenInAnalysis(selectedPatient)}
              className="w-full bg-[#2563eb] hover:bg-[#1d4ed8] text-white h-11 text-sm font-medium"
            >
              <ExternalLink className="w-4 h-4 mr-2" />
              {t.openInAnalysis}
            </Button>

            <Button
              variant="outline"
              className="w-full border-[#333] bg-transparent text-[#a3a3a3] hover:bg-[#2a2a2a] hover:text-[#e5e5e5] h-9 text-xs"
              onClick={async () => {
                if (!selectedPatient) return
                try {
                  // B2에서 기록 가져와 MD 생성
                  const res = await fetch(`http://localhost:8000/b2/download-record/${selectedPatient.id}`)
                  const mdText = res.ok
                    ? generateReportMd(await res.json())
                    : generateReportMd({
                        patientId: selectedPatient.id,
                        name: selectedPatient.name,
                        birthdate: selectedPatient.birthdate,
                        gender: selectedPatient.gender,
                        date: selectedPatient.date,
                        opinion: selectedPatient.opinion ?? "",
                        risk: selectedPatient.risk,
                      })
                  const blob = new Blob([mdText], { type: "text/markdown; charset=utf-8" })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement("a")
                  a.href = url
                  a.download = `${selectedPatient.id}_report.md`
                  a.click()
                  URL.revokeObjectURL(url)
                } catch {
                  alert("보고서 다운로드에 실패했습니다.")
                }
              }}
            >
              <Download className="w-3.5 h-3.5 mr-2" />{t.downloadReport}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Upload View
// ---------------------------------------------------------------------------
function UploadView({ t, uploadProgress, isUploading, isValidating, validateResult, analysisProgress, isRunningAnalysis, analysisStage, stageLogs, currentStep, folderName, fileInputRef, onFolderSelect, onRunAnalysis, onResetUpload, patientName, setPatientName, patientBirthdate, setPatientBirthdate, patientGender, setPatientGender }: {
  t: T
  uploadProgress: number
  isUploading: boolean
  isValidating: boolean
  validateResult: {valid: boolean, errors: any[], warnings: string[], valid_count: number, total: number} | null
  analysisProgress: number
  isRunningAnalysis: boolean
  analysisStage: string
  stageLogs: Array<{label: string, seconds: number}>
  currentStep: number
  folderName: string
  fileInputRef: React.RefObject<HTMLInputElement | null>
  onFolderSelect: (e: React.ChangeEvent<HTMLInputElement>) => void
  onRunAnalysis: () => void
  onResetUpload: () => void
  patientName: string
  setPatientName: (s: string) => void
  patientBirthdate: string
  setPatientBirthdate: (s: string) => void
  patientGender: string
  setPatientGender: (s: string) => void
}) {
  const analysisPhaseLabel = analysisStage || (analysisProgress < 50 ? t.phase1Label : t.phase2Label)

  return (
    <div className="h-full flex items-center justify-center p-8 bg-[#121212] overflow-hidden">
      <div className="w-full max-w-3xl space-y-6">
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          onChange={onFolderSelect}
          {...{ webkitdirectory: "", directory: "", multiple: true } as React.InputHTMLAttributes<HTMLInputElement>}
        />

        {!isUploading ? (
          <button
            onClick={() => fileInputRef.current?.click()}
            className="w-full p-24 border-2 border-dashed border-[#333] rounded-2xl hover:border-[#2563eb] hover:bg-[#2563eb]/5 transition-all group"
          >
            <div className="flex flex-col items-center gap-6">
              <div className="w-24 h-24 rounded-2xl bg-[#2563eb]/10 flex items-center justify-center group-hover:bg-[#2563eb]/20 transition-colors">
                <Upload className="w-12 h-12 text-[#2563eb]" />
              </div>
              <div className="text-center">
                <p className="font-semibold text-lg text-[#e5e5e5]">{t.selectFolder}</p>
                <p className="text-sm text-[#a3a3a3] mt-1.5">{t.dropHint}</p>
              </div>
            </div>
          </button>
        ) : (
          <div className="bg-[#1e1e1e] border border-[#333] rounded-2xl p-6 space-y-5">
            {/* Preprocessing progress */}
            <div>
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-[#a3a3a3]">{folderName || "CT_Folder"}</span>
                <span className="text-[#2563eb] font-medium tabular-nums">{Math.round(uploadProgress)}%</span>
              </div>
              <Progress value={uploadProgress} className="h-1.5" />
            </div>

            {/* 검증 결과 */}
            {isValidating && (
              <div className="flex items-center gap-2 px-3 py-2 bg-[#1a1a1a] rounded-lg border border-[#333]">
                <Loader2 className="w-3.5 h-3.5 text-[#2563eb] animate-spin shrink-0" />
                <span className="text-[11px] text-[#a3a3a3]">DICOM 파일 검증 중...</span>
              </div>
            )}
            {validateResult && !isValidating && (
              <div className={cn(
                "px-3 py-2.5 rounded-lg border space-y-1.5",
                validateResult.valid
                  ? "bg-green-900/20 border-green-800/40"
                  : "bg-red-900/20 border-red-800/40"
              )}>
                <div className="flex items-center gap-2">
                  {validateResult.valid
                    ? <Check className="w-3.5 h-3.5 text-green-400 shrink-0" />
                    : <X className="w-3.5 h-3.5 text-red-400 shrink-0" />}
                  <span className={cn("text-[11px] font-medium", validateResult.valid ? "text-green-400" : "text-red-400")}>
                    {validateResult.valid
                      ? `검증 완료 — ${validateResult.valid_count}개 정상`
                      : `검증 실패 — ${validateResult.errors.length}개 오류`}
                  </span>
                </div>
                {validateResult.warnings.map((w, i) => (
                  <div key={i} className="flex items-start gap-1.5">
                    <span className="text-[10px] text-yellow-400 mt-0.5">⚠</span>
                    <span className="text-[10px] text-yellow-400">{w}</span>
                  </div>
                ))}
                {validateResult.errors.slice(0, 3).map((e, i) => (
                  <div key={i} className="text-[10px] text-red-400 truncate">
                    {e.filename}: {e.message}
                  </div>
                ))}
              </div>
            )}

            {/* Dual-Stage AI Analysis block — shown after preprocessing */}
            {uploadProgress >= 100 && (
              <div className="space-y-3 pt-1">
                {isRunningAnalysis && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="text-[#a3a3a3]">{analysisPhaseLabel}</span>
                      <span className="text-[#2563eb] font-medium tabular-nums">{Math.round(analysisProgress)}%</span>
                    </div>
                    <Progress value={analysisProgress} className="h-1.5" />
                    {/* Phase indicators */}
                    <div className="space-y-1.5 pt-1">
                      {[
                        { label: "전처리: 폐 마스크 생성",      done: 15,  active: 0  },
                        { label: "전처리: 장기 제외",           done: 30,  active: 15 },
                        { label: "1단계: PaDiM 이상 탐지",     done: 55,  active: 30 },
                        { label: "2단계: RD4AD 재검증",        done: 85,  active: 55 },
                        { label: "결과 정리",                   done: 100, active: 85 },
                      ].map((step, i) => {
                        const isDone   = analysisProgress >= step.done
                        const isActive = analysisProgress >= step.active && analysisProgress < step.done
                        return (
                          <div key={i} className={cn(
                            "flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-[11px]",
                            isDone   ? "bg-green-900/20 text-green-400" :
                            isActive ? "bg-[#2563eb]/15 text-[#2563eb]" : "bg-[#2a2a2a] text-[#555]"
                          )}>
                            {isDone
                              ? <Check className="w-3 h-3 shrink-0" />
                              : isActive
                                ? <Loader2 className="w-3 h-3 shrink-0 animate-spin" />
                                : <div className="w-3 h-3 rounded-full border border-[#444] shrink-0" />}
                            <span>{step.label}</span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* 단계별 소요 시간 — 분석 완료 후 계속 표시 */}
                {stageLogs.length > 0 && (
                  <div className="space-y-1 pt-1">
                    <p className="text-[10px] font-semibold text-[#555] uppercase tracking-wide">단계별 소요 시간</p>
                    {stageLogs.map((log, i) => (
                      <div key={i} className="flex items-center justify-between px-2.5 py-1.5 bg-[#1a1a1a] rounded-lg">
                        <span className="text-[11px] text-[#a3a3a3] flex items-center gap-1.5">
                          <Check className="w-3 h-3 text-green-400 shrink-0" />
                          {log.label}
                        </span>
                        <span className="text-[11px] text-[#2563eb] font-mono tabular-nums">{log.seconds}s</span>
                      </div>
                    ))}
                  </div>
                )}

                {!isRunningAnalysis && (
                  <div className="space-y-2">
                    <div className="space-y-2 p-3 bg-[#1a1a1a] rounded-xl border border-[#333]">
                      <p className="text-[10px] font-semibold text-[#555] uppercase tracking-wide">환자 정보</p>
                      <input
                        value={patientName}
                        onChange={e => setPatientName(e.target.value)}
                        placeholder="환자 이름"
                        className="w-full h-8 text-xs bg-[#2a2a2a] border border-[#333] text-[#e5e5e5] placeholder:text-[#555] rounded-md px-3"
                      />
                      <input
                        value={patientBirthdate}
                        onChange={e => setPatientBirthdate(e.target.value)}
                        placeholder="생년월일"
                        type="date"
                        className="w-full h-8 text-xs bg-[#2a2a2a] border border-[#333] text-[#e5e5e5] rounded-md px-3"
                      />
                      <select
                        value={patientGender}
                        onChange={e => setPatientGender(e.target.value)}
                        className="w-full h-8 text-xs bg-[#2a2a2a] border border-[#333] text-[#e5e5e5] rounded-md px-2"
                      >
                        <option value="">성별 선택</option>
                        <option value="남성">남성</option>
                        <option value="여성">여성</option>
                      </select>
                    </div>
                    <Button
                      onClick={onRunAnalysis}
                      className="w-full bg-[#2563eb] hover:bg-[#1d4ed8] text-white h-10 font-medium"
                    >
                      <Activity className="w-4 h-4 mr-2" />
                      {t.runAnalysis}
                    </Button>
                    <Button
                      onClick={onResetUpload}
                      variant="outline"
                      className="w-full border-[#333] bg-transparent text-[#a3a3a3] hover:bg-[#2a2a2a] hover:text-[#e5e5e5] h-9 text-xs"
                    >
                      <Upload className="w-3.5 h-3.5 mr-2" />
                      새 환자 업로드
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// DICOM Canvas Renderer
// Parses raw .dcm binary, extracts pixel data, applies Lung Window W=1500 L=-600,
// draws grayscale ImageData, then renders heatmap / bounding-box overlays.
// ---------------------------------------------------------------------------

/** Locate a DICOM tag in a DataView (little-endian explicit VR). Returns byte offset of value, or -1. */
function findDicomTag(view: DataView, group: number, element: number): number {
  let i = 132 // skip 128-byte preamble + "DICM"
  while (i + 8 < view.byteLength) {
    const g = view.getUint16(i, true)
    const e = view.getUint16(i + 2, true)

    // Explicit VR 판별: 4~5번째 바이트가 대문자 알파벳 2개인지 확인
    const vrByte0 = view.getUint8(i + 4)
    const vrByte1 = view.getUint8(i + 5)
    const isExplicitVR = vrByte0 >= 65 && vrByte0 <= 90 && vrByte1 >= 65 && vrByte1 <= 90
    const vr = String.fromCharCode(vrByte0, vrByte1)

    let length: number
    let valueOffset: number

    if (isExplicitVR) {
      if (["OB","OW","OF","SQ","UC","UN","UR","UT"].includes(vr)) {
        // 4바이트 길이 (앞 2바이트는 예약)
        length = view.getUint32(i + 8, true)
        valueOffset = i + 12
      } else {
        // 2바이트 길이
        length = view.getUint16(i + 6, true)
        valueOffset = i + 8
      }
    } else {
      // Implicit VR: VR 없이 바로 4바이트 길이
      length = view.getUint32(i + 4, true)
      valueOffset = i + 8
    }

    if (g === group && e === element) return valueOffset

    // 0xFFFFFFFF는 undefined length (시퀀스 등) → 8바이트만 건너뛰고 계속
    if (length === 0xFFFFFFFF) {
      i = valueOffset
      continue
    }
    // 비정상적으로 큰 length → 4바이트씩 전진하며 태그 재탐색
    if (length < 0 || length > view.byteLength) {
      i += 2
      continue
    }
    i = valueOffset + length
  }
  console.log(`[DICOM] 태그 (${group.toString(16)},${element.toString(16)}) 못 찾음. 마지막 i=${i}, 파일크기=${view.byteLength}`)
  return -1
}

function readDicomUint16(view: DataView, group: number, element: number, def = 0): number {
  const off = findDicomTag(view, group, element)
  return off >= 0 ? view.getUint16(off, true) : def
}
function readDicomInt16(view: DataView, group: number, element: number, def = 0): number {
  const off = findDicomTag(view, group, element)
  return off >= 0 ? view.getInt16(off, true) : def
}
function readDicomFloat(view: DataView, group: number, element: number, def = 1): number {
  const off = findDicomTag(view, group, element)
  if (off < 0) return def
  const bytes: number[] = []
  for (let b = 0; b < Math.min(20, view.byteLength - off); b++) bytes.push(view.getUint8(off + b))
  const str = String.fromCharCode(...bytes).split("\\")[0].trim()
  return parseFloat(str) || def
}

function applyWindow(raw: number, rescaleSlope: number, rescaleIntercept: number, ww: number, wl: number): number {
  const hu = raw * rescaleSlope + rescaleIntercept
  const lo = wl - ww / 2
  const hi = wl + ww / 2
  if (hu <= lo) return 0
  if (hu >= hi) return 255
  return Math.round(((hu - lo) / ww) * 255)
}

function applyLungWindow(raw: number, rescaleSlope: number, rescaleIntercept: number): number {
  return applyWindow(raw, rescaleSlope, rescaleIntercept, 1500, -600)
}

function DicomCanvas({
  file, sliceIndex, totalSlices, score,
  showHeatmap, showBoxes, anomalyPatches, onWindowDetected, xaiBoundingBox,
  ctPngUrl,
}: {
  file: File | null
  sliceIndex: number
  totalSlices: number
  score: number
  showHeatmap: boolean
  showBoxes: boolean
  anomalyPatches: Array<{position: {y0:number, x0:number, y1:number, x1:number}, score:number, padim_score?: number}>
  onWindowDetected?: (ww: number, wl: number) => void
  xaiBoundingBox?: {y0:number, x0:number, y1:number, x1:number} | null
  ctPngUrl?: string | null
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const [activeWW, setActiveWW] = useState(1500)
  const [activeWL, setActiveWL] = useState(-600)

  // Parse + render DICOM pixel data whenever file/ctPngUrl changes
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // 백엔드 lung-window PNG가 있으면 우선 사용 (window 보장)
    if (ctPngUrl) {
      const img = new window.Image()
      img.onload = () => {
        canvas.width = img.naturalWidth || 512
        canvas.height = img.naturalHeight || 512
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
        onWindowDetected?.(1500, -600)
        setActiveWW(1500); setActiveWL(-600)
      }
      img.src = ctPngUrl
      return
    }

    if (!file) {
      // Draw anatomical SVG-style placeholder on canvas
      canvas.width = 512
      canvas.height = 512
      ctx.fillStyle = "#050505"
      ctx.fillRect(0, 0, 512, 512)
      // Body ellipse
      ctx.strokeStyle = "#1e1e1e"
      ctx.fillStyle = "#111"
      ctx.beginPath(); ctx.ellipse(256, 256, 225, 210, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke()
      // Left lung
      ctx.fillStyle = "#0a0a0a"; ctx.strokeStyle = "#252525"
      ctx.beginPath()
      ctx.moveTo(133, 166); ctx.bezierCurveTo(97, 166, 82, 205, 85, 256)
      ctx.bezierCurveTo(87, 307, 102, 358, 141, 371)
      ctx.bezierCurveTo(166, 379, 184, 345, 184, 307)
      ctx.lineTo(184, 230); ctx.bezierCurveTo(184, 192, 166, 166, 133, 166)
      ctx.fill(); ctx.stroke()
      // Right lung
      ctx.beginPath()
      ctx.moveTo(379, 166); ctx.bezierCurveTo(415, 166, 430, 205, 427, 256)
      ctx.bezierCurveTo(425, 307, 410, 358, 371, 371)
      ctx.bezierCurveTo(346, 379, 328, 345, 328, 307)
      ctx.lineTo(328, 230); ctx.bezierCurveTo(328, 192, 346, 166, 379, 166)
      ctx.fill(); ctx.stroke()
      // Trachea
      ctx.fillStyle = "#0d0d0d"; ctx.strokeStyle = "#222"
      ctx.beginPath(); ctx.roundRect(242, 77, 28, 77, 12); ctx.fill(); ctx.stroke()
      // Bronchi
      ctx.strokeStyle = "#222"; ctx.lineWidth = 3
      ctx.beginPath(); ctx.moveTo(256, 154); ctx.bezierCurveTo(218, 174, 184, 184, 184, 205); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(256, 154); ctx.bezierCurveTo(294, 174, 328, 184, 328, 205); ctx.stroke()
      ctx.lineWidth = 1
      return
    }

    const reader = new FileReader()
    reader.onload = (ev) => {
      const buf = ev.target?.result as ArrayBuffer
      if (!buf) return
      const view = new DataView(buf)

      // Read DICOM tags
      const rows = readDicomUint16(view, 0x0028, 0x0010, 512)
      const cols = readDicomUint16(view, 0x0028, 0x0011, 512)
      const bitsAllocated = readDicomUint16(view, 0x0028, 0x0100, 16)
      const pixelRepresentation = readDicomUint16(view, 0x0028, 0x0103, 0) // 0=uint,1=int
      const rescaleSlope = readDicomFloat(view, 0x0028, 0x1053, 1)
      const rescaleIntercept = readDicomFloat(view, 0x0028, 0x1052, -1024)
      // 항상 lung window 사용 (DICOM 내장 window 무시)
      const useWL = -600
      const useWW = 1500
      setActiveWW(useWW)
      setActiveWL(useWL)
      onWindowDetected?.(useWW, useWL)

      // Locate pixel data tag (7FE0,0010)
      const pixelOff = findDicomTag(view, 0x7FE0, 0x0010)
      if (pixelOff < 0) {
        console.error("[LUNAR] 픽셀 데이터를 찾지 못했습니다. DICOM 파싱 실패.")
        return
      }

      canvas.width = cols
      canvas.height = rows
      const imgData = ctx.createImageData(cols, rows)

      const totalPx = rows * cols
      for (let px = 0; px < totalPx; px++) {
        let raw: number
        if (bitsAllocated === 16) {
          const byteOff = pixelOff + px * 2
          if (byteOff + 1 >= buf.byteLength) break
          raw = pixelRepresentation === 1
            ? view.getInt16(byteOff, true)
            : view.getUint16(byteOff, true)
        } else {
          const byteOff = pixelOff + px
          if (byteOff >= buf.byteLength) break
          raw = view.getUint8(byteOff)
        }
        const v = applyWindow(raw, rescaleSlope, rescaleIntercept, useWW, useWL)
        const i = px * 4
        imgData.data[i] = v
        imgData.data[i + 1] = v
        imgData.data[i + 2] = v
        imgData.data[i + 3] = 255
      }
      ctx.putImageData(imgData, 0, 0)
    }
    reader.readAsArrayBuffer(file)
  }, [file, ctPngUrl])

  // xaiBoundingBox를 primitive key로 변환 — 렌더마다 새 object reference 생성을 deps에서 안전하게 처리
  const xaiKey = xaiBoundingBox ? `${xaiBoundingBox.y0},${xaiBoundingBox.x0},${xaiBoundingBox.y1},${xaiBoundingBox.x1}` : ""

  // Draw overlays whenever toggles or score change (separate canvas on top)
  useEffect(() => {
    const oc = overlayRef.current
    if (!oc) return
    const ctx = oc.getContext("2d")
    if (!ctx) return
    oc.width = 512
    oc.height = 512
    ctx.clearRect(0, 0, 512, 512)

    // XAI 선택 박스: 슬라이스가 바뀌어도 같은 위치에 유지
    if (xaiBoundingBox) {
      const { y0, x0, y1, x1 } = xaiBoundingBox
      const pad = 4
      ctx.strokeStyle = "#f59e0b"
      ctx.lineWidth = 2.5
      ctx.setLineDash([])
      ctx.globalAlpha = 1
      ctx.strokeRect(x0 - pad, y0 - pad, (x1 - x0) + pad * 2, (y1 - y0) + pad * 2)
      ctx.font = "bold 11px monospace"
      ctx.fillStyle = "#f59e0b"
      ctx.fillText("XAI", x0 - pad, y0 - pad - 4)
      ctx.setLineDash([])
      ctx.globalAlpha = 1
    }

    if (anomalyPatches.length === 0) return

    // score 상위 10개만 표시
    const topPatches = [...anomalyPatches]
      .sort((a, b) => b.score - a.score)
      .slice(0, 10)

    if (showHeatmap) {
      topPatches.forEach(patch => {
        const { y0, x0, y1, x1 } = patch.position
        const cx = (x0 + x1) / 2
        const cy = (y0 + y1) / 2
        const rx = Math.max((x1 - x0) / 2, 8)
        const ry = Math.max((y1 - y0) / 2, 8)
        const s = patch.score
        const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(rx, ry))
        const heatColor = s > 0.45 ? "180,0,255" : s > 0.35 ? "255,0,0" : s > 0.20 ? "255,160,0" : "255,220,0"
        gradient.addColorStop(0, `rgba(${heatColor}, 0.5)`)
        gradient.addColorStop(0.5, `rgba(${heatColor}, 0.2)`)
        gradient.addColorStop(1, `rgba(${heatColor}, 0)`)
        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.ellipse(cx, cy, rx * 1.5, ry * 1.5, 0, 0, Math.PI * 2)
        ctx.fill()
      })
    }

    if (showBoxes) {
      topPatches.forEach((patch, idx) => {
        const { y0, x0, y1, x1 } = patch.position
        const s = patch.score
        const pad = 3
        const color = idx === 0 ? "#ef4444" : s > 0.05 ? "#f97316" : "#eab308"
        ctx.strokeStyle = color
        ctx.lineWidth = idx === 0 ? 2 : 1.5
        ctx.setLineDash(idx === 0 ? [] : [4, 3])
        ctx.globalAlpha = 0.9
        ctx.strokeRect(x0 - pad, y0 - pad, (x1 - x0) + pad * 2, (y1 - y0) + pad * 2)
        ctx.font = "bold 10px monospace"
        ctx.fillStyle = color
        ctx.globalAlpha = 1
        ctx.fillText(`${s.toFixed(2)}`, x0 - pad, y0 - pad - 3)
      })
      ctx.setLineDash([])
      ctx.globalAlpha = 1
    }
  }, [showHeatmap, showBoxes, score, anomalyPatches, xaiKey])  // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="relative w-full h-full">
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-contain" style={{ imageRendering: "pixelated" }} />
      <canvas ref={overlayRef} className="absolute inset-0 w-full h-full pointer-events-none" style={{ imageRendering: "auto" }} />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Analysis View
// ---------------------------------------------------------------------------
function AnalysisView({
  t, filteredSlices, activeSlice, onSliceSelect, onSliceNavigate, sliceTab, setSliceTab, handleFetchCard,
  showHeatmap, setShowHeatmap, showBoxes, setShowBoxes, opinion, setOpinion,
  currentSlice, dcmFiles, volumeSliceCount, trackSummary, onSaveReport, slicePatchesMap, patientName, setPatientName, patientBirthdate, setPatientBirthdate, patientGender, setPatientGender,
  cardData, cardLoading, selectedRiskRegion, onClearRegion, analysisProgress
}: {
  t: T
  filteredSlices: { id: number; score: number }[]
  activeSlice: number
  onSliceSelect: (n: number) => void
  onSliceNavigate: (n: number) => void
  handleFetchCard: (n: number) => void
  sliceTab: "all" | "high"
  setSliceTab: (t: "all" | "high") => void
  selectedRiskRegion: {min: number, max: number} | null
  onClearRegion: () => void
  showHeatmap: boolean
  setShowHeatmap: (b: boolean) => void
  showBoxes: boolean
  setShowBoxes: (b: boolean) => void
  opinion: string
  setOpinion: (s: string) => void
  currentSlice: { id: number; score: number } | undefined
  dcmFiles: File[]
  volumeSliceCount: number
  trackSummary: any
  onSaveReport: () => void
  slicePatchesMap: Map<number, Array<{position: {y0:number, x0:number, y1:number, x1:number}, score:number, padim_score?: number}>>
  patientName: string
  setPatientName: (s: string) => void
  patientBirthdate: string
  setPatientBirthdate: (s: string) => void
  patientGender: string
  setPatientGender: (s: string) => void
  cardData: any
  cardLoading: boolean
  analysisProgress: number
}) {
  const [saving, setSaving] = useState(false)
  const [detectedWW, setDetectedWW] = useState(1500)
  const [detectedWL, setDetectedWL] = useState(-600)
  const [topK, setTopK] = useState(20)

  const TOP_K_OPTIONS = [1, 5, 10, 20, 30, 50] as const
  const TOP_K_ACCURACY: Record<number, number> = {1: 53.6, 5: 79.7, 10: 92.2, 20: 96.7, 30: 98.0, 50: 100}

  // 트랙별 z-range 집합: (y0,x0) key → {zSet, maxScore, peakZ}
  // peakZ = 트랙 내 최고 점수 슬라이스 (top10에 트랙당 1행만, 최고 슬라이스로 대표, #1)
  const trackMap = useMemo(() => {
    const map = new Map<string, {zSet: Set<number>, maxScore: number, peakZ: number}>()
    slicePatchesMap.forEach((patches, z) => {
      patches.forEach(p => {
        const key = `${p.position.y0},${p.position.x0}`
        if (!map.has(key)) map.set(key, {zSet: new Set(), maxScore: 0, peakZ: z})
        const entry = map.get(key)!
        entry.zSet.add(z)
        if (p.score > entry.maxScore) { entry.maxScore = p.score; entry.peakZ = z }
      })
    })
    return map
  }, [slicePatchesMap])

  const sortedTracks = useMemo(() =>
    Array.from(trackMap.entries()).sort((a, b) => b[1].maxScore - a[1].maxScore),
    [trackMap]
  )

  // 백엔드 트랙 집계(p5 정렬 + all_z) = 정본. 있으면 그걸 쓰고, 없으면 프론트 trackMap 폴백.
  const backendTracks: Array<{z:number, p5_score:number, track_len:number, all_z:number[]}> | undefined =
    trackSummary?.tracks_ranked

  // TOP K 트랙이 커버하는 슬라이스 합집합 (절약률 분자)
  const topKUnionZ = useMemo(() => {
    const s = new Set<number>()
    if (backendTracks && backendTracks.length > 0) {
      backendTracks.slice(0, topK).forEach(t => (t.all_z || []).forEach(z => s.add(z)))
    } else {
      sortedTracks.slice(0, topK).forEach(([, {zSet}]) => zSet.forEach(z => s.add(z)))
    }
    return s
  }, [backendTracks, sortedTracks, topK])

  // 고위험 탭: 슬라이스 단위가 아니라 트랙 단위로 표시 (#1)
  // 트랙당 1행(최고 점수 슬라이스 z) → 서로 다른 트랙이 top-K에 뜸.
  // 백엔드 tracks_ranked는 p5 정렬(사용자 정의), 폴백은 프론트 maxScore 정렬.
  // (기존엔 filteredSlices 슬라이스 점수순이라 한 병변의 연속 슬라이스가 리스트 독점)
  const displaySlices = sliceTab === "high"
    ? (backendTracks && backendTracks.length > 0
        ? backendTracks.slice(0, topK).map(t => ({ id: t.z + 1, score: t.p5_score }))
        : sortedTracks.slice(0, topK).map(([, v]) => ({ id: v.peakZ + 1, score: v.maxScore })))
    : filteredSlices

  const handleSave = () => {
    setSaving(true)
    setTimeout(() => {
      setSaving(false)
      onSaveReport()
    }, 1200)
  }

  // ----- Arrow key navigation -----
  useEffect(() => {
    const rangeMin = selectedRiskRegion ? selectedRiskRegion.min : 1
    const rangeMax = selectedRiskRegion ? selectedRiskRegion.max : (volumeSliceCount || dcmFiles.length || 1)
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement) return
      if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
        e.preventDefault()
        const next = Math.max(rangeMin, activeSlice - 1)
        if (next !== activeSlice) onSliceNavigate(next)
      } else if (e.key === "ArrowDown" || e.key === "ArrowRight") {
        e.preventDefault()
        const next = Math.min(rangeMax, activeSlice + 1)
        if (next !== activeSlice) onSliceNavigate(next)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [activeSlice, selectedRiskRegion, dcmFiles.length, onSliceNavigate])

  const score = currentSlice?.score ?? 0

  // Map activeSlice (1-based) to the sorted dcmFiles array (0-based)
  const activeDcmFile = dcmFiles.length > 0
    ? dcmFiles[Math.min(activeSlice - 1, dcmFiles.length - 1)] ?? null
    : null

  // slicePatchesMap에서 상/중/하엽 위험도 계산
  const totalSliceCount = volumeSliceCount || dcmFiles.length || 100
  const upperSlices = Array.from(slicePatchesMap.entries()).filter(([i]) => i < totalSliceCount / 3)
  const middleSlices = Array.from(slicePatchesMap.entries()).filter(([i]) => i >= totalSliceCount / 3 && i < (totalSliceCount / 3) * 2)
  const lowerSlices = Array.from(slicePatchesMap.entries()).filter(([i]) => i >= (totalSliceCount / 3) * 2)

  const getZoneRisk = (entries: [number, Array<{score:number}>][]) => {
    if (entries.length === 0) return "Low"
    let maxScore = 0
    for (const [, patches] of entries) {
      for (const p of patches) {
        if (p.score > maxScore) maxScore = p.score
      }
    }
    return maxScore > 0.45 ? "Critical" : maxScore > 0.35 ? "High" : maxScore > 0.20 ? "Medium" : "Low"
  }

  const resultRows = [
    { label: t.upper, risk: getZoneRisk(upperSlices) as "High"|"Medium"|"Low"|"Critical", zone: t.peripheral },
    { label: t.middle, risk: getZoneRisk(middleSlices) as "High"|"Medium"|"Low"|"Critical", zone: t.central },
    { label: t.lower, risk: getZoneRisk(lowerSlices) as "High"|"Medium"|"Low"|"Critical", zone: t.peripheral },
  ]

  return (
    <PanelGroup direction="horizontal" className="h-full">
      {/* Left: Slice Navigation */}
      <Panel defaultSize={22} minSize={14} maxSize={40}>
      <div className="h-full border-r border-[#333] flex flex-col bg-[#1a1a1a]">
        <div className="p-2.5 border-b border-[#333] shrink-0">
          <Tabs value={sliceTab} onValueChange={(v) => {
              setSliceTab(v as "all" | "high")
              if (v === "high" && filteredSlices.length > 0) {
                // 첫 번째 고위험 슬라이스의 connected region 자동 설정
                onSliceSelect(filteredSlices[0].id)
              }
            }}>
            <TabsList className="w-full bg-[#2a2a2a] h-8">
              <TabsTrigger value="all" className="flex-1 text-xs h-full data-[state=active]:bg-[#2563eb] data-[state=active]:text-white">
                {t.allSlices}
              </TabsTrigger>
              <TabsTrigger value="high" className="flex-1 text-xs h-full data-[state=active]:bg-[#2563eb] data-[state=active]:text-white">
                {t.highRisk}
              </TabsTrigger>
            </TabsList>
          </Tabs>
          {sliceTab === "high" && (
            <div className="mt-2">
              <div className="flex gap-1 flex-wrap">
                {TOP_K_OPTIONS.map(k => (
                  <button
                    key={k}
                    onClick={() => setTopK(k)}
                    className={cn(
                      "text-[10px] px-2 py-0.5 rounded border transition-colors",
                      topK === k
                        ? "bg-[#2563eb] border-[#2563eb] text-white"
                        : "bg-[#1a1a1a] border-[#444] text-[#888] hover:border-[#666]"
                    )}
                  >
                    Top-{k}
                  </button>
                ))}
              </div>
              <div className="mt-1.5 text-[10px] text-[#666] leading-tight">
                <span className="text-[#888]">리뷰 </span>
                <span className="text-[#a3a3a3] tabular-nums">{topKUnionZ.size}/{volumeSliceCount || dcmFiles.length || filteredSlices.length}슬</span>
                <span className="text-[#888]"> · 절약 </span>
                <span className="text-[#a3a3a3] tabular-nums">{(volumeSliceCount || dcmFiles.length || filteredSlices.length) > 0 ? (100 - (topKUnionZ.size / (volumeSliceCount || dcmFiles.length || filteredSlices.length)) * 100).toFixed(1) : "—"}%</span>
                <span className="ml-2 text-[#888]">병변 포함 </span>
                <span className="text-[#22c55e] font-semibold">{TOP_K_ACCURACY[topK]}%</span>
              </div>
            </div>
          )}
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-1 min-h-0">
          {displaySlices.map((slice, _i) => (
            <button
              key={`${slice.id}-${_i}`}
              onClick={() => onSliceSelect(slice.id)}
              className={cn(
                "w-full flex items-center gap-2.5 p-2 rounded-lg transition-colors text-left",
                activeSlice === slice.id ? "bg-[#2563eb]" : "hover:bg-[#2a2a2a]"
              )}
            >
              {/* Mini thumbnail */}
              <div className="w-10 h-10 bg-[#0a0a0a] rounded border border-[#333] flex items-center justify-center shrink-0 relative overflow-hidden">
                <div className="absolute inset-1 border border-[#222] rounded-full opacity-60" />
                <span className="relative z-10 text-xs text-[#555]">{slice.id}</span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm text-[#a3a3a3] mb-1">{t.slice} {slice.id}</p>
                <div className="flex items-center gap-1.5">
                  <div className="flex-1 h-1 bg-[#333] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${Math.min((slice.score / 3) * 100, 100)}%`,
                        background: slice.score > 0.45 ? "#7c3aed" : slice.score > 0.35 ? "#ef4444" : slice.score > 0.20 ? "#f59e0b" : "#22c55e"
                      }}
                    />
                  </div>
                  <span className="text-xs text-[#a3a3a3] tabular-nums w-9 text-right">{slice.score.toFixed(1)}</span>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>
      </Panel>

      <PanelResizeHandle className="w-1.5 bg-[#333] hover:bg-[#2563eb] transition-colors cursor-col-resize" />

      {/* Center: DICOM Canvas Viewport */}
      <Panel defaultSize={42} minSize={25}>
      <div className="h-full flex flex-col bg-[#0d0d0d] min-w-0">
        <div className="flex items-center gap-6 px-4 py-2.5 bg-[#1a1a1a] border-b border-[#333] shrink-0">
          <label className="flex items-center gap-2 cursor-pointer">
            <Switch checked={showHeatmap} onCheckedChange={setShowHeatmap} />
            <span className="text-xs text-[#a3a3a3]">{t.showHeatmap}</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <Switch checked={showBoxes} onCheckedChange={setShowBoxes} />
            <span className="text-xs text-[#a3a3a3]">{t.showBoxes}</span>
          </label>
          {dcmFiles.length > 0 && (
            <span className="ml-auto text-[10px] text-green-400 font-mono">
              DICOM {dcmFiles.length} slices loaded
            </span>
          )}
        </div>

        <div className="flex-1 flex flex-col items-center justify-center p-4 relative gap-3">
          {selectedRiskRegion && (
            <div className="w-full max-w-[806px] flex items-center justify-between">
              <span className="text-[10px] text-yellow-400 font-mono">
                ⚡ 위험 영역 제한 중: {selectedRiskRegion.min}~{selectedRiskRegion.max}
              </span>
              <button
                onClick={() => onClearRegion()}
                className="text-[10px] text-[#555] hover:text-[#999] transition-colors"
              >
                ✕ 해제
              </button>
            </div>
          )}
          {dcmFiles.length > 0 && (
            <div className="w-full max-w-[806px] flex items-center gap-2">
              <span className="text-xs text-[#555] font-mono w-5 text-right">
                {selectedRiskRegion ? selectedRiskRegion.min : 1}
              </span>
              <input
                type="range"
                min={selectedRiskRegion ? selectedRiskRegion.min : 1}
                max={selectedRiskRegion ? selectedRiskRegion.max : (volumeSliceCount || dcmFiles.length)}
                value={activeSlice}
                onChange={e => onSliceNavigate(Number(e.target.value))}
                className="flex-1 h-1.5 accent-[#2563eb] cursor-pointer"
              />
              <span className="text-xs text-[#555] font-mono">
                {selectedRiskRegion ? selectedRiskRegion.max : (volumeSliceCount || dcmFiles.length)}
              </span>
              <span className="text-xs text-[#2563eb] font-mono w-20 text-right">
                {activeSlice.toString().padStart(3, "0")} / {volumeSliceCount || dcmFiles.length}
              </span>
            </div>
          )}
          <div
            className="relative w-full max-w-[806px] aspect-square bg-[#050505] rounded-xl border border-[#222] overflow-hidden shadow-2xl cursor-ew-resize select-none"
            onMouseDown={e => {
              const startX = e.clientX
              const startSlice = activeSlice
              const rangeMin = selectedRiskRegion ? selectedRiskRegion.min : 1
              const rangeMax = selectedRiskRegion ? selectedRiskRegion.max : (dcmFiles.length || TOTAL_SLICES)
              const onMove = (me: MouseEvent) => {
                const delta = Math.round((me.clientX - startX) / 3)
                const next = Math.max(rangeMin, Math.min(rangeMax, startSlice + delta))
                onSliceNavigate(next)
              }
              const onUp = () => {
                window.removeEventListener("mousemove", onMove)
                window.removeEventListener("mouseup", onUp)
              }
              window.addEventListener("mousemove", onMove)
              window.addEventListener("mouseup", onUp)
            }}
          >
            <DicomCanvas
              file={activeDcmFile}
              sliceIndex={activeSlice - 1}
              totalSlices={dcmFiles.length || TOTAL_SLICES}
              score={score}
              showHeatmap={showHeatmap}
              showBoxes={showBoxes}
              anomalyPatches={slicePatchesMap.get(activeSlice - 1) ?? []}
              onWindowDetected={(ww, wl) => { setDetectedWW(ww); setDetectedWL(wl) }}
              xaiBoundingBox={cardData?.candidate ? {y0: cardData.candidate.y0, x0: cardData.candidate.x0, y1: cardData.candidate.y1, x1: cardData.candidate.x1} : null}
              ctPngUrl={analysisProgress >= 100 ? `http://localhost:8000/ct_slice/${activeSlice - 1}` : null}
            />
            {/* HUD overlay */}
            <div className="absolute bottom-3 left-3 flex items-center gap-2 pointer-events-none">
              <span className="text-xs text-[#666] bg-[#000]/80 px-2 py-1 rounded font-mono">
                {t.slice} {activeSlice.toString().padStart(3, "0")}
              </span>
              <span className={cn(
                "text-xs px-2 py-1 rounded font-mono bg-[#000]/80",
                score > 0.45 ? "text-purple-400" : score > 0.35 ? "text-red-400" : score > 0.20 ? "text-yellow-400" : "text-green-400"
              )}>
                {t.anomalyScore} {score.toFixed(2)}
              </span>
            </div>
            {dcmFiles.length > 0 && (
              <div className="absolute top-3 right-3 pointer-events-none flex flex-col items-end gap-1">
                <span className="text-[12px] text-white bg-[#2563eb]/90 px-2 py-0.5 rounded font-mono font-bold">
                  LUNG W
                </span>
                <span className="text-[11px] text-[#aaa] bg-[#000]/80 px-2 py-0.5 rounded font-mono">
                  WW:{detectedWW} WL:{detectedWL}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
      </Panel>

      <PanelResizeHandle className="w-1.5 bg-[#333] hover:bg-[#2563eb] transition-colors cursor-col-resize" />

      {/* Right: AI 이상 분석 + 소견 — 세로 분리 */}
      <Panel defaultSize={36} minSize={20} maxSize={55}>
      <div className="h-full border-l border-[#333] flex flex-col bg-[#1a1a1a]">

        {/* 상단: AI 이상 분석 (스크롤 가능) */}
        <div className="flex-1 overflow-y-auto min-h-0 border-b border-[#333]">
          <div className="p-4">
            <p className="text-xs font-semibold text-[#a3a3a3] uppercase tracking-widest mb-3">AI 이상 분석</p>

            {cardLoading && (
              <div className="flex items-center gap-2 py-3">
                <Loader2 className="w-4 h-4 animate-spin text-[#2563eb]" />
                <span className="text-sm text-[#a3a3a3]">설명 카드 로딩 중...</span>
              </div>
            )}
            {!cardLoading && !cardData && (
              <p className="text-sm text-[#555] py-3">슬라이스를 선택하면 AI 분석 카드가 표시됩니다.</p>
            )}
            {!cardLoading && cardData && (
              <>
                {/* 점수 + 위치 */}
                <div className="space-y-2 mb-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-[#a3a3a3]">Peak score</span>
                    <span className="text-lg font-semibold text-[#EF9F27]">{cardData.candidate?.score?.toFixed(4)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-[#a3a3a3]">위치</span>
                    <span className="text-sm text-[#85B7EB]">{cardData.position_bin}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-[#a3a3a3]">슬라이스</span>
                    <span className="text-sm text-[#a3a3a3]">z = {cardData.slice_index}</span>
                  </div>
                </div>

                {/* Panel 1: 후보 + 정상 비교 이미지 */}
                {cardData.normal_refs?.length > 0 && (
                  <div className="mb-4">
                    <p className="text-xs text-[#a3a3a3] uppercase tracking-widest mb-2">
                      Panel 1 · 후보 위치 vs 정상 비교
                    </p>
                    <div className="border-2 border-[#2563eb]/70 rounded-lg p-1.5">
                      <div className="grid grid-cols-2 gap-2">
                        {/* 후보 위치 슬롯 */}
                        <div className="space-y-1.5">
                          <div className="relative w-full aspect-square bg-[#111] rounded overflow-hidden">
                            {cardData.candidate?.crop_base64
                              ? <>
                                  <img src={`data:image/png;base64,${cardData.candidate.crop_base64}`} alt="candidate" className="w-full h-full object-contain rounded" />
                                  <div className="absolute pointer-events-none" style={{inset: "31.25%", border: "2px solid #E44B4A"}} />
                                </>
                              : <span className="text-xs text-[#E24B4A]">후보</span>
                            }
                          </div>
                          <p className="text-xs text-[#E24B4A] text-center font-medium">candidate (saved 후보)</p>
                        </div>
                        {/* 정상 ref 3장 */}
                        {cardData.normal_refs.map((ref: any, i: number) => (
                          <div key={i} className="space-y-1.5">
                            <div className="relative w-full aspect-square rounded overflow-hidden">
                              <img
                                src={`data:image/png;base64,${ref.image_base64}`}
                                alt={ref.alias}
                                className="w-full h-full object-contain rounded"
                              />
                              <div className="absolute pointer-events-none" style={{inset: "31.25%", border: "2px solid #EF9F27"}} />
                            </div>
                            <p className="text-xs text-[#888] text-center">
                              normal_patient_{i + 1}
                              {ref.lung_z_pct !== undefined && (
                                <span className="text-[#555] ml-1">zpct {ref.lung_z_pct.toFixed(3)} d {ref.distance?.toFixed(3)}</span>
                              )}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                    <p className="text-xs text-[#555] mt-2 text-center">
                      lung_z_pct 기준 위치 매칭 · same-z 아님
                    </p>
                  </div>
                )}

                {/* Panel 3: 3x3 패치 응답 */}
                {cardData.patch_3x3 && (
                  <div className="mb-4 border border-white/20 rounded-lg p-3">
                    <p className="text-xs text-[#a3a3a3] uppercase tracking-widest mb-2">
                      Panel 3 · 3×3 patch response
                    </p>
                    <div className="flex items-center gap-4">
                      <div className="grid grid-cols-3 gap-1.5" style={{width: 126}}>
                        {Object.entries(cardData.patch_3x3).map(([key, val]: [string, any]) => {
                          const isPeak = key === `${cardData.candidate?.y0},${cardData.candidate?.x0}`
                          return (
                            <div
                              key={key}
                              className={cn(
                                "rounded text-center py-2 text-sm font-mono",
                                isPeak
                                  ? "bg-[#EF9F27]/40 text-[#EF9F27] font-bold border border-[#EF9F27]/60"
                                  : val !== null
                                    ? "bg-[#2a2a2a] text-[#a3a3a3]"
                                    : "bg-[#1a1a1a] text-[#444]"
                              )}
                            >
                              {val !== null ? val.toFixed(2) : "N/A"}
                            </div>
                          )
                        })}
                      </div>
                      <div className="space-y-1.5">
                        <div className="text-base text-[#a3a3a3]">
                          Peak: <span className="text-[#EF9F27] font-semibold">{cardData.candidate?.score?.toFixed(2)}</span>
                        </div>
                        <div className="text-base text-[#a3a3a3]">
                          위치: [{cardData.candidate?.y0}, {cardData.candidate?.x0}]
                        </div>
                        <div className="text-base text-[#a3a3a3]">
                          side: {cardData.candidate?.side}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Panel 4: 임상 판독 보조 */}
                {cardData.panel4 && (
                  <div className="rounded-lg border border-[#333] overflow-hidden">
                    <div className="bg-[#1e1e1e] px-3 py-2 border-b border-[#333]">
                      <p className="text-sm font-semibold text-[#c0c0c0] uppercase tracking-widest">Panel 4 · 임상 판독 보조</p>
                    </div>
                    <div className="p-3 space-y-3">
                      {cardData.panel4.key_finding && (
                        <div>
                          <p className="text-sm font-semibold text-[#85B7EB] mb-1">[핵심 소견 / Key finding]</p>
                          <p className="text-sm text-[#e0e0e0] leading-relaxed font-medium">{cardData.panel4.key_finding.title}</p>
                          <p className="text-sm text-[#c0c0c0] leading-relaxed mt-1">{cardData.panel4.key_finding.body}</p>
                          <p className="text-sm text-[#a3a3a3] leading-relaxed mt-1">{cardData.panel4.key_finding.context}</p>
                        </div>
                      )}
                      {cardData.panel4.location_context && (
                        <div>
                          <p className="text-sm font-semibold text-[#85B7EB] mb-1">[위치 판독 보조 / Location context]</p>
                          <p className="text-sm text-[#c0c0c0] leading-relaxed">{cardData.panel4.location_context.position}</p>
                          <p className="text-sm text-[#EF9F27] leading-relaxed mt-1">{cardData.panel4.location_context.z_warning}</p>
                        </div>
                      )}
                      {cardData.panel4.fp_context && (
                        <div>
                          <p className="text-sm font-semibold text-[#EF9F27] mb-1">[오탐 가능성 / False-positive context]</p>
                          <p className="text-sm text-[#EF9F27] leading-relaxed">{cardData.panel4.fp_context}</p>
                        </div>
                      )}
                      <div className="bg-[#2a2a2a] rounded p-2">
                        <p className="text-sm font-semibold text-[#aaaaaa] mb-1">[Disclaimer]</p>
                        <p className="text-sm text-[#aaaaaa] leading-relaxed">{cardData.panel4.disclaimer}</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Grad-CAM / RD4AD 히트맵 */}
                {cardData.gradcam_base64 && (
                  <div className="bg-[#1e1e1e] border border-[#333] rounded-lg p-4">
                    <p className="text-sm font-semibold text-[#c0c0c0] uppercase tracking-widest mb-3">
                      {cardData.heatmap_type === "gradcam" ? "Grad-CAM · 비소세포암 주목 영역" : "RD4AD · 재구성 오차 맵"}
                    </p>
                    <div className="flex gap-4 items-start">
                      <div className="flex flex-col items-center gap-1">
                        <p className="text-xs text-[#888] mb-1">CT + 히트맵</p>
                        <div className="relative w-32 h-32 bg-black rounded border border-[#444]">
                          {/* 히트맵과 동일 96px FOV CT (B): 256px 후보 crop 대신 heatmap_ct_crop_b64 사용 → 오버레이 정합 */}
                          {(cardData.heatmap_ct_crop_b64 || cardData.candidate?.crop_base64) && (
                            <img
                              src={`data:image/png;base64,${cardData.heatmap_ct_crop_b64 || cardData.candidate.crop_base64}`}
                              className="absolute inset-0 w-full h-full object-contain"
                              style={{ imageRendering: "pixelated" }}
                              alt="heatmap ct crop"
                            />
                          )}
                          <img
                            src={`data:image/png;base64,${cardData.gradcam_base64}`}
                            className="absolute inset-0 w-full h-full object-contain"
                            style={{ imageRendering: "pixelated" }}
                            alt="gradcam"
                          />
                        </div>
                      </div>
                      <div className="flex flex-col items-center gap-1">
                        <p className="text-xs text-[#888] mb-1">히트맵 단독</p>
                        <div className="w-32 h-32 bg-black rounded border border-[#444] flex items-center justify-center">
                          <img
                            src={`data:image/png;base64,${cardData.gradcam_base64}`}
                            className="w-full h-full object-contain"
                            style={{ imageRendering: "pixelated" }}
                            alt="gradcam standalone"
                          />
                        </div>
                      </div>
                      <div className="flex-1 flex flex-col gap-2 pt-1">
                        {cardData.heatmap_type === "gradcam" ? (
                          <>
                            <p className="text-xs text-[#a0a0a0] leading-relaxed">
                              EfficientNet-B0 img_features[7] 기반 Grad-CAM.<br />
                              폐 마스크 내부 픽셀만 min-max 정규화. YlOrRd 컬러맵.
                            </p>
                            <p className="text-xs text-[#666]">P-C-NORMAL30b · epoch=6 · 연구용 보조 · 진단 아님</p>
                          </>
                        ) : (
                          <>
                            <p className="text-xs text-[#a0a0a0] leading-relaxed">
                              RD4AD E2 spatial anomaly map.<br />
                              Teacher-Student cosine 오차. 비소세포암 의심 확률 낮음(&lt;50%) 또는 미산출.
                            </p>
                            <p className="text-xs text-[#666]">RD4AD E2 · lung3ch · 연구용 보조 · 진단 아님</p>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* Panel 5: NSCLC 보조 분류기 */}
                {cardData.panel5 && (
                  <div className="rounded-lg border border-[#333] overflow-hidden">
                    <div className="bg-[#1a1a2e] px-3 py-2 border-b border-[#333] flex items-center gap-2">
                      <p className="text-sm font-semibold text-[#c0c0c0] uppercase tracking-widest">Panel 5 · 비소세포암 의심 점수</p>
                      {cardData.panel5.available && (
                        <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                          cardData.panel5.level === 'high'       ? 'bg-red-900 text-red-300' :
                          cardData.panel5.level === 'borderline' ? 'bg-yellow-900 text-yellow-300' :
                          cardData.panel5.level === 'low_nsclc'  ? 'bg-blue-900 text-blue-300' :
                                                                    'bg-green-900 text-green-300'
                        }`}>
                          {cardData.panel5.label}
                        </span>
                      )}
                    </div>
                    <div className="p-3 space-y-3">
                      {!cardData.panel5.available ? (
                        <p className="text-sm text-[#a3a3a3]">{cardData.panel5.reason}</p>
                      ) : (
                        <>
                          {/* 점수 바 */}
                          <div>
                            <div className="flex items-center justify-between mb-1">
                              <p className="text-sm font-semibold text-[#85B7EB]">비소세포암 의심 점수</p>
                              <span className={`text-lg font-bold ${
                                cardData.panel5.level === 'high'       ? 'text-red-400' :
                                cardData.panel5.level === 'borderline' ? 'text-yellow-400' :
                                cardData.panel5.level === 'low_nsclc'  ? 'text-blue-400' :
                                                                          'text-green-400'
                              }`}>{cardData.panel5.prob_pct}%</span>
                            </div>
                            <div className="w-full bg-[#2a2a2a] rounded-full h-2">
                              <div
                                className={`h-2 rounded-full transition-all ${
                                  cardData.panel5.level === 'high'       ? 'bg-red-500' :
                                  cardData.panel5.level === 'borderline' ? 'bg-yellow-500' :
                                  cardData.panel5.level === 'low_nsclc'  ? 'bg-blue-500' :
                                                                            'bg-green-500'
                                }`}
                                style={{ width: `${cardData.panel5.prob_pct}%` }}
                              />
                            </div>
                          </div>
                          <p className="text-sm text-[#e0e0e0] leading-relaxed">{cardData.panel5.interpretation}</p>
                          <div className="bg-[#2a1a1a] rounded p-2 border border-[#5a3030]">
                            <p className="text-xs font-semibold text-[#EF9F27] mb-1">⚠ 해석 주의</p>
                            <p className="text-xs text-[#EF9F27] leading-relaxed">{cardData.panel5.caution}</p>
                          </div>
                          <div className="text-xs text-[#666] space-y-0.5">
                            <p>{cardData.panel5.model_info}</p>
                            <p>{cardData.panel5.performance}</p>
                          </div>
                          <div className="bg-[#2a2a2a] rounded p-2">
                            <p className="text-xs text-[#888] leading-relaxed">{cardData.panel5.disclaimer}</p>
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {/* 하단: 소견 폼 (고정 높이) */}
        <div className="shrink-0 p-4 bg-[#1a1a1a]">
          <p className="text-xs font-semibold text-[#a3a3a3] uppercase tracking-widest mb-3">{t.medicalOpinion}</p>
          <Textarea
            value={opinion}
            onChange={e => setOpinion(e.target.value)}
            placeholder={t.clinicalNotes}
            className="min-h-[100px] bg-[#2a2a2a] border-[#333] resize-none text-[#e5e5e5] placeholder:text-[#555] text-sm focus:border-[#2563eb] mb-3"
          />
          <Button
            onClick={handleSave}
            disabled={saving || !opinion.trim()}
            className="w-full bg-[#2563eb] hover:bg-[#1d4ed8] text-white h-10 text-sm"
          >
            {saving
              ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Committing...</>
              : <><Globe className="w-4 h-4 mr-2" />{t.saveReport}</>}
          </Button>
        </div>

      </div>
      </Panel>
    </PanelGroup>
  )
}

// ---------------------------------------------------------------------------
// Patients View
// ---------------------------------------------------------------------------
function PatientsView({ t, filteredPatients, searchQuery, setSearchQuery, dateFilter, setDateFilter, setSelectedPatient, onDeletePatient, onLoadFromB2 }: {
  t: T
  filteredPatients: ReturnType<typeof generatePatients>
  searchQuery: string
  setSearchQuery: (s: string) => void
  dateFilter: string
  setDateFilter: (s: string) => void
  setSelectedPatient: (p: ReturnType<typeof generatePatients>[0] | null) => void
  onDeletePatient: (id: string) => void
  onLoadFromB2: () => void
}) {

  return (
    <div className="h-full flex flex-col p-5 bg-[#121212]">
      <div className="flex items-center gap-3 mb-4">
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[#555]" />
          <Input
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder={t.searchPatients}
            className="pl-9 bg-[#1e1e1e] border-[#333] text-[#e5e5e5] placeholder:text-[#555] h-9 text-sm"
          />
        </div>
        <Select value={dateFilter} onValueChange={setDateFilter}>
          <SelectTrigger className="w-[160px] bg-[#1e1e1e] border-[#333] text-[#e5e5e5] h-9 text-sm">
            <SelectValue placeholder={t.allDates} />
          </SelectTrigger>
          <SelectContent className="bg-[#1e1e1e] border-[#333] text-[#e5e5e5]">
            <SelectItem value="all">{t.allDates}</SelectItem>
            {Array.from({ length: 12 }, (_, i) => (
              <SelectItem key={i} value={String(i + 1)}>
                {new Date(2024, i).toLocaleString("en", { month: "long" })}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button
          variant="outline"
          onClick={onLoadFromB2}
          className="ml-auto border-[#333] bg-transparent text-[#a3a3a3] hover:bg-[#2a2a2a] hover:text-[#e5e5e5] h-9 text-xs whitespace-nowrap px-3"
        >
          <Download className="w-3.5 h-3.5 mr-1.5" />B2에서 불러오기
        </Button>
      </div>

      <div className="flex-1 bg-[#1e1e1e] border border-[#333] rounded-xl overflow-hidden min-h-0">
        <div className="overflow-auto h-full">
          <table className="w-full text-sm">
            <thead className="bg-[#252525] sticky top-0 z-10">
              <tr>
                {["ID", t.name, "생년월일", "성별", t.date, t.aiInspectionStatus, t.risk, ""].map((h, i) => (
                  <th key={i} className="text-left text-[11px] font-medium p-3 text-[#666] uppercase tracking-wide whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredPatients.map((patient, idx) => (
                <tr key={patient.id} className={cn(
                  "border-t border-[#2a2a2a] hover:bg-[#2a2a2a]/60 transition-colors",
                  idx % 2 === 0 && "bg-[#1a1a1a]/30"
                )}>
                  <td className="p-3 text-[#555] font-mono text-xs">{patient.id}</td>
                  <td className="p-3 font-medium text-[#e5e5e5]">{patient.name}</td>
                  <td className="p-3 text-[#a3a3a3]">{(patient as any).birthdate ?? "-"}</td>
                  <td className="p-3 text-[#a3a3a3]">{(patient as any).gender ?? "-"}</td>
                  <td className="p-3 text-[#a3a3a3] tabular-nums">{patient.date}</td>
                  <td className="p-3">
                    <div className="flex flex-col gap-1">
                      <Badge variant="outline" className={cn(
                        "text-xs",
                        patient.status === "Completed" ? "border-green-700 text-green-400" :
                        patient.status === "Pending" ? "border-yellow-700 text-yellow-400" : "border-blue-700 text-blue-400"
                      )}>
                        {tStatus(t, patient.status)}
                      </Badge>
                      {(patient as any).analysisStatus && (patient as any).analysisStatus !== "done" && (
                        <Badge variant="outline" className={cn(
                          "text-xs animate-pulse",
                          (patient as any).analysisStatus === "pending"     ? "border-yellow-600 text-yellow-400" :
                          (patient as any).analysisStatus === "downloading" ? "border-blue-600 text-blue-400" :
                          (patient as any).analysisStatus === "analyzing"   ? "border-purple-600 text-purple-400" :
                          (patient as any).analysisStatus === "error"       ? "border-red-700 text-red-400" : "border-gray-600 text-gray-400"
                        )}>
                          {(patient as any).analysisStatus === "pending"     ? "⏳ 대기중" :
                           (patient as any).analysisStatus === "downloading" ? "⬇ 다운로드" :
                           (patient as any).analysisStatus === "analyzing"   ? "🔬 분석중" :
                           (patient as any).analysisStatus === "error"       ? "⚠ 오류" : (patient as any).analysisStatus}
                        </Badge>
                      )}
                    </div>
                  </td>
                  <td className="p-3">
                    <Badge className={cn(
                      "text-xs text-white",
                      patient.risk === "Critical" ? "bg-purple-600" : patient.risk === "High" ? "bg-red-600" : patient.risk === "Medium" ? "bg-yellow-600" : "bg-green-600"
                    )}>
                      {tRisk(t, patient.risk)}
                    </Badge>
                  </td>
                  <td className="p-3">
                    <div className="flex items-center gap-1">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setSelectedPatient(patient)}
                        className="h-7 w-7 p-0 hover:bg-[#2563eb] hover:text-white text-[#555] transition-colors rounded-md"
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => onDeletePatient(patient.id)}
                        className="h-7 w-7 p-0 hover:bg-red-600 hover:text-white text-[#555] transition-colors rounded-md"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </Button>
                    </div>
                </td>
                </tr>
              ))}
              {filteredPatients.length === 0 && (
                <tr>
                  <td colSpan={6} className="p-12 text-center text-[#555] text-sm">{t.noPatients}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}