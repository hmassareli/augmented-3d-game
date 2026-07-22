/**
 * Headless lab audit: load WhatsApp test video → process all pipelines → dump TrackedPose vs MediaPipe.
 * Usage: npx tsx scripts/lab-depth-audit.mts
 */
import { spawn } from 'node:child_process'
import { cp, mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

type JointSnap = {
  wristDepth: number
  wristLift: number
  leftWrist: { x: number; y: number; z: number }
  avatarLeftWrist: { x: number; y: number; z: number }
  avatarRightWrist: { x: number; y: number; z: number }
}

type Audit = {
  ready: boolean
  frameCount: number
  summaries: Array<{
    pipeline: string
    meanWristDepth: number | null
    meanWristLift: number | null
    handsFront: number
    handsBack: number
    handsHigh: number
    vsMediaPipe: {
      frontBackFlips: number
      frames: number
      meanAbsDepthDelta: number
      meanLiftDelta: number
    } | null
  }>
  samples: Array<{
    index: number
    time: number
    byPipeline: Record<string, JointSnap | null>
  }>
}

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const webRoot = path.resolve(__dirname, '..')
const repoRoot = path.resolve(webRoot, '..')
const videoPath = path.join(repoRoot, 'WhatsApp Video 2026-07-21 at 15.32.46.mp4')
const outDir = path.join(webRoot, 'debug-out', 'depth-audit')
const PORT = 5179

async function waitForServer(url: string, timeoutMs = 60_000): Promise<void> {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    try {
      const response = await fetch(url)
      if (response.ok || response.status === 404) return
    } catch {
      // retry
    }
    await new Promise((resolve) => setTimeout(resolve, 400))
  }
  throw new Error(`Vite did not start at ${url}`)
}

async function main(): Promise<void> {
  await mkdir(outDir, { recursive: true })
  const ortDest = path.join(webRoot, 'public', 'ort')
  await mkdir(ortDest, { recursive: true })
  await cp(path.join(webRoot, 'node_modules', 'onnxruntime-web', 'dist'), ortDest, {
    recursive: true,
    filter: (source) => {
      const base = path.basename(source)
      return base === 'dist' || base.startsWith('ort-wasm-simd-threaded')
    },
  })

  const vite = spawn('npx vite --host 127.0.0.1 --port ' + String(PORT), {
    cwd: webRoot,
    shell: true,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, BROWSER: 'none' },
  })
  let viteLog = ''
  vite.stdout.on('data', (chunk) => { viteLog += String(chunk) })
  vite.stderr.on('data', (chunk) => { viteLog += String(chunk) })

  try {
    await waitForServer(`http://127.0.0.1:${PORT}/compare.html`)
    const browser = await chromium.launch({
      headless: true,
      args: [
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan,UseSkiaRenderer',
        '--use-angle=swiftshader',
        '--ignore-gpu-blocklist',
      ],
    })
    const page = await browser.newPage()
    page.setDefaultTimeout(600_000)
    page.on('console', (msg) => {
      const text = msg.text()
      if (
        text.includes('[lab]')
        || text.includes('Unable')
        || text.includes('Error')
        || text.includes('falha')
        || text.includes('WebGPU')
        || text.includes('ort')
        || text.includes('MediaPipe')
      ) {
        console.log(`[browser] ${text}`)
      }
    })
    page.on('pageerror', (error) => console.error('[pageerror]', error.message))

    await page.goto(`http://127.0.0.1:${PORT}/compare.html?wasm=1`, { waitUntil: 'domcontentloaded', timeout: 120_000 })
    await page.waitForSelector('#video-input', { state: 'attached' })
    await page.setInputFiles('#video-input', videoPath)

    await page.waitForFunction(() => {
      const video = document.querySelector('#source-video') as HTMLVideoElement | null
      const button = document.querySelector('#process-button') as HTMLButtonElement | null
      return Boolean(video && video.src && button && !button.disabled)
    }, { timeout: 120_000 })

    console.log('Starting processComparison…')
    const preflight = await page.evaluate(async () => {
      const urls = [
        '/assets/models/rtmpose-m.onnx',
        '/assets/models/motionagformer-xs.onnx',
        'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort-wasm-simd-threaded.mjs',
      ]
      const results: Record<string, string> = {}
      for (const url of urls) {
        try {
          const response = await fetch(url, { method: 'HEAD' })
          results[url] = `${response.status} ok=${response.ok}`
        } catch (error) {
          results[url] = `error: ${error instanceof Error ? error.message : String(error)}`
        }
      }
      return results
    })
    console.log('Preflight:', JSON.stringify(preflight, null, 2))
    await page.click('#process-button')

    const started = Date.now()
    let lastDetail = ''
    while (Date.now() - started < 600_000) {
      const snapshot = await page.evaluate(() => {
        const status = document.querySelector('#run-status')?.textContent ?? ''
        const detail = document.querySelector('#processing-detail')?.textContent ?? ''
        const percent = document.querySelector('#processing-percent')?.textContent ?? ''
        type AuditFn = () => { ready: boolean }
        const ready = Boolean((window as unknown as { __labAudit?: AuditFn }).__labAudit?.().ready)
        return { status, detail, percent, ready }
      })
      if (snapshot.detail !== lastDetail) {
        console.log(`  [${snapshot.percent}] ${snapshot.status} — ${snapshot.detail}`)
        lastDetail = snapshot.detail
      }
      if (snapshot.status.includes('pronta') || snapshot.ready) break
      if (snapshot.status.includes('falha')) {
        throw new Error(`Processing failed: ${snapshot.detail}`)
      }
      await new Promise((resolve) => setTimeout(resolve, 1500))
    }

    const status = await page.textContent('#run-status')
    if (!status?.includes('pronta')) {
      const detail = await page.textContent('#processing-detail')
      throw new Error(`Timed out waiting for comparison. status=${status} detail=${detail}`)
    }

    const audit = await page.evaluate(() => {
      type AuditFn = () => Audit
      const fn = (window as unknown as { __labAudit?: AuditFn }).__labAudit
      if (!fn) throw new Error('window.__labAudit missing')
      return fn()
    }) as Audit

    const reportPath = path.join(outDir, 'report.json')
    await writeFile(reportPath, JSON.stringify(audit, null, 2), 'utf8')
    console.log(`Wrote ${reportPath}`)

    console.log('\n=== SUMMARY (TrackedPose, −Z=front) ===')
    for (const row of audit.summaries) {
      const vs = row.vsMediaPipe
      const depth = row.meanWristDepth == null ? 'n/a' : row.meanWristDepth.toFixed(3)
      const lift = row.meanWristLift == null ? 'n/a' : row.meanWristLift.toFixed(3)
      const vsText = vs
        ? `flips=${vs.frontBackFlips}/${vs.frames} |ΔZ|=${vs.meanAbsDepthDelta.toFixed(3)} liftΔ=${vs.meanLiftDelta.toFixed(3)}`
        : ''
      console.log(
        row.pipeline.padEnd(14),
        `depth=${depth}`,
        `lift=${lift}`,
        `front=${row.handsFront} back=${row.handsBack} high=${row.handsHigh}`,
        vsText,
      )
    }

    console.log('\n=== SAMPLE FRAMES (TrackedPose + avatar IK space) ===')
    for (const sample of audit.samples) {
      console.log(`\nframe ${sample.index} t=${sample.time.toFixed(2)}s`)
      for (const [name, snap] of Object.entries(sample.byPipeline)) {
        if (!snap) {
          console.log(`  ${name.padEnd(14)} null`)
          continue
        }
        console.log(
          `  ${name.padEnd(14)} depth=${snap.wristDepth.toFixed(3)} lift=${snap.wristLift.toFixed(3)}`,
          `Lwrist=${JSON.stringify(snap.leftWrist)}`,
          `avatarL=${JSON.stringify(snap.avatarLeftWrist)}`,
          `avatarR=${JSON.stringify(snap.avatarRightWrist)}`,
        )
      }
    }

    await browser.close()
  } finally {
    vite.kill('SIGTERM')
    await writeFile(path.join(outDir, 'vite.log'), viteLog, 'utf8')
  }
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
