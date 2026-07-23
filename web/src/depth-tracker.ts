/**
 * DepthTracker — client for the DA3 metric depth backend.
 *
 * Sends webcam frames to the depth backend and receives metric depth
 * (in meters) at wrist pixel coordinates from MediaPipe.
 *
 * Architecture:
 *   MediaPipe Pose → (x, y) of wrists (normalized 0-1)
 *   DA3 Backend    → metric depth (meters) at those pixels
 *   Kalman filter  → smooth the Z signal
 *   PunchDetector → detect jabs/crosses/hooks from Z velocity
 */

export type DepthPoint = { x: number; y: number };

export type DepthResult = {
  /** Metric depth in meters at each requested point */
  depths: number[];
  /** Base64 depth visualization image */
  depthImage: string | null;
  /** Inference time in ms */
  inferenceMs: number;
};

export type WristDepth = {
  left: number | null;
  right: number | null;
};

/**
 * Simple 1D Kalman filter for smoothing depth values.
 * Depth can be noisy at wrist edges, so this stabilizes the signal.
 */
class KalmanFilter1D {
  private x: number;
  private p: number;
  private readonly q: number; // process noise
  private readonly r: number; // measurement noise

  constructor(initialValue: number, processNoise = 0.01, measurementNoise = 0.1) {
    this.x = initialValue;
    this.p = 1.0;
    this.q = processNoise;
    this.r = measurementNoise;
  }

  update(measurement: number): number {
    // Predict
    this.p += this.q;
    // Update
    const k = this.p / (this.p + this.r);
    this.x += k * (measurement - this.x);
    this.p *= (1 - k);
    return this.x;
  }

  get value(): number {
    return this.x;
  }

  reset(value: number): void {
    this.x = value;
    this.p = 1.0;
  }
}

/**
 * Punch detection state machine.
 *
 * A punch is detected when:
 * 1. Wrist depth decreases rapidly (hand moves toward camera)
 * 2. Wrist 2D velocity exceeds threshold
 * 3. The depth reaches a minimum (full extension) then starts returning
 *
 * This uses the metric depth from DA3, which actually changes when
 * the arm extends — unlike MediaPipe's relative Z which stays flat.
 */
export type PunchType = "jab" | "cross" | "left_hook" | "right_hook" | null;

export type PunchEvent = {
  type: PunchType;
  power: number; // 0-1 normalized
  timestamp: number;
};

type PunchState = {
  phase: "idle" | "extending" | "retracting";
  startDepth: number;
  minDepth: number;
  startTime: number;
  maxVelocity: number;
  side: "left" | "right";
};

export class PunchDetector {
  private state: PunchState = {
    phase: "idle",
    startDepth: 0,
    minDepth: 0,
    startTime: 0,
    maxVelocity: 0,
    side: "left",
  };

  private previousDepths: { left: number | null; right: number | null } = {
    left: null,
    right: null,
  };
  private previousTime: number = 0;

  // Thresholds (tunable)
  private readonly DEPTH_DELTA_THRESHOLD = 0.15; // meters — hand must come 15cm closer
  private readonly VELOCITY_THRESHOLD = 0.5; // m/s minimum
  private readonly EXTENSION_TIME_MAX = 500; // ms — punch must happen within 500ms
  private readonly COOLDOWN_MS = 300; // minimum time between punches per arm

  private lastPunchTime: { left: number; right: number } = { left: 0, right: 0 };

  detect(
    wristDepth: WristDepth,
    wristVelocity: { left: number; right: number }, // 2D velocity in px/s
    timestamp: number,
  ): PunchEvent | null {
    const dt = this.previousTime > 0 ? (timestamp - this.previousTime) / 1000 : 0.016;
    this.previousTime = timestamp;

    let event: PunchEvent | null = null;

    for (const side of ["left", "right"] as const) {
      const depth = wristDepth[side];
      const prevDepth = this.previousDepths[side];

      if (depth === null) {
        this.previousDepths[side] = null;
        continue;
      }

      if (prevDepth !== null) {
        const depthDelta = prevDepth - depth; // positive = hand moving toward camera
        const depthVelocity = depthDelta / dt; // m/s toward camera

        switch (this.state.phase) {
          case "idle": {
            // Check if a punch is starting
            if (
              depthDelta > 0.03 && // 3cm in one frame
              depthVelocity > this.VELOCITY_THRESHOLD &&
              timestamp - this.lastPunchTime[side] > this.COOLDOWN_MS
            ) {
              this.state = {
                phase: "extending",
                startDepth: prevDepth,
                minDepth: depth,
                startTime: timestamp,
                maxVelocity: depthVelocity,
                side,
              };
            }
            break;
          }

          case "extending": {
            // Track the extension
            if (depth < this.state.minDepth) {
              this.state.minDepth = depth;
            }
            if (depthVelocity > this.state.maxVelocity) {
              this.state.maxVelocity = depthVelocity;
            }

            // Check if extension is complete (hand starts returning)
            if (depth > this.state.minDepth + 0.02) {
              // Punch completed — classify
              const totalExtension = this.state.startDepth - this.state.minDepth;
              const duration = timestamp - this.state.startTime;

              if (
                totalExtension >= this.DEPTH_DELTA_THRESHOLD &&
                duration <= this.EXTENSION_TIME_MAX
              ) {
                const power = Math.min(
                  1.0,
                  (totalExtension / 0.5) * 0.5 +
                    (this.state.maxVelocity / 3.0) * 0.5,
                );

                // Classify punch type based on 2D velocity direction
                const v2d = wristVelocity[side];
                let type: PunchType = "jab";
                if (side === "right") {
                  type = v2d > 1.5 ? "right_hook" : "cross";
                } else {
                  type = v2d > 1.5 ? "left_hook" : "jab";
                }

                event = {
                  type,
                  power,
                  timestamp,
                };
                this.lastPunchTime[side] = timestamp;
              }

              this.state.phase = "retracting";
            }

            // Timeout — if extension takes too long, reset
            if (timestamp - this.state.startTime > this.EXTENSION_TIME_MAX) {
              this.state.phase = "idle";
            }
            break;
          }

          case "retracting": {
            // Wait for hand to return near guard position
            if (depth > this.state.startDepth - 0.05) {
              this.state.phase = "idle";
            }
            // Safety timeout
            if (timestamp - this.state.startTime > 1000) {
              this.state.phase = "idle";
            }
            break;
          }
        }
      }

      this.previousDepths[side] = depth;
    }

    return event;
  }

  reset(): void {
    this.state = {
      phase: "idle",
      startDepth: 0,
      minDepth: 0,
      startTime: 0,
      maxVelocity: 0,
      side: "left",
    };
    this.previousDepths = { left: null, right: null };
    this.previousTime = 0;
  }
}

/**
 * Main depth tracking client.
 *
 * Sends frames to the DA3 backend at a throttled rate (every N frames)
 * and maintains Kalman-filtered depth values for both wrists.
 */
export class DepthTracker {
  private backendUrl: string;
  private focalLength: number;
  private enabled: boolean = false;

  // Kalman filters for left/right wrist depth
  private leftKalman: KalmanFilter1D | null = null;
  private rightKalman: KalmanFilter1D | null = null;

  // Throttling: send every Nth frame to backend
  private frameCounter: number = 0;
  private readonly SEND_EVERY_N_FRAMES: number = 2; // send every 2nd frame

  // Last known depth values
  private currentDepths: WristDepth = { left: null, right: null };

  // Depth visualization image (for debug overlay)
  private depthImage: string | null = null;
  private lastInferenceMs: number = 0;

  // Pending request state
  private pendingRequest: boolean = false;

  // Punch detector
  public readonly punchDetector: PunchDetector;

  // Callbacks
  public onDepthUpdate: ((depths: WristDepth) => void) | null = null;
  public onPunch: ((event: PunchEvent) => void) | null = null;
  public onDepthImage: ((image: string) => void) | null = null;

  constructor(
    backendUrl: string = "http://127.0.0.1:8001",
    focalLength: number = 800,
  ) {
    this.backendUrl = backendUrl;
    this.focalLength = focalLength;
    this.punchDetector = new PunchDetector();
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) {
      this.currentDepths = { left: null, right: null };
      this.leftKalman = null;
      this.rightKalman = null;
    }
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  getLastInferenceMs(): number {
    return this.lastInferenceMs;
  }

  /**
   * Process a video frame. Called from the main render loop.
   *
   * @param video The HTMLVideoElement from the webcam
   * @param leftWrist Normalized (0-1) position of left wrist from MediaPipe
   * @param rightWrist Normalized (0-1) position of right wrist from MediaPipe
   * @param timestamp Current performance.now()
   * @param leftWristVel 2D velocity of left wrist (px/s)
   * @param rightWristVel 2D velocity of right wrist (px/s)
   */
  async processFrame(
    video: HTMLVideoElement,
    leftWrist: DepthPoint | null,
    rightWrist: DepthPoint | null,
    timestamp: number,
    leftWristVel: number,
    rightWristVel: number,
  ): Promise<void> {
    if (!this.enabled || this.pendingRequest) return;

    this.frameCounter++;
    if (this.frameCounter % this.SEND_EVERY_N_FRAMES !== 0) {
      // Still run punch detection with last known depths
      this.runPunchDetection(timestamp, leftWristVel, rightWristVel);
      return;
    }

    // Build points array
    const points: DepthPoint[] = [];
    if (leftWrist) points.push(leftWrist);
    if (rightWrist) points.push(rightWrist);

    if (points.length === 0) return;

    // Capture frame to canvas -> JPEG
    const canvas = document.createElement("canvas");
    const targetWidth = 320; // downscale for faster transfer
    const scale = targetWidth / video.videoWidth;
    const targetHeight = Math.round(video.videoHeight * scale);
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(video, 0, 0, targetWidth, targetHeight);

    const jpegBlob = await new Promise<Blob>((resolve) => {
      canvas.toBlob(
        (blob) => resolve(blob!),
        "image/jpeg",
        0.7,
      );
    });

    // Send to backend
    this.pendingRequest = true;
    try {
      const formData = new FormData();
      formData.append("image", jpegBlob, "frame.jpg");
      formData.append("points", JSON.stringify(points));
      formData.append("focal", String(this.focalLength));

      const response = await fetch(`${this.backendUrl}/api/depth`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        console.warn("[Depth] Backend error:", response.status);
        return;
      }

      const data: DepthResult = await response.json();
      this.lastInferenceMs = data.inferenceMs;
      this.depthImage = data.depthImage;
      if (this.onDepthImage && data.depthImage) {
        this.onDepthImage(data.depthImage);
      }

      // Map depths back to left/right
      let idx = 0;
      if (leftWrist && data.depths[idx] !== undefined) {
        const rawDepth = data.depths[idx];
        if (this.leftKalman === null) {
          this.leftKalman = new KalmanFilter1D(rawDepth);
        }
        const smoothed = this.leftKalman.update(rawDepth);
        this.currentDepths.left = smoothed;
        idx++;
      }
      if (rightWrist && data.depths[idx] !== undefined) {
        const rawDepth = data.depths[idx];
        if (this.rightKalman === null) {
          this.rightKalman = new KalmanFilter1D(rawDepth);
        }
        const smoothed = this.rightKalman.update(rawDepth);
        this.currentDepths.right = smoothed;
      }

      if (this.onDepthUpdate) {
        this.onDepthUpdate(this.currentDepths);
      }
    } catch (e) {
      console.warn("[Depth] Request failed:", e);
    } finally {
      this.pendingRequest = false;
    }

    // Run punch detection
    this.runPunchDetection(timestamp, leftWristVel, rightWristVel);
  }

  private runPunchDetection(
    timestamp: number,
    leftVel: number,
    rightVel: number,
  ): void {
    const event = this.punchDetector.detect(
      this.currentDepths,
      { left: leftVel, right: rightVel },
      timestamp,
    );
    if (event && this.onPunch) {
      this.onPunch(event);
    }
  }

  getDepths(): WristDepth {
    return this.currentDepths;
  }

  getDepthImage(): string | null {
    return this.depthImage;
  }

  /**
   * Check if the backend is available.
   */
  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.backendUrl}/api/health`);
      if (!response.ok) return false;
      const data = await response.json();
      return data.ready === true;
    } catch {
      return false;
    }
  }
}
