import { mat4, Mat4, quat, Quat, utils, vec3, Vec3 } from "wgpu-matrix";

enum MoveMode {
  NONE = "None",
  TUMBLE = "Tumble",
  TRACK = "Track",
  DOLLY = "Dolly",
}

export default class Camera {
  private controlled: boolean;
  private moveMode: MoveMode;
  private x: number;
  private y: number;
  private rotateAmplitude: number;
  private panAmplitude: number;
  private scrollAmplitude: number;
  private pitch: number;
  private yaw: number;
  private distance: number;
  private target: Vec3;
  private eye: Vec3;
  private up: Vec3;
  private right: Vec3;

  private div: HTMLDivElement;

  public constructor() {
    this.controlled = false;
    this.moveMode = MoveMode.NONE;
    this.rotateAmplitude = 0.1;
    this.panAmplitude = 0.01;
    this.scrollAmplitude = 0.01;
    this.x = 0;
    this.y = 0;
    this.pitch = 0.0;
    this.yaw = 0.0;
    this.distance = 10.0;
    this.target = vec3.create(0.0, 0.0, 0.0);
    this.eye = vec3.create(0.0, 0.0, 10.0);
    this.up = vec3.create(0.0, 1.0, 0.0);
    this.right = vec3.create(1.0, 0.0, 0.0);

    this.onKeyDown = this.onKeyDown.bind(this);
    this.onKeyUp = this.onKeyUp.bind(this);
    this.onMouseDown = this.onMouseDown.bind(this);
    this.onMouseUp = this.onMouseUp.bind(this);
    this.onMouseMove = this.onMouseMove.bind(this);

    document.addEventListener("contextmenu", (event: MouseEvent) => {
      if (this.controlled) {
        event.preventDefault();
      }
    });
    document.addEventListener("keydown", this.onKeyDown);
    document.addEventListener("keyup", this.onKeyUp);
    document.addEventListener("mousedown", this.onMouseDown);
    document.addEventListener("mouseup", this.onMouseUp);
    document.addEventListener("mousemove", this.onMouseMove);

    this.div = document.createElement("div");
    this.div.setAttribute(
      "style",
      "position: absolute; top: 10px; left: 10px; width: 200px; color: white; font-size: 16px; font-weight: bold;"
    );
    const body = document.querySelector("body");
    if (body) {
      body.appendChild(this.div);
    }
  }

  public get view(): Mat4 {
    return mat4.lookAt(this.eye, this.target, this.up);
  }

  public get position(): Vec3 {
    return this.eye;
  }

  private onKeyDown(event: KeyboardEvent): void {
    switch (event.key) {
      case "Alt":
        this.controlled = true;
        break;
      default:
        break;
    }
  }

  private onKeyUp(event: KeyboardEvent): void {
    switch (event.key) {
      case "Alt":
        this.controlled = false;
        this.div.innerText = "";
        break;
      default:
        break;
    }
  }

  private onMouseDown(event: MouseEvent): void {
    if (!this.controlled) {
      return;
    }

    this.x = event.clientX;
    this.y = event.clientY;

    switch (event.button) {
      case 0:
        this.moveMode = MoveMode.TUMBLE;
        this.div.innerText = `Tumble: Alt + LMB`;
        break;
      case 1:
        this.moveMode = MoveMode.TRACK;
        this.div.innerText = `Track: Alt + MMB`;
        break;
      case 2:
        this.moveMode = MoveMode.DOLLY;
        this.div.innerText = `Dolly: Alt + RMB`;
        break;
      default:
        this.moveMode = MoveMode.NONE;
        this.div.innerText = "";
        break;
    }
  }

  private onMouseUp(): void {
    if (!this.controlled) {
      return;
    }

    this.moveMode = MoveMode.NONE;
  }

  private onMouseMove(event: MouseEvent): void {
    if (!this.controlled || MoveMode.NONE === this.moveMode) {
      return;
    }

    const deltaX = event.clientX - this.x;
    this.x = event.clientX;

    const deltaY = event.clientY - this.y;
    this.y = event.clientY;

    switch (this.moveMode) {
      case MoveMode.TUMBLE:
        this.updateCameraRotation(deltaX, deltaY);
        break;
      case MoveMode.DOLLY:
        this.updateCameraDistance(deltaX, deltaY);
        break;
      case MoveMode.TRACK:
        this.updateCameraTarget(deltaX, deltaY);
        break;
      default:
        break;
    }
  }

  private updateCameraRotation(deltaX: number, deltaY: number): void {
    this.target = vec3.create(0.0, 0.0, 0.0);

    this.yaw = -deltaX * this.rotateAmplitude;
    this.pitch = -deltaY * this.rotateAmplitude;

    const quatYaw = quat.fromAxisAngle(this.up, utils.degToRad(this.yaw));

    const back = vec3.normalize(vec3.sub(this.eye, this.target));
    this.right = vec3.transformQuat(vec3.cross(this.up, back), quatYaw);
    const quatPitch = quat.fromAxisAngle(
      this.right,
      utils.degToRad(this.pitch)
    );

    const rotation = quat.mul(quatYaw, quatPitch);

    const newBack = vec3.transformQuat(back, rotation);
    this.up = vec3.normalize(vec3.cross(newBack, this.right));
    this.eye = vec3.add(this.target, vec3.mulScalar(newBack, this.distance));
  }

  private updateCameraDistance(deltaX: number, deltaY: number): void {
    this.distance -= (deltaX + deltaY) * this.scrollAmplitude;
    const back = vec3.normalize(vec3.sub(this.eye, this.target));
    this.eye = vec3.add(this.target, vec3.mulScalar(back, this.distance));
  }

  private updateCameraTarget(deltaX: number, deltaY: number) {
    let panOffset = vec3.create(0.0, 0.0, 0.0);
    panOffset = vec3.addScaled(
      panOffset,
      this.right,
      -deltaX * this.panAmplitude
    );
    panOffset = vec3.addScaled(panOffset, this.up, deltaY * this.panAmplitude);

    this.target = vec3.add(this.target, panOffset);
    this.eye = vec3.add(this.eye, panOffset);
  }
}
